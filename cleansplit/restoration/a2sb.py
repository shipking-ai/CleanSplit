"""NVIDIA A2SB (Audio-to-Audio Schrodinger Bridge) as a region-restricted CleanSplit restorer.

A2SB code and weights are NOT part of CleanSplit and are non-commercial:
  code    third_party/diffusion-audio-restoration (NVIDIA Source Code License-NC), imported at runtime
  weights models/a2sb/*.ckpt (NVIDIA OneWay Noncommercial License), SHA-256 pinned in models.checkpoints

What this module reproduces from NVIDIA's inference (A2SB_lightning_module_api.TimePartitionedPretrainedSTFTBridgeModel):
  * representation: STFT 2048/512/hann (center=True) -> [|X|^0.25, cos(phase), sin(phase)], DC bin dropped (3x1024xT)
  * corruption: masked cells replaced by 0.5 * N(0, 1) in all three channels (mask_with_noise, fill_noise_level 0.5)
  * sampler: t = linspace(1, 0.05, n_steps); per step pred_x0 = x_t - sigma_fwd(t) * f(x_t, emb(t)); pred_x0 is clamped to
    the observation outside the mask; x_{t-1} = SB posterior; outside the mask x_{t-1} = x_1 + sigma_SB(t-1) * noise
  * 2-split ensemble: network for t < 0.5 and for t >= 0.5 (beta_max 1.0), windows of 256 frames (= 130,560 samples)
  * inverse: magnitude ** 4, (cos, sin) projected onto the unit circle (equivalent to NVIDIA's SVD fix), DC = 0, iSTFT
Deviations (declared): mono per channel (A2SB is mono; stereo channels are generated independently, so inter-channel
coherence is NOT guaranteed); a fixed torch seed per region for reproducibility; TF-box masks instead of the time-slice
or high-band masks A2SB was trained on (out of distribution by construction, which is exactly what is being tested).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from ..audio.tf_edit import TFBox, box_mask
from ..models.checkpoints import A2SB_SHA256, A2SB_SIZE_BYTES, sha256_file
from .base import RestorationContext, RestorationProposal, Restorer

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPO = ROOT / "third_party" / "diffusion-audio-restoration"
DEFAULT_CKPT_DIR = ROOT / "models" / "a2sb"
WINDOW_FRAMES = 256
HOP = 512
N_FFT = 2048
SEGMENT = (WINDOW_FRAMES - 1) * HOP  # 130,560 samples -> exactly 256 frames with center=True


class A2SBModel:
    """Loads the 2-split ensemble once and samples masked 3x1024x256 windows."""

    def __init__(self, repo_dir=DEFAULT_REPO, ckpt_dir=DEFAULT_CKPT_DIR, precision="fp16", device="cuda", verify=True):
        import torch
        import yaml

        repo_dir, ckpt_dir = Path(repo_dir), Path(ckpt_dir)
        if not (repo_dir / "networks.py").is_file():
            raise FileNotFoundError(f"A2SB code not found at {repo_dir}")
        if str(repo_dir) not in sys.path:
            sys.path.insert(0, str(repo_dir))
        from diffusion import Diffusion
        from networks import AttnUNetF, SinusoidalTemporalEmbedding

        cfg = yaml.safe_load(open(repo_dir / "configs" / "ensemble_2split_sampling.yaml"))["model"]
        init = cfg["vf_model"]["init_args"]
        self.t_cutoff = float(cfg["t_cutoffs"][0])
        names = ["A2SB_twosplit_0.0_0.5_release.ckpt", "A2SB_twosplit_0.5_1.0_release.ckpt"]
        self.torch = torch
        self.device = device
        self.dtype = torch.float16 if precision == "fp16" else torch.float32
        self.precision = precision
        self.nets = []
        self.checkpoints = {}
        for name in names:
            p = ckpt_dir / name
            if not p.is_file():
                raise FileNotFoundError(f"A2SB checkpoint missing: {p}")
            if verify:
                if p.stat().st_size != A2SB_SIZE_BYTES or sha256_file(p) != A2SB_SHA256[name]:
                    raise ValueError(f"A2SB checkpoint {p} failed size/SHA-256 verification")
            sd = torch.load(p, map_location="cpu", weights_only=True)["state_dict"]
            net = AttnUNetF(**init)
            net.load_state_dict({k[len("vf_model."):]: v for k, v in sd.items() if k.startswith("vf_model.")}, strict=True)
            self.nets.append(net.eval().to(device, dtype=self.dtype))
            self.checkpoints[name] = A2SB_SHA256[name]
            del sd
        self.ddpm = Diffusion(beta_max=float(cfg["beta_max"]))
        self.t_to_emb = SinusoidalTemporalEmbedding(n_bands=int(cfg["n_timestep_channels"]) // 2, min_freq=0.5).to(device)
        self.window = torch.hann_window(N_FFT, device=device)

    # ---- representation ----
    def to_repr(self, wav):
        """wav (B, N) float32 tensor -> (B, 3, 1024, T)."""
        torch = self.torch
        X = torch.stft(wav, N_FFT, HOP, N_FFT, self.window, center=True, pad_mode="reflect", return_complex=True)
        mag = X.abs()
        ph = torch.angle(X)
        r = torch.stack([mag.pow(0.25), torch.cos(ph), torch.sin(ph)], dim=1)
        return r[:, :, 1:, :]

    def from_repr(self, r, length):
        torch = self.torch
        x = r[:, 0]
        mag = x * x.abs().pow(4) / (x.abs() + 1e-9)  # NVIDIA PowerScaleSpectrogram(power=4): sign kept (phase flip)
        cs = r[:, 1:3]
        norm = cs.norm(dim=1).clamp(min=1e-8)
        cos, sin = cs[:, 0] / norm, cs[:, 1] / norm
        X = torch.complex(mag * cos, mag * sin)
        X = torch.cat([torch.zeros_like(X[:, :1]), X], dim=1)
        return torch.istft(X, N_FFT, HOP, N_FFT, self.window, center=True, length=length)

    # ---- sampler (faithful to NVIDIA ddpm_sample for a single 256-frame window) ----
    def sample(self, x_1, mask, n_steps: int):
        torch = self.torch
        t_steps = torch.linspace(1.0, 0.05, n_steps, device=self.device)
        x_t = x_1.clone()
        pred_x0 = x_1
        B = x_1.shape[0]
        for i in range(n_steps - 1):
            t, t_prev = t_steps[i : i + 1], t_steps[i + 1 : i + 2]
            net = self.nets[1] if float(t) >= self.t_cutoff else self.nets[0]
            emb = self.t_to_emb(t).repeat(B, 1)
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.dtype == torch.float16):
                vf = net(x_t.to(self.dtype), emb.to(self.dtype)).float()
            pred_x0 = self.ddpm.get_pred_x0(t, x_t, vf)
            pred_x0 = pred_x0 * mask + (1 - mask) * x_1
            x_t = self.ddpm.p_posterior(t_prev, t, x_t, pred_x0, ot_ode=False)
            std_sb = self.ddpm.get_std_t(t_prev)
            x_t = (1.0 - mask) * (x_1 + std_sb * torch.randn_like(x_1)) + mask * x_t
        return pred_x0

    def inpaint(self, wav: np.ndarray, cell_mask: np.ndarray, n_steps: int, seed: int) -> np.ndarray:
        """wav (C, SEGMENT) float, cell_mask (1024, 256) bool (DC dropped) -> (C, SEGMENT) float64 regenerated audio."""
        torch = self.torch
        assert wav.shape[-1] == SEGMENT, wav.shape
        torch.manual_seed(seed)
        with torch.inference_mode():
            x = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32)).to(self.device)
            x_0 = self.to_repr(x)
            m = torch.from_numpy(cell_mask.astype(np.float32)).to(self.device)[None, None]
            x_1 = x_0 * (1 - m) + m * torch.randn_like(x_0) * 0.5
            pred = self.sample(x_1, m, n_steps)
            out = self.from_repr(pred.float(), SEGMENT)
        return out.double().cpu().numpy()


class A2SBConsistentRestorer(Restorer):
    """A2SB as a *prior* for how the mixture is split, not as the output.

    Raw A2SB inpainting regenerates a stem from its own context; the result is unconstrained by the mixture, so the
    sum of stems departs from O_ref (measured: +10..+43 dB local mixture error, all rejected). Here A2SB's inpainted
    magnitude for each target stem replaces that stem's current power estimate inside the box, and ALL stems are
    re-derived from the mixture with smoothed Wiener weights: S_k' = w_k * O_ref, w_k = P_k / sum_j P_j.
    Sum of stems equals O_ref in the box by construction; only the allocation comes from the generative model.
    """

    name = "a2sb_consistent"
    generative = True

    def __init__(self, **kw):
        self.inner = A2SBRestorer(**kw)

    def describe(self) -> dict:
        d = self.inner.describe()
        d.update(name=self.name, doc=(self.__doc__ or "").strip().splitlines()[0])
        return d

    def restore(self, stem, regions, context: RestorationContext):
        from scipy.ndimage import uniform_filter

        from ..audio.stft import istft, stft

        g = context.grid
        out = []
        for prop in self.inner.restore(stem, regions, context):
            if not prop.new_audio:
                out.append(prop)
                continue
            a, b = prop.excerpt
            O = stft(context.mixture[..., a:b].astype(np.float64), g, dtype=np.complex128)
            P = {}
            for k, s in context.stems.items():
                src = prop.new_audio[k] if k in prop.new_audio else s[..., a:b]
                S = stft(np.asarray(src, dtype=np.float64), g, dtype=np.complex128)
                P[k] = uniform_filter((np.abs(S) ** 2).sum(axis=0), size=(5, 9), mode="nearest")
            tot = sum(P.values()) + 1e-20
            new = {k: istft((P[k] / tot) * O, g, b - a) for k in context.stems}
            out.append(RestorationProposal(prop.region, new, (a, b), self.name, prop.notes + ["allocation from A2SB prior, stems = Wiener split of O_ref"]))
        return out


class A2SBRestorer(Restorer):
    """Generative inpainting of the flagged TF box in each affected stem with NVIDIA A2SB (non-commercial)."""

    name = "a2sb"
    generative = True

    def __init__(self, n_steps: int = 50, precision: str = "fp16", max_target_stems: int = 2, min_stem_share_db: float = -20.0, seed: int = 0, model: A2SBModel | None = None):
        self.n_steps = n_steps
        self.precision = precision
        self.max_target_stems = max_target_stems
        self.min_stem_share_db = min_stem_share_db
        self.seed = seed
        self._model = model

    @property
    def model(self) -> A2SBModel:
        if self._model is None:
            self._model = A2SBModel(precision=self.precision)
        return self._model

    def describe(self) -> dict:
        d = super().describe()
        d.update(n_steps=self.n_steps, precision=self.precision, license="NVIDIA OneWay Noncommercial (weights), NVIDIA Source Code License-NC (code)")
        return d

    def _targets(self, stem, box, ctx: RestorationContext, a, b):
        if stem != "mixture":
            return [stem]
        g = ctx.grid
        f0, f1 = g.hz_to_bin(box.freq_low_hz), g.hz_to_bin(box.freq_high_hz) + 1
        from ..audio.stft import stft

        energies = {}
        for k, s in ctx.stems.items():
            S = stft(s[..., a:b].astype(np.float64), g, dtype=np.complex128)
            t0 = max(0, g.time_to_frame(box.start_s) - a // g.hop)
            t1 = min(S.shape[-1], g.time_to_frame(box.end_s) - a // g.hop + 1)
            energies[k] = float(np.sum(np.abs(S[:, f0:f1, t0:t1]) ** 2))
        top = max(energies.values()) if energies else 0.0
        if top <= 0:
            return []
        ranked = sorted(energies, key=lambda k: -energies[k])
        return [k for k in ranked if 10 * np.log10(max(energies[k], 1e-30) / top) >= self.min_stem_share_db][: self.max_target_stems]

    def restore(self, stem, regions, context: RestorationContext):
        out = []
        n = context.mixture.shape[-1]
        g = context.grid
        for r in regions:
            box = TFBox(r.start_s, r.end_s, r.freq_low_hz, r.freq_high_hz)
            dur_frames = g.time_to_frame(r.end_s) - g.time_to_frame(r.start_s)
            if dur_frames > WINDOW_FRAMES - 32:
                out.append(RestorationProposal(r, {}, (0, 0), self.name, [f"region spans {dur_frames} frames > one A2SB window; skipped"]))
                continue
            center = 0.5 * (r.start_s + r.end_s) * g.sample_rate
            a = int(round((center - SEGMENT / 2) / HOP)) * HOP
            a = max(0, min(a, (n - SEGMENT) // HOP * HOP))
            b = a + SEGMENT
            if b > n:
                out.append(RestorationProposal(r, {}, (0, 0), self.name, ["signal shorter than one A2SB window; skipped"]))
                continue
            mask = box_mask(g, WINDOW_FRAMES, box, frame_offset=a // HOP, taper_bins=0, taper_frames=0)[1:] > 0
            targets = self._targets(stem, box, context, a, b)
            new = {}
            for i, k in enumerate(targets):
                new[k] = self.model.inpaint(context.stems[k][..., a:b], mask, self.n_steps, seed=self.seed + i)
            out.append(RestorationProposal(r, new, (a, b), self.name, [f"A2SB {self.n_steps} steps {self.precision}, targets {targets}"]))
        return out
