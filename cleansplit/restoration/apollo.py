"""Apollo (JusperLee) as a CleanSplit restorer: band-sequence codec-damage repair.

Apollo is the one generative candidate aimed at information the recording really lost — quantisation noise and the
band that an MP3/AAC encoder threw away — rather than at energy the separator put in the wrong stem. That distinction
matters: docs/04_results.md section 5 measures ~97% of the separator's error as misallocation, which no generative
prior can recover, so a restorer is only worth testing here if its target is genuine codec damage.

Code: third_party/apollo (vendored from github.com/JusperLee/Apollo, CC BY-SA 4.0, pinned commit in its header).
Weights: whatever the user already has in UVR's models/Apollo_Models (CC BY-SA 4.0), SHA-256 pinned in
models.checkpoints.APOLLO. CleanSplit downloads neither.

Two ways to use it, both measured in docs/04_results.md:
  * :class:`ApolloModel` - whole-signal enhancement, exactly the upstream inference path. Unrestricted: every sample
    changes. This is what UVR does, and it is the honest upper bound on what Apollo can do.
  * :class:`ApolloRestorer` - the CleanSplit way: Apollo's output is a *proposal* that the engine projects onto the
    flagged region's TF box, so audio outside the box stays bit-identical, and the mixture-consistency gate still
    decides. A whole-stem rewrite is never accepted on the strength of sounding smoother.

Deviations from upstream inference.py (declared): chunking is float32 on the same device with the same linear
crossfade and the same "pad each chunk with real audio, never silence" rule; CleanSplit works in (C, N) float64
arrays and converts once per call.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from ..audio.tf_edit import TFBox, excerpt_bounds
from ..models.checkpoints import find_apollo, sha256_file
from .base import RestorationContext, RestorationProposal, Restorer

ROOT = Path(__file__).resolve().parents[2]
VENDORED = ROOT / "third_party"
SAMPLE_RATE = 44100


class ApolloModel:
    """Loads one Apollo checkpoint and enhances whole signals with chunked overlap-add."""

    def __init__(self, checkpoint: str | Path | None = None, variant: str = "mp3_enhancer",
                 device: str = "cuda", verify: bool = True):
        import sys

        import torch

        path, known = find_apollo(variant, checkpoint)
        self.checkpoint = path
        self.variant = variant
        self.license_note = known.license_note
        self.sha256 = ""
        if verify:
            size = path.stat().st_size
            if size != known.size_bytes:
                raise ValueError(f"{path}: size {size} != pinned {known.size_bytes}")
            self.sha256 = sha256_file(path)
            if self.sha256 != known.sha256:
                raise ValueError(f"{path}: sha256 {self.sha256} != pinned {known.sha256}")

        if str(VENDORED) not in sys.path:
            sys.path.insert(0, str(VENDORED))
        from apollo import Apollo

        # Both released checkpoints are the configs/apollo.yaml shape; the state dict is checked against it below.
        self.torch = torch
        self.device = device
        self.model = Apollo(sr=SAMPLE_RATE, win=20, feature_dim=256, layer=6)
        state = self._load(path)
        state = state.get("state_dict", state) if isinstance(state, dict) else state
        self.model.load_state_dict(state, strict=True)
        self.model.to(device).eval()
        self.params_m = sum(p.numel() for p in self.model.parameters()) / 1e6

    @staticmethod
    def _load(path: Path):
        """weights_only load. Both checkpoints carry a torch version stamp, and the MP3 enhancer also carries
        omegaconf hyper-parameters, so those types are allowlisted; no checkpoint code is ever executed."""
        import collections
        import typing

        import torch
        from torch.torch_version import TorchVersion

        allowed = [TorchVersion, collections.defaultdict, int, dict, list, typing.Any]
        try:
            from omegaconf.base import ContainerMetadata, Metadata
            from omegaconf.dictconfig import DictConfig
            from omegaconf.listconfig import ListConfig
            from omegaconf.nodes import AnyNode

            allowed += [Metadata, ContainerMetadata, DictConfig, ListConfig, AnyNode]
        except ImportError:  # the tensors still load; only the unused hyper-parameter blob needs omegaconf
            pass
        with torch.serialization.safe_globals(allowed):
            return torch.load(path, map_location="cpu", weights_only=True)

    def enhance(self, x: np.ndarray, chunk_s: float = 5.0, overlap_s: float = 1.0, pad_s: float = 0.5) -> np.ndarray:
        """(C, N) float -> (C, N) float32.

        VRAM grows linearly with chunk length (measured ~410 MB per second of stereo audio), so the default 5 s
        chunk peaks near 2.1 GB and runs at 5x realtime on an 8 GB card; 20 s spills into shared memory and is
        50x slower. See docs/02_hardware_measurements.md.
        """
        torch = self.torch
        audio = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).unsqueeze(0)  # (1, C, N)
        n = audio.shape[-1]
        chunk = int(round(chunk_s * SAMPLE_RATE))
        overlap = int(round(overlap_s * SAMPLE_RATE))
        pad = int(round(pad_s * SAMPLE_RATE))
        with torch.inference_mode():
            if n <= chunk:
                out = self.model(audio.to(self.device)).detach().cpu()
            else:
                out_sum = torch.zeros_like(audio)
                w_sum = torch.zeros((1, 1, n), dtype=audio.dtype)
                hop = chunk - overlap
                starts = [0]
                while starts[-1] + chunk < n:
                    starts.append(starts[-1] + hop)
                padded = chunk + 2 * pad
                for i, start in enumerate(starts):
                    end = min(start + chunk, n)
                    valid = end - start
                    # Context on both sides comes from real audio; at the file's own edges the window slides
                    # inward instead of padding with silence, which upstream measured as degrading the edges.
                    p0 = max(0, start - pad)
                    if pad and end == n:
                        p0 = max(0, n - padded)
                    seg = audio[..., p0:p0 + padded]
                    if seg.shape[-1] < padded:
                        seg = torch.nn.functional.pad(seg, (0, padded - seg.shape[-1]))
                    y = self.model(seg.to(self.device)).detach().cpu()
                    y = y[..., start - p0:start - p0 + valid]
                    w = torch.ones(valid, dtype=audio.dtype)
                    fade = min(overlap, valid)
                    if fade:
                        ramp = torch.linspace(0.0, 1.0, fade, dtype=audio.dtype)
                        if i > 0:
                            w[:fade] = ramp
                        if i < len(starts) - 1:
                            w[-fade:] = torch.flip(ramp, dims=(0,))
                    w = w.view(1, 1, -1)
                    out_sum[..., start:end] += y * w
                    w_sum[..., start:end] += w
                if torch.any(w_sum <= 0):
                    raise RuntimeError("overlap-add left uncovered samples")
                out = out_sum / w_sum
        return out.squeeze(0).numpy().astype(np.float32)

    def describe(self) -> dict:
        return {"variant": self.variant, "checkpoint": str(self.checkpoint), "sha256": self.sha256,
                "params_m": round(self.params_m, 3), "license": self.license_note}


class ApolloRestorer(Restorer):
    """Apollo's enhancement of the whole stem, applied only inside the flagged region's TF box."""

    name = "apollo"
    generative = True

    def __init__(self, variant: str = "mp3_enhancer", device: str = "cuda", checkpoint=None,
                 max_target_stems: int = 2, min_stem_share_db: float = -20.0,
                 model: ApolloModel | None = None):
        self.variant = variant
        self.max_target_stems = max_target_stems
        self.min_stem_share_db = min_stem_share_db
        self._model = model
        self._device = device
        self._checkpoint = checkpoint
        self._cache: dict[str, np.ndarray] = {}

    @property
    def model(self) -> ApolloModel:
        if self._model is None:
            self._model = ApolloModel(self._checkpoint, variant=self.variant, device=self._device)
        return self._model

    def describe(self) -> dict:
        d = super().describe()
        d.update(variant=self.variant, max_target_stems=self.max_target_stems,
                 min_stem_share_db=self.min_stem_share_db, license="CC BY-SA 4.0 (code and weights)")
        return d

    def reset(self) -> None:
        """Drop the per-song enhanced-stem cache (call between songs)."""
        self._cache.clear()

    def _enhanced(self, stem: str, audio: np.ndarray) -> np.ndarray:
        """Apollo is a whole-signal model, so enhance the stem once per song and reuse it for every region.

        The key hashes the whole array, not a prefix: a later restoration pass may have changed the stem anywhere,
        and returning a stale enhancement would silently score the wrong audio. Hashing 80 MB costs ~0.1 s against
        a ~60 s model pass.
        """
        x = np.ascontiguousarray(audio)
        key = f"{stem}:{x.shape}:{hashlib.blake2b(x.tobytes(), digest_size=16).hexdigest()}"
        if key not in self._cache:
            self._cache[key] = self.model.enhance(x)
        return self._cache[key]

    def _targets(self, stem: str, box: TFBox, ctx: RestorationContext, a: int, b: int) -> list[str]:
        """For a per-stem region, that stem. For a mixture-level region, the stems that actually own the box."""
        if stem != "mixture":
            return [stem] if stem in ctx.stems else []
        from ..audio.stft import stft

        g = ctx.grid
        f0, f1 = g.hz_to_bin(box.freq_low_hz), g.hz_to_bin(box.freq_high_hz) + 1
        energies = {}
        for k, s in ctx.stems.items():
            S = stft(s[..., a:b].astype(np.float64), g, dtype=np.complex128)
            t0 = max(0, g.time_to_frame(box.start_s) - a // g.hop)
            t1 = min(S.shape[-1], g.time_to_frame(box.end_s) - a // g.hop + 1)
            energies[k] = float(np.sum(np.abs(S[:, f0:f1, t0:t1]) ** 2))
        top = max(energies.values(), default=0.0)
        if top <= 0:
            return []
        ranked = sorted(energies, key=lambda k: -energies[k])
        return [k for k in ranked
                if 10 * np.log10(max(energies[k], 1e-30) / top) >= self.min_stem_share_db][: self.max_target_stems]

    def restore(self, stem: str, regions, context: RestorationContext) -> list[RestorationProposal]:
        proposals = []
        n = context.mixture.shape[-1]
        for region in regions:
            box = TFBox(region.start_s, region.end_s, region.freq_low_hz, region.freq_high_hz)
            a, b = excerpt_bounds(n, context.grid, box)
            targets = self._targets(stem, box, context, a, b)
            new = {k: self._enhanced(k, np.asarray(context.stems[k], dtype=np.float64))[:, a:b].astype(np.float64)
                   for k in targets}
            proposals.append(RestorationProposal(
                region=region,
                new_audio=new,
                excerpt=(a, b),
                restorer=self.name,
                notes=[f"apollo:{self.variant}, targets {targets}"],
            ))
        return proposals
