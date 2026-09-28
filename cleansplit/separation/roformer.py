"""BS-RoFormer SW six-stem separator (local checkpoint, chunked overlap-add inference).

Overlap-add is done here (not in a third-party helper) because the reconstruction analysis
depends on its exact properties:

* Every chunk yields all stems from the *same* forward pass, and all stems are blended with
  the *same* weights, so ``sum(stems)`` equals the blend of per-chunk ``sum(stems)``. Chunking
  therefore does not by itself break mixture consistency; any residual comes from the masks.
* Chunk start positions are returned so seam artifacts can be tested for explicitly.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import yaml

from ..models import checkpoints
from ..models.device import resolve_device
from .base import SeparationResult, Separator

log = logging.getLogger(__name__)


class _TupleLoader(yaml.SafeLoader):
    pass


_TupleLoader.add_constructor("tag:yaml.org,2002:python/tuple", lambda loader, node: tuple(loader.construct_sequence(node)))


def load_msst_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=_TupleLoader)


def fade_window(chunk: int, fade: int, fade_in: bool, fade_out: bool) -> np.ndarray:
    w = np.ones(chunk, dtype=np.float64)
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade + 2, dtype=np.float64)[1:-1]  # strictly inside (0, 1)
        if fade_in:
            w[:fade] = ramp
        if fade_out:
            w[-fade:] = ramp[::-1]
    return w


class BSRoformerSeparator(Separator):
    name = "bs_roformer_sw"

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        model: str = "bs_roformer_sw",
        device: str = "auto",
        chunk_size: int | None = None,
        num_overlap: int = 4,  # best measured (docs/04 section 14.5); 2 is ~2x faster and sub-0.1 dB worse
        fp16: bool = False,
        verify_hash: bool = True,
        tta: bool = False,
    ):
        """tta: average SW over the original, channel-swapped and polarity-inverted inputs (outputs mapped back).
        Measured on real music with known stems (docs/04_results.md): vocals +0.08/+0.32 dB, other +0.04/+0.25 dB,
        drums and bass within +-0.01 dB; 3x separation time. Averaging is linear, so mixture consistency is unchanged."""
        self.tta = bool(tta)
        if model not in checkpoints.KNOWN:
            raise KeyError(f"unknown BS-RoFormer model '{model}'; known: {sorted(checkpoints.KNOWN)}")
        self.ck = checkpoints.KNOWN[model]
        self.name = model
        self.ckpt_path, self.cfg_path = checkpoints.find_checkpoint(self.ck, checkpoint)
        self.cfg = load_msst_config(self.cfg_path)
        self.sample_rate = int(self.cfg["audio"]["sample_rate"])
        self.channels = int(self.cfg["audio"]["num_channels"])
        self.instruments = [s.lower() for s in self.cfg["training"]["instruments"]]
        # Single-target models (num_stems == 1) predict only the target; the complement is mixture - target (MSST/UVR).
        self.target = (self.cfg["training"].get("target_instrument") or "").lower() or None
        self.model_stems = [self.target] if int(self.cfg["model"].get("num_stems", 1)) == 1 and self.target else self.instruments
        self.stems = tuple(self.instruments)
        self.chunk_size = int(chunk_size or self.cfg["audio"]["chunk_size"])
        if num_overlap < 1:
            raise ValueError("num_overlap must be >= 1")
        self.num_overlap = int(num_overlap)
        self.fp16 = bool(fp16)
        self.requested_device = device
        self.verify_hash = verify_hash
        self._model = None
        self._device = None
        self._sha256 = None

    def cache_key(self) -> dict:
        return {
            **self.describe(),
            "checkpoint_sha256": self.ck.sha256,
            "chunk_size": self.chunk_size,
            "num_overlap": self.num_overlap,
            "fp16": self.fp16,
            "tta": self.tta,
        }

    def _load(self):
        if self._model is not None:
            return
        import torch

        from .bs_roformer.model import BSRoformer

        if self.verify_hash:
            self._sha256 = checkpoints.verify(self.ck, self.ckpt_path)
        self._device = resolve_device(self.requested_device)
        state = torch.load(self.ckpt_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model = BSRoformer(**self.cfg["model"])
        model.load_state_dict(state, strict=True)
        if model.num_stems != len(self.model_stems):
            raise ValueError(f"config num_stems {model.num_stems} does not match predicted stems {self.model_stems}")
        self._model = model.eval().to(self._device)

    def _run_chunks(self, padded, starts, chunk, step):
        import torch

        n_stems = len(self.model_stems)
        total = padded.shape[-1]
        acc = np.zeros((n_stems, self.channels, total), dtype=np.float64)
        wsum = np.zeros(total, dtype=np.float64)
        fade = chunk // 10 if self.num_overlap > 1 else 0
        use_amp = self.fp16 and self._device.startswith("cuda")
        for idx, s in enumerate(starts):
            part = padded[:, s : s + chunk]
            valid = part.shape[-1]
            if valid < chunk:
                part = np.pad(part, ((0, 0), (0, chunk - valid)))
            x = torch.from_numpy(np.ascontiguousarray(part, dtype=np.float32))[None].to(self._device)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                y = self._model(x)
            y = y[0].float().cpu().numpy().astype(np.float64)[..., :valid]
            if not np.all(np.isfinite(y)):
                raise FloatingPointError(f"non-finite model output in chunk {idx} (fp16={self.fp16})")
            w = fade_window(chunk, fade, fade_in=idx > 0, fade_out=idx < len(starts) - 1)[:valid]
            acc[..., s : s + valid] += y * w
            wsum[s : s + valid] += w
        if np.any(wsum <= 0):
            raise RuntimeError("overlap-add weight vanished; chunk/overlap configuration is invalid")
        return acc / wsum

    def separate(self, mixture: np.ndarray, sample_rate: int) -> SeparationResult:
        if not self.tta:
            return self._separate_once(mixture, sample_rate)
        passes = [
            ("original", mixture, lambda y: y),
            ("channel_swap", mixture[::-1].copy(), lambda y: y[::-1]),
            ("polarity_invert", -mixture, lambda y: -y),
        ]
        acc, first, seconds = None, None, 0.0
        for label, x, back in passes:
            res = self._separate_once(x, sample_rate)
            seconds += res.metadata["seconds"]
            mapped = {k: back(v.astype(np.float64)) for k, v in res.stems.items()}
            acc = mapped if acc is None else {k: acc[k] + mapped[k] for k in acc}
            first = first or res
        stems = {k: (v / len(passes)).astype(np.float32) for k, v in acc.items()}
        meta = {**first.metadata, "tta": [p[0] for p in passes], "seconds": round(seconds, 2)}
        return SeparationResult(stems, first.sample_rate, self.name, meta, first.chunk_starts, first.chunk_size)

    def _separate_once(self, mixture: np.ndarray, sample_rate: int) -> SeparationResult:
        if sample_rate != self.sample_rate:
            raise ValueError(f"separator expects {self.sample_rate} Hz, got {sample_rate}")
        if mixture.shape[0] != self.channels:
            raise ValueError(f"separator expects {self.channels} channels, got {mixture.shape[0]}")
        import torch

        self._load()
        n = mixture.shape[-1]
        chunk = self.chunk_size
        t0 = time.time()
        attempts = []
        while True:
            step = max(1, chunk // self.num_overlap)
            border = chunk - step
            mode = "reflect" if n > border else "constant"
            padded = np.pad(mixture.astype(np.float64), ((0, 0), (border, border)), mode=mode)
            starts = list(range(0, max(padded.shape[-1] - border, 1), step))
            try:
                if self._device.startswith("cuda"):
                    torch.cuda.reset_peak_memory_stats()
                out = self._run_chunks(padded, starts, chunk, step)
                break
            except torch.OutOfMemoryError:
                attempts.append(chunk)
                torch.cuda.empty_cache()
                if chunk <= 44100 * 4:
                    raise
                chunk = int(chunk * 0.6)
                log.warning("CUDA OOM; retrying with chunk_size=%d (differs from model's native context)", chunk)
        out = out[..., border : border + n]
        peak_mb = torch.cuda.max_memory_allocated() / 2**20 if self._device.startswith("cuda") else None
        stems = {name: out[i].astype(np.float32) for i, name in enumerate(self.model_stems)}
        if self.model_stems != self.instruments:
            for other in self.instruments:
                if other not in stems:
                    stems[other] = (mixture.astype(np.float64) - out[0]).astype(np.float32)
        meta = {
            "checkpoint": str(self.ckpt_path),
            "checkpoint_sha256": self._sha256 or "not verified",
            "config": str(self.cfg_path),
            "provenance": self.ck.provenance,
            "license_note": self.ck.license_note,
            "device": self._device,
            "fp16": self.fp16,
            "chunk_size": chunk,
            "native_chunk_size": int(self.cfg["audio"]["chunk_size"]),
            "oom_retries_from": attempts,
            "num_overlap": self.num_overlap,
            "zero_dc": bool(self.cfg["model"].get("zero_dc", True)),
            "stft": {
                "n_fft": self.cfg["model"]["stft_n_fft"],
                "hop": self.cfg["model"]["stft_hop_length"],
                "win": self.cfg["model"]["stft_win_length"],
            },
            "seconds": round(time.time() - t0, 2),
            "peak_vram_mb": peak_mb,
            "model_stem_order": self.model_stems,
            "complement_stems": [k for k in self.instruments if k not in self.model_stems],
        }
        chunk_starts = [s - border for s in starts]
        return SeparationResult(stems, self.sample_rate, self.name, meta, chunk_starts, chunk)
