"""MDX23C InstVoc HQ: a vocals/instrumental model with a different architecture from both BS-RoFormers.

Why it is here: the only measured gain in this project is ensembling (docs/04_results.md section 7), and ensembles
gain from models whose errors differ. SW and ep317 are both BS-RoFormers; MDX23C is a convolutional TFC-TDF U-Net
on an 8192-point STFT, so its mistakes should be less correlated with theirs. Whether that turns into a better
vocal stem is measured, not assumed (tools/ensemble_experiment.py, docs/04 section 10).

Inference follows MSST's demix exactly as ``roformer.py`` does: native chunk (261,120 samples), step = chunk /
num_overlap, reflect padding of chunk - step on both sides, the same linear fade over chunk // 10. The model's own
default is num_overlap = 8; that is kept unless the caller overrides it.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from ..models import checkpoints
from ..models.device import resolve_device
from .base import SeparationResult, Separator
from .roformer import fade_window, load_msst_config

log = logging.getLogger(__name__)


class AttrDict(dict):
    """Nested dict with attribute access, standing in for ml_collections.ConfigDict in the vendored model."""

    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return AttrDict(v) if isinstance(v, dict) else v


class MDX23CSeparator(Separator):
    name = "mdx23c_instvoc_hq"

    def __init__(self, checkpoint: str | Path | None = None, device: str = "auto", num_overlap: int | None = None,
                 verify_hash: bool = True):
        self.ck = checkpoints.MDX23C_INSTVOC_HQ
        self.ckpt_path, self.cfg_path = checkpoints.find_checkpoint(self.ck, checkpoint)
        self.cfg = load_msst_config(self.cfg_path)
        self.sample_rate = int(self.cfg["audio"]["sample_rate"])
        self.channels = int(self.cfg["audio"]["num_channels"])
        self.instruments = [s.lower() for s in self.cfg["training"]["instruments"]]  # ["vocals", "instrumental"]
        self.stems = tuple(self.instruments)
        self.chunk_size = int(self.cfg["audio"]["chunk_size"])
        self.num_overlap = int(num_overlap or self.cfg.get("inference", {}).get("num_overlap", 4))
        if self.num_overlap < 1:
            raise ValueError("num_overlap must be >= 1")
        self.requested_device = device
        self.verify_hash = verify_hash
        self._model = None
        self._device = None
        self._sha256 = None

    def cache_key(self) -> dict:
        return {**self.describe(), "checkpoint_sha256": self.ck.sha256, "chunk_size": self.chunk_size,
                "num_overlap": self.num_overlap}

    def _load(self):
        if self._model is not None:
            return
        import torch

        from .mdx23c.model import TFC_TDF_net

        if self.verify_hash:
            self._sha256 = checkpoints.verify(self.ck, self.ckpt_path)
        self._device = resolve_device(self.requested_device)
        state = torch.load(self.ckpt_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model = TFC_TDF_net(AttrDict(self.cfg))
        model.load_state_dict(state, strict=True)
        self._model = model.eval().to(self._device)

    def separate(self, mixture: np.ndarray, sample_rate: int) -> SeparationResult:
        if sample_rate != self.sample_rate:
            raise ValueError(f"separator expects {self.sample_rate} Hz, got {sample_rate}")
        if mixture.shape[0] != self.channels:
            raise ValueError(f"separator expects {self.channels} channels, got {mixture.shape[0]}")
        import torch

        self._load()
        t0 = time.time()
        n = mixture.shape[-1]
        chunk = self.chunk_size
        step = max(1, chunk // self.num_overlap)
        border = chunk - step
        padded = np.pad(mixture.astype(np.float64), ((0, 0), (border, border)), mode="reflect" if n > border else "constant")
        starts = list(range(0, max(padded.shape[-1] - border, 1), step))
        total = padded.shape[-1]
        acc = np.zeros((len(self.instruments), self.channels, total), dtype=np.float64)
        wsum = np.zeros(total, dtype=np.float64)
        fade = chunk // 10 if self.num_overlap > 1 else 0
        if self._device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        for idx, s in enumerate(starts):
            part = padded[:, s:s + chunk]
            valid = part.shape[-1]
            if valid < chunk:
                part = np.pad(part, ((0, 0), (0, chunk - valid)))
            x = torch.from_numpy(np.ascontiguousarray(part, dtype=np.float32))[None].to(self._device)
            with torch.inference_mode():
                y = self._model(x)
            y = y[0].float().cpu().numpy().astype(np.float64)[..., :valid]
            if not np.all(np.isfinite(y)):
                raise FloatingPointError(f"non-finite model output in chunk {idx}")
            w = fade_window(chunk, fade, fade_in=idx > 0, fade_out=idx < len(starts) - 1)[:valid]
            acc[..., s:s + valid] += y * w
            wsum[s:s + valid] += w
        if np.any(wsum <= 0):
            raise RuntimeError("overlap-add weight vanished; chunk/overlap configuration is invalid")
        out = (acc / wsum)[..., border:border + n]
        peak_mb = torch.cuda.max_memory_allocated() / 2**20 if self._device.startswith("cuda") else None
        stems = {name: out[i].astype(np.float32) for i, name in enumerate(self.instruments)}
        meta = {
            "checkpoint": str(self.ckpt_path),
            "checkpoint_sha256": self._sha256 or "not verified",
            "config": str(self.cfg_path),
            "provenance": self.ck.provenance,
            "license_note": self.ck.license_note,
            "device": self._device,
            "chunk_size": chunk,
            "num_overlap": self.num_overlap,
            "zero_dc": False,
            "seconds": round(time.time() - t0, 2),
            "peak_vram_mb": peak_mb,
        }
        return SeparationResult(stems, self.sample_rate, self.name, meta, [s - border for s in starts], chunk)
