"""Meta HTDemucs (fine-tuned bag, 4 sources) from the local UVR model folder, via the `demucs` package (MIT).

Follows demucs' own separate.py: normalise by the mono reference's mean/std, apply_model(split=True, overlap=0.25),
de-normalise. Stems are NOT constrained to sum to the mixture (Demucs has no such constraint); the reconstruction
analysis measures that rather than assuming it.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..models import checkpoints
from ..models.device import resolve_device
from .base import SeparationResult, Separator


def uvr_demucs_repo() -> Path:
    for root in checkpoints.uvr_models_dirs():
        p = root / "Demucs_Models" / "v3_v4_repo"
        if p.is_dir():
            return p
    raise FileNotFoundError("UVR Demucs v3_v4_repo folder not found")


class HTDemucsSeparator(Separator):
    name = "htdemucs_ft"
    sample_rate = 44100
    channels = 2
    stems = ("drums", "bass", "other", "vocals")

    def __init__(self, repo: str | Path | None = None, device: str = "auto", shifts: int = 1, overlap: float = 0.25, verify_hash: bool = True):
        self.repo = Path(repo) if repo else uvr_demucs_repo()
        self.requested_device = device
        self.shifts = shifts
        self.overlap = overlap
        self.verify_hash = verify_hash
        self._model = None

    def cache_key(self) -> dict:
        return {**self.describe(), "weights_sha256": checkpoints.HTDEMUCS_FT_SHA256, "shifts": self.shifts, "overlap": self.overlap}

    def _load(self):
        if self._model is not None:
            return
        from demucs.pretrained import get_model

        if self.verify_hash:
            for fname, digest in checkpoints.HTDEMUCS_FT_SHA256.items():
                if checkpoints.sha256_file(self.repo / fname) != digest:
                    raise ValueError(f"{self.repo / fname} failed SHA-256 verification")
        self._device = resolve_device(self.requested_device)
        # Demucs pickles store a class reference plus args (demucs.states). Keep torch.load's weights_only safety and
        # allow exactly the globals these hash-verified files contain (checked with get_unsafe_globals_in_checkpoint).
        import fractions

        import numpy
        import torch
        from demucs.htdemucs import HTDemucs

        try:
            scalar = numpy._core.multiarray.scalar  # numpy >= 2
        except AttributeError:  # pragma: no cover
            scalar = numpy.core.multiarray.scalar
        # the pickles name the NumPy 1.x path; map it explicitly to the NumPy 2 function
        allowed = [HTDemucs, fractions.Fraction, (scalar, "numpy.core.multiarray.scalar"), numpy.dtype]
        allowed += [type(numpy.dtype(t)) for t in ("float32", "float64", "int64", "int32", "bool")]
        with torch.serialization.safe_globals(allowed):
            model = get_model("htdemucs_ft", repo=self.repo)
        model.eval()
        self._model = model
        self.stems = tuple(model.sources)

    def separate(self, mixture: np.ndarray, sample_rate: int) -> SeparationResult:
        import torch
        from demucs.apply import apply_model

        if sample_rate != self.sample_rate or mixture.shape[0] != self.channels:
            raise ValueError("htdemucs_ft expects 44.1 kHz stereo")
        self._load()
        t0 = time.time()
        wav = torch.from_numpy(np.ascontiguousarray(mixture, dtype=np.float32))
        ref = wav.mean(0)
        mean, std = ref.mean(), ref.std() + 1e-8
        with torch.inference_mode():
            out = apply_model(
                self._model, ((wav - mean) / std)[None], device=self._device, shifts=self.shifts,
                split=True, overlap=self.overlap, progress=False,
            )[0]
        out = out * std + mean
        stems = {name: out[i].cpu().numpy().astype(np.float32) for i, name in enumerate(self._model.sources)}
        meta = {"repo": str(self.repo), "device": self._device, "shifts": self.shifts, "overlap": self.overlap,
                "seconds": round(time.time() - t0, 2), "zero_dc": False, "license_note": "demucs code MIT; Meta weights"}
        return SeparationResult(stems, self.sample_rate, self.name, meta)
