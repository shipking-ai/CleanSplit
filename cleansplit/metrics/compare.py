"""Objective A/B comparison of two audio files (B measured against A as reference)."""

from __future__ import annotations

import numpy as np

from ..audio.conform import conform_channels, resample
from ..audio.io import load_audio
from ..reconstruction.core import estimate_lag
from . import signal as M


def compare_arrays(a: np.ndarray, b: np.ndarray, sr: int) -> dict:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    lag, corr = estimate_lag(a, b, min(sr // 2, a.shape[-1] // 2))
    n = min(a.shape[-1], b.shape[-1])
    a, b = a[..., :n], b[..., :n]
    diff = b - a
    identical = bool(np.array_equal(a, b))
    return {
        "identical": identical,
        "lag_samples": lag,
        "correlation_at_lag": corr,
        "snr_db": M.snr_db(a, b),
        "si_sdr_db": M.si_sdr_db(a, b),
        "difference_energy_rel_db": M.residual_energy_rel_db(a, diff),
        "max_abs_difference": float(np.max(np.abs(diff))) if n else 0.0,
        "log_spectral_distance": M.log_spectral_distance(a, b) if not identical else 0.0,
        "multi_mel_snr_db": M.multi_mel_snr_db(a, b, sr),
        "band_difference_db": M.band_residual_db(a, diff),
    }


def compare_files(path_a, path_b) -> dict:
    A, B = load_audio(path_a), load_audio(path_b)
    a, b = A.audio, B.audio
    notes = []
    if B.sample_rate != A.sample_rate:
        b = resample(b, B.sample_rate, A.sample_rate)
        notes.append(f"B resampled {B.sample_rate}->{A.sample_rate}")
    if b.shape[0] != a.shape[0]:
        b, conv = conform_channels(b, a.shape[0])
        notes.append(f"B channels: {conv}")
    if a.shape[-1] != b.shape[-1]:
        notes.append(f"length differs: A={a.shape[-1]} B={b.shape[-1]} (compared over common length)")
    return {"a": str(path_a), "b": str(path_b), "sample_rate": A.sample_rate, "notes": notes, **compare_arrays(a, b, A.sample_rate)}
