"""Validation and *format* normalization.

Deliberately no loudness normalization, dithering, limiting or DC removal: CleanSplit must
compare stems against the mixture exactly as it was given. The only transformations are
channel conformance and (if needed) a single high-quality rational resample, both recorded.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly


@dataclass
class ValidationReport:
    sample_rate_in: int
    sample_rate_out: int
    channels_in: int
    channels_out: int
    samples_in: int
    samples_out: int
    duration_s: float
    peak: float
    rms_dbfs: float
    dc_offset: list[float]
    clipped_samples: int
    clipped_fraction: float
    nonfinite_samples: int
    silent: bool
    resampled: bool
    channel_conversion: str | None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def db(x: float, floor: float = -200.0) -> float:
    return float(10.0 * np.log10(x)) if x > 0 else floor


def resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Polyphase resampling (Kaiser window, beta 5.0 as scipy default).

    Returned in float64. Resampling is linear, so resampling stems and summing equals
    summing then resampling up to float rounding; tests assert this.
    """
    if sr_in == sr_out:
        return audio.astype(np.float64, copy=False)
    frac = Fraction(sr_out, sr_in)
    return resample_poly(audio.astype(np.float64, copy=False), frac.numerator, frac.denominator, axis=-1)


def conform_channels(audio: np.ndarray, channels: int) -> tuple[np.ndarray, str | None]:
    c = audio.shape[0]
    if c == channels:
        return audio, None
    if c == 1 and channels == 2:
        return np.repeat(audio, 2, axis=0), "mono->stereo (duplicated)"
    if c == 2 and channels == 1:
        return audio.mean(axis=0, keepdims=True), "stereo->mono (mean)"
    if c > 2 and channels == 2:
        # Do not guess a downmix matrix for surround material.
        raise ValueError(f"{c}-channel input: provide a stereo mix explicitly")
    raise ValueError(f"unsupported channel conversion {c}->{channels}")


def count_clipped(audio: np.ndarray, threshold: float = 0.9999, min_run: int = 3) -> int:
    """Samples in runs of >= min_run consecutive near-full-scale samples (a clipping signature)."""
    total = 0
    for ch in np.abs(audio) >= threshold:
        if not ch.any():
            continue
        d = np.diff(np.concatenate(([0], ch.view(np.int8), [0])))
        starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        runs = ends - starts
        total += int(runs[runs >= min_run].sum())
    return total


def validate_and_conform(
    audio: np.ndarray, sample_rate: int, target_sr: int = 44100, target_channels: int = 2
) -> tuple[np.ndarray, ValidationReport]:
    if audio.ndim != 2:
        raise ValueError(f"expected (C, N) audio, got {audio.shape}")
    warnings: list[str] = []
    nonfinite = int(np.size(audio) - np.count_nonzero(np.isfinite(audio)))
    if nonfinite:
        raise ValueError(f"input contains {nonfinite} non-finite samples")
    c_in, n_in = audio.shape
    if n_in < target_sr // 10:
        raise ValueError("input shorter than 100 ms")

    clipped = count_clipped(audio)
    if clipped:
        warnings.append(
            f"input has {clipped} samples in full-scale runs (likely clipped); separators and the "
            "residual behave non-linearly around clipped peaks"
        )

    x, conv = conform_channels(audio, target_channels)
    if conv:
        warnings.append(f"channel conversion: {conv}")
    resampled = sample_rate != target_sr
    if resampled:
        warnings.append(f"resampled {sample_rate} Hz -> {target_sr} Hz; analysis refers to the resampled mixture")
        x = resample(x, sample_rate, target_sr)
    x64 = x.astype(np.float64, copy=False)
    peak = float(np.max(np.abs(x64)))
    ms = float(np.mean(x64 * x64))
    silent = ms < 1e-10  # below -100 dBFS
    if silent:
        warnings.append("input is effectively silent (< -100 dBFS RMS)")
    if resampled and peak > 1.0:
        warnings.append(f"resampled peak {peak:.4f} exceeds 1.0 (inter-sample overs); kept in float, not clipped")

    report = ValidationReport(
        sample_rate_in=int(sample_rate),
        sample_rate_out=int(target_sr),
        channels_in=int(c_in),
        channels_out=int(target_channels),
        samples_in=int(n_in),
        samples_out=int(x.shape[1]),
        duration_s=float(x.shape[1] / target_sr),
        peak=peak,
        rms_dbfs=db(ms),
        dc_offset=[float(v) for v in x64.mean(axis=1)],
        clipped_samples=clipped,
        clipped_fraction=float(clipped / audio.size),
        nonfinite_samples=nonfinite,
        silent=silent,
        resampled=resampled,
        channel_conversion=conv,
        warnings=warnings,
    )
    return x.astype(np.float32), report
