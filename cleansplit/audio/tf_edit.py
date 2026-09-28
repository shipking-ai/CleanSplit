"""Region-restricted time-frequency edits with a hard locality guarantee.

``tf_region_delta`` returns the time-domain *difference* produced by an edit confined to a TF box:
the edit is expressed as a complex spectrogram difference that is exactly zero outside the (tapered) box, so
its inverse STFT is exactly zero outside the frames that touch the box. ``apply_delta`` adds it to a signal;
samples outside the support are copied, not recomputed, so they stay bit-identical.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .stft import STFTGrid, istft, stft


@dataclass(frozen=True)
class TFBox:
    start_s: float
    end_s: float
    freq_low_hz: float
    freq_high_hz: float


def box_mask(grid: STFTGrid, n_frames: int, box: TFBox, frame_offset: int = 0, taper_bins: int = 2, taper_frames: int = 2) -> np.ndarray:
    """(F, T) mask in [0, 1] with raised-cosine taper *inside* the box edges."""
    f0 = grid.hz_to_bin(box.freq_low_hz)
    f1 = grid.hz_to_bin(box.freq_high_hz)
    t0 = grid.time_to_frame(box.start_s) - frame_offset
    t1 = grid.time_to_frame(box.end_s) - frame_offset

    def ramp(lo, hi, n, taper):
        v = np.zeros(n)
        lo_c, hi_c = max(lo, 0), min(hi, n - 1)
        if hi_c < lo_c:
            return v
        v[lo_c : hi_c + 1] = 1.0
        L = hi - lo + 1
        tp = min(taper, max((L - 1) // 2, 0))
        for i in range(tp):
            w = 0.5 - 0.5 * np.cos(np.pi * (i + 1) / (tp + 1))
            for idx in (lo + i, hi - i):
                if 0 <= idx < n:
                    v[idx] = min(v[idx], w)
        return v

    return np.outer(ramp(f0, f1, grid.n_bins, taper_bins), ramp(t0, t1, n_frames, taper_frames))


def excerpt_bounds(n_samples: int, grid: STFTGrid, box: TFBox, margin_s: float = 0.5) -> tuple[int, int]:
    """Sample range to process, aligned to the hop so excerpt frames coincide with full-signal frames."""
    hop = grid.hop
    a = int(np.floor(max(box.start_s - margin_s, 0.0) * grid.sample_rate / hop)) * hop
    b = int(np.ceil(min(box.end_s + margin_s, n_samples / grid.sample_rate) * grid.sample_rate / hop)) * hop
    return a, min(b, n_samples)


def tf_region_delta(
    x: np.ndarray,
    grid: STFTGrid,
    box: TFBox,
    edit: Callable[[np.ndarray, np.ndarray], np.ndarray],
    margin_s: float = 0.5,
) -> tuple[np.ndarray, int, int]:
    """edit(spec (C,F,T) complex128, mask (F,T)) -> edited spec. Returns (delta excerpt (C, b-a), a, b).

    The applied change is ``mask * (edited - spec)``; anything the edit does outside the mask is discarded.
    """
    a, b = excerpt_bounds(x.shape[-1], grid, box, margin_s)
    seg = x[..., a:b].astype(np.float64)
    S = stft(seg, grid, dtype=np.complex128)
    mask = box_mask(grid, S.shape[-1], box, frame_offset=a // grid.hop)
    D = (edit(S, mask) - S) * mask
    delta = istft(D, grid, b - a)
    return delta, a, b


def apply_delta(x: np.ndarray, delta: np.ndarray, a: int, b: int) -> np.ndarray:
    y = x.copy()
    y[..., a:b] = (x[..., a:b].astype(np.float64) + delta).astype(x.dtype)
    return y


def changed_support(before: np.ndarray, after: np.ndarray) -> tuple[int, int] | None:
    """First and last+1 sample index where signals differ (bitwise), or None if identical."""
    diff = np.any(before != after, axis=0)
    idx = np.flatnonzero(diff)
    if idx.size == 0:
        return None
    return int(idx[0]), int(idx[-1] + 1)
