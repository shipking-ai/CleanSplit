"""NumPy STFT/ISTFT matching ``torch.stft(center=True, pad_mode='reflect', window=hann_window(periodic))``.

The analysis grid defaults to the BS-RoFormer SW grid (n_fft 2048, hop 512, 44.1 kHz) so that
artifact regions map 1:1 onto the separator's (and A2SB's) time-frequency bins.
Computation is done in blocks to bound memory on long songs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.fft
from numpy.lib.stride_tricks import sliding_window_view


def hann_periodic(n: int, dtype=np.float64) -> np.ndarray:
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)).astype(dtype)


@dataclass(frozen=True)
class STFTGrid:
    sample_rate: int = 44100
    n_fft: int = 2048
    hop: int = 512

    @property
    def n_bins(self) -> int:
        return self.n_fft // 2 + 1

    def n_frames(self, n_samples: int) -> int:
        return 1 + n_samples // self.hop

    def bin_hz(self, k) -> np.ndarray:
        return np.asarray(k, dtype=np.float64) * self.sample_rate / self.n_fft

    def hz_to_bin(self, hz: float) -> int:
        return int(np.clip(np.round(hz * self.n_fft / self.sample_rate), 0, self.n_bins - 1))

    def frame_time(self, t) -> np.ndarray:
        """Centre time (s) of frame t (center=True convention)."""
        return np.asarray(t, dtype=np.float64) * self.hop / self.sample_rate

    def time_to_frame(self, seconds: float) -> int:
        return int(np.round(seconds * self.sample_rate / self.hop))

    def freqs(self) -> np.ndarray:
        return self.bin_hz(np.arange(self.n_bins))


def stft(x: np.ndarray, grid: STFTGrid, dtype=np.complex64, block_frames: int = 4096) -> np.ndarray:
    """x: (..., N) -> (..., F, T) complex."""
    x = np.asarray(x)
    lead = x.shape[:-1]
    n = x.shape[-1]
    pad = grid.n_fft // 2
    if n <= pad:
        raise ValueError(f"signal too short for STFT: {n} samples, need > {pad}")
    real_dtype = np.float32 if dtype == np.complex64 else np.float64
    xp = np.pad(x.reshape(-1, n).astype(real_dtype, copy=False), ((0, 0), (pad, pad)), mode="reflect")
    win = hann_periodic(grid.n_fft, real_dtype)
    n_frames = grid.n_frames(n)
    frames_view = sliding_window_view(xp, grid.n_fft, axis=-1)[:, :: grid.hop][:, :n_frames]
    out = np.empty((xp.shape[0], grid.n_bins, n_frames), dtype=dtype)
    for s in range(0, n_frames, block_frames):
        e = min(s + block_frames, n_frames)
        spec = scipy.fft.rfft(frames_view[:, s:e] * win, axis=-1)  # (B, t, F)
        out[:, :, s:e] = np.moveaxis(spec, -1, -2)
    return out.reshape(*lead, grid.n_bins, n_frames)


def istft(spec: np.ndarray, grid: STFTGrid, length: int, dtype=np.float64) -> np.ndarray:
    """(..., F, T) -> (..., length). Least-squares overlap-add (same as torch.istft)."""
    lead = spec.shape[:-2]
    n_bins, n_frames = spec.shape[-2:]
    if n_bins != grid.n_bins:
        raise ValueError("bin count does not match grid")
    spec = spec.reshape(-1, n_bins, n_frames)
    win = hann_periodic(grid.n_fft, dtype)
    pad = grid.n_fft // 2
    total = grid.n_fft + grid.hop * (n_frames - 1)
    out = np.zeros((spec.shape[0], total), dtype=dtype)
    wsum = np.zeros(total, dtype=dtype)
    for t in range(n_frames):
        wsum[t * grid.hop : t * grid.hop + grid.n_fft] += win * win
    block = 2048
    for s in range(0, n_frames, block):
        e = min(s + block, n_frames)
        frames = scipy.fft.irfft(np.moveaxis(spec[:, :, s:e], -2, -1), n=grid.n_fft, axis=-1).real.astype(dtype)
        frames *= win
        for i in range(e - s):
            o = (s + i) * grid.hop
            out[:, o : o + grid.n_fft] += frames[:, i]
    out = out[:, pad : pad + length]
    w = wsum[pad : pad + length]
    if out.shape[-1] < length:
        raise ValueError("requested length exceeds what the frames cover")
    if np.any(w < 1e-11):
        raise ValueError("window envelope vanishes inside requested length (NOLA violated)")
    return (out / w).reshape(*lead, length)


def power(spec: np.ndarray, channel_axis: int | None = 0) -> np.ndarray:
    """|X|^2 summed over channels (float32). Energy-preserving channel combination."""
    p = spec.real.astype(np.float32) ** 2 + spec.imag.astype(np.float32) ** 2
    if channel_axis is not None and p.ndim >= 3:
        p = p.sum(axis=channel_axis)
    return p
