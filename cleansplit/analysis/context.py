"""Shared, lazily computed features for detectors.

Conventions
-----------
* Power spectra are summed over channels (energy-preserving), float32, shape (F, T).
* ``mix`` refers to the model-matched reference O_ref (== O when the separator applies no known
  deterministic processing). Detectors compare stems against what the separator could have produced.
* Smoothed powers use a TF box filter; ratios are always ratios of smoothed powers (never smoothed ratios).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
from scipy.ndimage import uniform_filter

from ..audio.stft import STFTGrid, stft
from ..config.settings import AnalysisConfig

TINY = 1e-20


def to_db(x, floor=TINY):
    return 10.0 * np.log10(np.maximum(x, floor))


def log_band_edges(fmin: float, fmax: float, n: int) -> np.ndarray:
    return np.geomspace(fmin, fmax, n + 1)


@dataclass
class AnalysisContext:
    mixture: np.ndarray  # O_ref (C, N) float64/float32
    stems: dict[str, np.ndarray]  # (C, N)
    sample_rate: int
    config: AnalysisConfig
    chunk_starts: list[int] = field(default_factory=list)
    chunk_size: int | None = None

    @property
    def grid(self) -> STFTGrid:
        return STFTGrid(self.sample_rate, self.config.n_fft, self.config.hop)

    @property
    def stem_names(self) -> list[str]:
        return list(self.stems)

    @cached_property
    def n_frames(self) -> int:
        return self.grid.n_frames(self.mixture.shape[-1])

    # ---- complex spectra: computed once, stems' complex spectra are only summed then discarded ----
    @cached_property
    def _spectra(self):
        g = self.grid
        mix = stft(self.mixture, g)
        P_stems = {}
        R = np.zeros_like(mix)
        for k, s in self.stems.items():
            S = stft(s, g)
            R += S
            P_stems[k] = (S.real**2 + S.imag**2).sum(axis=0).astype(np.float32)
            del S
        P_mix = (mix.real**2 + mix.imag**2).sum(axis=0).astype(np.float32)
        P_recon = (R.real**2 + R.imag**2).sum(axis=0).astype(np.float32)
        E = mix - R
        P_res = (E.real**2 + E.imag**2).sum(axis=0).astype(np.float32)
        return P_mix, P_stems, P_recon, P_res

    @property
    def P_mix(self):
        return self._spectra[0]

    @property
    def P_stems(self) -> dict[str, np.ndarray]:
        return self._spectra[1]

    @property
    def P_recon(self):
        return self._spectra[2]

    @property
    def P_res(self):
        return self._spectra[3]

    def smooth(self, P: np.ndarray) -> np.ndarray:
        c = self.config
        return uniform_filter(P.astype(np.float32), size=(c.smooth_bins, c.smooth_frames), mode="nearest")

    @cached_property
    def Ps_mix(self):
        return self.smooth(self.P_mix)

    @cached_property
    def Ps_stems(self):
        return {k: self.smooth(v) for k, v in self.P_stems.items()}

    @cached_property
    def Ps_recon(self):
        return self.smooth(self.P_recon)

    @cached_property
    def Ps_res(self):
        return self.smooth(self.P_res)

    # ---- audibility gates ----
    @cached_property
    def mix_audible(self) -> np.ndarray:
        ref = float(self.Ps_mix.max())
        return self.Ps_mix > max(ref * 10 ** (-self.config.audibility_range_db / 10), self.config.abs_floor_power)

    def stem_audible(self, stem: str) -> np.ndarray:
        P = self.Ps_stems[stem]
        ref = float(P.max())
        return P > max(ref * 10 ** (-self.config.audibility_range_db / 10), self.config.abs_floor_power)

    # ---- log-band pooled envelopes (for modulation / leakage) ----
    @cached_property
    def band_edges_hz(self) -> np.ndarray:
        c = self.config
        return log_band_edges(c.band_fmin_hz, min(c.band_fmax_hz, self.sample_rate / 2), c.n_bands)

    @cached_property
    def _band_index(self):
        f = self.grid.freqs()
        edges = self.band_edges_hz
        idx = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = np.flatnonzero((f >= lo) & (f < hi))
            if sel.size == 0:  # narrow low band: take nearest bin
                sel = np.array([int(np.argmin(np.abs(f - (lo + hi) / 2)))])
            idx.append(sel)
        return idx

    def band_power(self, P: np.ndarray) -> np.ndarray:
        """(F, T) -> (B, T) summed power in log-spaced bands."""
        return np.stack([P[sel].sum(axis=0) for sel in self._band_index])

    @cached_property
    def B_mix(self):
        return self.band_power(self.P_mix)

    @cached_property
    def B_stems(self):
        return {k: self.band_power(v) for k, v in self.P_stems.items()}

    def frame_bounds(self) -> np.ndarray:
        g = self.grid
        t = g.frame_time(np.arange(self.n_frames))
        half = 0.5 * g.hop / g.sample_rate
        return np.stack([np.maximum(t - half, 0.0), t + half], axis=1)

    def bin_bounds(self) -> np.ndarray:
        g = self.grid
        f = g.freqs()
        half = 0.5 * g.sample_rate / g.n_fft
        return np.stack([np.maximum(f - half, 0.0), f + half], axis=1)

    def band_bounds(self) -> np.ndarray:
        e = self.band_edges_hz
        return np.stack([e[:-1], e[1:]], axis=1)
