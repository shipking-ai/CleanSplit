from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ...analysis.context import TINY, AnalysisContext
from .base import Detector, EvidenceMap, logistic


def _modulation_spectra(Bp: np.ndarray, win: int, hop: int, fr: float, fmin: float, fmax: float):
    """(B, T) linear band power -> complex spectra of the windowed power envelope in [fmin, fmax] (B, K, M),
    and window mean power (B, K)."""
    T = Bp.shape[1]
    if T < win:
        Bp = np.pad(Bp, ((0, 0), (0, win - T)), mode="edge")
    seg = sliding_window_view(Bp.astype(np.float64), win, axis=-1)[:, ::hop]
    mean = seg.mean(axis=-1)
    w = np.hanning(win)
    spec = np.fft.rfft((seg - mean[..., None]) * w, axis=-1)
    f = np.fft.rfftfreq(win, 1.0 / fr)
    sel = (f >= fmin) & (f <= fmax)
    return spec[..., sel] * np.sqrt(2.0 / np.sum(w * w) / win), mean


def _pool_bands(x: np.ndarray, reach: int) -> np.ndarray:
    """Sum over bands b-reach..b+reach (axis 0)."""
    c = np.cumsum(np.concatenate([np.zeros_like(x[:1]), x], axis=0), axis=0)
    B = x.shape[0]
    lo = np.clip(np.arange(B) - reach, 0, B)
    hi = np.clip(np.arange(B) + reach + 1, 0, B)
    return c[hi] - c[lo]


class ModulationDetector(Detector):
    """Warble / "underwater": 3-20 Hz fluctuation of a stem's band power that the mixture does not carry.

    Band powers add (cross terms average out over a band), so a genuine fluctuation of a stem (tremolo, onsets,
    chorus beating) appears in the mixture's power envelope *with the same waveform*, whatever other stems do.
    We regress the mixture's modulation onto the stem's:  beta = Re<D_s, D_m> / <D_s, D_s>  (complex modulation
    spectra, pooled over +-1 band and a 1 s window). Genuine: beta ~ 1. Artifact whose complement went to another
    stem (mixture unchanged) or that the mixture never had: beta ~ 0.  measure = 1 - clip(beta, 0, 1).

    The estimate is only trusted where the stem's own modulation is strong relative to the mixture's
    (``modulation_min_rel_db``): under a much busier mixture, beta is dominated by unrelated fluctuation and the
    mixture cannot arbitrate. That is a stated blind spot, not something to paper over.
    """

    name = "modulation"
    version = "5"
    min_depth = 0.15  # RMS power fluctuation / mean power
    reach = 1
    # Synthetic benchmark (5 seeds, randomized boxes): warble recall 5/10 at ~19.5 false-positive regions/min on
    # clean stems; no threshold improved both. Capped like leakage: a hypothesis, not a finding.
    max_confidence = 0.7

    def detect(self, ctx: AnalysisContext) -> list[EvidenceMap]:
        c = ctx.config
        th = c.modulation
        fr = ctx.sample_rate / c.hop
        win = max(8, int(round(c.modulation_window_s * fr)))
        hop = max(1, int(round(c.modulation_hop_s * fr)))
        Dm, Mm = _modulation_spectra(ctx.B_mix, win, hop, fr, c.modulation_fmin_hz, c.modulation_fmax_hz)
        Em = _pool_bands(np.sum(np.abs(Dm) ** 2, axis=-1), self.reach)
        K = Dm.shape[1]
        centers = (np.arange(K) * hop + win / 2.0) / fr
        quarter = c.modulation_window_s / 4.0
        cols = np.stack([np.maximum(centers - quarter, 0.0), centers + quarter], axis=1)
        rows = ctx.band_bounds()
        out = []
        for stem in ctx.stem_names:
            Bs = ctx.B_stems[stem]
            Ds, Ms = _modulation_spectra(Bs, win, hop, fr, c.modulation_fmin_hz, c.modulation_fmax_hz)
            Es_band = np.sum(np.abs(Ds) ** 2, axis=-1)
            Es = _pool_bands(Es_band, self.reach)
            cross = _pool_bands(np.sum((Ds * np.conj(Dm)).real, axis=-1), self.reach)
            beta = cross / np.maximum(Es, TINY)
            m = (1.0 - np.clip(beta, 0.0, 1.0)).astype(np.float32)
            depth = np.sqrt(Es_band) / np.maximum(Ms, TINY)
            rel_db = 10 * np.log10(np.maximum(Es, TINY) / np.maximum(Em, TINY))
            audible = Ms > float(Bs.max()) * 10 ** (-c.audibility_range_db / 10)
            trusted = (rel_db >= c.modulation_min_rel_db) & (depth >= self.min_depth) & audible
            ev = np.where(trusted, logistic(m, th.center, th.width), 0.0).astype(np.float32)
            out.append(
                EvidenceMap(
                    self.name, stem, np.minimum(ev, self.max_confidence), m, th.unit, rows, cols,
                    ["unnatural_modulation", "warble"], max_confidence=self.max_confidence,
                    extra={"min_cells": 2, "min_rows": 2},
                )
            )
        return out
