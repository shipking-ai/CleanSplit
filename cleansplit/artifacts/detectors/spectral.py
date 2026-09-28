from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter

from ...analysis.context import TINY, AnalysisContext, to_db
from .base import Detector, EvidenceMap, logistic


def _blocks(P: np.ndarray, r0: int, br: int, bc: int) -> np.ndarray:
    """Sum (F, T) over non-overlapping blocks starting at row r0 -> (R, K)."""
    Q = P[r0:]
    R, K = Q.shape[0] // br, Q.shape[1] // bc
    return Q[: R * br, : K * bc].reshape(R, br, K, bc).sum(axis=(1, 3))


def _block_bounds(ctx: AnalysisContext, r0: int, br: int, bc: int, R: int, K: int):
    g = ctx.grid
    df = g.sample_rate / g.n_fft
    rows = np.array([[(r0 + i * br - 0.5) * df, (r0 + (i + 1) * br - 0.5) * df] for i in range(R)])
    dt = g.hop / g.sample_rate
    cols = np.array([[k * bc * dt, (k + 1) * bc * dt] for k in range(K)])
    return np.maximum(rows, 0.0), cols


class MusicalNoiseDetector(Detector):
    """Musical noise ("twinkling"): energy in *mask spikes*: stem cells that are strong isolated peaks in time
    AND frequency, where the mixture at the same cell is NOT an isolated peak, and the stem carries a large share
    (>= -6 dB) of that mixture cell.

    Rationale: S = M*O. A spiky mask on a smooth mixture creates exactly this pattern. Genuine sparse content
    (a hi-hat onset, a partial) is either an isolated peak in the mixture too, persists in time, or is
    broadband, or is masked (small share). A 10x (10 dB) spike factor keeps the natural fluctuation of noise-like
    content (exponential bin powers exceed 10x their median ~0.1% of the time) from counting.

    "Isolated" is judged against a *ring* of neighbours (3-5 frames / bins away), not adjacent cells: after
    inverse STFT and re-analysis with 75% overlap, a one-cell spike necessarily spreads over about +-2 frames and
    +-1-2 bins (STFT consistency), so adjacent cells can never be 10 dB below it.
    Kurtosis-style measures correlate only moderately with perceived musical noise (arXiv:2105.13079); weak cue.
    """

    name = "musical_noise"
    version = "4"
    # Isolation must hold on EACH side separately: a note onset is quiet before but strong after (and a note end
    # the reverse), so a symmetric ring would call every fast-decaying onset a spike.
    _BEFORE = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=bool)  # time offsets -5..-3
    _AFTER = _BEFORE[:, ::-1].copy()  # +3..+5
    _BELOW = _BEFORE.T.copy()
    _ABOVE = _AFTER.T.copy()

    @classmethod
    def _spikes(cls, P, k):
        ok = np.ones(P.shape, dtype=bool)
        for fp in (cls._BEFORE, cls._AFTER, cls._BELOW, cls._ABOVE):
            ok &= P > k * median_filter(P, footprint=fp, mode="nearest")
        return ok

    @classmethod
    def _stationary_in_time(cls, P, ratio=2.0):
        """Cell not more than ``ratio`` x the median power 3-5 frames before AND after (mixture is steady there).
        A partial sweeping through the bin (vibrato) makes the *mixture* non-stationary; a mask spike does not."""
        before = median_filter(P, footprint=cls._BEFORE, mode="nearest")
        after = median_filter(P, footprint=cls._AFTER, mode="nearest")
        return P <= ratio * np.minimum(before, after)

    def detect(self, ctx: AnalysisContext) -> list[EvidenceMap]:
        c = ctx.config
        th = c.musical_noise
        k = c.musical_noise_spike_factor
        r0 = ctx.grid.hz_to_bin(c.musical_noise_fmin_hz)
        br, bc = c.musical_noise_patch_bins, c.musical_noise_patch_frames
        mix_ok = self._stationary_in_time(ctx.P_mix)
        tot_m = _blocks(ctx.P_mix, r0, br, bc)
        rows, cols = _block_bounds(ctx, r0, br, bc, *tot_m.shape)
        out = []
        for stem in ctx.stem_names:
            P = ctx.P_stems[stem]
            cells = self._spikes(P, k) & mix_ok & (P >= 0.25 * ctx.P_mix)
            iso_s = _blocks(np.where(cells, P, 0.0), r0, br, bc)
            tot_s = _blocks(P, r0, br, bc)
            m = (to_db(iso_s / np.maximum(tot_s, TINY) + 1e-6)).astype(np.float32)
            audible = tot_s > tot_s.max() * 10 ** (-c.audibility_range_db / 10)
            share_ok = to_db(tot_s + TINY) >= to_db(tot_m + TINY) - 40.0
            ev = np.where(audible & share_ok, logistic(m, th.center, th.width), 0.0).astype(np.float32)
            out.append(EvidenceMap(self.name, stem, ev, m, th.unit, rows, cols, ["musical_noise"], extra={"min_cells": 1}))
        return out


class HFNoiseDetector(Detector):
    """High-frequency noise: stem HF (default >= 8 kHz) markedly flatter (more noise-like) than the mixture HF
    in the same patch, where the stem carries a non-negligible share of that HF energy.

    Known confound: legitimately noisy HF content (sibilance, breath) against tonal HF in the mix.
    """

    name = "hf_noise"
    version = "1"

    @staticmethod
    def _flatness(P, r0, br, bc):
        Q = P[r0:]
        R, K = Q.shape[0] // br, Q.shape[1] // bc
        Q = Q[: R * br, : K * bc].reshape(R, br, K, bc).astype(np.float64) + 1e-12
        gmean = np.exp(np.mean(np.log(Q), axis=1))  # over bins -> (R, K, bc)
        amean = np.mean(Q, axis=1)
        # frame-energy-weighted flatness: near-silent frames must not dominate
        flat = np.sum((gmean / amean) * amean, axis=-1) / np.sum(amean, axis=-1)
        return flat, Q.sum(axis=(1, 3))

    def detect(self, ctx: AnalysisContext) -> list[EvidenceMap]:
        c = ctx.config
        th = c.hf_noise
        g = ctx.grid
        if c.hf_noise_fmin_hz >= g.sample_rate / 2:
            return []
        r0 = g.hz_to_bin(c.hf_noise_fmin_hz)
        br = g.hz_to_bin(c.hf_noise_fmin_hz + 4000.0) - r0  # 4 kHz wide patches
        bc = max(1, int(round(0.25 * g.sample_rate / g.hop)))
        flat_m, tot_m = self._flatness(ctx.P_mix, r0, br, bc)
        if flat_m.size == 0:
            return []
        rows, cols = _block_bounds(ctx, r0, br, bc, *flat_m.shape)
        out = []
        for stem in ctx.stem_names:
            flat_s, tot_s = self._flatness(ctx.P_stems[stem], r0, br, bc)
            m = (flat_s - flat_m).astype(np.float32)
            stem_total = float(ctx.P_stems[stem].sum()) / max(ctx.P_stems[stem].shape[1] // bc, 1)
            audible = tot_s > stem_total * 10 ** (-45.0 / 10)
            share_ok = to_db(tot_s + TINY) >= to_db(tot_m + TINY) + c.hf_min_share_db
            ev = np.where(audible & share_ok, logistic(m, th.center, th.width), 0.0).astype(np.float32)
            out.append(EvidenceMap(self.name, stem, ev, m, th.unit, rows, cols, ["high_frequency_noise"], extra={"min_cells": 1}))
        return out
