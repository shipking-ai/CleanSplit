from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from ...analysis.context import TINY, AnalysisContext, to_db
from ...audio.stft import STFTGrid, stft
from .base import Detector, EvidenceMap, logistic

BANDS_HZ = ((150.0, 1000.0), (1000.0, 4000.0), (4000.0, 10000.0), (10000.0, 18000.0))


class TransientDetector(Detector):
    """Transient smearing / pre-echo, measured at mixture onsets on a short STFT (default 512/128).

    attack = dB(max power in [n, n+4]) - dB(mean power in [n-12, n-3]). For an onset owned by the stem (stem
    carries >= ownership share of the post-onset mixture energy), a faithful stem has attack close to the
    mixture's attack: the mixture contains the stem, so the stem's pre-onset energy cannot legitimately exceed the
    mixture's. measure = attack_mix - attack_stem; large positive values mean energy smeared before the onset.
    """

    name = "transient"
    version = "1"

    def detect(self, ctx: AnalysisContext) -> list[EvidenceMap]:
        c = ctx.config
        th = c.transient
        g = STFTGrid(ctx.sample_rate, c.transient_n_fft, c.transient_hop)
        f = g.freqs()
        sels = [(f >= lo) & (f < min(hi, ctx.sample_rate / 2)) for lo, hi in BANDS_HZ]

        def band_db(x):
            S = stft(x, g)
            P = (S.real**2 + S.imag**2).sum(axis=0)
            return np.stack([P[s].sum(axis=0) for s in sels]).astype(np.float64)

        Pm = band_db(ctx.mixture)
        T = Pm.shape[1]
        Lm = to_db(Pm + TINY)
        pre_a, pre_b, post_len = 12, 3, 4
        fr = ctx.sample_rate / g.hop
        onsets = []
        for b in range(len(BANDS_HZ)):
            rise = np.full(T, -np.inf)
            rise[2:] = Lm[b, 2:] - Lm[b, :-2]
            floor_ok = Pm[b] > Pm[b].max() * 10 ** (-c.audibility_range_db / 10)
            rise = np.where(floor_ok, rise, -np.inf)
            peaks, _ = find_peaks(np.nan_to_num(rise, neginf=-1e9), height=c.transient_onset_rise_db, distance=max(1, int(0.05 * fr)))
            onsets.append(peaks[(peaks >= pre_a) & (peaks < T - post_len)])

        cols = np.stack([np.arange(T) * g.hop / g.sample_rate, (np.arange(T) + 1) * g.hop / g.sample_rate], axis=1)
        rows = np.array(BANDS_HZ, dtype=np.float64)
        rows[:, 1] = np.minimum(rows[:, 1], ctx.sample_rate / 2)

        def attack(P, b, n):
            pre = P[b, n - pre_a : n - pre_b].mean()
            post = P[b, n : n + post_len].max()
            return to_db(post + TINY) - to_db(pre + TINY), post

        out = []
        for stem in ctx.stem_names:
            Ps = band_db(ctx.stems[stem])
            ev = np.zeros((len(BANDS_HZ), T), dtype=np.float32)
            meas = np.zeros_like(ev)
            n_eval = 0
            for b, peaks in enumerate(onsets):
                for n in peaks:
                    a_mix, post_mix = attack(Pm, b, n)
                    a_stem, post_stem = attack(Ps, b, n)
                    if to_db(post_stem + TINY) < to_db(post_mix + TINY) + c.transient_ownership_db:
                        continue
                    n_eval += 1
                    m = a_mix - a_stem
                    e = float(logistic(m, th.center, th.width))
                    sl = slice(n - pre_a, n + post_len)
                    ev[b, sl] = np.maximum(ev[b, sl], e)
                    meas[b, sl] = np.where(ev[b, sl] == e, m, meas[b, sl])
            out.append(
                EvidenceMap(
                    self.name, stem, ev, meas, th.unit, rows, cols, ["transient_smearing"],
                    extra={"min_cells": 1, "onsets_evaluated": n_eval},
                )
            )
        return out
