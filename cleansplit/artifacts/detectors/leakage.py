from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ...analysis.context import TINY, AnalysisContext, to_db
from .base import Detector, EvidenceMap, logistic


class LeakageDetector(Detector):
    """Possible leakage: in a band and 0.5 s window, a weaker stem's power envelope is a *constant-ratio copy* of a
    stronger stem's (std of the dB level difference is small while both envelopes move), at a level ratio typical
    of bleed (default -35..-8 dB).

    Co-timed but distinct parts (a bass note on the kick) share onsets but not decay shapes, so their level
    ratio varies; a leaked copy does not. Still ambiguous for exact doublings, so confidence is capped (default
    0.7) and regions are labelled ``possible_leakage`` with the likely source. The mixture residual cannot reveal
    leakage at all, which is why this detector exists.
    measure = std over the window of (L_receiver - L_source) in dB, on frames where both are >= 20 dB above floor.

    The source envelope must have non-linear fine structure (std >= ``leakage_min_env_std_db`` after removing a
    linear trend): exponential decays are straight lines in dB, so two unrelated decaying notes can hold a
    nearly constant level ratio for half a second. Regions must span >= 2 adjacent bands (bleed is broadband
    relative to a single partial).
    """

    name = "leakage"
    version = "3"
    max_ratio_std_db = 1.5

    def detect(self, ctx: AnalysisContext) -> list[EvidenceMap]:
        c = ctx.config
        th_cap = c.leakage_max_confidence
        fr = ctx.sample_rate / c.hop
        win = max(8, int(round(0.5 * fr)))
        hop = max(1, win // 2)
        names = ctx.stem_names
        L, V, M = {}, {}, {}
        for k in names:
            Bp = ctx.B_stems[k]
            if Bp.shape[1] < win:
                Bp = np.pad(Bp, ((0, 0), (0, win - Bp.shape[1])), mode="edge")
            floor = float(Bp.max()) * 1e-6
            env = to_db(Bp + floor + TINY)
            L[k] = sliding_window_view(env, win, axis=-1)[:, ::hop]  # (B, K, W)
            V[k] = L[k] >= to_db(floor) + 20.0  # valid (clearly above floor) frames
            M[k] = to_db(sliding_window_view(Bp, win, axis=-1)[:, ::hop].mean(-1) + TINY)
        K = next(iter(L.values())).shape[1]
        centers = (np.arange(K) * hop + win / 2) / fr
        cols = np.stack([np.maximum(centers - 0.25, 0), centers + 0.25], axis=1)
        rows = ctx.band_bounds()
        lo, hi = c.leakage_level_range_db
        out = []
        for j in names:  # candidate receiver
            best = np.zeros(L[j].shape[:2], dtype=np.float32)
            best_m = np.full(best.shape, np.inf, dtype=np.float32)
            winner = np.full(best.shape, -1, dtype=np.int32)
            others = [n for n in names if n != j]
            for idx, i in enumerate(others):
                both = V[i] & V[j]
                n_valid = both.sum(-1)
                w = both.astype(np.float64)
                d = L[j] - L[i]
                mu = (d * w).sum(-1) / np.maximum(n_valid, 1)
                sd = np.sqrt(((d - mu[..., None]) ** 2 * w).sum(-1) / np.maximum(n_valid, 1))
                si = self._detrended_std(L[i], w, n_valid)
                level = M[j] - M[i]
                ok = (n_valid >= win // 2) & (si >= c.leakage_min_env_std_db) & (level >= lo) & (level <= hi)
                e = np.where(ok, logistic(-sd, -self.max_ratio_std_db, 0.4), 0.0)
                better = e > best
                best = np.where(better, e, best)
                best_m = np.where(better, sd, best_m)
                winner = np.where(better, idx, winner)
            type_masks = {f"leakage_from:{i}": winner == idx for idx, i in enumerate(others)}
            ev = np.minimum(best, th_cap).astype(np.float32)
            meas = np.where(np.isfinite(best_m), best_m, 0.0).astype(np.float32)
            out.append(
                EvidenceMap(
                    self.name, j, ev, meas, "dB std of receiver-source level difference", rows, cols, ["possible_leakage"],
                    type_masks=type_masks, max_confidence=th_cap, extra={"min_cells": 2, "min_rows": 2},
                )
            )
        return out

    @staticmethod
    def _detrended_std(Lw: np.ndarray, w: np.ndarray, n_valid: np.ndarray) -> np.ndarray:
        """Weighted std of each window after removing a weighted least-squares line. Lw, w: (B, K, W)."""
        W = Lw.shape[-1]
        t = np.arange(W, dtype=np.float64) - (W - 1) / 2.0
        sw = np.maximum(w.sum(-1), 1.0)
        mt = (w * t).sum(-1) / sw
        my = (w * Lw).sum(-1) / sw
        tc = t - mt[..., None]
        yc = Lw - my[..., None]
        slope = (w * tc * yc).sum(-1) / np.maximum((w * tc * tc).sum(-1), 1e-9)
        resid = yc - slope[..., None] * tc
        return np.sqrt((w * resid**2).sum(-1) / np.maximum(n_valid, 1))
