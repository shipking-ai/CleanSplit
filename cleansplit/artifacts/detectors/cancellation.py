from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter

from ...analysis.context import TINY, AnalysisContext, to_db
from .base import Detector, EvidenceMap, logistic


class CancellationDetector(Detector):
    """Inter-stem cancellation: a stem locally louder than the sum of all stems.

    If sources in a TF cell have unrelated phases, E|sum S_j|^2 = sum E|S_j|^2 >= |S_i|^2, so a stem exceeding the
    *sum* by several dB means some of its content only exists because another stem cancels it. Soloed, such a
    stem sounds phasey/flanged/hollow even though the mix still nulls; the residual detector cannot see this.
    Measured against the sum of stems (not O), so it is independent of mask-sum error.

    Smoothing is wider than the shared context default (~200 Hz x ~100 ms): a single stationary partial collision
    between two genuinely different sources can be anti-phase in one bin, but not coherently across a band and
    over time, which is what separator-induced cancellation looks like.
    """

    name = "cancellation"
    version = "2"
    smooth_bins = 9
    smooth_frames = 9

    def detect(self, ctx: AnalysisContext) -> list[EvidenceMap]:
        th = ctx.config.cancellation
        size = (self.smooth_bins, self.smooth_frames)
        den_db = to_db(uniform_filter(ctx.P_recon, size=size, mode="nearest") + TINY)
        rows, cols = ctx.bin_bounds(), ctx.frame_bounds()
        out = []
        for stem in ctx.stem_names:
            m = to_db(uniform_filter(ctx.P_stems[stem], size=size, mode="nearest") + TINY) - den_db
            ev = logistic(m, th.center, th.width)
            ev = np.where(ctx.stem_audible(stem), ev, 0.0).astype(np.float32)
            out.append(
                EvidenceMap(
                    self.name, stem, ev, m.astype(np.float32), th.unit, rows, cols, ["phase_cancellation", "phasing"],
                    extra={"min_cells": self.smooth_bins * 2},
                )
            )
        return out
