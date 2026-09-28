from __future__ import annotations

import numpy as np

from ...analysis.context import TINY, AnalysisContext, to_db
from .base import Detector, EvidenceMap, logistic


class ResidualDetector(Detector):
    """Reconstruction discrepancy: local energy of E = O_ref - sum(stems) relative to O_ref.

    For a mask-based separator E(f,t) = O(f,t)(1 - sum_i M_i(f,t)), so this is a direct, exact map of mask-sum
    error. It is blind to energy that is misassigned *between* stems (leakage keeps the sum intact).
    Cells are labelled ``missing_energy`` when |R| < |O_ref| locally, ``added_energy`` otherwise.
    """

    name = "residual"
    version = "1"

    def detect(self, ctx: AnalysisContext) -> list[EvidenceMap]:
        th = ctx.config.residual
        m = to_db(ctx.Ps_res + TINY) - to_db(ctx.Ps_mix + TINY)
        ev = logistic(m, th.center, th.width)
        gate = ctx.mix_audible | (ctx.Ps_recon > ctx.Ps_mix.max() * 10 ** (-ctx.config.audibility_range_db / 10))
        ev = np.where(gate, ev, 0.0).astype(np.float32)
        added = ctx.Ps_recon > ctx.Ps_mix
        return [
            EvidenceMap(
                detector=self.name,
                stem="mixture",
                evidence=ev,
                measure=m.astype(np.float32),
                measure_name=th.unit,
                row_bounds_hz=ctx.bin_bounds(),
                col_bounds_s=ctx.frame_bounds(),
                artifact_types=["reconstruction_discrepancy"],
                type_masks={"added_energy": added, "missing_energy": ~added},
            )
        ]
