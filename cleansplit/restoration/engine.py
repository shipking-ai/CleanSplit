"""Apply restoration proposals under hard constraints and a mixture-consistency acceptance test.

For every proposal:
 1. Projection: delta_TF = mask_region * (STFT(proposed) - STFT(current)); only this is applied (via inverse STFT of
    the masked difference), so samples outside the frames touching the region are copied bit-for-bit.
 2. Measurements in the region box (TF energy sums over the excerpt STFT):
      err_before/after      = ||O_ref - R||^2 vs ||O_ref - R'||^2           (did reconstruction get worse?)
      stem_change           = sum_i ||S_i' - S_i||^2 relative to sum_i ||S_i||^2
      added / removed       = positive / negative parts of sum_i (|S_i'|^2 - |S_i|^2)
      unexplained_added     = ||sum_i D_i||^2 - ||E_before||^2 (net energy added to the mix beyond what was missing).
                              NOT sum_i(|S_i'|^2 - |S_i|^2): when stems are too quiet, adding the residual in phase
                              raises stem energy through the coherent cross term 2Re(S* D), which is correct.
      redistributed         = sum_i ||D_i||^2 - ||sum_i D_i||^2 (energy moved between stems; reported)
      evidence before/after = the flagging detector's evidence inside the box, recomputed on the excerpt
 3. Accept only if ALL hold:
      err_after <= err_before * 10^(tol_db/10)
      unexplained_added <= max_new_energy_ratio * max(stem energy in box, err_before)
      artifact evidence did not increase
      locality check passes (bitwise identical outside the support)
Otherwise reject and keep the current audio. Nothing is assumed to be an improvement because it "sounds smoother".

LIMIT (measured, docs/04_results.md): this gate only checks consistency with the mixture. A restorer that re-splits
mixture energy among the wrong stems passes it; region_wiener was accepted 84/85 times while moving stems +1.98 dB
further from the ground truth. Judge restorers on ground-truth stems (metrics.restoration_experiment) first.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from ..analysis.context import AnalysisContext
from ..artifacts import detectors as detector_registry
from ..audio.stft import istft, stft
from ..audio.tf_edit import TFBox, box_mask, changed_support
from ..config.settings import AnalysisConfig
from .base import RestorationContext, RestorationProposal


@dataclass
class GateConfig:
    tolerance_db: float = 0.05
    max_new_energy_ratio: float = 0.1
    require_evidence_decrease: bool = True
    evidence_margin_s: float = 2.0


@dataclass
class Decision:
    region_id: str
    stem: str
    restorer: str
    accepted: bool
    reasons: list[str]
    stems_changed: list[str]
    err_before: float
    err_after: float
    err_delta_db: float
    stem_energy_in_box: float
    stem_change_rel_db: float | None
    added_energy: float
    removed_energy: float
    unexplained_added_energy: float
    redistributed_energy: float
    evidence_before: float | None
    evidence_after: float | None
    projection_discarded_rel_db: float | None
    support_samples: tuple[int, int] | None
    notes: list[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def _db_ratio(a, b):
    if b <= 0:
        return None if a <= 0 else float("inf")
    return float(10 * np.log10(max(a, 1e-30) / b))


def _box_sum(P, mask):
    return float(np.sum(P * mask[None] if P.ndim == 3 else P * mask))


def _evidence_in_box(stems, mixture, sr, cfg: AnalysisConfig, detector: str, stem: str, box: TFBox, lo: int, hi: int) -> float | None:
    """Recompute ``detector`` on [lo, hi) and return its max evidence inside the box (None if not applicable)."""
    if detector not in detector_registry.available():
        return None
    seg = {k: v[..., lo:hi] for k, v in stems.items()}
    ctx = AnalysisContext(mixture[..., lo:hi], seg, sr, cfg)
    t_off = lo / sr
    best = 0.0
    for em in detector_registry.create([detector])[0].detect(ctx):
        if em.stem != stem:
            continue
        cols = (em.col_bounds_s[:, 0] + t_off < box.end_s) & (em.col_bounds_s[:, 1] + t_off > box.start_s)
        rows = (em.row_bounds_hz[:, 0] < box.freq_high_hz) & (em.row_bounds_hz[:, 1] > box.freq_low_hz)
        if cols.any() and rows.any():
            best = max(best, float(em.evidence[np.ix_(rows, cols)].max()))
    return best


def evaluate_and_apply(
    proposal: RestorationProposal,
    ctx: RestorationContext,
    gate: GateConfig,
    analysis_cfg: AnalysisConfig,
    collector: list | None = None,
) -> Decision:
    r = proposal.region
    box = TFBox(r.start_s, r.end_s, r.freq_low_hz, r.freq_high_hz)
    a, b = proposal.excerpt
    g = ctx.grid
    O = stft(ctx.mixture[..., a:b].astype(np.float64), g, dtype=np.complex128)
    mask = box_mask(g, O.shape[-1], box, frame_offset=a // g.hop)
    S = {k: stft(v[..., a:b].astype(np.float64), g, dtype=np.complex128) for k, v in ctx.stems.items()}
    R = sum(S.values())
    err_before = _box_sum(np.abs(O - R) ** 2, mask)
    stem_energy = sum(_box_sum(np.abs(s) ** 2, mask) for s in S.values())

    changed = [k for k in proposal.new_audio if k in S]
    base = dict(
        region_id=r.id, stem=r.stem, restorer=proposal.restorer, stems_changed=changed, err_before=err_before,
        stem_energy_in_box=stem_energy, notes=list(proposal.notes),
    )
    if not changed:
        return Decision(accepted=False, reasons=["no change proposed"], err_after=err_before, err_delta_db=0.0, stem_change_rel_db=None,
                        added_energy=0.0, removed_energy=0.0, unexplained_added_energy=0.0, redistributed_energy=0.0, evidence_before=None, evidence_after=None,
                        projection_discarded_rel_db=None, support_samples=None, **base)

    new_specs, deltas, discarded, total_prop = {}, {}, 0.0, 0.0
    for k in changed:
        prop = proposal.new_audio[k]
        if prop.shape != (ctx.stems[k].shape[0], b - a) or not np.all(np.isfinite(prop)):
            return Decision(accepted=False, reasons=[f"invalid proposal for {k} (shape/finite)"], err_after=err_before, err_delta_db=0.0,
                            stem_change_rel_db=None, added_energy=0.0, removed_energy=0.0, unexplained_added_energy=0.0, redistributed_energy=0.0,
                            evidence_before=None, evidence_after=None, projection_discarded_rel_db=None, support_samples=None, **base)
        D_full = stft(prop.astype(np.float64), g, dtype=np.complex128) - S[k]
        D = D_full * mask
        discarded += float(np.sum(np.abs(D_full - D) ** 2))
        total_prop += float(np.sum(np.abs(D_full) ** 2))
        new_specs[k] = S[k] + D
        deltas[k] = istft(D, g, b - a)

    R2 = R + sum(new_specs[k] - S[k] for k in changed)
    err_after = _box_sum(np.abs(O - R2) ** 2, mask)
    change = sum(_box_sum(np.abs(new_specs[k] - S[k]) ** 2, mask) for k in changed)
    per_cell = sum((np.abs(new_specs[k]) ** 2 - np.abs(S[k]) ** 2).sum(axis=0) * mask for k in changed)
    added = float(np.sum(np.maximum(per_cell, 0)))
    removed = float(-np.sum(np.minimum(per_cell, 0)))
    D_sum = sum(new_specs[k] - S[k] for k in changed)
    net_added = _box_sum(np.abs(D_sum) ** 2, mask)
    unexplained = max(0.0, net_added - err_before)
    redistributed = max(0.0, change - net_added)

    # candidate new stems (time domain), locality verified bitwise
    new_stems = dict(ctx.stems)
    support = None
    for k in changed:
        y = ctx.stems[k].copy()
        y[..., a:b] = (ctx.stems[k][..., a:b].astype(np.float64) + deltas[k]).astype(ctx.stems[k].dtype)
        sup = changed_support(ctx.stems[k], y)
        if sup is not None:
            support = sup if support is None else (min(support[0], sup[0]), max(support[1], sup[1]))
        new_stems[k] = y

    reasons = []
    tol = 10 ** (gate.tolerance_db / 10)
    if err_after > err_before * tol + 1e-12:
        reasons.append(f"mixture error in region increased by {_db_ratio(err_after, err_before):+.2f} dB")
    if unexplained > gate.max_new_energy_ratio * max(stem_energy, err_before):
        reasons.append(f"adds energy not explained by the missing residual ({_db_ratio(unexplained, max(stem_energy, err_before)):+.1f} dB)")
    if support is not None and (support[0] < a or support[1] > b):
        reasons.append("locality violated: samples outside the excerpt changed")

    ev_before = ev_after = None
    det = next(iter(r.detectors), None)
    if gate.require_evidence_decrease and det is not None:
        lo = max(0, int((r.start_s - gate.evidence_margin_s) * ctx.sample_rate))
        hi = min(ctx.mixture.shape[-1], int((r.end_s + gate.evidence_margin_s) * ctx.sample_rate))
        ev_before = _evidence_in_box(ctx.stems, ctx.mixture, ctx.sample_rate, analysis_cfg, det, r.stem, box, lo, hi)
        if ev_before is not None:
            ev_after = _evidence_in_box(new_stems, ctx.mixture, ctx.sample_rate, analysis_cfg, det, r.stem, box, lo, hi)
            if ev_after > ev_before + 1e-6:
                reasons.append(f"{det} evidence increased ({ev_before:.2f} -> {ev_after:.2f})")

    accepted = not reasons
    if change <= 0:
        accepted, reasons = False, ["projection left no change inside the region"]
    if collector is not None:  # projected candidate, i.e. exactly what would be applied (for ground-truth scoring)
        collector.append({"region_id": r.id, "accepted": accepted, "excerpt": (a, b), "box": box,
                          "audio": {k: new_stems[k][..., a:b].astype(np.float64) for k in changed}})
    if accepted:
        ctx.stems.update({k: new_stems[k] for k in changed})
    return Decision(
        accepted=accepted,
        reasons=reasons or ["all acceptance criteria met"],
        err_after=err_after,
        err_delta_db=_db_ratio(err_after, err_before) if err_before > 0 else 0.0,
        stem_change_rel_db=_db_ratio(change, stem_energy),
        added_energy=added,
        removed_energy=removed,
        unexplained_added_energy=unexplained,
        redistributed_energy=redistributed,
        evidence_before=ev_before,
        evidence_after=ev_after,
        projection_discarded_rel_db=_db_ratio(discarded, total_prop) if total_prop > 0 else None,
        support_samples=support,
        **base,
    )
