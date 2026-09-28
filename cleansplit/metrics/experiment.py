"""Ground-truth restoration experiment: the harness that can prove the core hypothesis wrong.

Mixture consistency is necessary but NOT sufficient: simply distributing the residual to stems improves consistency
by construction. Whether a restorer makes stems *more correct* can only be measured against true stems. For each
restorer this reports, per stem and against the reference:
  * global SDR / SI-SDR before vs after,
  * error energy restricted to the samples the restorer actually changed (before vs after),
  * how many proposals the mixture-consistency gate accepted/rejected.
A restorer "helps" a stem only if the in-region reference error decreases; everything else is reported as-is.
"""

from __future__ import annotations

import time

import numpy as np

from ..analysis.pipeline import analyze_separation
from ..config.settings import AnalysisConfig
from ..restoration.engine import GateConfig
from ..restoration.pipeline import restore_in_memory
from ..separation.base import SeparationResult
from . import signal as M


def _err(ref, est, mask=None):
    d = est.astype(np.float64) - ref.astype(np.float64)
    r = ref.astype(np.float64)
    if mask is not None:
        d, r = d[:, mask], r[:, mask]
    return float(np.sum(d * d)), float(np.sum(r * r))


def run_experiment(
    mixture: np.ndarray,
    references: dict[str, np.ndarray],
    separation: SeparationResult,
    restorers=("identity", "residual_reallocation", "region_wiener"),
    analysis_cfg: AnalysisConfig | None = None,
    gate: GateConfig | None = None,
    min_confidence: float = 0.6,
    log=lambda m: None,
) -> dict:
    analysis_cfg = analysis_cfg or AnalysisConfig()
    t0 = time.time()
    zero_dc = bool(separation.metadata.get("zero_dc", False))
    analysis = analyze_separation(mixture, separation, analysis_cfg, zero_dc=zero_dc)
    est = analysis.reconstruction.stems
    common = [k for k in est if k in references]
    baseline = {}
    for k in common:
        silent = float(np.mean(references[k].astype(np.float64) ** 2)) < 1e-10
        baseline[k] = {
            "reference_silent": silent,
            "sdr_db": None if silent else M.snr_db(references[k], est[k]),
            "si_sdr_db": None if silent else M.si_sdr_db(references[k], est[k]),
        }
    log(f"analysis: {len(analysis.artifact_map.regions)} regions")
    results = {}
    for name in restorers:
        tr = time.time()
        out = restore_in_memory(
            mixture, est, analysis.artifact_map, name, separation.sample_rate, zero_dc, min_confidence, gate, analysis_cfg
        )
        restored = out["stems"]
        decisions = [d for p in out["summary"]["passes"] for d in p.get("decisions", [])]
        per_stem = {}
        for k in common:
            changed = np.any(restored[k] != est[k], axis=0)
            e_before, ref_e = _err(references[k], est[k], changed) if changed.any() else (0.0, 0.0)
            e_after, _ = _err(references[k], restored[k], changed) if changed.any() else (0.0, 0.0)
            entry = {
                "changed_samples": int(changed.sum()),
                "global_sdr_db_after": None if baseline[k]["reference_silent"] else M.snr_db(references[k], restored[k]),
                "global_si_sdr_db_after": None if baseline[k]["reference_silent"] else M.si_sdr_db(references[k], restored[k]),
                "region_error_before_db": M.energy_db(np.sqrt(e_before)) if changed.any() else None,
                "region_error_change_db": (10 * np.log10(max(e_after, 1e-30) / e_before)) if changed.any() and e_before > 0 else None,
                "region_reference_energy_db": M.energy_db(np.sqrt(ref_e)) if changed.any() else None,
            }
            ch = entry["region_error_change_db"]
            entry["verdict"] = "unchanged" if ch is None else ("closer_to_truth" if ch < -0.1 else ("further_from_truth" if ch > 0.1 else "neutral"))
            per_stem[k] = entry
        results[name] = {
            "accepted": sum(d["accepted"] for d in decisions),
            "rejected": sum(not d["accepted"] for d in decisions),
            "mixture": out["summary"]["reconstruction"],
            "stems": per_stem,
            "seconds": round(time.time() - tr, 1),
        }
        log(f"{name}: accepted {results[name]['accepted']}, rejected {results[name]['rejected']}")
    return {
        "separator": separation.separator,
        "regions": len(analysis.artifact_map.regions),
        "regions_by_detector": _by_detector(analysis.artifact_map),
        "baseline": baseline,
        "restorers": results,
        "seconds": round(time.time() - t0, 1),
        "interpretation": "Only 'region_error_change_db' against true stems measures correctness. Mixture metrics measure consistency only.",
    }


def _by_detector(amap):
    out = {}
    for r in amap.regions:
        for d in r.detectors:
            out[d] = out.get(d, 0) + 1
    return out
