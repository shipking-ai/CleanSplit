"""`cleansplit restore`: run a restorer on analyzed regions, gate every change, write before/after outputs.

Iteration: after a pass, the restored stems are re-analyzed; the loop stops when a pass accepts nothing, when the
number of eligible regions stops decreasing, or at ``max_iterations``.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np

from ..analysis.pipeline import analyze_separation, song_slug, write_json
from ..artifacts.region import ArtifactMap
from ..audio.io import load_audio, save_audio
from ..audio.stft import STFTGrid
from ..config.settings import AnalysisConfig
from ..metrics import signal as M
from ..reconstruction.core import model_matched_reference
from ..separation.base import SeparationResult
from .base import RestorationContext
from .baselines import create
from .engine import GateConfig, evaluate_and_apply


def restore_in_memory(
    mixture: np.ndarray,
    stems: dict[str, np.ndarray],
    artifact_map: ArtifactMap,
    restorer_name: str,
    sample_rate: int = 44100,
    zero_dc: bool = False,
    min_confidence: float = 0.6,
    gate: GateConfig | None = None,
    analysis_cfg: AnalysisConfig | None = None,
    max_iterations: int = 1,
    log=lambda m: None,
    max_regions: int | None = None,
    record_proposals: bool = False,
    region_scope: str = "all",
) -> dict:
    """region_scope: 'all', 'mixture' (reconstruction-level regions only) or 'stems' (per-stem artifact regions only)."""
    if region_scope not in ("all", "mixture", "stems"):
        raise ValueError(f"region_scope must be all|mixture|stems, got {region_scope!r}")
    gate = gate or GateConfig()
    analysis_cfg = analysis_cfg or AnalysisConfig()
    grid = STFTGrid(sample_rate, analysis_cfg.n_fft, analysis_cfg.hop)
    O_ref = model_matched_reference(mixture, grid, zero_dc) if zero_dc else mixture.astype(np.float64)
    restorer = create(restorer_name)
    current = {k: v.copy() for k, v in stems.items()}
    amap = artifact_map
    passes = []
    collector = [] if record_proposals else None
    prev_eligible = None
    for it in range(max_iterations):
        eligible = sorted(
            [r for r in amap.regions if r.confidence >= min_confidence
             and (region_scope == "all" or (region_scope == "mixture") == (r.stem == "mixture"))],
            key=lambda r: -r.confidence,
        )
        if max_regions is not None:
            eligible = eligible[:max_regions]
        if prev_eligible is not None and len(eligible) >= prev_eligible:
            passes.append({"iteration": it, "stopped": "eligible regions did not decrease", "eligible": len(eligible)})
            break
        ctx = RestorationContext(O_ref, current, sample_rate, grid, amap)
        decisions = []
        for r in eligible:
            for prop in restorer.restore(r.stem, [r], ctx):
                d = evaluate_and_apply(prop, ctx, gate, analysis_cfg, collector=collector if it == 0 else None)
                decisions.append(d)
        accepted = sum(d.accepted for d in decisions)
        log(f"pass {it}: {len(eligible)} regions, {accepted} accepted")
        passes.append({"iteration": it, "eligible": len(eligible), "accepted": accepted, "decisions": [d.to_dict() for d in decisions]})
        current = ctx.stems
        prev_eligible = len(eligible)
        if accepted == 0:
            passes[-1]["stopped"] = "no proposal accepted"
            break
        if it + 1 < max_iterations:
            amap = analyze_separation(mixture, SeparationResult(current, sample_rate, "restored", {"zero_dc": zero_dc}), analysis_cfg).artifact_map

    def recon(st):
        R = np.zeros(mixture.shape, dtype=np.float64)
        for s in st.values():
            R += s
        return R

    R0, R1 = recon(stems), recon(current)
    summary = {
        "restorer": restorer.describe(),
        "gate": gate.__dict__,
        "min_confidence": min_confidence,
        "reconstruction": {
            "before": {"snr_db": M.snr_db(O_ref, R0), "residual_rel_db": M.residual_energy_rel_db(O_ref, O_ref - R0)},
            "after": {"snr_db": M.snr_db(O_ref, R1), "residual_rel_db": M.residual_energy_rel_db(O_ref, O_ref - R1)},
        },
        "stem_changes": {
            k: {
                "bit_identical": bool(np.array_equal(stems[k], current[k])),
                "changed_samples": int(np.count_nonzero(np.any(stems[k] != current[k], axis=0))),
                "change_energy_rel_db": M.residual_energy_rel_db(stems[k], current[k].astype(np.float64) - stems[k]),
            }
            for k in stems
        },
        "passes": passes,
    }
    return {"stems": current, "summary": summary, "proposals_audio": collector}


def run_restore(input_path, out_root, restorer_name="residual_reallocation", min_confidence=0.6, tolerance_db=0.05, max_iterations=1, max_regions=None) -> dict:
    t0 = time.time()
    out = Path(out_root) / song_slug(input_path)
    amap_path = out / "analysis" / "artifact_map.json"
    if not amap_path.is_file():
        raise FileNotFoundError(f"{amap_path} not found; run `cleansplit analyze` first")
    amap = ArtifactMap.load(amap_path)
    orig = load_audio(out / "original.wav")
    manifest = json.loads((out / "stems" / "manifest.json").read_text(encoding="utf-8")) if (out / "stems" / "manifest.json").is_file() else {}
    zero_dc = bool(manifest.get("metadata", {}).get("zero_dc", False))
    stem_dir = out / "stems_aligned" if (out / "stems_aligned").is_dir() else out / "stems"
    stems = {name: load_audio(stem_dir / f"{name}.wav").audio for name in amap.source["stems"]}
    res = restore_in_memory(
        orig.audio, stems, amap, restorer_name, orig.sample_rate, zero_dc, min_confidence,
        GateConfig(tolerance_db=tolerance_db), max_iterations=max_iterations, max_regions=max_regions,
        log=lambda m: print(m, flush=True, file=__import__("sys").stderr),
    )
    rdir = out / "restoration" / _folder_name(restorer_name)
    for k, v in res["stems"].items():
        save_audio(rdir / "stems" / f"{k}.wav", v, orig.sample_rate)
    R = np.zeros(orig.audio.shape, dtype=np.float64)
    for v in res["stems"].values():
        R += v
    save_audio(rdir / "reconstructed.wav", R, orig.sample_rate)
    save_audio(rdir / "residual.wav", orig.audio.astype(np.float64) - R, orig.sample_rate)
    summary = res["summary"]
    summary["seconds"] = round(time.time() - t0, 1)
    write_json(rdir / "restoration_report.json", summary)
    last = summary["passes"][-1] if summary["passes"] else {}
    return {
        "output_dir": str(rdir),
        "reconstruction": summary["reconstruction"],
        "passes": [{k: v for k, v in p.items() if k != "decisions"} for p in summary["passes"]],
        "stems_bit_identical": {k: v["bit_identical"] for k, v in summary["stem_changes"].items()},
        "rejection_reasons": _reason_counts(last.get("decisions", [])),
    }


def _folder_name(restorer_name: str) -> str:
    """Restorer names may carry options ("apollo:variant=vocal_restore"); ':' and '=' are illegal in Windows paths."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", restorer_name).strip("_")


def _reason_counts(decisions):
    counts = {}
    for d in decisions:
        for reason in d["reasons"]:
            key = reason.split("(")[0].split(" by ")[0].strip()
            counts[key] = counts.get(key, 0) + 1
    return counts
