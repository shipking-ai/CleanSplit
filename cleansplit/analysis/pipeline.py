"""V0 pipeline: validate -> separate -> align -> reconstruct -> residual -> artifact analysis -> map -> report."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import __version__
from ..artifacts import detectors as detector_registry
from ..artifacts.extract import annotate_hf_character, annotate_levels, cap_and_number, merge_regions, regions_from_evidence
from ..artifacts.region import ArtifactMap
from ..audio.conform import validate_and_conform
from ..audio.io import load_audio, save_audio
from ..audio.stft import STFTGrid
from ..config.settings import AnalysisConfig
from ..metrics import signal as M
from ..reconstruction.core import Reconstruction, peak_report, reconstruct
from ..separation.base import SeparationResult, Separator
from .context import TINY, AnalysisContext, to_db

log = logging.getLogger(__name__)


def song_slug(path: str | Path) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", Path(path).stem).strip("_") or "song"


def audio_digest(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x, dtype=np.float32).tobytes()).hexdigest()


def _json_safe(o):
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if np.isfinite(f) else M.finite_or_none(f)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _json_safe(o.tolist())
    return o


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(obj), indent=2, allow_nan=False), encoding="utf-8")


@dataclass
class AnalysisResult:
    reconstruction: Reconstruction
    artifact_map: ArtifactMap
    metrics: dict
    evidence_stats: dict = field(default_factory=dict)


def mixture_metrics(rec: Reconstruction, sr: int) -> dict:
    O, R, ref = rec.mixture, rec.reconstructed, rec.reference
    return {
        "full_band": {
            "snr_db": M.snr_db(O, R),
            "si_sdr_db": M.si_sdr_db(O, R),
            "residual_energy_rel_db": M.residual_energy_rel_db(O, rec.residual),
            "band_residual_db": M.band_residual_db(O, rec.residual),
        },
        "model_matched": {
            "note": "reference = mixture with the separator's deterministic processing applied (BS-RoFormer: STFT bin 0 zeroed)",
            "snr_db": M.snr_db(ref, R),
            "si_sdr_db": M.si_sdr_db(ref, R),
            "residual_energy_rel_db": M.residual_energy_rel_db(ref, rec.residual_matched),
            "band_residual_db": M.band_residual_db(ref, rec.residual_matched),
            "multi_mel_snr_db": M.multi_mel_snr_db(ref, R, sr),
            "log_spectral_distance": M.log_spectral_distance(ref, R),
        },
    }


def stem_metrics(rec: Reconstruction) -> dict:
    O = rec.mixture
    e_mix = float(np.sum(O**2)) + TINY
    out = {}
    for k, s in rec.stems.items():
        s64 = s.astype(np.float64)
        e = float(np.sum(s64**2))
        out[k] = {
            "energy_rel_mixture_db": float(to_db(e + TINY) - to_db(e_mix)),
            "rms_dbfs": float(to_db(np.mean(s64**2) + TINY)),
            "silent": e / s.size < 1e-10,
        }
    return out


def seam_diagnostics(ctx: AnalysisContext, chunk_starts: list[int], chunk_size: int | None) -> dict:
    """Residual and cancellation levels inside overlap-add crossfade zones vs elsewhere (broadband, per frame)."""
    if not chunk_starts or not chunk_size or len(chunk_starts) < 2:
        return {"available": False}
    g = ctx.grid
    audible = ctx.mix_audible
    res_db = to_db(ctx.Ps_res.sum(0) + TINY) - to_db(ctx.Ps_mix.sum(0) + TINY)
    canc = np.max(
        np.stack([to_db(v.sum(0) + TINY) - to_db(ctx.Ps_recon.sum(0) + TINY) for v in ctx.Ps_stems.values()]), axis=0
    )
    fade = chunk_size // 10
    zone = np.zeros(ctx.n_frames, dtype=bool)
    for s in chunk_starts[1:]:
        for a, b in ((s, s + fade),):
            zone[max(0, g.time_to_frame(a / g.sample_rate)) : min(ctx.n_frames, g.time_to_frame(b / g.sample_rate) + 1)] = True
        e = s + chunk_size  # fade-out of this chunk
        zone[max(0, g.time_to_frame((e - fade) / g.sample_rate)) : min(ctx.n_frames, g.time_to_frame(e / g.sample_rate) + 1)] = True
    active = audible.any(axis=0)
    if not (zone & active).any() or not (~zone & active).any():
        return {"available": False}
    return {
        "available": True,
        "crossfade_frames": int((zone & active).sum()),
        "residual_db_median_in_crossfades": float(np.median(res_db[zone & active])),
        "residual_db_median_elsewhere": float(np.median(res_db[~zone & active])),
        "max_stem_over_sum_db_median_in_crossfades": float(np.median(canc[zone & active])),
        "max_stem_over_sum_db_median_elsewhere": float(np.median(canc[~zone & active])),
    }


def analyze_separation(
    mixture: np.ndarray,
    sep: SeparationResult,
    config: AnalysisConfig | None = None,
    zero_dc: bool | None = None,
) -> AnalysisResult:
    config = config or AnalysisConfig()
    sr = sep.sample_rate
    if zero_dc is None:
        zero_dc = bool(sep.metadata.get("zero_dc", False))
    grid = STFTGrid(sr, config.n_fft, config.hop)
    t0 = time.time()
    rec = reconstruct(mixture, sep.stems, sample_rate=sr, zero_dc_grid=grid if zero_dc else None)
    ctx = AnalysisContext(rec.reference, rec.stems, sr, config, sep.chunk_starts, sep.chunk_size)

    regions, ev_stats, summaries = [], {}, {k: {} for k in list(rec.stems) + ["mixture"]}
    for det in detector_registry.create(config.enabled_detectors):
        td = time.time()
        maps = det.detect(ctx)
        for em in maps:
            rs, st = regions_from_evidence(em, config)
            st["seconds"] = round(time.time() - td, 2)
            st.update({k: v for k, v in em.extra.items() if k != "min_cells"})
            ev_stats[f"{det.name}/{em.stem}"] = st
            regions.extend(rs)
            flagged_s = _union_duration([(r.start_s, r.end_s) for r in rs])
            summaries[em.stem][det.name] = {
                "regions": len(rs),
                "flagged_cell_fraction": st["cells_high"] / max(st["cells"], 1),
                "flagged_duration_s": round(flagged_s, 3),
                "max_confidence": max((r.confidence for r in rs), default=0.0),
            }
        log.info("detector %s done in %.1fs", det.name, time.time() - td)

    regions = merge_regions(regions)
    annotate_levels(regions, ctx)
    annotate_hf_character(regions, ctx)
    regions, truncated = cap_and_number(regions, config)
    warnings = list(rec.alignment.warnings)
    if truncated:
        warnings.append(f"region lists truncated (kept highest confidence x area): {truncated}")

    amap = ArtifactMap(
        regions=regions,
        source={"separator": sep.separator, "sample_rate": sr, "stems": list(rec.stems)},
        analysis={
            "cleansplit_version": __version__,
            "grid": {"n_fft": config.n_fft, "hop": config.hop, "sample_rate": sr},
            "detectors": [d.describe() for d in detector_registry.create(config.enabled_detectors)],
            "config": config.to_dict(),
            "confidence_semantics": "heuristic evidence score in [0,1]; not a calibrated probability",
        },
        stem_summaries=summaries,
        warnings=warnings,
    )
    metrics = {
        "mixture": mixture_metrics(rec, sr),
        "stems": stem_metrics(rec),
        "peaks": peak_report({"mixture": rec.mixture, "reconstructed": rec.reconstructed, **rec.stems}),
        "alignment": rec.alignment.to_dict(),
        "chunk_seams": seam_diagnostics(ctx, sep.chunk_starts, sep.chunk_size),
        "analysis_seconds": round(time.time() - t0, 2),
    }
    return AnalysisResult(rec, amap, metrics, ev_stats)


def _union_duration(intervals) -> float:
    total, cur_s, cur_e = 0.0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


# ---------------------------------------------------------------- file-level orchestration

def load_cached_stems(stem_dir: Path, key: dict, names) -> dict | None:
    man = stem_dir / "manifest.json"
    if not man.is_file():
        return None
    try:
        data = json.loads(man.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if data.get("key") != _json_safe(key):
        return None
    stems = {}
    for n in names:
        p = stem_dir / f"{n}.wav"
        if not p.is_file():
            return None
        stems[n] = load_audio(p).audio
    return {"stems": stems, "manifest": data}


def run_analyze(
    input_path: str | Path,
    out_root: str | Path,
    separator: Separator,
    config: AnalysisConfig | None = None,
    reuse_stems: bool = True,
) -> dict:
    t_start = time.time()
    config = config or AnalysisConfig()
    out = Path(out_root) / song_slug(input_path)
    dec = load_audio(input_path)
    O, vrep = validate_and_conform(dec.audio, dec.sample_rate, separator.sample_rate, separator.channels)
    save_audio(out / "original.wav", O, separator.sample_rate)

    stem_dir = out / "stems"
    key = {"input_sha256": audio_digest(O), "separator": separator.cache_key()}
    cached = load_cached_stems(stem_dir, key, separator.stems) if reuse_stems else None
    if cached:
        m = cached["manifest"]
        sep = SeparationResult(cached["stems"], separator.sample_rate, m["separator"], m["metadata"], m.get("chunk_starts", []), m.get("chunk_size"))
        sep.metadata = {**sep.metadata, "reused_cached_stems": True}
    else:
        sep = separator.separate(O, separator.sample_rate)
        for name, s in sep.stems.items():
            save_audio(stem_dir / f"{name}.wav", s, sep.sample_rate)
        write_json(
            stem_dir / "manifest.json",
            {"key": key, "separator": sep.separator, "metadata": sep.metadata, "chunk_starts": sep.chunk_starts, "chunk_size": sep.chunk_size},
        )

    res = analyze_separation(O, sep, config)
    rec = res.reconstruction
    save_audio(out / "reconstruction" / "reconstructed.wav", rec.reconstructed, sep.sample_rate)
    save_audio(out / "reconstruction" / "residual.wav", rec.residual, sep.sample_rate)
    save_audio(out / "reconstruction" / "residual_model_matched.wav", rec.residual_matched, sep.sample_rate)
    if rec.alignment.lag_applied or rec.alignment.gain_applied or rec.alignment.polarity_fixed:
        for name, s in rec.stems.items():
            save_audio(out / "stems_aligned" / f"{name}.wav", s, sep.sample_rate)

    res.artifact_map.source.update({"input": str(Path(input_path).resolve()), "input_sha256_conformed": key["input_sha256"]})
    res.artifact_map.save(out / "analysis" / "artifact_map.json")
    write_json(out / "analysis" / "metrics.json", res.metrics)

    from ..models.device import probe

    amap = res.artifact_map
    by_type, by_stem = {}, {}
    for r in amap.regions:
        by_stem[r.stem] = by_stem.get(r.stem, 0) + 1
        for t in r.artifact_types:
            by_type[t] = by_type.get(t, 0) + 1
    top = sorted(amap.regions, key=lambda r: r.confidence, reverse=True)[:25]
    mm = res.metrics["mixture"]
    report = {
        "cleansplit_version": __version__,
        "input": {"path": str(Path(input_path).resolve()), "decoder": dec.decoder, "subtype": dec.source_subtype, "validation": vrep.to_dict()},
        "separator": {"name": sep.separator, "metadata": sep.metadata},
        "reconstruction_headline": {
            "snr_full_band_db": mm["full_band"]["snr_db"],
            "residual_full_band_rel_db": mm["full_band"]["residual_energy_rel_db"],
            "snr_model_matched_db": mm["model_matched"]["snr_db"],
            "residual_model_matched_rel_db": mm["model_matched"]["residual_energy_rel_db"],
        },
        "artifacts": {"regions_total": len(amap.regions), "regions_by_stem": by_stem, "regions_by_type": by_type, "top_regions": [r.to_dict() for r in top]},
        "stem_summaries": amap.stem_summaries,
        "evidence_stats": res.evidence_stats,
        "warnings": vrep.warnings + amap.warnings,
        "environment": probe().to_dict(),
        "limitations": [
            "Detector confidences are heuristic evidence scores, not calibrated probabilities.",
            "Mixture residual only measures mask-sum error; misassignment between stems is invisible to it.",
            "Leakage and modulation cues are ambiguous with genuine musical content (unison parts, tremolo).",
            "No ground-truth stems for real songs: findings are hypotheses to verify by listening or reference data.",
        ],
        "seconds_total": round(time.time() - t_start, 2),
    }
    write_json(out / "analysis" / "report.json", report)
    return {"output_dir": str(out), "report": report}
