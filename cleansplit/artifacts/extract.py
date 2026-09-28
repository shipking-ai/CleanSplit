"""EvidenceMap -> ArtifactRegion: hysteresis thresholding, 8-connected components, bounding boxes, then
conservative cross-detector merging (only strongly overlapping boxes merge, so regions never chain into
whole-song boxes)."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from ..analysis.context import TINY, AnalysisContext, to_db
from ..config.settings import AnalysisConfig
from .detectors.base import EvidenceMap
from .region import ArtifactRegion, DetectorEvidence


def regions_from_evidence(em: EvidenceMap, cfg: AnalysisConfig) -> tuple[list[ArtifactRegion], dict]:
    ev = em.evidence
    low = ev >= cfg.hysteresis_low
    stats = {"cells": int(ev.size), "cells_high": int(np.count_nonzero(ev >= cfg.hysteresis_high)), "components": 0, "kept": 0}
    if not low.any():
        return [], stats
    labels, n = ndimage.label(low, structure=np.ones((3, 3), dtype=bool))
    stats["components"] = int(n)
    idx = np.arange(1, n + 1)
    peak = ndimage.maximum(ev, labels, idx)
    mean = ndimage.mean(ev, labels, idx)
    count = ndimage.sum_labels(np.ones_like(ev, dtype=np.int32), labels, idx)
    mpeak = ndimage.maximum(em.measure, labels, idx)
    mmean = ndimage.mean(em.measure, labels, idx)
    type_frac = {t: ndimage.mean(mask.astype(np.float32), labels, idx) for t, mask in em.type_masks.items()}
    min_cells = int(em.extra.get("min_cells", cfg.min_region_cells))
    min_rows = int(em.extra.get("min_rows", 1))
    objs = ndimage.find_objects(labels)
    out = []
    for k, sl in enumerate(objs):
        if sl is None or peak[k] < cfg.hysteresis_high or count[k] < min_cells:
            continue
        rs, cs = sl
        if rs.stop - rs.start < min_rows:
            continue
        conf = float(min(em.max_confidence, 0.5 * (peak[k] + mean[k])))
        types = list(em.artifact_types)
        types += [t for t, fr in type_frac.items() if fr[k] > 0.5]
        out.append(
            ArtifactRegion(
                stem=em.stem,
                start_s=float(em.col_bounds_s[cs.start, 0]),
                end_s=float(em.col_bounds_s[cs.stop - 1, 1]),
                freq_low_hz=float(em.row_bounds_hz[rs.start, 0]),
                freq_high_hz=float(em.row_bounds_hz[rs.stop - 1, 1]),
                confidence=round(conf, 4),
                artifact_types=types,
                detectors={
                    em.detector: DetectorEvidence(
                        confidence=round(conf, 4),
                        measure_name=em.measure_name,
                        measure_peak=round(float(mpeak[k]), 3),
                        measure_mean=round(float(mmean[k]), 3),
                        extra={"cells": int(count[k])},
                    )
                },
            )
        )
    stats["kept"] = len(out)
    return out, stats


def _area(r: ArtifactRegion) -> float:
    return max(r.duration_s, 1e-6) * max(r.freq_high_hz - r.freq_low_hz, 1e-6)


def _intersection(a: ArtifactRegion, b: ArtifactRegion) -> float:
    dt = min(a.end_s, b.end_s) - max(a.start_s, b.start_s)
    df = min(a.freq_high_hz, b.freq_high_hz) - max(a.freq_low_hz, b.freq_low_hz)
    return dt * df if dt > 0 and df > 0 else 0.0


def merge_regions(regions: list[ArtifactRegion], min_overlap: float = 0.6) -> list[ArtifactRegion]:
    """Merge boxes of the same stem from different detectors when intersection/smaller-area >= min_overlap."""
    regions = sorted(regions, key=lambda r: (r.stem, r.start_s))
    merged: list[ArtifactRegion] = []
    active: list[ArtifactRegion] = []
    for r in regions:
        active = [a for a in active if a.stem == r.stem and a.end_s > r.start_s]
        target = None
        for a in active:
            if set(a.detectors) & set(r.detectors):
                continue
            inter = _intersection(a, r)
            if inter > 0 and inter / min(_area(a), _area(r)) >= min_overlap:
                target = a
                break
        if target is None:
            merged.append(r)
            active.append(r)
            continue
        target.start_s = min(target.start_s, r.start_s)
        target.end_s = max(target.end_s, r.end_s)
        target.freq_low_hz = min(target.freq_low_hz, r.freq_low_hz)
        target.freq_high_hz = max(target.freq_high_hz, r.freq_high_hz)
        target.confidence = max(target.confidence, r.confidence)
        target.artifact_types = sorted(set(target.artifact_types) | set(r.artifact_types))
        target.detectors.update(r.detectors)
    return merged


def annotate_levels(regions: list[ArtifactRegion], ctx: AnalysisContext) -> None:
    g = ctx.grid
    mix_max = float(ctx.Ps_mix.max()) + TINY
    stem_max = {k: float(v.max()) + TINY for k, v in ctx.Ps_stems.items()}
    for r in regions:
        f0, f1 = g.hz_to_bin(r.freq_low_hz), g.hz_to_bin(r.freq_high_hz) + 1
        t0, t1 = g.time_to_frame(r.start_s), g.time_to_frame(r.end_s) + 1
        t1 = min(max(t1, t0 + 1), ctx.n_frames)
        mix_cell = float(ctx.Ps_mix[f0:f1, t0:t1].mean()) if t0 < ctx.n_frames else 0.0
        r.mixture_level_db = round(float(to_db(mix_cell + TINY) - to_db(mix_max)), 2)
        if r.stem in ctx.Ps_stems and t0 < ctx.n_frames:
            s_cell = float(ctx.Ps_stems[r.stem][f0:f1, t0:t1].mean())
            r.stem_level_db = round(float(to_db(s_cell + TINY) - to_db(stem_max[r.stem])), 2)


def annotate_hf_character(regions: list[ArtifactRegion], ctx: AnalysisContext) -> None:
    """Label regions lying above ``hf_label_min_hz`` whose stem content is noise-like (energy-weighted spectral
    flatness) as ``high_frequency_noise``. This labels what physical detectors found; it detects nothing itself."""
    c = ctx.config
    g = ctx.grid
    for r in regions:
        if r.stem not in ctx.P_stems or r.freq_low_hz < c.hf_label_min_hz:
            continue
        f0, f1 = g.hz_to_bin(r.freq_low_hz), g.hz_to_bin(r.freq_high_hz) + 1
        t0, t1 = g.time_to_frame(r.start_s), min(g.time_to_frame(r.end_s) + 1, ctx.n_frames)
        if f1 - f0 < 4 or t1 <= t0:
            continue
        Q = ctx.P_stems[r.stem][f0:f1, t0:t1].astype(np.float64) + 1e-12
        am = Q.mean(axis=0)
        flat = float(np.sum(np.exp(np.log(Q).mean(axis=0))) / max(np.sum(am), 1e-30))
        if flat >= c.hf_label_min_flatness:
            r.artifact_types = sorted(set(r.artifact_types) | {"high_frequency_noise"})
            r.notes.append(f"stem content in region is noise-like (flatness {flat:.2f})")


def cap_and_number(regions: list[ArtifactRegion], cfg: AnalysisConfig) -> tuple[list[ArtifactRegion], dict]:
    by_stem: dict[str, list[ArtifactRegion]] = {}
    for r in regions:
        by_stem.setdefault(r.stem, []).append(r)
    kept, truncated = [], {}
    for stem, rs in by_stem.items():
        rs.sort(key=lambda r: r.confidence * np.log1p(_area(r)), reverse=True)
        if len(rs) > cfg.max_regions_per_stem:
            truncated[stem] = len(rs) - cfg.max_regions_per_stem
            rs = rs[: cfg.max_regions_per_stem]
        rs.sort(key=lambda r: (r.start_s, r.freq_low_hz))
        for i, r in enumerate(rs):
            r.id = f"{stem}-{i:05d}"
        kept.extend(rs)
    kept.sort(key=lambda r: (r.start_s, r.stem))
    return kept, truncated
