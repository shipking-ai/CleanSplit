"""Ground-truth restoration experiment.

    known stems -> mix -> real separator -> analysis -> restorer + gate -> fidelity vs the TRUE stems

This is the test that can say "no": a restoration may be mixture-consistent and still move stems away from the
truth. Every restorer (including the identity control) sees the same regions and is scored identically.

Truth may be coarser than the separator's stems. ``groups`` maps each truth stem to the separator stems whose SUM
estimates it (e.g. truth 'other' <- guitar + piano + other). All scores are computed on these group sums.

Primary score (per proposal, per affected group), computed for EVERY proposal whether accepted or not:
  box_err_change_db = 10log10( ||STFT(G' - T)||^2 / ||STFT(G - T)||^2 ) inside the region's TF box
  (negative = closer to truth). Restricting to the box matters: region edits touch a few hundred Hz, and a
  time-domain measure over all frequencies dilutes any effect toward 0 dB.
Secondary: whole-stem SNR-form SDR of each group before/after the accepted changes.

Caveats: synthetic caricature stems are out of domain for real separators; stems from another separator
(e.g. a commercial service) are real music but carry that separator's own artifacts in the "truth".
"""

from __future__ import annotations

import time

import numpy as np

from ..analysis.pipeline import analyze_separation
from ..audio.stft import STFTGrid, stft
from ..audio.synthetic import make_song
from ..audio.tf_edit import box_mask
from ..config.settings import AnalysisConfig
from ..restoration.pipeline import restore_in_memory
from . import signal as M

IDENTITY_GROUPS = None  # truth stem names == separator stem names


def _group_sum(stems: dict, members) -> np.ndarray:
    out = None
    for k in members:
        x = stems[k].astype(np.float64)
        out = x if out is None else out + x
    return out


def _resolve_groups(truth: dict, est_names, groups):
    if groups is None:
        return {k: [k] for k in truth}
    missing = [m for ms in groups.values() for m in ms if m not in est_names]
    if missing:
        raise KeyError(f"groups reference unknown separator stems {missing}")
    return groups


def _box_truth_changes(proposals, truth, before, groups, grid: STFTGrid):
    if not proposals:
        return []
    out = []
    for p in proposals:
        a, b = p["excerpt"]
        box = p["box"]
        changed = set(p["audio"])
        for tname, members in groups.items():
            if not changed & set(members):
                continue
            T = truth[tname][..., a:b].astype(np.float64)
            old = sum(before[m][..., a:b].astype(np.float64) for m in members)
            new = sum((p["audio"][m] if m in p["audio"] else before[m][..., a:b].astype(np.float64)) for m in members)
            ST = stft(T, grid, dtype=np.complex128)
            mask = box_mask(grid, ST.shape[-1], box, frame_offset=a // grid.hop)
            eb = float(np.sum(np.abs((stft(old, grid, dtype=np.complex128) - ST) * mask) ** 2))
            ea = float(np.sum(np.abs((stft(new, grid, dtype=np.complex128) - ST) * mask) ** 2))
            tb = float(np.sum(np.abs(ST * mask) ** 2))
            out.append({
                "region_id": p["region_id"], "truth_stem": tname, "accepted": p["accepted"],
                "box_err_change_db": 10 * np.log10(max(ea, 1e-30) / max(eb, 1e-30)),
                "box_err_before_rel_truth_db": 10 * np.log10(max(eb, 1e-30) / max(tb, 1e-30)),
                "err_before": eb, "err_after": ea, "truth_energy": tb,
            })
    return out


ERR_FLOOR_REL_TRUTH = 1e-3  # errors below -30 dB of the true energy in the box count as "already perfect"


def _box_total_changes(box_rows):
    """Per region: 10log10((sum_groups err_after + floor) / (sum_groups err_before + floor)), floor = 1e-3 * truth energy.
    Summing over groups keeps near-silent groups from dominating; the floor keeps already-perfect regions from
    producing arbitrarily large ratios."""
    by = {}
    for x in box_rows:
        r = by.setdefault(x["region_id"], {"accepted": x["accepted"], "eb": 0.0, "ea": 0.0, "tb": 0.0})
        r["eb"] += x["err_before"]
        r["ea"] += x["err_after"]
        r["tb"] += x["truth_energy"]
    out = []
    for k, v in by.items():
        fl = ERR_FLOOR_REL_TRUTH * v["tb"] + 1e-30
        out.append({
            "region_id": k, "accepted": v["accepted"],
            "total_err_change_db": 10 * np.log10((v["ea"] + fl) / (v["eb"] + fl)),
            "total_err_before_rel_truth_db": 10 * np.log10(max(v["eb"], 1e-30) / max(v["tb"], 1e-30)),
            "err_before": v["eb"], "err_after": v["ea"], "truth_energy": v["tb"],
        })
    return out


def pooled_change_db(totals, accepted_only=False):
    """10log10(sum err_after / sum err_before) over regions (floored per region). None if no regions."""
    rows = [t for t in totals if t["accepted"] or not accepted_only]
    if not rows:
        return None
    ea = sum(t["err_after"] + ERR_FLOOR_REL_TRUTH * t["truth_energy"] for t in rows)
    eb = sum(t["err_before"] + ERR_FLOOR_REL_TRUTH * t["truth_energy"] for t in rows)
    return float(10 * np.log10(max(ea, 1e-30) / max(eb, 1e-30)))


def _fmt(x):
    return "n/a" if x is None else f"{x:+.2f} dB"


def run_experiment(
    separator,
    restorers=("identity", "residual_reallocation", "region_wiener"),
    seeds=(0, 1, 2),
    duration_s: float = 20.0,
    songs=None,
    groups=IDENTITY_GROUPS,
    min_confidence: float = 0.6,
    max_regions: int | None = 200,
    config: AnalysisConfig | None = None,
    log=print,
    region_scope: str = "all",
) -> dict:
    """``songs``: optional iterable of (name, mixture, truth_stems); otherwise synthetic songs per seed."""
    config = config or AnalysisConfig()
    grid = STFTGrid(44100, config.n_fft, config.hop)
    if songs is None:
        songs = [(f"synthetic_seed{s}", *make_song(duration_s, seed=s)) for s in seeds]
    results = []
    t0 = time.time()
    for name, mix, truth in songs:
        sep = separator.separate(mix, separator.sample_rate)
        g = _resolve_groups(truth, list(sep.stems), groups)
        zero_dc = bool(sep.metadata.get("zero_dc", False))
        analysis = analyze_separation(mix, sep, config)
        n_regions = sum(r.confidence >= min_confidence and (region_scope == "all" or (region_scope == "mixture") == (r.stem == "mixture")) for r in analysis.artifact_map.regions)
        log(f"{name}: separated, {n_regions} eligible regions, mixture SNR {analysis.metrics['mixture']['model_matched']['snr_db']:.1f} dB")
        entry = {
            "song": name if region_scope == "all" else f"{name}[{region_scope}]",
            "region_scope": region_scope,
            "separator": sep.separator,
            "groups": g,
            "mixture_snr_model_matched_db": analysis.metrics["mixture"]["model_matched"]["snr_db"],
            "eligible_regions": int(n_regions),
            "separation_sdr_db": {t: M.snr_db(truth[t].astype(np.float64), _group_sum(sep.stems, ms)) for t, ms in g.items()},
            "restorers": {},
        }
        for rname in restorers:
            tr = time.time()
            res = restore_in_memory(
                mix, sep.stems, analysis.artifact_map, rname, sep.sample_rate, zero_dc, min_confidence,
                analysis_cfg=config, max_regions=max_regions, record_proposals=True, region_scope=region_scope,
            )
            decisions = res["summary"]["passes"][0].get("decisions", []) if res["summary"]["passes"] else []
            box = _box_truth_changes(res.get("proposals_audio"), truth, sep.stems, g, grid)
            sdr = {
                t: {"sdr_before_db": M.snr_db(truth[t].astype(np.float64), _group_sum(sep.stems, ms)),
                    "sdr_after_db": M.snr_db(truth[t].astype(np.float64), _group_sum(res["stems"], ms))}
                for t, ms in g.items()
            }
            totals = _box_total_changes(box)
            acc = [x["total_err_change_db"] for x in totals if x["accepted"]]
            allp = [x["total_err_change_db"] for x in totals]
            entry["restorers"][rname] = {
                "accepted": int(sum(d["accepted"] for d in decisions)),
                "rejected": int(sum(not d["accepted"] for d in decisions)),
                "decisions": [
                    {k: d[k] for k in ("region_id", "stem", "accepted", "reasons", "stems_changed", "err_delta_db", "stem_change_rel_db",
                                       "unexplained_added_energy", "stem_energy_in_box", "err_before", "evidence_before", "evidence_after", "notes")}
                    for d in decisions
                ],
                "box_truth_changes": box,
                "box_total_changes": totals,
                "pooled_change_db_accepted": pooled_change_db(totals, True),
                "pooled_change_db_all_proposals": pooled_change_db(totals),
                "whole_stem_sdr": sdr,
                "reconstruction": res["summary"]["reconstruction"],
                "seconds": round(time.time() - tr, 1),
            }
            log(
                f"  {rname:22s} accepted {entry['restorers'][rname]['accepted']:3d}/{len(decisions):<3d} "
                f"stem error vs truth in boxes, pooled: accepted {_fmt(pooled_change_db(totals, True))} (n={len(acc)}), "
                f"all proposals {_fmt(pooled_change_db(totals))}, closer {sum(v < -0.05 for v in allp)}/{len(allp)}, worse {sum(v > 0.05 for v in allp)}/{len(allp)}"
            )
        results.append(entry)
    return {"results": results, "seconds": round(time.time() - t0, 1), "doc": __doc__}


def stem_folder_songs(stem_paths: dict, excerpts, derive: dict | None = None, drop=()):
    """Build (name, mixture, truth) excerpts from real stem files. The mixture is the exact sum of the truth stems.

    stem_paths: name -> audio path. derive: new truth name -> (plus, minus) name lists, e.g.
    {"other": (["instrumentals"], ["drums", "bass"])}. drop: source names removed from truth after deriving
    (e.g. "instrumentals", which overlaps drums and bass). excerpts: list of (start_s, dur_s)."""
    from ..audio.conform import conform_channels
    from ..audio.io import load_audio

    raw = {}
    for k, p in stem_paths.items():
        d = load_audio(p)
        if d.sample_rate != 44100:
            raise ValueError(f"{p}: expected 44.1 kHz")
        raw[k] = conform_channels(d.audio, 2)[0].astype(np.float64)
    truth_full = dict(raw)
    for name, (plus, minus) in (derive or {}).items():
        truth_full[name] = sum(raw[k] for k in plus) - sum(raw[k] for k in minus)
    for k in drop:
        truth_full.pop(k)
    songs = []
    for start, dur in excerpts:
        a, b = int(start * 44100), int((start + dur) * 44100)
        truth = {k: v[:, a:b].astype(np.float32) for k, v in truth_full.items()}
        mix = sum(v[:, a:b] for v in truth_full.values())
        peak = float(np.max(np.abs(mix)))
        gain = min(1.0, 10 ** (-1 / 20) / peak) if peak > 0 else 1.0  # same gain on mix and truth: exact decomposition kept
        truth = {k: (v * gain).astype(np.float32) for k, v in truth.items()}
        mixture = np.zeros_like(mix)
        for v in truth.values():
            mixture += v.astype(np.float64)
        songs.append((f"excerpt_{start:.0f}s", mixture.astype(np.float32), truth))
    return songs
