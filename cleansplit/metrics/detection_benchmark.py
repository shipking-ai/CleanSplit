"""Synthetic detection benchmark: how well does artifact analysis find known, injected corruptions, and how much
does it report on clean ground-truth stems? This is the objective check that stands in for listening while the
detectors are deterministic heuristics."""

from __future__ import annotations

import time

import numpy as np

from ..analysis.pipeline import analyze_separation
from ..artifacts import corruptions as C
from ..audio.synthetic import make_song
from ..audio.tf_edit import TFBox
from ..config.settings import AnalysisConfig
from ..separation.base import SeparationResult

# (name, factory(stems, rng) -> (stems, GroundTruth), expected [(stem_role, detector_family)])
# stem_role: "target" = gt.stem, "sink" = gt.sink, "mixture"


def _box(rng, t_len=(0.8, 1.5), f_lo=(300.0, 3000.0), f_ratio=(2.0, 4.0), dur=8.0):
    L = rng.uniform(*t_len)
    t0 = rng.uniform(0.8, dur - L - 0.8)
    lo = rng.uniform(*f_lo)
    return TFBox(t0, t0 + L, lo, min(lo * rng.uniform(*f_ratio), 16000.0))


SCENARIOS = [
    ("warble_conserve", lambda s, r: C.warble(s, "vocals", _box(r, f_lo=(600, 1500)), conserve=True, sink="other", seed=int(r.integers(1 << 30))), [("target", "modulation")]),
    ("warble_lossy", lambda s, r: C.warble(s, "vocals", _box(r, f_lo=(600, 1500)), conserve=False, seed=int(r.integers(1 << 30))), [("target", "modulation"), ("mixture", "residual")]),
    ("cancellation", lambda s, r: C.cancellation(s, "guitar", "piano", _box(r, f_lo=(800, 3000)), seed=int(r.integers(1 << 30))), [("target", "cancellation"), ("sink", "cancellation")]),
    ("smear_conserve", lambda s, r: C.smear(s, "drums", _box(r, f_lo=(1000, 2000), f_ratio=(6, 8)), conserve=True, sink="other"), [("target", "transient")]),
    ("smear_lossy", lambda s, r: C.smear(s, "drums", _box(r, f_lo=(1000, 2000), f_ratio=(6, 8)), conserve=False), [("target", "transient"), ("mixture", "residual")]),
    ("hf_noise_conserve", lambda s, r: C.hf_noise(s, "vocals", _box(r, f_lo=(8000, 10000), f_ratio=(1.3, 1.5)), level_db=-10, conserve=True, sink="drums", seed=int(r.integers(1 << 30))), [("target", "cancellation")]),
    ("hf_noise_lossy", lambda s, r: C.hf_noise(s, "vocals", _box(r, f_lo=(8000, 10000), f_ratio=(1.3, 1.5)), level_db=-10, seed=int(r.integers(1 << 30))), [("mixture", "residual")]),
    ("leakage_drums_into_vocals", lambda s, r: C.leakage(s, "drums", "vocals", _box(r, t_len=(1.5, 2.5), f_lo=(2000, 4000), f_ratio=(3, 4)), gain_db=-12), [("target", "leakage")]),
    ("dropout_lossy", lambda s, r: C.dropout(s, "piano", _box(r, t_len=(0.3, 0.6), f_lo=(200, 600), f_ratio=(3, 6))), [("mixture", "residual")]),
    ("musical_noise_into_vocals", lambda s, r: C.musical_noise(s, "vocals", "drums", _box(r, f_lo=(3000, 5000), f_ratio=(2, 3)), seed=int(r.integers(1 << 30))), [("target", "musical_noise")]),
]


def _hits(amap, stem, gt, detector):
    out = []
    for reg in amap.regions:
        if reg.stem != stem or detector not in reg.detectors:
            continue
        if not (reg.start_s < gt.end_s and gt.start_s < reg.end_s):
            continue
        if stem != "mixture" and not (reg.freq_low_hz < gt.freq_high_hz and gt.freq_low_hz < reg.freq_high_hz):
            continue
        out.append(reg)
    return out


def run_benchmark(seeds=(0, 1, 2), duration_s: float = 8.0, config: AnalysisConfig | None = None, threshold: float = 0.6, log=print) -> dict:
    config = config or AnalysisConfig()
    t0 = time.time()
    clean_fp = {}
    clean_regions = 0
    scen = {name: {"trials": 0, "detected": 0, "per_expectation": {}} for name, _, _ in SCENARIOS}
    for seed in seeds:
        mix, stems = make_song(duration_s, seed=seed)
        res = analyze_separation(mix, SeparationResult(stems, 44100, "oracle", {"zero_dc": False}), config)
        for reg in res.artifact_map.regions:
            if reg.confidence >= threshold:
                clean_regions += 1
                for d in reg.detectors:
                    clean_fp[d] = clean_fp.get(d, 0) + 1
        log(f"seed {seed}: clean stems -> {sum(r.confidence >= threshold for r in res.artifact_map.regions)} regions >= {threshold}")
        rng = np.random.default_rng(1000 + seed)
        for name, factory, expectations in SCENARIOS:
            corrupted, gt = factory(stems, rng)
            res = analyze_separation(mix, SeparationResult(corrupted, 44100, "oracle", {"zero_dc": False}), config)
            ok_all = True
            for role, det in expectations:
                stem = {"target": gt.stem, "sink": gt.sink, "mixture": "mixture"}[role]
                hits = [h for h in _hits(res.artifact_map, stem, gt, det) if h.detectors[det].confidence >= threshold]
                key = f"{role}:{det}"
                pe = scen[name]["per_expectation"].setdefault(key, {"trials": 0, "hits": 0})
                pe["trials"] += 1
                pe["hits"] += bool(hits)
                ok_all &= bool(hits)
            scen[name]["trials"] += 1
            scen[name]["detected"] += ok_all
            log(f"  {name:28s} {'DETECTED' if ok_all else 'missed'}  box={gt.start_s:.2f}-{gt.end_s:.2f}s {gt.freq_low_hz:.0f}-{gt.freq_high_hz:.0f}Hz")
    minutes = len(seeds) * duration_s / 60.0
    for v in scen.values():
        v["recall"] = v["detected"] / max(v["trials"], 1)
    return {
        "seeds": list(seeds),
        "duration_s": duration_s,
        "confidence_threshold": threshold,
        "enabled_detectors": list(config.enabled_detectors),
        "clean": {"regions": clean_regions, "regions_per_minute": clean_regions / minutes, "by_detector": clean_fp},
        "scenarios": scen,
        "seconds": round(time.time() - t0, 1),
        "caveat": "Synthetic caricature instruments with injected corruptions; indicates detector mechanics, not real-world accuracy.",
    }
