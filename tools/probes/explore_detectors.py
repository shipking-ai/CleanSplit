"""Dev exploration: detector hits on ground truth and false positives (confidence >= 0.6) elsewhere.

Usage: python tools/probes/explore_detectors.py [seed] [filter-substring]
"""
import sys

from cleansplit.analysis.pipeline import analyze_separation
from cleansplit.artifacts import corruptions as C
from cleansplit.audio.synthetic import make_song
from cleansplit.audio.tf_edit import TFBox
from cleansplit.separation.base import SeparationResult

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
flt = sys.argv[2] if len(sys.argv) > 2 else ""
mix, stems = make_song(8.0, seed=seed)


def ov(r, gt):
    return r.stem == gt.stem and r.start_s < gt.end_s and gt.start_s < r.end_s and r.freq_low_hz < gt.freq_high_hz and gt.freq_low_hz < r.freq_high_hz


def run(label, st, gt=None):
    res = analyze_separation(mix, SeparationResult(st, 44100, "oracle", {"zero_dc": False}))
    fp, hits = {}, {}
    for r in res.artifact_map.regions:
        if gt is not None and (ov(r, gt) or (r.stem == "mixture" and r.start_s < gt.end_s and gt.start_s < r.end_s)):
            for d, e in r.detectors.items():
                hits[d] = max(hits.get(d, 0), e.confidence)
        else:
            for d, e in r.detectors.items():
                if e.confidence >= 0.6:
                    fp.setdefault(f"{r.stem}/{d}", []).append(round(e.confidence, 2))
    print(f"{label:30s} hits={ {k: round(v, 2) for k, v in hits.items()} }  FP={ {k: (len(v), max(v)) for k, v in sorted(fp.items())} }", flush=True)


cases = [
    ("warble conserve vocals", lambda: C.warble(stems, "vocals", TFBox(3.0, 4.2, 800, 3000), conserve=True, sink="other")),
    ("warble lossy vocals", lambda: C.warble(stems, "vocals", TFBox(3.0, 4.2, 800, 3000), conserve=False)),
    ("warble conserve other", lambda: C.warble(stems, "other", TFBox(2.0, 3.5, 200, 1500), conserve=True, sink="piano")),
    ("cancellation guitar/piano", lambda: C.cancellation(stems, "guitar", "piano", TFBox(4.5, 5.2, 1500, 5000))),
    ("smear drums conserve", lambda: C.smear(stems, "drums", TFBox(2.0, 3.5, 1000, 16000), conserve=True, sink="other")),
    ("smear drums lossy", lambda: C.smear(stems, "drums", TFBox(2.0, 3.5, 1000, 16000), conserve=False)),
    ("hf_noise vocals lossy", lambda: C.hf_noise(stems, "vocals", TFBox(5.0, 6.0, 9000, 13000), level_db=-10)),
    ("hf_noise vocals conserve", lambda: C.hf_noise(stems, "vocals", TFBox(5.0, 6.0, 9000, 13000), level_db=-10, conserve=True, sink="drums")),
    ("leakage drums->vocals", lambda: C.leakage(stems, "drums", "vocals", TFBox(1.0, 3.0, 2000, 12000), gain_db=-12)),
    ("leakage vocals->guitar", lambda: C.leakage(stems, "vocals", "guitar", TFBox(4.0, 6.0, 500, 3000), gain_db=-15)),
    ("musical_noise guitar<-drums", lambda: C.musical_noise(stems, "guitar", "drums", TFBox(1.0, 3.0, 2000, 8000))),
    ("musical_noise vocals<-drums", lambda: C.musical_noise(stems, "vocals", "drums", TFBox(1.0, 3.0, 3000, 12000))),
    ("dropout piano lossy", lambda: C.dropout(stems, "piano", TFBox(6.0, 6.5, 200, 2000))),
    ("dropout piano conserve", lambda: C.dropout(stems, "piano", TFBox(6.0, 6.5, 200, 2000), conserve=True, sink="guitar")),
]
if not flt or "clean" in flt:
    run("clean", stems)
for label, make in cases:
    if flt and flt not in label:
        continue
    st, gt = make()
    run(label, st, gt)
