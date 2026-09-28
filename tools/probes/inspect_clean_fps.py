"""Dev: list every region reported on clean oracle stems (should ideally be none)."""
import sys

from cleansplit.analysis.pipeline import analyze_separation
from cleansplit.audio.synthetic import make_song
from cleansplit.separation.base import SeparationResult

for seed in [int(s) for s in (sys.argv[1:] or ["0"])]:
    mix, stems = make_song(8.0, seed=seed)
    res = analyze_separation(mix, SeparationResult(stems, 44100, "oracle", {"zero_dc": False}))
    print(f"seed {seed}: {len(res.artifact_map.regions)} regions")
    for r in res.artifact_map.regions:
        if r.confidence < 0.6:
            continue
        ev = {d: (e.confidence, e.measure_peak, e.measure_mean) for d, e in r.detectors.items()}
        print(f"  {r.stem:7s} {r.start_s:5.2f}-{r.end_s:5.2f}s {r.freq_low_hz:6.0f}-{r.freq_high_hz:6.0f}Hz types={r.artifact_types} stem_lvl={r.stem_level_db} {ev}")
