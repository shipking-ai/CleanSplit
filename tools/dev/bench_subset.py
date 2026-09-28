"""Dev: synthetic benchmark restricted to some scenarios, with config overrides.

Usage: python bench_subset.py warble,smear [seeds...] [key=value ...]   (DetectorThresholds: name.center=0.7)
"""
import sys

from cleansplit.config.settings import AnalysisConfig
from cleansplit.metrics import detection_benchmark as B

keep = sys.argv[1].split(",")
seeds = tuple(int(a) for a in sys.argv[2:] if a.isdigit()) or (0, 1, 2, 3, 4)
cfg = AnalysisConfig()
for a in sys.argv[2:]:
    if "=" not in a:
        continue
    k, v = a.split("=")
    if "." in k:
        obj, attr = k.split(".")
        setattr(getattr(cfg, obj), attr, float(v))
    else:
        setattr(cfg, k, float(v))
B.SCENARIOS = [s for s in B.SCENARIOS if any(k in s[0] for k in keep)]
res = B.run_benchmark(seeds=seeds, config=cfg, log=lambda m: None)
print(" ".join(sys.argv[2:]), "| clean/min:", round(res["clean"]["regions_per_minute"], 1), res["clean"]["by_detector"],
      "|", {n: v["recall"] for n, v in res["scenarios"].items()})
