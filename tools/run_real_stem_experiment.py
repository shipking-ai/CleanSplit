"""Ground-truth restoration experiment on real music built from known stems (default: the local Treblo 'BUH' stems).

Truth = {vocals, drums, bass, other = instrumentals - drums - bass}; mixture = sum(truth) (exact).
SW groups: vocals<-vocals, drums<-drums, bass<-bass, other<-guitar+piano+other.

Usage: python tools/run_real_stem_experiment.py <restorers> <start_s> [<start_s> ...] [--dur 30] [--max-regions 10]
"""
import argparse
from pathlib import Path

from cleansplit.analysis.pipeline import write_json
from cleansplit.metrics.restoration_experiment import run_experiment, stem_folder_songs
from cleansplit.separation.roformer import BSRoformerSeparator

ap = argparse.ArgumentParser()
ap.add_argument("restorers")
ap.add_argument("starts", type=float, nargs="+")
ap.add_argument("--dur", type=float, default=30.0)
ap.add_argument("--max-regions", type=int, default=10)
ap.add_argument("--scope", default="all", choices=["all", "mixture", "stems"])
ap.add_argument("--pattern", default=str(Path.home() / "Downloads" / "BUH - {} - Treblo.wav"))
args = ap.parse_args()

paths = {k: args.pattern.format(k) for k in ("vocals", "drums", "bass", "instrumentals")}
songs = stem_folder_songs(paths, [(s, args.dur) for s in args.starts], derive={"other": (["instrumentals"], ["drums", "bass"])}, drop=("instrumentals",))
for name, mix, truth in songs:
    assert set(truth) == {"vocals", "drums", "bass", "other"}, truth.keys()
groups = {"vocals": ["vocals"], "drums": ["drums"], "bass": ["bass"], "other": ["guitar", "piano", "other"]}
res = run_experiment(
    BSRoformerSeparator(), restorers=tuple(args.restorers.split(",")), songs=songs, groups=groups,
    max_regions=args.max_regions, region_scope=args.scope, log=lambda m: print(m, flush=True),
)
tag = args.restorers.replace(",", "_") + "_" + "-".join(f"{s:.0f}" for s in args.starts) + ("" if args.scope == "all" else f"_{args.scope}")
out = Path("outputs/_benchmarks") / f"real_stem_experiment_BUH_{tag}.json"
write_json(out, res)
print("written", out)
