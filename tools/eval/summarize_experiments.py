"""Aggregate ground-truth restoration experiments by restorer, recomputing the pooled, floored box metric from the
raw per-group energies stored in each JSON (so runs made with older metric code are scored identically).

Usage: python tools/eval/summarize_experiments.py <glob> [<glob> ...]   (default: outputs/_benchmarks/real_stem_experiment_*.json)
Latest file wins for each (song, restorer).
"""
import glob
import json
import sys
from pathlib import Path

from cleansplit.metrics.restoration_experiment import _box_total_changes, pooled_change_db

patterns = sys.argv[1:] or ["outputs/_benchmarks/real_stem_experiment_*.json"]
files = sorted({f for p in patterns for f in glob.glob(p)}, key=lambda f: Path(f).stat().st_mtime)
latest = {}
for f in files:
    for e in json.loads(Path(f).read_text()).get("results", []):
        for name, v in e["restorers"].items():
            if v.get("box_truth_changes") is None or (v["box_truth_changes"] and "err_before" not in v["box_truth_changes"][0]):
                continue
            latest[(e["song"], name)] = (v, f)

agg = {}
print(f"{'song':14s} {'restorer':22s} {'accepted':>9s} {'pooled(acc)':>12s} {'pooled(all)':>12s} closer/worse(all)")
for (song, name), (v, f) in sorted(latest.items()):
    totals = _box_total_changes(v["box_truth_changes"])
    pa, pall = pooled_change_db(totals, True), pooled_change_db(totals)
    closer = sum(t["total_err_change_db"] < -0.05 for t in totals)
    worse = sum(t["total_err_change_db"] > 0.05 for t in totals)
    n = len(v["decisions"])
    fmt = lambda x: "n/a" if x is None else f"{x:+.2f} dB"
    print(f"{song:14s} {name:22s} {v['accepted']:3d}/{n:<5d} {fmt(pa):>12s} {fmt(pall):>12s} {closer}/{worse} of {len(totals)}")
    a = agg.setdefault(name, {"totals": [], "accepted": 0, "proposals": 0, "songs": []})
    a["totals"] += totals
    a["accepted"] += v["accepted"]
    a["proposals"] += n
    a["songs"].append(song)

print("\nPooled over all songs:")
summary = {}
for name, a in sorted(agg.items()):
    pa, pall = pooled_change_db(a["totals"], True), pooled_change_db(a["totals"])
    closer = int(sum(t["total_err_change_db"] < -0.05 for t in a["totals"]))
    worse = int(sum(t["total_err_change_db"] > 0.05 for t in a["totals"]))
    print(f"  {name:22s} songs {len(a['songs'])}  accepted {a['accepted']}/{a['proposals']}  pooled(accepted) {pa if pa is None else round(pa, 2)} dB  "
          f"pooled(all proposals) {pall if pall is None else round(pall, 2)} dB  closer {closer} / worse {worse} of {len(a['totals'])}")
    summary[name] = {"songs": a["songs"], "accepted": a["accepted"], "proposals": a["proposals"], "pooled_accepted_db": pa,
                     "pooled_all_proposals_db": pall, "regions_closer": closer, "regions_worse": worse, "regions_scored": len(a["totals"])}

# Per truth group: pooled error change of that group only (floored by that group's truth energy). Groups where the
# 'truth' separator and the evaluated separator disagree strongly (low separation SDR) are less trustworthy as truth.
print("\nPer truth group (pooled over regions and songs, accepted proposals; raw a2sb: all proposals):")
for name in sorted(agg):
    rows = [x for (song, n), (v, f) in latest.items() if n == name for x in v["box_truth_changes"]]
    groups = sorted({x["truth_stem"] for x in rows})
    parts = []
    for gname in groups:
        g = [x for x in rows if x["truth_stem"] == gname and (x["accepted"] or name == "a2sb")]
        if not g:
            continue
        ea = sum(x["err_after"] + 1e-3 * x["truth_energy"] for x in g)
        eb = sum(x["err_before"] + 1e-3 * x["truth_energy"] for x in g)
        parts.append(f"{gname} {10 * __import__('math').log10(max(ea, 1e-30) / max(eb, 1e-30)):+.2f} dB (n={len(g)})")
        summary[name].setdefault("per_group_pooled_db", {})[gname] = 10 * __import__("math").log10(max(ea, 1e-30) / max(eb, 1e-30))
    print(f"  {name:22s} " + ", ".join(parts))
Path("outputs/_benchmarks/real_stem_experiment_summary.json").write_text(json.dumps(summary, indent=1))
