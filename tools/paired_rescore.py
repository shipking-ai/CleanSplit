"""Re-derive the verdict in an existing benchmark JSON as a PAIRED statistic, without re-running any GPU work.

    python tools/paired_rescore.py outputs/_benchmarks/musdb18hq_artifact_reduction.json --baseline ensemble
    python tools/paired_rescore.py outputs/_benchmarks/musdb18hq_overlap.json --baseline sw_tta

Why this exists. tools/artifact_reduction.py and tools/overlap_experiment.py both decided their verdicts on
`median(candidate) - median(baseline)` with no win counts at all -- the unpaired statistic that has already manufactured
two false results in this project (docs/04 sections 14.5, 14.6). Those two tools produced the four rejections in
section 14.2 and the overlap adoption in section 14.5, so every one of those published numbers needs re-deriving.

It does not need a GPU: both tools recorded a `song` field on every row, so the per-song pairing can be recovered from
the JSON they already wrote. This reads any file shaped `rows[arm][stem] = [{song, sdr, sir, sar}, ...]`.

What it reports, per candidate and stem: the paired median difference (median of per-song differences, songs matched by
name over the intersection of the two arms), the win count, and whether the original unpaired delta and the paired one
DISAGREE IN SIGN -- which is the case that silently changes a verdict.

This tool only re-states the evidence. It deliberately does not re-apply any adoption rule, because the rules in those
two files are written in terms of bss_eval SAR, and section 14.4 demoted bss_eval SAR from deciding anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from fullband_check import paired  # noqa: E402

METRICS = ("sar", "sdr", "sir")


def main(path: Path, baseline: str, metrics: tuple[str, ...]) -> None:
    doc = json.loads(path.read_text())
    rows = doc.get("rows")
    if not isinstance(rows, dict):
        raise SystemExit(f"{path.name} has no 'rows' table to re-pair")
    if baseline not in rows:
        raise SystemExit(f"baseline {baseline!r} not in {sorted(rows)}")

    print(f"{path.name}: paired re-derivation, baseline {baseline}")
    print(f"  {'candidate':16s} {'stem':7s} {'metric':6s} {'unpaired':>9s} {'paired':>8s} {'won':>7s}  note")
    flips, out = [], {}
    for arm, per in rows.items():
        if arm == baseline:
            continue
        for stem, vals in per.items():
            if stem not in rows[baseline]:
                continue
            base_vals = rows[baseline][stem]
            for m in metrics:
                cand = {r["song"]: r[m] for r in vals if np.isfinite(r.get(m, np.nan))}
                base = {r["song"]: r[m] for r in base_vals if np.isfinite(r.get(m, np.nan))}
                if not cand or not base:
                    continue
                unp = float(np.median(list(cand.values()))) - float(np.median(list(base.values())))
                d, won, n = paired(cand, base)
                if not n:
                    continue
                note = ""
                if np.sign(unp) != np.sign(d) and abs(unp) > 5e-3 and abs(d) > 5e-3:
                    note = "** SIGN FLIP **"
                    flips.append((arm, stem, m, unp, d, won, n))
                elif won <= n / 2 and d > 0:
                    note = "gain, but loses on most songs"
                elif won > n / 2 and d < 0:
                    note = "loss, but wins on most songs"
                print(f"  {arm:16s} {stem:7s} {m:6s} {unp:+9.2f} {d:+8.2f} {won:3d}/{n:<3d}  {note}")
                out.setdefault(arm, {}).setdefault(stem, {})[m] = {
                    "unpaired_delta_db": unp, "paired_delta_db": d, "won": won, "n": n}
    print()
    if flips:
        print(f"  {len(flips)} sign flip(s): the unpaired and paired statistics disagree about the DIRECTION")
        for arm, stem, m, unp, d, won, n in flips:
            print(f"    {arm} {stem} {m}: unpaired {unp:+.2f} dB, paired {d:+.2f} dB, wins {won}/{n}")
    else:
        print("  no sign flips: every unpaired delta pointed the same way the paired one does")
    dest = path.with_name(path.stem + "_paired.json")
    dest.write_text(json.dumps({"source": path.name, "baseline": baseline, "paired": out,
                                "sign_flips": [dict(zip(("arm", "stem", "metric", "unpaired", "paired", "won", "n"), f))
                                               for f in flips]}, indent=1))
    print(f"  written {dest}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("path", type=Path)
    p.add_argument("--baseline", required=True)
    p.add_argument("--metrics", nargs="+", default=list(METRICS))
    a = p.parse_args()
    main(a.path, a.baseline, tuple(a.metrics))
