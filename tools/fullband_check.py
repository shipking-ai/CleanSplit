"""The filter-free full-band gate that docs/04 section 14.4 requires of every SAR candidate.

    python tools/fullband_check.py sw_tta sw_tta_ov4 sw_tta_ov8 [--limit 20] [--baseline sw_tta]

Why this exists: section 14.2's SAR numbers come from `bss_eval`, which fits an optimal 512-tap distortion filter
before scoring, so error shaped like an EQ curve is absorbed into the target term instead of charged as damage. That
was caught in section 14.4 by a falsified prediction: `+mdx23c` gained +0.22 dB SAR while being WORSE full-band on
16/20 songs. The rule written there, before any overlap result was seen: a SAR-only gain is not evidence of a cleaner
stem -- a candidate must also improve the plain, filter-free, full-rate SNR.

This is that check, and nothing else: median `snr_db` at the native 44.1 kHz over all four stems, the same metric as
sections 3, 6, 7 and 11, no downsampling and no filter allowance. Arms are cache keys under data/musdb_cache/.

GATE (fixed here, independent of any candidate's numbers): median full-band SNR must improve on the vocals, drums AND
bass stems, and `other` is reported but not gated (it is the remainder, so it absorbs the other three's mistakes).

Output: outputs/_benchmarks/musdb18hq_fullband.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import musdb_eval as M  # noqa: E402

OUT = ROOT / "outputs" / "_benchmarks" / "musdb18hq_fullband.json"
GATED = ("vocals", "drums", "bass")


def main(arms: list[str], baseline: str, limit: int | None, out: Path) -> None:
    from cleansplit.metrics.signal import snr_db

    rows: dict[str, dict[str, list[float]]] = {}
    for name, _mix, truth in M.songs(limit):
        for arm in arms:
            f = M.CACHE / arm / f"{name}.npz"
            if not f.is_file():
                continue
            s = np.load(f)
            for g in M.GROUPS:
                t = truth[g].astype(np.float64)
                if float(np.sum(t**2)) <= 1e-9:
                    continue
                v = float(snr_db(t, s[g].astype(np.float64)))
                if np.isfinite(v):
                    rows.setdefault(arm, {}).setdefault(g, []).append(v)
        print(f"{name}: done", flush=True)

    med = {a: {g: float(np.median(v)) for g, v in per.items()} for a, per in rows.items()}
    base = med[baseline]
    print(f"\nMUSDB18-HQ full band at {M.SR} Hz, filter-free SNR. Median dB, change vs {baseline}")
    print(f"  {'arm':14s} {'songs':>5s} {'stem':7s} {'SNR':>7s} {'dSNR':>7s} {'won':>7s}")
    verdict = {}
    for arm, per in med.items():
        for g in M.GROUPS:
            if g not in per:
                continue
            won = sum(a > b for a, b in zip(rows[arm][g], rows[baseline][g][: len(rows[arm][g])]))
            print(f"  {arm:14s} {len(rows[arm][g]):5d} {g:7s} {per[g]:7.2f} {per[g] - base[g]:+7.2f} "
                  f"{won:3d}/{len(rows[arm][g]):<3d}")
        if arm != baseline:
            ok = all(per[g] > base[g] for g in GATED if g in per)
            verdict[arm] = {"fullband_better_on_vocals_drums_bass": ok,
                            "delta_db": {g: per[g] - base[g] for g in per}}
    print()
    for arm, v in verdict.items():
        print(f"  {arm:14s} full-band gate: {'PASS' if v['fullband_better_on_vocals_drums_bass'] else 'FAIL'}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"baseline": baseline, "median": med, "verdict": verdict, "rows": rows}, indent=1))
    print(f"written {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("arms", nargs="+")
    p.add_argument("--baseline", default="sw_tta")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args()
    main(a.arms, a.baseline, a.limit, a.out)
