"""Is there a FAST recipe that is still nearly as good as the best one? Measure the whole cost/quality curve.

    python tools/quality_tiers.py [--limit 20]

The user's question: the best recipe takes ~30 minutes a song; is there something much faster that is almost as good?
docs/04 section 14.8 already says where to look. Ranked by dB per doubling of compute, ensembling a second model is
~40x better value than TTA and ~12x better than doubling the overlap. So the cheap-but-good recipe should be the
ENSEMBLE WITHOUT TTA -- keeping the one lever that pays, dropping the two that barely do.

That costs no new GPU time to test, because every arm already exists in data/musdb_cache/:

  tier        recipe                                            passes x overlap = units
  fast        SW alone, overlap 2, no TTA                       1 x 2  =  2
  balanced    mean(SW, ep317), overlap 2, no TTA                2 x 2  =  4      <- the candidate
  best        mean(SW+TTA, ep317), overlap 4                    4 x 4  = 16

`other` is the exact remainder in each tier, as the shipped separators compute it, so the stems sum to the mixture.

PRE-REGISTERED, written before the numbers were seen. `balanced` is worth shipping as a tier only if it recovers at
least **70% of best's gain over fast**, on vocals, measured as paired medians. That threshold is a judgement about what
"almost as good" means at 4x less compute, fixed here so it cannot be moved afterwards to make the answer come out
nicely. The per-stem numbers and win counts are reported either way.

FALSIFIABLE PREDICTION: section 14.8's costs say TTA is worth about +0.04 dB and overlap 2 -> 4 about +0.08 dB, so
dropping both should cost roughly 0.12 dB of best's ~0.45 dB vocal gain -- leaving `balanced` at roughly 70-75% of it.
Falsified if `balanced` recovers less than 70%, which would mean TTA and overlap contribute much more inside the
ensemble than they do to SW alone, and that the section 14.8 ranking does not transfer.

Output: outputs/_benchmarks/musdb18hq_quality_tiers.json
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
from fullband_check import paired  # noqa: E402

OUT = ROOT / "outputs" / "_benchmarks" / "musdb18hq_quality_tiers.json"
STEMS = ("vocals", "drums", "bass", "other")
UNITS = {"fast": 2, "balanced": 4, "best": 16}
RECOVERY_FLOOR = 0.70


def main(limit: int | None, out: Path) -> None:
    from cleansplit.metrics.signal import snr_db

    snr: dict[str, dict[str, dict[str, float]]] = {t: {} for t in UNITS}
    for name, mix, truth in M.songs(limit):
        f = {k: M.CACHE / k / f"{name}.npz" for k in ("sw", "ep317", "sw_tta_ov4", "ep317_ov4")}
        if not all(v.is_file() for v in f.values()):
            print(f"{name}: skipped (missing {[k for k, v in f.items() if not v.is_file()]})", flush=True)
            continue
        sw = {k: v.astype(np.float64) for k, v in np.load(f["sw"]).items()}
        ep = np.load(f["ep317"])["vocals"].astype(np.float64)
        sw4 = {k: v.astype(np.float64) for k, v in np.load(f["sw_tta_ov4"]).items()}
        ep4 = np.load(f["ep317_ov4"])["vocals"].astype(np.float64)
        mixf = mix.astype(np.float64)

        tiers = {}
        tiers["fast"] = {"vocals": sw["vocals"], "drums": sw["drums"], "bass": sw["bass"]}
        tiers["balanced"] = {"vocals": 0.5 * (sw["vocals"] + ep), "drums": sw["drums"], "bass": sw["bass"]}
        tiers["best"] = {"vocals": 0.5 * (sw4["vocals"] + ep4), "drums": sw4["drums"], "bass": sw4["bass"]}
        for t, d in tiers.items():
            d["other"] = mixf - d["vocals"] - d["drums"] - d["bass"]
            for g in STEMS:
                tr = truth[g].astype(np.float64)
                if float(np.sum(tr**2)) > 1e-9:
                    snr[t].setdefault(g, {})[name] = float(snr_db(tr, d[g]))
        print(f"{name}: scored", flush=True)

    if not snr["fast"]:
        raise SystemExit("no song has all four caches (sw, ep317, sw_tta_ov4, ep317_ov4)")

    print()
    print("MUSDB18-HQ, paired median full-band 44.1 kHz filter-free SNR. Compute in forward-pass units (section 14.8)")
    print(f"  {'stem':7s} {'best-fast':>10s} {'bal-fast':>9s} {'won':>7s} {'recovered':>10s} {'best-bal':>9s}")
    rec = {}
    for g in STEMS:
        d_best, w_best, n = paired(snr["best"][g], snr["fast"][g])
        d_bal, w_bal, _ = paired(snr["balanced"][g], snr["fast"][g])
        d_gap = paired(snr["best"][g], snr["balanced"][g])[0]
        frac = (d_bal / d_best) if abs(d_best) > 1e-9 else float("nan")
        rec[g] = {"best_minus_fast_db": d_best, "balanced_minus_fast_db": d_bal, "best_minus_balanced_db": d_gap,
                  "recovered_fraction": frac, "won_balanced_vs_fast": w_bal, "won_best_vs_fast": w_best, "n": n}
        print(f"  {g:7s} {d_best:+10.3f} {d_bal:+9.3f} {w_bal:3d}/{n:<3d} {frac:9.0%} {d_gap:+9.3f}")

    v = rec["vocals"]
    ok = bool(v["recovered_fraction"] >= RECOVERY_FLOOR)
    print()
    print(f"  balanced costs {UNITS['balanced']} units vs best's {UNITS['best']}: "
          f"{UNITS['best'] / UNITS['balanced']:.0f}x cheaper, {UNITS['balanced'] / UNITS['fast']:.0f}x the fast tier")
    print(f"  RULE was: ship `balanced` only if it recovers >= {RECOVERY_FLOOR:.0%} of best's vocal gain over fast")
    print(f"  -> recovered {v['recovered_fraction']:.0%}  =>  {'SHIP IT' if ok else 'do not ship'}")
    print(f"  PREDICTION was 70-75% -> {'CONFIRMED' if ok else '** FALSIFIED **'}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"units": UNITS, "recovery_floor": RECOVERY_FLOOR, "per_stem": rec,
                               "adopt_balanced": ok, "snr": snr}, indent=1))
    print(f"  written {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args()
    main(a.limit, a.out)
