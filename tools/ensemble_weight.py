"""Is the vocal ensemble's equal weighting optimal? The only quality lever that costs nothing at inference.

    python tools/ensemble_weight.py [--limit 20] [--overlap 4]

Every other lever in docs/04 section 14.8 buys quality with compute: a second model costs 1.33x, overlap costs 2x, TTA
costs 3x. The blend WEIGHT costs nothing at all -- `vocals = (sw + ep317) / 2` is one constant in
`cleansplit/separation/ensemble.py`, and 0.5 was chosen because it is the obvious value, never because it measured
best. If some other weight is better, it is free quality.

THE HAZARD, and why this script is built the way it is. Choosing a weight that maximises a score on the same 20 songs
the project reports its results on is fitting the test set, and it would inflate every future number measured against
it. So the protocol is fixed BEFORE any weight is scored:

  * FIT half   = the first 10 songs in the fixed alphabetical order used everywhere in this project.
  * HELD-OUT half = the remaining 10. Not looked at while choosing the weight.
  * The weight is chosen ONLY on the fit half, by paired median full-band 44.1 kHz filter-free SNR on vocals.

PRE-REGISTERED DECISION RULE: the chosen weight w* replaces 0.5 only if, on the HELD-OUT half, its paired median gain
over w=0.5 is at least +0.02 dB (the project-wide floor, tools/fullband_check.py) AND it wins on more than half of
those songs. A gain that appears on the fit half and vanishes on the held-out half is overfitting, and is reported as
such rather than quietly dropped.

FALSIFIABLE PREDICTION, recorded before the run. Averaging works here because the two models' errors are largely
uncorrelated (section 14.1, +1.12 dB SAR), and for two uncorrelated errors of similar size the variance-minimising
weight is near 0.5 and the optimum is FLAT nearby. So: w* on the fit half lands in [0.4, 0.6], and the held-out gain is
below the +0.02 dB floor -- i.e. equal weighting is already right and this returns nothing.
Falsified if w* is outside [0.4, 0.6], or if the held-out gain clears the floor: either would mean one model deserves
more say than the other, and that the free lever was being left on the table.

Also reported, as a ceiling rather than a proposal: the per-song oracle weight, which reads the truth and so cannot
ship, bounding what ANY fixed or adaptive weighting could deliver.

Output: outputs/_benchmarks/musdb18hq_ensemble_weight.json
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

OUT = ROOT / "outputs" / "_benchmarks" / "musdb18hq_ensemble_weight.json"
MIN_DELTA_DB = 0.02
GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)


def main(overlap: int, limit: int | None, out: Path) -> None:
    from cleansplit.metrics.signal import snr_db

    # song -> weight -> vocal SNR, plus the effect on `other`, which is the exact remainder and absorbs any change.
    voc: dict[str, dict[float, float]] = {}
    oth: dict[str, dict[float, float]] = {}
    order: list[str] = []
    for name, mix, truth in M.songs(limit):
        fsw = M.CACHE / f"sw_tta_ov{overlap}" / f"{name}.npz"
        fep = M.CACHE / f"ep317_ov{overlap}" / f"{name}.npz"
        if not (fsw.is_file() and fep.is_file()):
            print(f"{name}: skipped (missing cache)", flush=True)
            continue
        sw = {k: v.astype(np.float64) for k, v in np.load(fsw).items()}
        ep = np.load(fep)["vocals"].astype(np.float64)
        tv, to = truth["vocals"].astype(np.float64), truth["other"].astype(np.float64)
        if float(np.sum(tv**2)) <= 1e-9:
            continue
        order.append(name)
        rest = sw["drums"] + sw["bass"]
        for w in GRID:
            v = w * sw["vocals"] + (1.0 - w) * ep
            voc.setdefault(name, {})[float(w)] = float(snr_db(tv, v))
            if float(np.sum(to**2)) > 1e-9:
                oth.setdefault(name, {})[float(w)] = float(snr_db(to, mix.astype(np.float64) - v - rest))
        print(f"{name}: swept {len(GRID)} weights", flush=True)

    if len(order) < 4:
        raise SystemExit(f"only {len(order)} songs scored; need the caches filled first")
    half = len(order) // 2
    fit, held = order[:half], order[half:]
    print()
    print(f"Fit half ({len(fit)} songs): {fit[0]} .. {fit[-1]}")
    print(f"Held-out half ({len(held)} songs): {held[0]} .. {held[-1]}")

    def col(songs, w):
        return {n: voc[n][w] for n in songs if w in voc.get(n, {})}

    print()
    print("  w = weight on SW+TTA; 1-w on ep317. Paired median vs w=0.50, on the FIT half only")
    print(f"  {'w':>5s} {'fit dSNR':>9s} {'won':>7s}")
    fit_rows = {}
    for w in GRID:
        d, won, n = paired(col(fit, float(w)), col(fit, 0.5))
        fit_rows[float(w)] = {"delta_db": d, "won": won, "n": n}
        star = "  <- best" if False else ""
        print(f"  {w:5.2f} {d:+9.3f} {won:3d}/{n:<3d}{star}")
    w_star = max(fit_rows, key=lambda w: fit_rows[w]["delta_db"])
    print(f"\n  w* chosen on the fit half = {w_star:.2f} ({fit_rows[w_star]['delta_db']:+.3f} dB there)")

    d_h, won_h, n_h = paired(col(held, w_star), col(held, 0.5))
    adopt = bool(w_star != 0.5 and d_h >= MIN_DELTA_DB and won_h > n_h / 2)
    print(f"  HELD-OUT check at w={w_star:.2f}: paired {d_h:+.3f} dB, wins {won_h}/{n_h} -> "
          f"{'ADOPT' if adopt else 'keep 0.50'}")

    # ceiling: per-song best weight, reading the truth
    orc = [max(voc[n].values()) - voc[n][0.5] for n in order]
    print(f"  oracle per-song weight ceiling: paired median {float(np.median(orc)):+.3f} dB "
          f"(reads the truth; cannot ship)")

    d_all, won_all, n_all = paired(col(order, w_star), col(order, 0.5))
    print(f"  (all {n_all} songs at w={w_star:.2f}, for the record: {d_all:+.3f} dB, {won_all}/{n_all})")

    oth_note = {}
    if oth:
        d_o, won_o, n_o = paired({n: oth[n][w_star] for n in oth}, {n: oth[n][0.5] for n in oth})
        oth_note = {"delta_db": d_o, "won": won_o, "n": n_o}
        print(f"  effect on `other` (the exact remainder) at w={w_star:.2f}: {d_o:+.3f} dB, {won_o}/{n_o}")

    lo, hi = 0.4, 0.6
    pred_ok = (lo <= w_star <= hi) and not adopt
    print()
    print(f"  PREDICTION was: w* in [{lo}, {hi}] AND held-out gain below the {MIN_DELTA_DB:+.2f} dB floor")
    print(f"  -> {'CONFIRMED' if pred_ok else '** FALSIFIED **'}: w*={w_star:.2f}, held-out {d_h:+.3f} dB")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"overlap": overlap, "grid": [float(w) for w in GRID], "fit": fit, "held_out": held,
                               "fit_rows": fit_rows, "w_star": w_star, "min_delta_db": MIN_DELTA_DB,
                               "held_out_delta_db": d_h, "held_out_won": won_h, "held_out_n": n_h, "adopt": adopt,
                               "oracle_ceiling_db": float(np.median(orc)), "other_effect": oth_note,
                               "prediction_confirmed": pred_ok, "vocal_snr": voc}, indent=1))
    print(f"  written {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--overlap", type=int, default=4)
    p.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args()
    main(a.overlap, a.limit, a.out)
