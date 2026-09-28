"""Can any TRUTH-FREE combiner capture part of the oracle selector's +0.38 dB? The user's idea, in its measurable form.

    python tools/experiments/tile_combine.py [--limit 20]

tools/eval/oracle_headroom.py measured the ceiling on "listen to several models and decide who got it right": a per-TF-tile
oracle that reads the truth beats the shipped average by **+0.38 dB paired, 20/20 songs**. That is not a small budget --
it is most of the +0.45 dB the entire two-model ensemble is worth. So unlike the weighting axis (section 14.9, capped at
0.013 dB) this one is worth an attempt.

This tries to collect some of it WITHOUT the truth. Every combiner here is parameter-free, so nothing is fitted to these
songs and no held-out split is needed; each is a fixed formula on the two complex STFTs, costs nothing at inference
(both models already run), and is evaluated against what ships today, the plain complex average.

Combiners (A = SW+TTA, B = ep317, per TF bin):
  average          0.5(A+B)                                  what ships; the baseline
  min_mag          whichever of A, B has the smaller |.|     artifacts usually ADD energy, so prefer the quieter claim
  mag_mean_phase   |.| = 0.5(|A|+|B|), phase from arg(A+B)   keeps the average's phase, drops its magnitude cancellation
  geo_mag          |.| = sqrt(|A||B|), phase from arg(A+B)    geometric mean punishes disagreement harder than the mean
  consensus_gate   0.5(A+B) * |A+B| / (|A|+|B|)               shrink where the two models disagree in phase

`other` is the exact remainder in CleanSplit, so any change to vocals moves it too; its SNR is reported as a guard.

PRE-REGISTERED DECISION RULE: a combiner replaces the average only if its paired median full-band 44.1 kHz filter-free
vocal SNR gains at least +0.02 dB (the project-wide floor), AND it wins on more than half the songs, AND `other` does not
lose more than 0.02 dB. All three, or it is rejected.

FALSIFIABLE PREDICTION, written before the run. For two zero-mean errors of equal variance that are uncorrelated with
each other, the plain average is the minimum-variance unbiased combination -- and section 14.9 established these two
models ARE of near-equal strength (the optimal blend weight is exactly 0.50, with a symmetric flat optimum). Every
combiner above deviates from that average using only the DISAGREEMENT between the models, which says how much error
there is but not WHICH model carries it. So: all four lose, and `min_mag` loses worst, because systematically choosing
the quieter estimate biases the stem's energy downward.
FALSIFIED if any combiner clears the rule above -- which would mean disagreement alone does carry usable information
about who is wrong, and that a learned per-tile selector is worth building after all.

Output: outputs/_benchmarks/musdb18hq_tile_combine.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "eval"))
import musdb_eval as M
from fullband_check import paired

from cleansplit.audio.stft import STFTGrid, istft, stft

OUT = ROOT / "outputs" / "_benchmarks" / "musdb18hq_tile_combine.json"
GRID = STFTGrid(n_fft=4096, hop=1024)
MIN_DELTA_DB = 0.02
EPS = 1e-12


def combiners(A: np.ndarray, B: np.ndarray) -> dict[str, np.ndarray]:
    avg = 0.5 * (A + B)
    mA, mB = np.abs(A), np.abs(B)
    ph = np.exp(1j * np.angle(avg))
    quieter = np.where(mA <= mB, A, B)
    denom = mA + mB + EPS
    return {
        "average": avg,
        "min_mag": quieter,
        "mag_mean_phase": 0.5 * (mA + mB) * ph,
        "geo_mag": np.sqrt(mA * mB) * ph,
        "consensus_gate": avg * (np.abs(A + B) / denom),
    }


def main(limit: int | None, out: Path) -> None:
    from cleansplit.metrics.signal import snr_db

    voc: dict[str, dict[str, float]] = {}
    oth: dict[str, dict[str, float]] = {}
    for name, mix, truth in M.songs(limit):
        fsw, fep = M.CACHE / "sw_tta" / f"{name}.npz", M.CACHE / "ep317" / f"{name}.npz"
        if not (fsw.is_file() and fep.is_file()):
            print(f"{name}: skipped (missing cache)", flush=True)
            continue
        sw = {k: v.astype(np.float64) for k, v in np.load(fsw).items()}
        ep = np.load(fep)["vocals"].astype(np.float64)
        tv, to = truth["vocals"].astype(np.float64), truth["other"].astype(np.float64)
        if float(np.sum(tv**2)) <= 1e-9:
            continue
        n = tv.shape[-1]
        A = stft(sw["vocals"], GRID, dtype=np.complex128)
        B = stft(ep, GRID, dtype=np.complex128)
        rest = sw["drums"] + sw["bass"]
        line = []
        for k, Z in combiners(A, B).items():
            v = istft(Z, GRID, n)
            voc.setdefault(k, {})[name] = float(snr_db(tv, v))
            if float(np.sum(to**2)) > 1e-9:
                oth.setdefault(k, {})[name] = float(snr_db(to, mix.astype(np.float64) - v - rest))
            line.append(f"{k} {voc[k][name]:.2f}")
        print(f"  {name[:32]:32s} " + "  ".join(line), flush=True)

    if "average" not in voc:
        raise SystemExit("no songs scored; fill the sw_tta and ep317 caches first")
    print()
    print("MUSDB18-HQ vocals, paired median full-band 44.1 kHz filter-free SNR vs the shipped complex average")
    print(f"  {'combiner':16s} {'dSNR':>8s} {'won':>7s} {'d other':>8s} verdict")
    verdict = {}
    for k in voc:
        if k == "average":
            continue
        d, won, n = paired(voc[k], voc["average"])
        d_o = paired(oth[k], oth["average"])[0] if k in oth and "average" in oth else float("nan")
        ok = bool(d >= MIN_DELTA_DB and won > n / 2 and (np.isnan(d_o) or d_o >= -MIN_DELTA_DB))
        verdict[k] = {"delta_snr_db": d, "won": won, "n": n, "delta_other_db": d_o, "adopt": ok}
        print(f"  {k:16s} {d:+8.3f} {won:3d}/{n:<3d} {d_o:+8.3f} {'ADOPT' if ok else 'no'}")
    print()
    any_ok = any(v["adopt"] for v in verdict.values())
    worst = min(verdict, key=lambda k: verdict[k]["delta_snr_db"])
    print("  PREDICTION was: all four lose, and min_mag loses worst")
    print(f"  -> {'** FALSIFIED **' if any_ok else 'CONFIRMED'}: "
          f"{'a combiner cleared the rule' if any_ok else 'none adopted'}; worst is {worst} "
          f"({verdict[worst]['delta_snr_db']:+.3f} dB)"
          + ("" if worst == "min_mag" else "  <- NOT min_mag: that half of the prediction is wrong"))
    print("  for scale: the per-tile ORACLE that reads the truth is worth +0.38 dB (oracle_headroom.py)")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"min_delta_db": MIN_DELTA_DB, "verdict": verdict,
                               "prediction_confirmed": not any_ok, "worst": worst,
                               "vocal_snr": voc, "other_snr": oth}, indent=1))
    print(f"  written {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args()
    main(a.limit, a.out)
