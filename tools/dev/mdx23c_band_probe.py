"""Why do docs/04 section 11 and section 14 disagree about MDX23C? Split the vocal error by frequency band.

    python tools/dev/mdx23c_band_probe.py [--limit 20] [--cut 8000]

Section 11 (full-band 44.1 kHz SNR, no filter allowance): adding MDX23C to the vocal average was WORSE on 16/20
songs, so it was removed. Section 14 (16 kHz, bss_eval, which allows an optimal distortion filter): +0.22 dB SAR,
+0.13 dB SDR. Both cannot be a complete description of the same estimates.

HYPOTHESIS, written before the run: MDX23C helps below the 16 kHz run's 8 kHz ceiling and hurts above it, so each
measurement is right about the band it can see.
PREDICTION: delta SNR(ensemble+mdx23c vs ensemble) > 0 below the cut and < 0 above it.
FALSIFIED IF: the delta has the same sign in both bands. Then the band story is wrong and section 14's gain is an
artifact of bss_eval's filter allowance, which would also mean the SAR column cannot be read as "this candidate is
audibly cleaner" without a full-band check.

Diagnostic only: nothing is adopted or rejected on this number, so there is no threshold to tune. Per-band SNR is
computed from the STFT error by Parseval; the window's COLA constant cancels in the ratio.

Output: outputs/_benchmarks/mdx23c_band_probe.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import musdb_eval as M

from cleansplit.audio.stft import STFTGrid, stft

OUT = ROOT / "outputs" / "_benchmarks" / "mdx23c_band_probe.json"
GRID = STFTGrid(n_fft=4096, hop=1024)


def band_snr(truth: np.ndarray, est: np.ndarray, lo: int, hi: int) -> float:
    T = stft(truth.astype(np.float64), GRID, dtype=np.complex128)[..., lo:hi, :]
    E = stft(est.astype(np.float64), GRID, dtype=np.complex128)[..., lo:hi, :]
    num, den = float(np.sum(np.abs(T) ** 2)), float(np.sum(np.abs(T - E) ** 2))
    return 10 * np.log10(num / den) if den > 0 else float("inf")


def main(limit: int | None, cut: int) -> None:
    k = int(round(cut * GRID.n_fft / M.SR))  # first bin at or above the cut
    rows = []
    for name, _mix, truth in M.songs(limit):
        e = {m: np.load(M.CACHE / m / f"{name}.npz")["vocals"].astype(np.float64)
             for m in ("sw_tta", "ep317", "mdx23c")}
        two = (e["sw_tta"] + e["ep317"]) / 2
        three = (e["sw_tta"] + e["ep317"] + e["mdx23c"]) / 3
        t = truth["vocals"].astype(np.float64)
        r = {"song": name}
        for label, est in (("ensemble", two), ("+mdx23c", three)):
            r[f"{label}_low"] = band_snr(t, est, 0, k)
            r[f"{label}_high"] = band_snr(t, est, k, GRID.n_fft // 2 + 1)
            r[f"{label}_full"] = band_snr(t, est, 0, GRID.n_fft // 2 + 1)
        rows.append(r)
        print(f"{name}: low {r['+mdx23c_low'] - r['ensemble_low']:+.2f} dB, "
              f"high {r['+mdx23c_high'] - r['ensemble_high']:+.2f} dB", flush=True)

    d = {b: [r[f"+mdx23c_{b}"] - r[f"ensemble_{b}"] for r in rows] for b in ("low", "high", "full")}
    print(f"\nVocals, {len(rows)} songs. Adding MDX23C to the vocal average, SNR change vs the 2-model ensemble")
    print(f"  {'band':22s} {'median':>8s} {'mean':>8s} {'songs improved':>16s}")
    summary = {}
    for b, label in (("low", f"below {cut} Hz"), ("high", f"above {cut} Hz"), ("full", "full band")):
        v = np.array(d[b])
        summary[b] = {"median": float(np.median(v)), "mean": float(np.mean(v)),
                      "improved": int(np.sum(v > 0)), "n": len(v)}
        print(f"  {label:22s} {summary[b]['median']:+8.2f} {summary[b]['mean']:+8.2f} "
              f"{summary[b]['improved']:>13d}/{len(v)}")
    same_sign = np.sign(summary["low"]["median"]) == np.sign(summary["high"]["median"])
    print(f"\n  prediction (helps below the cut, hurts above): "
          f"{'FALSIFIED - same sign in both bands' if same_sign else 'HELD'}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"cut_hz": cut, "summary": summary, "prediction_held": not bool(same_sign),
                               "rows": rows}, indent=1))
    print(f"written {OUT}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--cut", type=int, default=8000)
    a = p.parse_args()
    main(a.limit, a.cut)
