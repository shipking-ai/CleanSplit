"""Can the artifacts be reduced? Candidates scored on SAR against the shipped ensemble (MUSDB18-HQ, 20 songs).

    python tools/artifact_reduction.py [--limit 20] [--sr 16000]

tools/artifact_metrics.py showed the error inside a stem is dominated by ARTIFACTS, not bleed (vocals: SIR 24 dB vs
SAR 16 dB). So the thing to optimise is SAR. Everything here is computed from the estimates tools/musdb_eval.py
already cached — no GPU, no new separation.

Candidates (all start from the shipped `ensemble`: vocals = mean(SW+TTA, ep317), drums/bass = SW+TTA):
  ensemble            the baseline, as shipped
  +mdx23c             vocals = mean(SW+TTA, ep317, MDX23C). Rejected on SDR in docs/04 section 11; averaging is what
                      cancels artifacts, so it gets a second hearing on SAR.
  wiener1 / wiener2   every stem re-derived as a filter of the ORIGINAL mixture:
                      S_i' = (|S_i|^a / sum_j |S_j|^a) * X, a = 1 or 2, STFT 4096/1024.
                      The classic artifact reducer (Open-Unmix/norbert, Demucs' --wiener): the output can only be a
                      re-weighting of sound that is really in the mixture, so invented energy cannot survive.
  wiener2_smooth      wiener2 with the weights smoothed over ~5 frames, which is what stops mask flutter (warble).

PRE-REGISTERED, written before the run. A candidate replaces the default only if, over the 20 songs:
    median SAR improves on the vocals, drums AND bass stems, AND
    median SDR drops by no more than 0.10 dB on any of them.
The SDR guard is there because docs/04 section 4 already measured the failure mode this invites: a mixture-consistent
re-weighting can sound smoother while moving the stems further from the truth.

Output: outputs/_benchmarks/musdb18hq_artifact_reduction.json
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

from cleansplit.audio.stft import STFTGrid, istft, stft  # noqa: E402

OUT = ROOT / "outputs" / "_benchmarks" / "musdb18hq_artifact_reduction.json"
GRID = STFTGrid(n_fft=4096, hop=1024)
SAR_STEMS = ("vocals", "drums", "bass")


def wiener(stems: dict[str, np.ndarray], mixture: np.ndarray, power: float = 2.0, smooth: int = 0) -> dict:
    from scipy.ndimage import uniform_filter1d

    X = stft(mixture.astype(np.float64), GRID, dtype=np.complex128)
    n = mixture.shape[-1]
    P = {}
    for k, s in stems.items():
        S = stft(np.asarray(s, dtype=np.float64), GRID, dtype=np.complex128)
        p = np.abs(S) ** power
        P[k] = uniform_filter1d(p, size=smooth, axis=-1, mode="nearest") if smooth else p
    tot = sum(P.values()) + 1e-20
    return {k: istft((P[k] / tot) * X, GRID, n) for k in stems}


def main(sr: int, limit: int | None) -> None:
    import mir_eval
    import soxr

    rows: dict[str, dict[str, list]] = {}
    for name, mix, truth in M.songs(limit):
        e = {k: {g: v.astype(np.float64) for g, v in np.load(M.CACHE / k / f"{name}.npz").items()}
             for k in ("sw_tta", "ep317", "mdx23c")}
        mix64 = mix.astype(np.float64)
        base = {"vocals": (e["sw_tta"]["vocals"] + e["ep317"]["vocals"]) / 2,
                "drums": e["sw_tta"]["drums"], "bass": e["sw_tta"]["bass"]}
        base["other"] = mix64 - base["vocals"] - base["drums"] - base["bass"]
        three = dict(base)
        three["vocals"] = (e["sw_tta"]["vocals"] + e["ep317"]["vocals"] + e["mdx23c"]["vocals"]) / 3
        three["other"] = mix64 - three["vocals"] - three["drums"] - three["bass"]
        cands = {
            "ensemble": base,
            "+mdx23c": three,
            "wiener1": wiener(base, mix64, power=1.0),
            "wiener2": wiener(base, mix64, power=2.0),
            "wiener2_smooth": wiener(base, mix64, power=2.0, smooth=5),
        }
        names = list(M.GROUPS)
        ref = np.stack([soxr.resample(truth[g].astype(np.float64).mean(0), 44100, sr) for g in names])
        keep = [i for i, r in enumerate(ref) if np.sum(r**2) > 1e-9]
        if len(keep) < 2:
            continue
        for c, stems in cands.items():
            est = np.stack([soxr.resample(stems[g].mean(0), 44100, sr) for g in names])
            sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(ref[keep], est[keep], compute_permutation=False)
            for i, k in enumerate(keep):
                rows.setdefault(c, {}).setdefault(names[k], []).append(
                    {"song": name, "sdr": float(sdr[i]), "sir": float(sir[i]), "sar": float(sar[i])})
        print(f"{name}: done", flush=True)

    med = {c: {s: {m: float(np.median([v[m] for v in vals])) for m in ("sdr", "sir", "sar")}
               for s, vals in per.items()} for c, per in rows.items()}
    base_med = med["ensemble"]
    print(f"\nMUSDB18-HQ, {len(rows['ensemble']['vocals'])} songs at {sr} Hz. Median dB, change vs the shipped ensemble")
    print(f"  {'candidate':16s} {'stem':7s} {'SAR':>7s} {'dSAR':>7s} {'SDR':>7s} {'dSDR':>7s} {'SIR':>7s} {'dSIR':>7s}")
    verdict = {}
    for c, per in med.items():
        for s in names:
            if s not in per:
                continue
            d = {m: per[s][m] - base_med[s][m] for m in ("sdr", "sir", "sar")}
            print(f"  {c:16s} {s:7s} {per[s]['sar']:7.2f} {d['sar']:+7.2f} {per[s]['sdr']:7.2f} {d['sdr']:+7.2f} "
                  f"{per[s]['sir']:7.2f} {d['sir']:+7.2f}")
        if c != "ensemble":
            sar_ok = all(per[s]["sar"] > base_med[s]["sar"] for s in SAR_STEMS if s in per)
            sdr_ok = all(per[s]["sdr"] - base_med[s]["sdr"] >= -0.10 for s in SAR_STEMS if s in per)
            verdict[c] = {"sar_better_on_all_three": sar_ok, "sdr_within_0.1dB": sdr_ok, "adopt": bool(sar_ok and sdr_ok)}
    print()
    for c, v in verdict.items():
        print(f"  {c:16s} SAR better on vocals+drums+bass: {v['sar_better_on_all_three']}, "
              f"SDR guard held: {v['sdr_within_0.1dB']} -> {'ADOPT' if v['adopt'] else 'no'}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"sample_rate": sr, "median": med, "verdict": verdict, "rows": rows}, indent=1))
    print(f"written {OUT}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--limit", type=int, default=20)
    a = p.parse_args()
    main(a.sr, a.limit)
