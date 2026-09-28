"""Does more overlap-add averaging reduce artifacts? SW at num_overlap 2 (shipped) vs 4 vs 8, scored on SAR.

    python tools/overlap_experiment.py [--limit 20] [--overlaps 4 8] [--sr 16000] [--out FILE]

tools/artifact_metrics.py showed the error left in a stem is dominated by ARTIFACTS, not bleed. Chunked inference is
one known source of them: each chunk is denoised independently, so the seams between chunks carry discontinuities and
each chunk's own invented energy is uncorrelated with its neighbour's. Raising num_overlap makes every output sample
the average of more independent passes, which is the same mechanism that makes model ensembling cancel artifacts.
docs/02 section on overlap measured the cost: 2 -> 3.6x realtime, 4 -> 3.1x, 8 -> 1.5x. Nobody has measured whether
it buys anything, because until now the target was SDR.

Isolated on purpose: SW alone with TTA (not the shipped ensemble), so the only thing that changes between arms is
the overlap factor. All four stems come from the one model, so the BSS decomposition sees a complete separation.

PRE-REGISTERED, written before the run. An overlap replaces the shipped 2 only if, over the songs scored:
    median SAR improves on the vocals, drums AND bass stems, AND
    median SDR drops by no more than 0.10 dB on any of them.
Identical bar to tools/artifact_reduction.py, for the same reason (docs/04 section 4). The GPU cost is reported but
is not part of the rule: if it works, the speed/quality trade is the user's to make, not mine to hide.

Estimates are cached per overlap under data/musdb_cache/sw_tta_ov<N>/, so the run is resumable. An arm whose cache
is incomplete is scored over the songs it has, which is NOT comparable to a full arm -- the printed song count per arm
tells you which is which. `--out` exists so a partial arm can be scored in one process while another fills its cache.
Output: outputs/_benchmarks/musdb18hq_overlap.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import musdb_eval as M

OUT = ROOT / "outputs" / "_benchmarks" / "musdb18hq_overlap.json"
SAR_STEMS = ("vocals", "drums", "bass")


def separate(overlaps: list[int], limit: int | None) -> dict[int, float]:
    """Fill the cache for each overlap. Returns seconds of audio per second of GPU time, per overlap."""
    import torch

    from cleansplit.separation import registry

    items = [(n, m) for n, m, _ in M.songs(limit)]
    speed = {}
    for ov in overlaps:
        key = f"sw_tta_ov{ov}"
        todo = [(n, m) for n, m in items if not (M.CACHE / key / f"{n}.npz").is_file()]
        if not todo:
            print(f"{key}: cached", flush=True)
            continue
        sep = registry.create("bs_roformer_sw", tta=True, num_overlap=ov)
        t0, audio_s = time.time(), 0.0
        for i, (name, mix) in enumerate(todo, 1):
            g = M._grouped("sw_tta", {k: v.astype(np.float32) for k, v in sep.separate(mix, M.SR).stems.items()})
            (M.CACHE / key).mkdir(parents=True, exist_ok=True)
            np.savez(M.CACHE / key / f"{name}.npz", **g)
            audio_s += mix.shape[-1] / M.SR
            print(f"{key}: {i}/{len(todo)} {name} ({audio_s / (time.time() - t0):.2f}x realtime)", flush=True)
        speed[ov] = audio_s / (time.time() - t0)
        sep._model = None
        torch.cuda.empty_cache()
    return speed


def score(overlaps: list[int], limit: int | None, sr: int) -> tuple[dict, dict]:
    import mir_eval
    import soxr

    arms = ["sw_tta"] + [f"sw_tta_ov{ov}" for ov in overlaps]
    names = list(M.GROUPS)
    rows: dict[str, dict[str, list]] = {}
    for name, _mix, truth in M.songs(limit):
        ref = np.stack([soxr.resample(truth[g].astype(np.float64).mean(0), M.SR, sr) for g in names])
        keep = [i for i, r in enumerate(ref) if np.sum(r**2) > 1e-9]  # silent truth stems break bss_eval
        if len(keep) < 2:
            continue
        for arm in arms:
            f = M.CACHE / arm / f"{name}.npz"
            if not f.is_file():
                continue
            s = np.load(f)
            est = np.stack([soxr.resample(s[g].astype(np.float64).mean(0), M.SR, sr) for g in names])
            sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(ref[keep], est[keep], compute_permutation=False)
            for i, k in enumerate(keep):
                rows.setdefault(arm, {}).setdefault(names[k], []).append(
                    {"song": name, "sdr": float(sdr[i]), "sir": float(sir[i]), "sar": float(sar[i])})
        print(f"{name}: scored", flush=True)
    med = {a: {s: {m: float(np.median([v[m] for v in vals])) for m in ("sdr", "sir", "sar")}
               for s, vals in per.items()} for a, per in rows.items()}
    return med, rows


def main(overlaps: list[int], limit: int | None, sr: int, out: Path) -> None:
    speed = separate(overlaps, limit)
    med, rows = score(overlaps, limit, sr)
    base = med["sw_tta"]
    n = len(rows["sw_tta"]["vocals"])
    print(f"\nMUSDB18-HQ, {n} songs at {sr} Hz. SW+TTA alone. Median dB, change vs the shipped num_overlap=2")
    for arm, per in rows.items():
        print(f"  {arm}: {len(per['vocals'])} songs scored")
    print(f"  {'arm':14s} {'stem':7s} {'SAR':>7s} {'dSAR':>7s} {'SDR':>7s} {'dSDR':>7s} {'SIR':>7s} {'dSIR':>7s}")
    verdict = {}
    for arm, per in med.items():
        for s in names_of(per):
            d = {m: per[s][m] - base[s][m] for m in ("sdr", "sir", "sar")}
            print(f"  {arm:14s} {s:7s} {per[s]['sar']:7.2f} {d['sar']:+7.2f} {per[s]['sdr']:7.2f} {d['sdr']:+7.2f} "
                  f"{per[s]['sir']:7.2f} {d['sir']:+7.2f}")
        if arm != "sw_tta":
            sar_ok = all(per[s]["sar"] > base[s]["sar"] for s in SAR_STEMS if s in per)
            sdr_ok = all(per[s]["sdr"] - base[s]["sdr"] >= -0.10 for s in SAR_STEMS if s in per)
            verdict[arm] = {"sar_better_on_all_three": sar_ok, "sdr_within_0.1dB": sdr_ok,
                            "adopt": bool(sar_ok and sdr_ok), "x_realtime": speed.get(int(arm.split("ov")[1]))}
    print()
    for arm, v in verdict.items():
        cost = f", {v['x_realtime']:.2f}x realtime" if v["x_realtime"] else ""
        print(f"  {arm:14s} SAR better on vocals+drums+bass: {v['sar_better_on_all_three']}, "
              f"SDR guard held: {v['sdr_within_0.1dB']}{cost} -> {'ADOPT' if v['adopt'] else 'no'}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"sample_rate": sr, "songs": n, "x_realtime": speed, "median": med,
                               "verdict": verdict, "rows": rows}, indent=1))
    print(f"written {out}")


def names_of(per: dict) -> list[str]:
    return [g for g in M.GROUPS if g in per]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--overlaps", type=int, nargs="+", default=[4, 8])
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args()
    main(a.overlaps, a.limit, a.sr, a.out)
