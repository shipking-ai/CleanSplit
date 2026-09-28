"""Where does the damage in a stem come from: bleed, or invented artifacts?

    python tools/eval/artifact_metrics.py [--sr 16000] [--limit N]

SDR alone cannot say. The BSS decomposition splits a stem estimate into
    target + interference (other sources leaking in) + artifacts (energy explained by NO source)
and reports SIR (bleed; higher is cleaner) and **SAR (artifacts; higher is cleaner)** separately.
Artifacts are exactly what "split with no artifacts" means: warble, smearing, musical noise — sound the model
invented that is in none of the true sources.

Scored on the MUSDB18-HQ estimates already cached by tools/eval/musdb_eval.py (songs 1-20, 30 s each), for the models
and the shipped ensembles. Downsampled to `--sr` (16 kHz by default) because bss_eval's filter projections are
O(n^2)-ish; this changes absolute numbers slightly but not the comparison between candidates on the same audio.

Output: outputs/_benchmarks/musdb18hq_artifacts.json
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

OUT = ROOT / "outputs" / "_benchmarks" / "musdb18hq_artifacts.json"


def candidates(est: dict) -> dict[str, dict[str, np.ndarray]]:
    """Full four-stem sets, so the decomposition sees a complete separation."""
    sw, tta, ep, mdx, dm = (est[k] for k in ("sw", "sw_tta", "ep317", "mdx23c", "demucs"))
    ens = {"vocals": (tta["vocals"] + ep["vocals"]) / 2, "drums": tta["drums"], "bass": tta["bass"]}
    ens_dm = {"vocals": ens["vocals"], "drums": (tta["drums"] + dm["drums"]) / 2,
              "bass": (tta["bass"] + dm["bass"]) / 2}
    out = {"sw": {k: sw[k] for k in M.GROUPS},
           "sw_tta": {k: tta[k] for k in M.GROUPS},
           "demucs": {k: dm[k] for k in M.GROUPS},
           "ensemble": ens, "ensemble_demucs": ens_dm}
    return out


def main(sr: int, limit: int | None) -> None:
    import mir_eval
    import soxr

    rows = {}
    for name, mix, truth in M.songs(limit):
        e = {k: {g: v.astype(np.float64) for g, v in np.load(M.CACHE / k / f"{name}.npz").items()}
             for k in ("sw", "sw_tta", "ep317", "mdx23c", "demucs")}
        cands = candidates(e)
        for c, stems in cands.items():
            if "other" not in stems:  # the ensembles define 'other' as the exact remainder
                rest = sum(stems.values())
                stems = {**stems, "other": mix.astype(np.float64) - rest}
            names = list(M.GROUPS)
            ref = np.stack([soxr.resample(truth[g].astype(np.float64).mean(0), 44100, sr) for g in names])
            est = np.stack([soxr.resample(stems[g].mean(0), 44100, sr) for g in names])
            keep = [i for i, r in enumerate(ref) if np.sum(r**2) > 1e-9]  # silent truth stems break bss_eval
            if len(keep) < 2:
                continue
            sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(ref[keep], est[keep], compute_permutation=False)
            for i, k in enumerate(keep):
                rows.setdefault(c, {}).setdefault(names[k], []).append(
                    {"song": name, "sdr": float(sdr[i]), "sir": float(sir[i]), "sar": float(sar[i])})
        print(f"{name}: done", flush=True)

    print(f"\nMUSDB18-HQ, {len(next(iter(rows['sw'].values())))} songs at {sr} Hz. Median dB (higher is cleaner)")
    print(f"  {'candidate':18s} {'stem':7s} {'SDR':>7s} {'SIR (bleed)':>12s} {'SAR (artifacts)':>16s}")
    summary = {}
    for c, per_stem in rows.items():
        for stem, vals in per_stem.items():
            med = {m: float(np.median([v[m] for v in vals])) for m in ("sdr", "sir", "sar")}
            summary.setdefault(c, {})[stem] = med
            print(f"  {c:18s} {stem:7s} {med['sdr']:7.2f} {med['sir']:12.2f} {med['sar']:16.2f}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"sample_rate": sr, "summary": summary, "rows": rows}, indent=1))
    print(f"written {OUT}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--limit", type=int, default=20)
    a = p.parse_args()
    main(a.sr, a.limit)
