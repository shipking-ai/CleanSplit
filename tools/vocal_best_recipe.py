"""What is the ABSOLUTE BEST vocal stem this project can make? The one untested lever on the flagship stem.

    python tools/vocal_best_recipe.py [--limit 20] [--overlap 4]

Vocals are the only stem the ensemble actually improves: on MUSDB18-HQ, going from fast single-pass SW to the shipped
ensemble is +0.39 dB on vocals and +0.02..+0.05 dB on drums, bass and other (docs/04 section 11). All of that gain
comes from averaging a SECOND model, ep317, into the vocal stem -- and ep317 is run WITHOUT test-time augmentation,
while its partner SW is run with it. Nobody has measured whether that asymmetry costs anything. It is the last
untested lever on the stem that matters most, and docs/04 section 14.5 lists it as open work rather than guessing.

Arms (all at the same --overlap, so overlap is not a confound):
  sw_tta            SW, 3-pass TTA                      (already cached by tools/overlap_experiment.py at overlap 4)
  ep317             ep317, single pass                  the shipped ensemble's second vocal model
  ep317_tta         ep317, 3-pass TTA                   new
Candidates:
  shipped           mean(sw_tta, ep317)                 what ships today
  both_tta          mean(sw_tta, ep317_tta)             the question
  ep317_tta_alone   ep317_tta                           a control: is the averaging doing the work, or just TTA?

PRE-REGISTERED, written before the run. `both_tta` replaces the shipped recipe only if BOTH hold:
    median full-band 44.1 kHz filter-free SNR improves on the vocal stem, AND it wins on more than half the songs.
The full-band filter-free metric is primary because section 14.4 caught a candidate (+mdx23c) that gained on
bss_eval's SAR while being worse on 16/20 songs full-band; SAR is reported alongside but does not decide.
Cost if it passes: ep317 goes from 1 pass to 3, so the ensemble goes from ~4 passes to ~6 (+50% GPU time).

Estimates cache under data/musdb_cache/<arm>_ov<N>/, so the run is resumable.
Output: outputs/_benchmarks/musdb18hq_vocal_recipe.json
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

OUT = ROOT / "outputs" / "_benchmarks" / "musdb18hq_vocal_recipe.json"
ARMS = {"ep317": dict(tta=False), "ep317_tta": dict(tta=True)}


def fill(overlap: int, limit: int | None) -> None:
    import torch

    from cleansplit.separation import registry

    items = [(n, m) for n, m, _ in M.songs(limit)]
    for arm, kw in ARMS.items():
        key = f"{arm}_ov{overlap}"
        todo = [(n, m) for n, m in items if not (M.CACHE / key / f"{n}.npz").is_file()]
        if not todo:
            print(f"{key}: cached", flush=True)
            continue
        sep = registry.create("bs_roformer_ep317", num_overlap=overlap, **kw)
        for i, (name, mix) in enumerate(todo, 1):
            v = sep.separate(mix, M.SR).stems["vocals"].astype(np.float32)
            (M.CACHE / key).mkdir(parents=True, exist_ok=True)
            np.savez(M.CACHE / key / f"{name}.npz", vocals=v)
            print(f"{key}: {i}/{len(todo)} {name}", flush=True)
        sep._model = None
        torch.cuda.empty_cache()


def main(overlap: int, limit: int | None, sr: int, out: Path) -> None:
    import mir_eval
    import soxr

    from cleansplit.metrics.signal import snr_db

    fill(overlap, limit)
    per: dict[str, list[dict]] = {}
    for name, mix, truth in M.songs(limit):
        f = {k: M.CACHE / k / f"{name}.npz" for k in (f"sw_tta_ov{overlap}", f"ep317_ov{overlap}", f"ep317_tta_ov{overlap}")}
        if not all(p.is_file() for p in f.values()):
            print(f"{name}: skipped, missing {[k for k, p in f.items() if not p.is_file()]}", flush=True)
            continue
        sw = {g: v.astype(np.float64) for g, v in np.load(f[f"sw_tta_ov{overlap}"]).items()}
        e1 = np.load(f[f"ep317_ov{overlap}"])["vocals"].astype(np.float64)
        e3 = np.load(f[f"ep317_tta_ov{overlap}"])["vocals"].astype(np.float64)
        cands = {"shipped": (sw["vocals"] + e1) / 2, "both_tta": (sw["vocals"] + e3) / 2, "ep317_tta_alone": e3}
        t = truth["vocals"].astype(np.float64)
        if float(np.sum(t**2)) <= 1e-9:
            continue
        names = list(M.GROUPS)
        ref = np.stack([soxr.resample(truth[g].astype(np.float64).mean(0), M.SR, sr) for g in names])
        keep = [i for i, r in enumerate(ref) if np.sum(r**2) > 1e-9]
        for c, v in cands.items():
            stems = {"vocals": v, "drums": sw["drums"], "bass": sw["bass"]}
            stems["other"] = mix.astype(np.float64) - v - sw["drums"] - sw["bass"]
            est = np.stack([soxr.resample(stems[g].mean(0), M.SR, sr) for g in names])
            _, _, sar, _ = mir_eval.separation.bss_eval_sources(ref[keep], est[keep], compute_permutation=False)
            i = keep.index(names.index("vocals"))
            per.setdefault(c, []).append({"song": name, "snr": float(snr_db(t, v)), "sar": float(sar[i])})
        print(f"{name}: done", flush=True)

    base = per["shipped"]
    print(f"\nMUSDB18-HQ vocals, {len(base)} songs, overlap {overlap}. Full-band filter-free SNR is the deciding metric")
    print(f"  {'candidate':16s} {'SNR':>7s} {'dSNR':>7s} {'won':>7s} {'SAR':>7s} {'dSAR':>7s}")
    med = lambda rows, k: float(np.median([r[k] for r in rows]))
    verdict = {}
    for c, rows in per.items():
        won = sum(a["snr"] > b["snr"] for a, b in zip(rows, base))
        d_snr, d_sar = med(rows, "snr") - med(base, "snr"), med(rows, "sar") - med(base, "sar")
        print(f"  {c:16s} {med(rows,'snr'):7.2f} {d_snr:+7.2f} {won:3d}/{len(rows):<3d} {med(rows,'sar'):7.2f} {d_sar:+7.2f}")
        if c != "shipped":
            verdict[c] = {"delta_snr_db": d_snr, "won": won, "n": len(rows), "delta_sar_db": d_sar,
                          "adopt": bool(d_snr > 0 and won > len(rows) / 2)}
    print()
    for c, v in verdict.items():
        print(f"  {c:16s} median {v['delta_snr_db']:+.2f} dB, wins {v['won']}/{v['n']} -> "
              f"{'ADOPT' if v['adopt'] else 'no'}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"overlap": overlap, "sample_rate_sar": sr, "verdict": verdict, "rows": per}, indent=1))
    print(f"written {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--overlap", type=int, default=4)
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args()
    main(a.overlap, a.limit, a.sr, a.out)
