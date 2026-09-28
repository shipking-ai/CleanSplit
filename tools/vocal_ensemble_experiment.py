"""Does MDX23C InstVoc HQ improve the vocal ensemble? Scored against known vocal stems (BUH excerpts).

    python tools/vocal_ensemble_experiment.py [start_s ...]      (default: 30 60 90 120 150, 30 s each)

Candidates: SW with 3-pass TTA, BS-RoFormer ep317, MDX23C InstVoc HQ (native overlap 8). Every equal-weight
combination is scored — no weights are tuned, because tuning on the same excerpts that score it would be fitting
the test. Only vocals are scored: in docs/04 section 7 the BUH vocal stem is the only truth that is reliable
(its derived 'other' contains inverted drums).

The decision rule is set before running: MDX23C joins the default ensemble only if avg(sw_tta, ep317, mdx23c)
beats the current avg(sw_tta, ep317) on the pooled score AND on a majority of excerpts. Otherwise it stays
available as a separator and the result is written down either way.
Output: outputs/_benchmarks/vocal_ensemble_BUH.json
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

from cleansplit.metrics import signal as M
from cleansplit.metrics.restoration_experiment import stem_folder_songs
from cleansplit.separation import registry

starts = [float(s) for s in sys.argv[1:]] or [30.0, 60.0, 90.0, 120.0, 150.0]
pat = str(Path.home() / "Downloads" / "BUH - {} - Treblo.wav")
paths = {k: pat.format(k) for k in ("vocals", "drums", "bass", "instrumentals")}
songs = stem_folder_songs(paths, [(s, 30.0) for s in starts], derive={"other": (["instrumentals"], ["drums", "bass"])},
                          drop=("instrumentals",))

members = {"sw_tta": registry.create("bs_roformer_sw", tta=True), "ep317": registry.create("bs_roformer_ep317"),
           "mdx23c": registry.create("mdx23c_instvoc_hq")}


def release(sep):
    sep._model = None
    import torch

    torch.cuda.empty_cache()


# Run one model over every excerpt before loading the next: one model on the 8 GB card at a time.
est = {name: {} for name, _, _ in songs}
for key, sep in members.items():
    for name, mix, _ in songs:
        est[name][key] = sep.separate(mix, 44100).stems["vocals"].astype(np.float64)
    release(sep)
    print(f"{key}: done", flush=True)

combos = [c for r in (1, 2, 3) for c in itertools.combinations(members, r)]
label = lambda c: c[0] if len(c) == 1 else f"avg({','.join(c)})"
report, err, ref = {}, {label(c): 0.0 for c in combos}, 0.0
for name, _, truth in songs:
    t = truth["vocals"].astype(np.float64)
    ref += float(np.sum(t**2))
    row = {}
    for c in combos:
        e = sum(est[name][k] for k in c) / len(c)
        row[label(c)] = M.snr_db(t, e)
        err[label(c)] += float(np.sum((t - e) ** 2))
    report[name] = {k: float(v) for k, v in row.items()}

pooled = {k: float(10 * np.log10(ref / v)) for k, v in err.items()}
base, cand = "avg(sw_tta,ep317)", "avg(sw_tta,ep317,mdx23c)"
wins = int(sum(report[n][cand] > report[n][base] for n in report))
decision = bool(pooled[cand] > pooled[base] and wins > len(report) / 2)

print("\nvocals SDR (dB) per excerpt; last column pooled over all excerpts")
names = list(report)
print(f"{'':28s}" + "".join(f"{n[-10:]:>11s}" for n in names) + f"{'pooled':>10s}")
for c in combos:
    k = label(c)
    print(f"{k:28s}" + "".join(f"{report[n][k]:11.2f}" for n in names) + f"{pooled[k]:10.2f}")
print(f"\n{cand} vs {base}: pooled {pooled[cand] - pooled[base]:+.2f} dB, better on {wins}/{len(report)} excerpts "
      f"-> {'JOIN the default ensemble' if decision else 'do NOT change the default ensemble'}")

out = Path("outputs/_benchmarks/vocal_ensemble_BUH.json")
out.write_text(json.dumps({"excerpts": report, "pooled_db": pooled, "baseline": base, "candidate": cand,
                           "candidate_wins": wins, "n_excerpts": len(report), "adopt": decision}, indent=1))
print(f"written {out}")
