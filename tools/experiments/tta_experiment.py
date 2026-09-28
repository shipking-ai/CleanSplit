"""Test-time augmentation for BS-RoFormer SW, scored against known stems.

Variants (each output mapped back to the original orientation, then averaged per stem):
  base      SW(x)
  swap      channel-swapped: swap(SW(swap(x)))
  invert    polarity-inverted: -SW(-x)
  shift     input shifted by half a hop (256 samples): SW(shift(x)) unshifted (different STFT frame alignment)
Averages of stems remain mixture-consistent in the same way the single pass is (averaging is linear).

Usage: python tools/experiments/tta_experiment.py [start_s ...]
"""
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

from cleansplit.metrics import signal as M
from cleansplit.metrics.restoration_experiment import stem_folder_songs
from cleansplit.separation.roformer import BSRoformerSeparator

starts = [float(s) for s in sys.argv[1:]] or [60.0, 120.0]
pat = str(Path.home() / "Downloads" / "BUH - {} - Treblo.wav")
paths = {k: pat.format(k) for k in ("vocals", "drums", "bass", "instrumentals")}
songs = stem_folder_songs(paths, [(s, 30.0) for s in starts], derive={"other": (["instrumentals"], ["drums", "bass"])}, drop=("instrumentals",))
groups = {"vocals": ["vocals"], "drums": ["drums"], "bass": ["bass"], "other": ["guitar", "piano", "other"]}
sep = BSRoformerSeparator()
SHIFT = 256


def run(x):
    return {k: v.astype(np.float64) for k, v in sep.separate(x.astype(np.float32), 44100).stems.items()}


report = {}
for name, mix, truth in songs:
    t0 = time.time()
    outs = {"base": run(mix)}
    s = run(mix[::-1].copy())
    outs["swap"] = {k: v[::-1] for k, v in s.items()}
    s = run(-mix)
    outs["invert"] = {k: -v for k, v in s.items()}
    shifted = np.pad(mix, ((0, 0), (SHIFT, 0)))[:, : mix.shape[1]]
    s = run(shifted)
    outs["shift"] = {k: np.concatenate([v[:, SHIFT:], np.zeros((2, SHIFT))], axis=1) for k, v in s.items()}
    edge = slice(SHIFT, mix.shape[1] - SHIFT)  # exclude the zero-filled shift edges from scoring for every variant
    print(f"\n== {name} ({time.time() - t0:.0f} s for 4 SW passes)")

    def score(stems):
        return {g: M.snr_db(truth[g][:, edge].astype(np.float64), sum(stems[m][:, edge] for m in ms)) for g, ms in groups.items()}

    rows = {}
    names = list(outs)
    for r in range(1, len(names) + 1):
        for combo in combinations(names, r):
            if r == 1 and combo[0] != "base":
                rows["+".join(combo) + " (alone)"] = score(outs[combo[0]])
                continue
            avg = {k: sum(outs[c][k] for c in combo) / len(combo) for k in outs["base"]}
            rows["+".join(combo)] = score(avg)
    base = rows["base"]
    print(f"   {'variant':28s} " + " ".join(f"{g:>16s}" for g in groups) + f" {'mean gain':>10s}")
    for vname, sc in rows.items():
        gains = [sc[g] - base[g] for g in groups]
        print(f"   {vname:28s} " + " ".join(f"{sc[g]:7.2f} ({sc[g] - base[g]:+5.2f})" for g in groups) + f" {np.mean(gains):+10.3f}")
    report[name] = rows

Path("outputs/_benchmarks/tta_experiment_BUH.json").write_text(json.dumps(report, indent=1))
print("\nwritten outputs/_benchmarks/tta_experiment_BUH.json")
