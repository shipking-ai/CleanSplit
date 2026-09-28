"""Where does SW's missing energy belong, and how much could ANY reallocation of the residual recover?

Per truth group i (group sums of SW stems), in the STFT domain (bin 0 excluded: SW zeroes it by design):
  M_i = T_i - S_i            what stem i is missing (or has in excess)
  E   = O - sum_i S_i = sum_i M_i   (exact: O = sum_i T_i)
Allocating the residual with weights a_i (sum a_i = 1) gives error sum_i |M_i - a_i E|^2. Compared strategies:
  none          a_i = 0 (current stems)                      -> sum |M_i|^2
  proportional  a_i = smoothed P(S_i) / sum_j P(S_j)          (what residual_reallocation does)
  oracle_cell   a_i = Re(M_i E*) / |E|^2 per TF cell            (upper bound; uses the truth)
  oracle_block  oracle a_i averaged over 5x9 cells              (upper bound at the restorer's resolution)
  confident     proportional, but only in cells where max_i a_i >= tau (else a = 0): tests "act only when owner is clear"
Reported per frequency band and overall, in dB relative to 'none' (negative = better).

Usage: python tools/experiments/diagnose_missing_energy.py [start_s ...]
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter

from cleansplit.audio.stft import STFTGrid, stft
from cleansplit.metrics.restoration_experiment import stem_folder_songs
from cleansplit.separation.roformer import BSRoformerSeparator

starts = [float(s) for s in sys.argv[1:]] or [60.0, 120.0]
pat = str(Path.home() / "Downloads" / "BUH - {} - Treblo.wav")
paths = {k: pat.format(k) for k in ("vocals", "drums", "bass", "instrumentals")}
songs = stem_folder_songs(paths, [(s, 30.0) for s in starts], derive={"other": (["instrumentals"], ["drums", "bass"])}, drop=("instrumentals",))
groups = {"vocals": ["vocals"], "drums": ["drums"], "bass": ["bass"], "other": ["guitar", "piano", "other"]}
g = STFTGrid()
bands = [(20, 250), (250, 2000), (2000, 6000), (6000, 12000), (12000, 22050)]
freqs = g.freqs()
sep = BSRoformerSeparator()
report = {}

for name, mix, truth in songs:
    res = sep.separate(mix, 44100)
    S = {k: stft(sum(res.stems[m].astype(np.float64) for m in ms), g, dtype=np.complex128)[:, 1:] for k, ms in groups.items()}
    T = {k: stft(truth[k].astype(np.float64), g, dtype=np.complex128)[:, 1:] for k in groups}
    f = freqs[1:]
    M = {k: T[k] - S[k] for k in groups}
    E = sum(M.values())
    E2 = (np.abs(E) ** 2).sum(axis=0) + 1e-20  # (F, T) summed over channels

    P = {k: uniform_filter((np.abs(S[k]) ** 2).sum(axis=0), size=(5, 9), mode="nearest") for k in groups}
    tot = sum(P.values()) + 1e-20
    a_prop = {k: P[k] / tot for k in groups}
    a_cell = {k: (M[k] * np.conj(E)).real.sum(axis=0) / E2 for k in groups}
    a_block = {k: uniform_filter(a_cell[k], size=(5, 9), mode="nearest") for k in groups}
    amax = np.max(np.stack([a_prop[k] for k in groups]), axis=0)

    def err(alloc, cells=None):
        out = {}
        for k in groups:
            e = (np.abs(M[k] - alloc[k][None] * E) ** 2).sum(axis=0) if alloc is not None else (np.abs(M[k]) ** 2).sum(axis=0)
            out[k] = e if cells is None else e * cells
        return out

    strategies = {"none": err(None), "proportional": err(a_prop), "oracle_cell": err(a_cell), "oracle_block": err(a_block)}
    for tau in (0.8, 0.95):
        conf = amax >= tau
        strategies[f"confident_{tau}"] = err({k: a_prop[k] * conf for k in groups})

    rep = {"separation_sdr_db": {}, "missing_energy_share": {}, "bands": {}}
    for k in groups:
        t = float(np.sum(np.abs(T[k]) ** 2))
        rep["separation_sdr_db"][k] = 10 * np.log10(t / float(np.sum(np.abs(M[k]) ** 2)))
    e_tot = float(E2.sum())
    rep["residual_rel_mixture_db"] = 10 * np.log10(e_tot / float(sum((np.abs(T[k]) ** 2).sum() for k in groups)))
    # how much of total stem error is 'missing energy' (E) vs 'misallocation' (errors that cancel in the sum)
    total_err = float(sum(np.sum(np.abs(M[k]) ** 2) for k in groups))
    rep["residual_share_of_total_stem_error"] = e_tot / total_err

    def band_sel(lo, hi):
        return ((f >= lo) & (f < hi))[:, None]

    for (lo, hi) in [*bands, (0, 22050)]:
        sel = band_sel(lo, hi)
        base = sum(float(np.sum(strategies["none"][k] * sel)) for k in groups)
        row = {}
        for sname, e in strategies.items():
            val = sum(float(np.sum(e[k] * sel)) for k in groups)
            row[sname] = 10 * np.log10(val / base) if base > 0 else None
        row["vocals_only"] = {sname: 10 * np.log10(float(np.sum(e["vocals"] * sel)) / float(np.sum(strategies["none"]["vocals"] * sel))) for sname, e in strategies.items()}
        row["residual_share_of_stem_error"] = float(np.sum(E2 * sel)) / base if base > 0 else None
        rep["bands"][f"{lo}-{hi}Hz"] = row
    rep["confident_cell_fraction"] = {str(tau): float(np.mean(amax >= tau)) for tau in (0.8, 0.95)}
    report[name] = rep

    print(f"\n== {name}: separation SDR {{{', '.join(f'{k} {v:.1f}' for k, v in rep['separation_sdr_db'].items())}}} dB, "
          f"residual {rep['residual_rel_mixture_db']:.1f} dB rel mix; residual = {100 * rep['residual_share_of_total_stem_error']:.1f}% of total stem error energy")
    print(f"   {'band':14s} {'resid share':>11s} " + " ".join(f"{s:>14s}" for s in strategies))
    for bname, row in rep["bands"].items():
        print(f"   {bname:14s} {100 * row['residual_share_of_stem_error']:10.1f}% " + " ".join(f"{row[s]:+13.2f}dB" for s in strategies))
    print("   vocals only (all bands): " + ", ".join(f"{s} {v:+.2f} dB" for s, v in rep["bands"]["0-22050Hz"]["vocals_only"].items()))
    print("   confident-owner cell fraction:", rep["confident_cell_fraction"])

Path("outputs/_benchmarks/missing_energy_diagnosis_BUH.json").write_text(json.dumps(report, indent=1))
print("\nwritten outputs/_benchmarks/missing_energy_diagnosis_BUH.json")
