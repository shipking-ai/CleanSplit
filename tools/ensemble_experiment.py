"""Per-stem ensembles of local separators, scored against known stems (BUH excerpts).

Separators: SW (6 stems), SW with 3-pass TTA, BS-RoFormer ep317 (vocals + complement), HTDemucs_ft (4 stems).
Truth groups: vocals, drums, bass, other (SW other <- guitar + piano + other).
Only equal-weight averages are evaluated (no weight tuning on two excerpts). Every full ensemble is made
mixture-consistent by giving 'other' the remainder: other = mix - vocals - drums - bass.

CAVEAT: the 'truth' comes from an undisclosed commercial separator; if it is Demucs-like, Demucs will score better
than it deserves. Report per-source results, not a winner.
Usage: python tools/ensemble_experiment.py [start_s ...]
"""
import json
import sys
from pathlib import Path

import numpy as np

from cleansplit.metrics import signal as M
from cleansplit.metrics.restoration_experiment import stem_folder_songs
from cleansplit.separation import registry

starts = [float(s) for s in sys.argv[1:]] or [60.0, 120.0]
pat = str(Path.home() / "Downloads" / "BUH - {} - Treblo.wav")
paths = {k: pat.format(k) for k in ("vocals", "drums", "bass", "instrumentals")}
songs = stem_folder_songs(paths, [(s, 30.0) for s in starts], derive={"other": (["instrumentals"], ["drums", "bass"])}, drop=("instrumentals",))

seps = {
    "sw": registry.create("bs_roformer_sw"),
    "sw_tta": registry.create("bs_roformer_sw", tta=True),
    "ep317": registry.create("bs_roformer_ep317"),
    "demucs": registry.create("htdemucs_ft"),
}
report = {}
for name, mix, truth in songs:
    mix64 = mix.astype(np.float64)
    out = {k: {s: v.astype(np.float64) for s, v in sep.separate(mix, 44100).stems.items()} for k, sep in seps.items()}
    g = {}  # group estimates per source
    for k in ("sw", "sw_tta"):
        st = out[k]
        g[k] = {"vocals": st["vocals"], "drums": st["drums"], "bass": st["bass"], "other": st["guitar"] + st["piano"] + st["other"]}
    g["ep317"] = {"vocals": out["ep317"]["vocals"]}
    g["demucs"] = {s: out["demucs"][s] for s in ("vocals", "drums", "bass", "other")}

    def sdr(t, est):
        return M.snr_db(truth[t].astype(np.float64), est)

    rows = {}
    for t in ("vocals", "drums", "bass", "other"):
        cand = {k: v[t] for k, v in g.items() if t in v}
        r = {k: sdr(t, v) for k, v in cand.items()}
        keys = list(cand)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                if {keys[i], keys[j]} == {"sw", "sw_tta"}:
                    continue
                r[f"avg({keys[i]},{keys[j]})"] = sdr(t, (cand[keys[i]] + cand[keys[j]]) / 2)
        if len(keys) >= 3:
            ks = [k for k in keys if k != "sw"]
            r[f"avg({','.join(ks)})"] = sdr(t, sum(cand[k] for k in ks) / len(ks))
        rows[t] = r

    # full, mixture-consistent ensembles
    def full(v, d, b):
        return {"vocals": v, "drums": d, "bass": b, "other": mix64 - v - d - b}

    fulls = {
        "sw (single pass)": {t: g["sw"][t] for t in ("vocals", "drums", "bass", "other")},
        "sw_tta": {t: g["sw_tta"][t] for t in ("vocals", "drums", "bass", "other")},
        "E1: V=avg(sw_tta,ep317) D,B=sw_tta": full((g["sw_tta"]["vocals"] + g["ep317"]["vocals"]) / 2, g["sw_tta"]["drums"], g["sw_tta"]["bass"]),
        "E2: V=avg(sw_tta,ep317) D,B=avg(sw_tta,demucs)": full((g["sw_tta"]["vocals"] + g["ep317"]["vocals"]) / 2,
                                                              (g["sw_tta"]["drums"] + g["demucs"]["drums"]) / 2, (g["sw_tta"]["bass"] + g["demucs"]["bass"]) / 2),
        "E3: V=avg(sw_tta,ep317,demucs) D,B=avg(sw_tta,demucs)": full((g["sw_tta"]["vocals"] + g["ep317"]["vocals"] + g["demucs"]["vocals"]) / 3,
                                                                     (g["sw_tta"]["drums"] + g["demucs"]["drums"]) / 2, (g["sw_tta"]["bass"] + g["demucs"]["bass"]) / 2),
    }
    frows = {}
    for fname, est in fulls.items():
        frows[fname] = {t: sdr(t, est[t]) for t in est}
        frows[fname]["mixture_snr_db"] = M.snr_db(mix64, sum(est.values()))
    report[name] = {"per_group": rows, "full_ensembles": frows}

    print(f"\n== {name}")
    for t, r in rows.items():
        base = r["sw"]
        print(f"  {t:7s} " + "  ".join(f"{k} {v:.2f}({v - base:+.2f})" for k, v in r.items()))
    print("  full ensembles (SDR per group, mixture SNR):")
    b = frows["sw (single pass)"]
    for fname, fr in frows.items():
        print(f"    {fname:55s} " + " ".join(f"{t} {fr[t]:6.2f}({fr[t] - b[t]:+.2f})" for t in ("vocals", "drums", "bass", "other")) + f"  mix {fr['mixture_snr_db']:.1f} dB")

Path("outputs/_benchmarks/ensemble_experiment_BUH.json").write_text(json.dumps(report, indent=1))
print("\nwritten outputs/_benchmarks/ensemble_experiment_BUH.json")
