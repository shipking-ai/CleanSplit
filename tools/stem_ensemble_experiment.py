"""Close the biggest structural gap: give drums, bass and 'other' the ensemble partner only vocals have.

    python tools/stem_ensemble_experiment.py [--limit 20] [--overlap 4]

Today `vocals = mean(SW+TTA, ep317)` and that average is worth **+1.12 dB SAR** (docs/04 section 14.1), the largest
quality gain in this project. Drums, bass, guitar and piano come from SW alone, so three quarters of the output gets
none of it. SCNet XL IHF is the candidate partner: a four-stem CONVOLUTIONAL model (arXiv:2401.13276), architecturally
unlike a band-split transformer, which is the property that makes averaging cancel artifacts instead of just blurring.

PRE-REGISTERED, written before the run. For each stem independently, mean(SW+TTA, SCNet) replaces SW+TTA only if:
    (1) paired median full-band 44.1 kHz filter-free SNR improves, AND
    (2) it wins on more than half the songs, AND
    (3) median SAR does not drop.
(1) and (2) are the primary gate because docs/04 section 14.4 caught a candidate that gained on bss_eval's SAR while
being worse full-band on 16/20 songs; (3) keeps the artifact axis honest. All deltas are PAIRED (median of per-song
differences), because on this project's own data the unpaired form once claimed ep317 alone beat the shipped average
by +0.32 dB when it actually LOSES on 17/20 songs.

FALSIFIABLE PREDICTION, recorded before any number is seen. Three third models have been rejected here (MDX23C,
HTDemucs_ft, MDX-Net Inst HQ 3) and all three were >2 dB weaker than the pair they joined, giving the rule: a member
more than ~1-2 dB below its partner drags the average down. SCNet's author-reported MUSDB drums SDR is 11.81 against
SW's ~12.4 here, i.e. inside that band, so the rule predicts SCNet HELPS on drums. If SCNet lands within 2 dB of SW on
a stem and still hurts it, the comparable-strength rule is wrong and docs/01 section 8.2 must be corrected.

'other' is compared on SW's own 'other' group (guitar + piano + other) against SCNet's single 'other', which is the
same four-group convention as tools/musdb_eval.py. Vocals are included as a control: SCNet's vocals are expected to be
too weak to help the existing pair.

Estimates cache under data/musdb_cache/scnet_ov<N>/, so the run is resumable.
Output: outputs/_benchmarks/musdb18hq_stem_ensemble.json
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

OUT = ROOT / "outputs" / "_benchmarks" / "musdb18hq_stem_ensemble.json"
STEMS = ("drums", "bass", "other", "vocals")


def fill(overlap: int, limit: int | None) -> None:
    import torch

    from cleansplit.separation import registry

    key = f"scnet_ov{overlap}"
    items = [(n, m) for n, m, _ in M.songs(limit)]
    todo = [(n, m) for n, m in items if not (M.CACHE / key / f"{n}.npz").is_file()]
    if not todo:
        print(f"{key}: cached", flush=True)
        return
    sep = registry.create("scnet_xl_ihf", num_overlap=overlap)
    for i, (name, mix) in enumerate(todo, 1):
        st = sep.separate(mix, M.SR).stems
        (M.CACHE / key).mkdir(parents=True, exist_ok=True)
        np.savez(M.CACHE / key / f"{name}.npz", **{k: v.astype(np.float32) for k, v in st.items()})
        print(f"{key}: {i}/{len(todo)} {name}", flush=True)
    sep._model = None
    torch.cuda.empty_cache()


def main(overlap: int, limit: int | None, sr: int, out: Path) -> None:
    import mir_eval
    import soxr

    from cleansplit.metrics.signal import snr_db

    fill(overlap, limit)
    rows: dict[str, list[dict]] = {}
    for name, mix, truth in M.songs(limit):
        f_sw = M.CACHE / f"sw_tta_ov{overlap}" / f"{name}.npz"
        f_sc = M.CACHE / f"scnet_ov{overlap}" / f"{name}.npz"
        if not (f_sw.is_file() and f_sc.is_file()):
            print(f"{name}: skipped (missing cache)", flush=True)
            continue
        sw = {k: v.astype(np.float64) for k, v in np.load(f_sw).items()}
        sc = {k: v.astype(np.float64) for k, v in np.load(f_sc).items()}
        ep = np.load(M.CACHE / "ep317" / f"{name}.npz")["vocals"].astype(np.float64)
        pair_voc = (sw["vocals"] + ep) / 2
        # per stem: what ships today, versus adding SCNet to it
        base = {"drums": sw["drums"], "bass": sw["bass"], "other": sw["other"], "vocals": pair_voc}
        cand = {"drums": (sw["drums"] + sc["drums"]) / 2, "bass": (sw["bass"] + sc["bass"]) / 2,
                "other": (sw["other"] + sc["other"]) / 2, "vocals": (sw["vocals"] + ep + sc["vocals"]) / 3}
        names = list(M.GROUPS)
        ref = np.stack([soxr.resample(truth[g].astype(np.float64).mean(0), M.SR, sr) for g in names])
        keep = [i for i, r in enumerate(ref) if np.sum(r**2) > 1e-9]
        sar = {}
        for label, stems in (("base", base), ("cand", cand)):
            est = np.stack([soxr.resample(stems[g].mean(0), M.SR, sr) for g in names])
            _, _, s, _ = mir_eval.separation.bss_eval_sources(ref[keep], est[keep], compute_permutation=False)
            sar[label] = {names[k]: float(s[i]) for i, k in enumerate(keep)}
        for g in STEMS:
            t = truth[g].astype(np.float64)
            if float(np.sum(t**2)) <= 1e-9:
                continue
            rows.setdefault(g, []).append({
                "song": name, "base_snr": float(snr_db(t, base[g])), "cand_snr": float(snr_db(t, cand[g])),
                "scnet_solo_snr": float(snr_db(t, sc[g])), "sw_solo_snr": float(snr_db(t, sw[g])),
                "base_sar": sar["base"].get(g), "cand_sar": sar["cand"].get(g)})
        print(f"{name}: scored", flush=True)

    print(f"\nMUSDB18-HQ, overlap {overlap}. Adding SCNet XL IHF to each stem. PAIRED medians, SNR full-band 44.1 kHz")
    print(f"  {'stem':7s} {'n':>3s} {'SW solo':>8s} {'SCNet':>8s} {'gap':>6s} {'dSNR':>7s} {'won':>7s} {'dSAR':>7s} verdict")
    verdict = {}
    for g in STEMS:
        r = rows.get(g)
        if not r:
            continue
        d_snr = float(np.median([x["cand_snr"] - x["base_snr"] for x in r]))
        won = int(sum(x["cand_snr"] > x["base_snr"] for x in r))
        pairs = [(x["cand_sar"], x["base_sar"]) for x in r if x["cand_sar"] is not None and x["base_sar"] is not None]
        d_sar = float(np.median([a - b for a, b in pairs])) if pairs else float("nan")
        sw_solo = float(np.median([x["sw_solo_snr"] for x in r]))
        sc_solo = float(np.median([x["scnet_solo_snr"] for x in r]))
        ok = bool(d_snr > 0 and won > len(r) / 2 and (np.isnan(d_sar) or d_sar >= 0))
        verdict[g] = {"delta_snr_db": d_snr, "won": won, "n": len(r), "delta_sar_db": d_sar,
                      "sw_solo_snr": sw_solo, "scnet_solo_snr": sc_solo, "strength_gap_db": sw_solo - sc_solo,
                      "adopt": ok}
        print(f"  {g:7s} {len(r):3d} {sw_solo:8.2f} {sc_solo:8.2f} {sw_solo - sc_solo:6.2f} {d_snr:+7.2f} "
              f"{won:3d}/{len(r):<3d} {d_sar:+7.2f} {'ADOPT' if ok else 'no'}")
    print("\n  comparable-strength rule (docs/01 8.2): a member >1-2 dB weaker should HURT.")
    for g, v in verdict.items():
        pred = "help" if v["strength_gap_db"] <= 2.0 else "hurt"
        got = "helped" if v["adopt"] else "hurt"
        mark = "consistent" if (pred == "help") == v["adopt"] else "** RULE CONTRADICTED **"
        print(f"    {g:7s} gap {v['strength_gap_db']:+.2f} dB -> predicted {pred}, {got}: {mark}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"overlap": overlap, "sample_rate_sar": sr, "verdict": verdict, "rows": rows}, indent=1))
    print(f"written {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--overlap", type=int, default=4)
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args()
    main(a.overlap, a.limit, a.sr, a.out)
