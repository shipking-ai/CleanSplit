"""Is there anything left for a "listening AI" to win? Measure the CEILING before building the thing.

    python tools/oracle_headroom.py [--limit 20] [--tile 8 8]

The user's plan is: split, then have an AI listen to the stems and fix the damaged parts. Two generative repairers were
built and both lost (docs/04 sections 4, 9), because ~97% of the error is misallocation, not missing information
(section 5). But that does not rule out the *other* reading of the same idea: instead of inventing audio, LISTEN to
several models and decide, region by region, which one got it right. Averaging already does a dumb version of this and
it is the single biggest measured win in the project (+1.12 dB vocal SAR, section 14.1).

Before writing a selector, measure whether a selector could pay. This computes an ORACLE: for every
time-frequency tile it picks whichever candidate is closest to the TRUE stem, using the truth to choose. That is
cheating, and deliberately so -- it is an upper bound on any possible selector, learned or otherwise:

    oracle - average  =  the ENTIRE budget a perfect listener could ever win.

If that gap is large, a selector is worth building and this says how much is on the table. If it is small, the idea is
dead on arrival and no amount of model quality changes that. Either way it is a real answer, obtained without training
anything.

Two things this number is NOT, stated so nobody quotes it as a result:
  * It is not achievable. It consumes the truth, which no shipped tool has.
  * It is optimistic even as a bound: reassembling from tiles adds its own seam artifacts, and the oracle is credited
    for none of them. The gap here is a ceiling on the ceiling.

All comparisons are PAIRED (the median of the per-song difference, plus how many songs it wins), never the difference
of two medians. That is not pedantry: on this data the unpaired form claims ep317 alone beats the shipped average by
+0.32 dB, while the paired form shows it LOSING by 0.30 dB on 17 of 20 songs. The two medians simply come from
different songs.

Candidates per tile: sw_tta, ep317, their average (what ships). Vocals only -- it is the only stem the ensemble
changes and the only one with two independent estimates. Runs off the cache, no GPU.
Output: outputs/_benchmarks/musdb18hq_oracle_headroom.json
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

OUT = ROOT / "outputs" / "_benchmarks" / "musdb18hq_oracle_headroom.json"
GRID = STFTGrid(n_fft=4096, hop=1024)


def tiled_oracle(truth: np.ndarray, cands: dict[str, np.ndarray], tf: int, tt: int) -> tuple[np.ndarray, dict]:
    """Per-tile pick of the candidate with the smallest squared error to the truth, in the STFT domain."""
    T = stft(truth, GRID, dtype=np.complex128)
    C = {k: stft(v, GRID, dtype=np.complex128) for k, v in cands.items()}
    keys = list(C)
    err = np.stack([np.abs(C[k] - T) ** 2 for k in keys])          # [cand, ch, freq, frame]
    nf, nt = err.shape[-2], err.shape[-1]
    pf, pt = -(-nf // tf) * tf - nf, -(-nt // tt) * tt - nt
    e = np.pad(err, ((0, 0), (0, 0), (0, pf), (0, pt)))
    # sum the error inside each tile, over channels too: one decision per tile for the whole stereo image
    agg = e.reshape(len(keys), e.shape[1], e.shape[2] // tf, tf, e.shape[3] // tt, tt).sum((1, 3, 5))
    pick = agg.argmin(0)                                            # [tile_f, tile_t]
    full = np.repeat(np.repeat(pick, tf, 0), tt, 1)[:nf, :nt]
    out = np.zeros_like(T)
    for i, k in enumerate(keys):
        m = (full == i)
        out += C[k] * m
    counts = {k: float((pick == i).mean()) for i, k in enumerate(keys)}
    return istft(out, GRID, truth.shape[-1]), counts


def main(limit: int | None, tf: int, tt: int, out: Path) -> None:
    from cleansplit.metrics.signal import snr_db

    rows = []
    for name, _mix, truth in M.songs(limit):
        t = truth["vocals"].astype(np.float64)
        if float(np.sum(t**2)) <= 1e-9:
            continue
        e = {k: np.load(M.CACHE / k / f"{name}.npz")["vocals"].astype(np.float64) for k in ("sw_tta", "ep317")}
        avg = (e["sw_tta"] + e["ep317"]) / 2
        cands = {"sw_tta": e["sw_tta"], "ep317": e["ep317"], "average": avg}
        oracle, counts = tiled_oracle(t, cands, tf, tt)
        r = {"song": name, "sw_tta": float(snr_db(t, e["sw_tta"])), "ep317": float(snr_db(t, e["ep317"])),
             "average": float(snr_db(t, avg)), "oracle": float(snr_db(t, oracle)), "picked": counts}
        rows.append(r)
        print(f"  {name[:38]:38s} avg {r['average']:6.2f} -> oracle {r['oracle']:6.2f} "
              f"({r['oracle'] - r['average']:+.2f} dB)  picks sw/ep/avg "
              f"{counts['sw_tta']:.0%}/{counts['ep317']:.0%}/{counts['average']:.0%}", flush=True)

    med = lambda k: float(np.median([r[k] for r in rows]))
    # PAIRED, and this matters: the median of one arm minus the median of another is NOT the typical per-song
    # difference, because the two medians can come from different songs. On this very data the unpaired form said
    # ep317 alone beat the shipped average by +0.32 dB while the paired form says it LOSES by 0.30 dB on 17/20 songs.
    # Every delta below is the median of the per-song difference, with the win count beside it.
    paired = lambda a, b: (float(np.median([r[a] - r[b] for r in rows])),
                           int(sum(r[a] > r[b] for r in rows)))
    gap, gap_wins = paired("oracle", "average")
    print(f"\nMUSDB18-HQ vocals, {len(rows)} songs, {tf}x{tt}-bin tiles ({tf * M.SR / GRID.n_fft:.0f} Hz x "
          f"{tt * GRID.hop / M.SR * 1000:.0f} ms). Median full-band SNR")
    for k in ("sw_tta", "ep317", "average", "oracle"):
        print(f"  {k:10s} {med(k):6.2f} dB")
    print(f"\n  paired deltas vs the shipped average (median per-song difference, songs won):")
    for k in ("sw_tta", "ep317", "oracle"):
        d, w = paired(k, "average")
        print(f"    {k:10s} {d:+6.2f} dB  {w:2d}/{len(rows)}")
    print(f"\n  HEADROOM a perfect listener could win: {gap:+.2f} dB over what ships, on {gap_wins}/{len(rows)} songs")
    print(f"  for scale: the entire two-model ensemble is worth +0.39 dB, and every generative repairer tested lost.")
    picks = {k: float(np.mean([r['picked'][k] for r in rows])) for k in rows[0]["picked"]}
    print(f"  oracle picked: " + ", ".join(f"{k} {v:.0%}" for k, v in picks.items()))
    print("  NOT achievable: it reads the truth to choose, and is not charged for the seams it creates.")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"tile": [tf, tt],
                               "median": {k: med(k) for k in ("sw_tta", "ep317", "average", "oracle")},
                               "paired_vs_average": {k: {"median_delta_db": paired(k, "average")[0],
                                                         "wins": paired(k, "average")[1], "n": len(rows)}
                                                     for k in ("sw_tta", "ep317", "oracle")},
                               "headroom_db": gap, "headroom_wins": gap_wins,
                               "pick_fraction": picks, "rows": rows}, indent=1))
    print(f"written {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--tile", type=int, nargs=2, default=[8, 8], metavar=("FREQ_BINS", "FRAMES"))
    p.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args()
    main(a.limit, a.tile[0], a.tile[1], a.out)
