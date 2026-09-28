"""The filter-free full-band gate that docs/04 section 14.4 requires of every SAR candidate.

    python tools/fullband_check.py sw_tta sw_tta_ov4 sw_tta_ov8 [--limit 20] [--baseline sw_tta]

Why this exists: section 14.2's SAR numbers come from `bss_eval`, which fits an optimal 512-tap distortion filter
before scoring, so error shaped like an EQ curve is absorbed into the target term instead of charged as damage. That
was caught in section 14.4 by a falsified prediction: `+mdx23c` gained +0.22 dB SAR while being WORSE full-band on
16/20 songs. The rule written there, before any overlap result was seen: a SAR-only gain is not evidence of a cleaner
stem -- a candidate must also improve the plain, filter-free, full-rate SNR.

This is that check, and nothing else: median `snr_db` at the native 44.1 kHz over all four stems, the same metric as
sections 3, 6, 7 and 11, no downsampling and no filter allowance. Arms are cache keys under data/musdb_cache/.

GATE (fixed here, independent of any candidate's numbers): the PAIRED median full-band SNR must improve on the
vocals, drums AND bass stems, AND the candidate must win on more than half the songs on each of those three. `other`
is reported but not gated (it is the remainder, so it absorbs the other three's mistakes). A gain must also clear
--min-delta, default +0.02 dB.

THE FLOOR, and when it was added, because the order matters: the gate originally required only delta > 0. On the
paired form overlap 8 then PASSED against overlap 4 -- a real, consistent +0.01 dB, winning 12-16 of 20 songs -- for
double the GPU time. A gate with no effect-size floor will adopt any infinitesimal consistent gain, so the floor was
added AFTER seeing that, taking +0.02 dB from the rule already pre-registered in tools/cache_arm.py before this run.
Overlap 8 is therefore rejected on effect size against cost, NOT for being indistinguishable from overlap 4.

PAIRED, and the gate depends on it: the median of one arm minus the median of another is NOT the typical per-song
difference, because the two medians can come from different songs. That unpaired form has already manufactured two
false results in this project (docs/04 sections 14.5, 14.6 and tools/oracle_headroom.py), so every delta here is the
median of per-song differences over the songs BOTH arms have, and songs are matched BY NAME, never by list position:
these caches fill incrementally, so an arm missing a song in the middle would otherwise silently compare song i of one
arm against a different song i of the other.

Output: outputs/_benchmarks/musdb18hq_fullband.json
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

OUT = ROOT / "outputs" / "_benchmarks" / "musdb18hq_fullband.json"
GATED = ("vocals", "drums", "bass")


def paired(cand: dict[str, float], base: dict[str, float]) -> tuple[float, int, int]:
    """Median per-song difference, wins, and the number of songs both arms scored.

    Songs are matched BY NAME. Matching by list position instead is a silent corruption once arms have different
    coverage -- and these caches fill incrementally, so that is the normal state, not an edge case.
    """
    shared = sorted(set(cand) & set(base))
    if not shared:
        return float("nan"), 0, 0
    d = [cand[n] - base[n] for n in shared]
    return float(np.median(d)), int(sum(x > 0 for x in d)), len(shared)


def main(arms: list[str], baseline: str, limit: int | None, out: Path, min_delta: float) -> None:
    from cleansplit.metrics.signal import snr_db

    # arm -> stem -> {song: snr}. Keyed by song so arms with different coverage still compare correctly.
    rows: dict[str, dict[str, dict[str, float]]] = {}
    for name, _mix, truth in M.songs(limit):
        for arm in arms:
            f = M.CACHE / arm / f"{name}.npz"
            if not f.is_file():
                continue
            s = np.load(f)
            for g in M.GROUPS:
                t = truth[g].astype(np.float64)
                if float(np.sum(t**2)) <= 1e-9:
                    continue
                v = float(snr_db(t, s[g].astype(np.float64)))
                if np.isfinite(v):
                    rows.setdefault(arm, {}).setdefault(g, {})[name] = v
        print(f"{name}: done", flush=True)

    if baseline not in rows:
        raise SystemExit(f"baseline arm {baseline!r} has no scored songs in data/musdb_cache/")
    med = {a: {g: float(np.median(list(v.values()))) for g, v in per.items()} for a, per in rows.items()}
    print()
    print(f"MUSDB18-HQ full band at {M.SR} Hz, filter-free SNR. PAIRED median change vs {baseline}")
    print(f"  a gain counts only if it clears {min_delta:+.2f} dB paired and wins more than half the songs")
    print(f"  {'arm':14s} {'n':>3s} {'stem':7s} {'SNR':>7s} {'dSNR':>7s} {'won':>7s}")
    verdict = {}
    for arm, per in rows.items():
        deltas, wins, ns = {}, {}, {}
        for g in M.GROUPS:
            if g not in per or g not in rows[baseline]:
                continue
            d_med, won, n = paired(per[g], rows[baseline][g])
            if not n:
                continue
            deltas[g], wins[g], ns[g] = d_med, won, n
            print(f"  {arm:14s} {n:3d} {g:7s} {med[arm][g]:7.2f} {d_med:+7.2f} {won:3d}/{n:<3d}")
        if arm == baseline:
            continue
        gated = [g for g in GATED if g in deltas]
        ok = bool(gated) and all(deltas[g] >= min_delta and wins[g] > ns[g] / 2 for g in gated)
        verdict[arm] = {"fullband_better_on_vocals_drums_bass": ok, "min_delta_db": min_delta,
                        "paired_delta_db": deltas,
                        "won": wins, "n": ns, "median_db": med[arm],
                        "songs_short_of_baseline": {g: len(rows[baseline][g]) - ns[g] for g in ns}}
    print()
    for arm, v in verdict.items():
        short = {g: n for g, n in v["songs_short_of_baseline"].items() if n}
        note = f"  (compared on {min(v['n'].values())} shared songs, {max(short.values())} fewer than {baseline})" if short else ""
        print(f"  {arm:14s} full-band gate: {'PASS' if v['fullband_better_on_vocals_drums_bass'] else 'FAIL'}{note}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"baseline": baseline, "median": med, "verdict": verdict, "rows": rows}, indent=1))
    print(f"written {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("arms", nargs="+")
    p.add_argument("--baseline", default="sw_tta")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--min-delta", type=float, default=0.02,
                   help="smallest paired median gain worth adopting, dB (see the module docstring)")
    a = p.parse_args()
    main(a.arms, a.baseline, a.limit, a.out, a.min_delta)
