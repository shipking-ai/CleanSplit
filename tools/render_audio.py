"""Turn the measured MUSDB caches into WAVs you can actually listen to, including the error signal.

    python tools/render_audio.py                 # 3 songs: worst, median and best vocal SNR
    python tools/render_audio.py --all           # all 20
    python tools/render_audio.py --songs "Arise" # substring match

Why this exists: every number in docs/04 section 14 was measured on arrays in data/musdb_cache/ and never written to
audio, so for eleven days nothing listenable reflected the current defaults. These are the EXACT signals behind the
published numbers -- not a re-render, the same arrays -- so what you hear is what was scored.

Per song, 30 s excerpt centred on the track midpoint (the fixed protocol in tools/musdb_eval.py), 44.1 kHz float WAV:

  mixture.wav              the input
  truth/<stem>.wav         the real studio stem: the ceiling, what a perfect separator would output
  best/<stem>.wav          the CURRENT DEFAULT: ensemble (SW+TTA + ep317) at overlap 4
  fast/<stem>.wav          the opt-out: single-pass SW at overlap 2, no TTA
  error_best/<stem>.wav    best minus truth -- everything the separator got wrong, on its own
  error_fast/<stem>.wav    same for the fast path
  error_*_loud/<stem>.wav  the same error normalised to -3 dBFS, because the true-scale version is very quiet

The error signal is the point. A stem can sound fine in isolation and still carry the artifacts these numbers track;
soloing `error_best/vocals.wav` is the most direct way to hear what 0.45 dB of SNR actually buys.

`other` is the exact remainder (mixture minus the other three), exactly as the shipped separators compute it, so the
four stems sum back to the mixture sample-for-sample.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import musdb_eval as M

from cleansplit.audio.io import save_audio

OUT = ROOT / "outputs" / "listen"
STEMS = ("vocals", "drums", "bass", "other")


def peak_db(x: np.ndarray) -> float:
    p = float(np.max(np.abs(x))) if x.size else 0.0
    return 20.0 * np.log10(p) if p > 0 else -np.inf


def main(limit: int | None, which: str | None, take_all: bool, out: Path) -> None:
    from cleansplit.metrics.signal import snr_db

    have = []
    for name, mix, truth in M.songs(limit):
        f_best = M.CACHE / "sw_tta_ov4" / f"{name}.npz"
        f_ep = M.CACHE / "ep317_ov4" / f"{name}.npz"
        f_fast = M.CACHE / "sw" / f"{name}.npz"
        if not (f_best.is_file() and f_ep.is_file() and f_fast.is_file()):
            continue
        have.append((name, mix, truth, f_best, f_ep, f_fast))
    if not have:
        raise SystemExit("no song has all three caches (sw_tta_ov4, ep317_ov4, sw) filled")

    scored = []
    for name, mix, truth, f_best, f_ep, f_fast in have:
        sw = {k: v.astype(np.float64) for k, v in np.load(f_best).items()}
        ep = np.load(f_ep)["vocals"].astype(np.float64)
        v = 0.5 * (sw["vocals"] + ep)
        tv = truth["vocals"].astype(np.float64)
        scored.append((float(snr_db(tv, v)) if np.sum(tv**2) > 1e-9 else np.nan, name))
    scored = [s for s in scored if np.isfinite(s[0])]
    scored.sort()

    if which:
        chosen = {n for _, n in scored if which.lower() in n.lower()}
        if not chosen:
            raise SystemExit(f"no cached song matches {which!r}; have:\n  " + "\n  ".join(n for _, n in scored))
    elif take_all:
        chosen = {n for _, n in scored}
    else:
        picks = [scored[0], scored[len(scored) // 2], scored[-1]]
        chosen = {n for _, n in picks}
        print("Picked by vocal SNR of the current default (worst / median / best), so you can hear both ends:")
        for lbl, (s, n) in zip(("worst ", "median", "best  "), picks, strict=True):
            print(f"  {lbl}  {s:6.2f} dB  {n}")
        print()

    for name, mix, truth, f_best, f_ep, f_fast in have:
        if name not in chosen:
            continue
        d = out / name.replace("/", "_")
        sw = {k: v.astype(np.float64) for k, v in np.load(f_best).items()}
        ep = np.load(f_ep)["vocals"].astype(np.float64)
        fast = {k: v.astype(np.float64) for k, v in np.load(f_fast).items()}
        mixf = mix.astype(np.float64)

        best = {"vocals": 0.5 * (sw["vocals"] + ep), "drums": sw["drums"], "bass": sw["bass"]}
        best["other"] = mixf - best["vocals"] - best["drums"] - best["bass"]
        fastd = {"vocals": fast["vocals"], "drums": fast["drums"], "bass": fast["bass"]}
        fastd["other"] = mixf - fastd["vocals"] - fastd["drums"] - fastd["bass"]

        save_audio(d / "mixture.wav", mixf, M.SR)
        print(f"{name}")
        for arm, stems in (("best", best), ("fast", fastd)):
            for g in STEMS:
                save_audio(d / arm / f"{g}.wav", stems[g], M.SR)
        for g in STEMS:
            t = truth[g].astype(np.float64)
            save_audio(d / "truth" / f"{g}.wav", t, M.SR)
            if float(np.sum(t**2)) <= 1e-9:
                continue
            for arm, stems in (("best", best), ("fast", fastd)):
                err = stems[g] - t
                save_audio(d / f"error_{arm}" / f"{g}.wav", err, M.SR)
                pk = peak_db(err)
                if np.isfinite(pk):
                    gain = 10 ** ((-3.0 - pk) / 20.0)
                    save_audio(d / f"error_{arm}_loud" / f"{g}.wav", err * gain, M.SR)
            sb, sf_ = float(snr_db(t, best[g])), float(snr_db(t, fastd[g]))
            print(f"  {g:7s} best {sb:6.2f} dB   fast {sf_:6.2f} dB   best is {sb - sf_:+.2f} dB   "
                  f"error peak {peak_db(best[g] - t):+.0f} dBFS")
        print(f"  -> {d}")
    print(f"\nWritten under {out}")
    print("Listen to error_best_loud/vocals.wav against error_fast_loud/vocals.wav: that is the difference, isolated.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--songs", default=None, help="substring of a song name")
    p.add_argument("--all", dest="take_all", action="store_true")
    p.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args()
    main(a.limit, a.songs, a.take_all, a.out)
