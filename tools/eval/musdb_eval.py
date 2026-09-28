"""Real multitrack truth: score the separators and ensembles on the MUSDB18-HQ test set (50 songs).

    python tools/eval/musdb_eval.py extract            # unzip only test/ from data/musdb18hq/musdb18hq.zip
    python tools/eval/musdb_eval.py run   [--limit N]  # separate (cached per model and song, resumable), then score
    python tools/eval/musdb_eval.py score              # score from the cache only

Why: every ensemble result so far (docs/04 sections 7 and 10) comes from ONE song whose "truth" was made by an
undisclosed commercial separator. MUSDB18-HQ has the real studio stems of 50 held-out songs.

Protocol (fixed before any result is seen):
  * One 30 s excerpt per song, centred on the track midpoint. No selection by content: choosing loud-vocal excerpts
    would flatter the vocal models.
  * Mixture = exact sum of the four truth stems, one gain for all to peak at -1 dBFS (as in docs/04 section 3).
  * Metric: SDR = 10 log10(|s|^2 / |s - s_hat|^2) per song (the same snr_db as every other section). Reported as
    median over songs (the SiSEC convention), mean, and pooled; plus how many songs the candidate wins.
  * Equal-weight averages only. Nothing is tuned on this set.
  * Groups: vocals, drums, bass, other (SW's guitar + piano + other). In the full ensembles 'other' is the exact
    remainder, as in the shipped separators.

Scope actually run (2026-09-21): the FIRST 20 test songs in alphabetical order (`--limit 20`), at the user's request
to save GPU time. Alphabetical order is fixed and content-blind, so it is not a selection; it is a smaller sample.

Pre-registered questions (a "yes" needs a better median AND wins on more than half the songs scored):
  Q1  Does the shipped `ensemble` vocal stem beat single-pass SW?
  Q2  Does adding MDX23C (the section 10 change) still help across 50 songs?
  Q3  Does averaging drums and bass with HTDemucs_ft help (the `ensemble_demucs` recipe)?
  Q4  Is that average better than HTDemucs_ft ALONE for drums and bass? Added after the smoke test on two BUH
      excerpts (not MUSDB data), where Demucs alone beat the average; written in before any MUSDB song was scored.

Known bias, stated up front: HTDemucs_ft was trained with the MUSDB18-HQ test set held out. The training data of
the community models (SW, ep317, MDX23C) is undisclosed and may include these songs, which would inflate their
scores. That bias works AGAINST Demucs, so a "yes" on Q3 is robust and a "no" is inconclusive.
"""

from __future__ import annotations

import argparse
import json
import os
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "musdb18hq"
ZIP = DATA / "musdb18hq.zip"
# CLEANSPLIT_MUSDB_TEST points at any folder of <song>/{vocals,drums,bass,other}.wav (used for the smoke test).
TEST = Path(os.environ.get("CLEANSPLIT_MUSDB_TEST", DATA / "test"))
CACHE = Path(os.environ.get("CLEANSPLIT_MUSDB_CACHE", ROOT / "data" / "musdb_cache"))
OUT = Path(os.environ.get("CLEANSPLIT_MUSDB_OUT", ROOT / "outputs" / "_benchmarks" / "musdb18hq_test.json"))
GROUPS = ("vocals", "drums", "bass", "other")
EXCERPT_S = 30.0
SR = 44100


def extract() -> None:
    with zipfile.ZipFile(ZIP) as z:
        members = [m for m in z.namelist() if m.startswith("test/")]
        print(f"extracting {len(members)} files from test/ ...", flush=True)
        z.extractall(DATA, members=members)
    print(f"{len(list(TEST.iterdir()))} test songs in {TEST}")


def songs(limit: int | None = None):
    import soundfile as sf

    from cleansplit.metrics.restoration_experiment import stem_folder_songs

    dirs = sorted(p for p in TEST.iterdir() if p.is_dir())[:limit]
    for d in dirs:
        dur = sf.info(str(d / "vocals.wav")).duration
        start = max(0.0, dur / 2 - EXCERPT_S / 2)
        ((_, mix, truth),) = stem_folder_songs({g: d / f"{g}.wav" for g in GROUPS}, [(start, EXCERPT_S)])
        yield d.name, mix, truth


def _separators():
    from cleansplit.separation import registry

    return {
        "sw": lambda: registry.create("bs_roformer_sw"),
        "sw_tta": lambda: registry.create("bs_roformer_sw", tta=True),
        "ep317": lambda: registry.create("bs_roformer_ep317"),
        "mdx23c": lambda: registry.create("mdx23c_instvoc_hq"),
        "demucs": lambda: registry.create("htdemucs_ft"),
    }


def _grouped(key: str, stems: dict) -> dict:
    if key in ("sw", "sw_tta"):
        return {"vocals": stems["vocals"], "drums": stems["drums"], "bass": stems["bass"],
                "other": stems["guitar"] + stems["piano"] + stems["other"]}
    if key in ("ep317", "mdx23c"):
        return {"vocals": stems["vocals"]}
    return {g: stems[g] for g in GROUPS}


def run(limit: int | None) -> None:
    import torch

    items = list(songs(limit))
    print(f"{len(items)} songs, {EXCERPT_S:.0f} s each", flush=True)
    for key, make in _separators().items():
        todo = [(n, m) for n, m, _ in items if not (CACHE / key / f"{n}.npz").is_file()]
        if not todo:
            print(f"{key}: cached", flush=True)
            continue
        sep = make()
        for i, (name, mix) in enumerate(todo, 1):
            g = _grouped(key, {k: v.astype(np.float32) for k, v in sep.separate(mix, SR).stems.items()})
            (CACHE / key).mkdir(parents=True, exist_ok=True)
            np.savez(CACHE / key / f"{name}.npz", **g)
            print(f"{key}: {i}/{len(todo)} {name}", flush=True)
        sep._model = None
        torch.cuda.empty_cache()
    score(limit)


def score(limit: int | None) -> dict:
    from cleansplit.metrics.signal import snr_db

    items = list(songs(limit))
    load = lambda k, n: {g: v.astype(np.float64) for g, v in np.load(CACHE / k / f"{n}.npz").items()}
    avg = lambda *xs: sum(xs) / len(xs)
    per_song: dict[str, dict[str, float]] = {}
    err: dict[str, float] = {}
    ref: dict[str, float] = {g: 0.0 for g in GROUPS}
    for name, mix, truth in items:
        e = {k: load(k, name) for k in _separators()}
        mix64 = mix.astype(np.float64)
        v2 = avg(e["sw_tta"]["vocals"], e["ep317"]["vocals"])
        v3 = avg(e["sw_tta"]["vocals"], e["ep317"]["vocals"], e["mdx23c"]["vocals"])
        d_dm = avg(e["sw_tta"]["drums"], e["demucs"]["drums"])
        b_dm = avg(e["sw_tta"]["bass"], e["demucs"]["bass"])
        cands = {
            "vocals": {"sw": e["sw"]["vocals"], "sw_tta": e["sw_tta"]["vocals"], "ep317": e["ep317"]["vocals"],
                       "mdx23c": e["mdx23c"]["vocals"], "demucs": e["demucs"]["vocals"],
                       "avg(sw_tta,ep317)": v2, "avg(sw_tta,ep317,mdx23c) [ensemble]": v3,
                       "avg(sw_tta,ep317,mdx23c,demucs)": avg(e["sw_tta"]["vocals"], e["ep317"]["vocals"],
                                                              e["mdx23c"]["vocals"], e["demucs"]["vocals"])},
            "drums": {"sw": e["sw"]["drums"], "sw_tta": e["sw_tta"]["drums"], "demucs": e["demucs"]["drums"],
                      "avg(sw_tta,demucs) [ensemble_demucs]": d_dm},
            "bass": {"sw": e["sw"]["bass"], "sw_tta": e["sw_tta"]["bass"], "demucs": e["demucs"]["bass"],
                     "avg(sw_tta,demucs) [ensemble_demucs]": b_dm},
            "other": {"sw": e["sw"]["other"], "sw_tta": e["sw_tta"]["other"], "demucs": e["demucs"]["other"],
                      "remainder [ensemble]": mix64 - v3 - e["sw_tta"]["drums"] - e["sw_tta"]["bass"],
                      "remainder [ensemble_demucs]": mix64 - v3 - d_dm - b_dm},
        }
        row = {}
        for g, cs in cands.items():
            t = truth[g].astype(np.float64)
            ref[g] += float(np.sum(t**2))
            for c, est in cs.items():
                row[f"{g}|{c}"] = float(snr_db(t, est))
                err[f"{g}|{c}"] = err.get(f"{g}|{c}", 0.0) + float(np.sum((t - est) ** 2))
        per_song[name] = row

    keys = list(next(iter(per_song.values())))
    summary = {}
    for k in keys:
        vals = np.array([per_song[n][k] for n in per_song])
        finite = vals[np.isfinite(vals)]
        g = k.split("|")[0]
        summary[k] = {"median": float(np.median(finite)), "mean": float(np.mean(finite)),
                      "pooled": float(10 * np.log10(ref[g] / err[k])), "n": int(finite.size)}

    def question(g, cand, base):
        a, b = f"{g}|{cand}", f"{g}|{base}"
        wins = int(sum(per_song[n][a] > per_song[n][b] for n in per_song))
        d_med = summary[a]["median"] - summary[b]["median"]
        return {"candidate": a, "baseline": b, "delta_median_db": d_med,
                "delta_pooled_db": summary[a]["pooled"] - summary[b]["pooled"],
                "wins": wins, "n": len(per_song), "yes": bool(d_med > 0 and wins > len(per_song) / 2)}

    answers = {
        "Q1 ensemble vocals vs SW": question("vocals", "avg(sw_tta,ep317,mdx23c) [ensemble]", "sw"),
        "Q2 adding MDX23C": question("vocals", "avg(sw_tta,ep317,mdx23c) [ensemble]", "avg(sw_tta,ep317)"),
        "Q3a demucs averaging, drums": question("drums", "avg(sw_tta,demucs) [ensemble_demucs]", "sw_tta"),
        "Q3b demucs averaging, bass": question("bass", "avg(sw_tta,demucs) [ensemble_demucs]", "sw_tta"),
        "Q4a average vs demucs alone, drums": question("drums", "avg(sw_tta,demucs) [ensemble_demucs]", "demucs"),
        "Q4b average vs demucs alone, bass": question("bass", "avg(sw_tta,demucs) [ensemble_demucs]", "demucs"),
    }

    print(f"\nMUSDB18-HQ test, {len(per_song)} songs x {EXCERPT_S:.0f} s. SDR in dB (median / mean / pooled)")
    for k in keys:
        s = summary[k]
        print(f"  {k:48s} {s['median']:7.2f} {s['mean']:7.2f} {s['pooled']:7.2f}")
    print()
    for q, a in answers.items():
        print(f"  {q:38s} median {a['delta_median_db']:+.2f} dB, pooled {a['delta_pooled_db']:+.2f} dB, "
              f"wins {a['wins']}/{a['n']} -> {'YES' if a['yes'] else 'no'}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"protocol": __doc__, "excerpt_s": EXCERPT_S, "summary": summary,
                               "answers": answers, "per_song": per_song}, indent=1))
    print(f"\nwritten {OUT}")
    return answers


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["extract", "run", "score"])
    p.add_argument("--limit", type=int, default=None, help="first N songs only (smoke test)")
    a = p.parse_args()
    {"extract": lambda: extract(), "run": lambda: run(a.limit), "score": lambda: score(a.limit)}[a.cmd]()
