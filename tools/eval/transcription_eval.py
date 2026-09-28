"""Does transcribing separated stems beat transcribing the full mix? Scored against exact MIDI (BabySlakh).

    python tools/eval/transcription_eval.py extract
    python tools/eval/transcription_eval.py run      # separates, transcribes every route (cached per route), scores
    python tools/eval/transcription_eval.py score    # re-score from the cache

Data: BabySlakh (Zenodo 4603870, CC BY 4.0): the first 20 Slakh2100 tracks, 16 kHz, each with the rendered stems
and the exact MIDI they were rendered from. One 30 s excerpt per track, centred on the midpoint (content-blind).

Routes:
  A  full mix   -> MuScriptor medium, unconstrained. What a plain audio-to-MIDI tool does.
  C  CleanSplit -> `ensemble` separation (the shipped recipe) -> every stem to MuScriptor medium with that stem's
                   hard instrument constraint (muscriptor_backend.STEM_INSTRUMENTS).
  C2 as C, but the piano stem goes to Transkun (piano specialist, velocities) instead of MuScriptor.
  D  oracle     -> the TRUE stems summed into CleanSplit's six groups -> MuScriptor constrained. Upper bound: the gap
                   D - C is what separation errors cost; D - A is what stem-wise transcription could gain at best.

Metrics (mir_eval.transcription, onset tolerance 50 ms, offsets ignored — the MT3/MuScriptor convention):
  multi-F1   a note counts only if pitch, onset AND instrument group match (MuScriptor's MT3_FULL_PLUS groups).
             Matching is done within each group, true positives summed over groups.
  onset-F1   pitch + onset, instrument ignored (pitched notes only).
  drum-F1    drum hits, GM keys collapsed to kick / snare / hi-hat / tom / cymbal / other.
All pooled over the 20 excerpts (sum TP, ref, est), plus the per-track multi-F1 for the win count.

PRE-REGISTERED (written before any route was transcribed):
  Q1  C beats A on pooled multi-F1 AND on more than 10/20 tracks  -> ship per-stem transcription as the default.
  Q2  C2 beats C on piano-group onset-F1 pooled                  -> route piano stems to Transkun.
  Reported, not decided on: onset-F1, drum-F1, D.

ADDED 2026-09-21 after the first results, before route C2n was run (prompted by a real song whose drum stem sat at
-48 dBFS and came out nearly empty; MuScriptor does not normalise its input):
  C2n as C2, but every stem sent to MuScriptor is first scaled to -20 dBFS RMS (silent stems still skipped).
  Q3  C2n beats C2 on pooled multi-F1 AND on more than 10/20 tracks -> normalise stems by default.

Known biases, stated up front:
  * BabySlakh is 16 kHz. The separators were trained on 44.1 kHz and here see audio with nothing above 8 kHz.
    This works AGAINST route C. MuScriptor itself runs at 16 kHz, so route A loses nothing.
  * The audio is rendered from MIDI with sample libraries, not played. MuScriptor was pre-trained on synthetic
    data and may have seen Slakh; that flatters every MuScriptor route alike.
  * Slakh has no singing, so the vocal route (voice constraint) is not tested here at all.
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data" / "babyslakh"
TAR = DATA / "babyslakh_16k.tar.gz"
CACHE = ROOT / "data" / "transcription_cache"
OUT = ROOT / "outputs" / "_benchmarks" / "transcription_babyslakh.json"
EXCERPT_S = 30.0
ONSET_TOL = 0.05

# muscriptor 0.3.0, tokenizer/mt3.py, MT3_FULL_PLUS: group index -> GM programs. Names from MT3_FULL_PLUS_GROUP_NAMES.
_GROUPS = {0: [0, 1, 3, 6, 7], 1: [2, 4, 5], 2: list(range(8, 16)), 3: list(range(16, 24)), 4: [24, 25],
           5: [26, 27, 28], 6: [29, 30, 31], 7: [32, 35], 8: [33, 34, 36, 37, 38, 39], 9: [40], 10: [41],
           11: [42], 12: [43], 13: [46], 14: [47], 15: [48, 49, 44, 45], 16: [50, 51], 17: [52, 53, 54],
           18: [55], 19: [56, 59], 20: [57], 21: [58], 22: [60], 23: [61, 62, 63], 24: [64, 65], 25: [66],
           26: [67], 27: [68], 28: [69], 29: [70], 30: [71], 31: list(range(72, 80)), 32: list(range(80, 88)),
           33: list(range(88, 96)), 34: [100], 35: [101]}
_NAMES = ["acoustic_piano", "electric_piano", "chromatic_percussion", "organ", "acoustic_guitar",
          "clean_electric_guitar", "distorted_electric_guitar", "acoustic_bass", "electric_bass", "violin", "viola",
          "cello", "contrabass", "orchestral_harp", "timpani", "string_ensemble", "synth_strings", "voice",
          "orchestra_hit", "trumpet", "trombone", "tuba", "french_horn", "brass_section", "soprano_and_alto_sax",
          "tenor_sax", "baritone_sax", "oboe", "english_horn", "bassoon", "clarinet", "flutes", "synth_lead",
          "synth_pad", "group_34", "group_35"]
PROGRAM_GROUP = {p: _NAMES[g] for g, ps in _GROUPS.items() for p in ps}
PIANO_GROUPS = {"acoustic_piano", "electric_piano"}


def cleansplit_stem(program: int, is_drum: bool) -> str:
    """Which of CleanSplit's six stems a Slakh source belongs to (for the oracle route D)."""
    if is_drum:
        return "drums"
    if program < 8:
        return "piano"
    if 24 <= program < 32:
        return "guitar"
    if 32 <= program < 40:
        return "bass"
    return "other"  # Slakh has no singing; everything else is 'other'


def drum_class(key: int) -> str:
    if key in (35, 36):
        return "kick"
    if key in (37, 38, 39, 40):
        return "snare"
    if key in (42, 44, 46):
        return "hihat"
    if key in (41, 43, 45, 47, 48, 50):
        return "tom"
    if key in (49, 51, 52, 53, 55, 57, 59):
        return "cymbal"
    return "other"


def extract() -> None:
    with tarfile.open(TAR) as t:
        t.extractall(DATA, filter="data")
    print(sorted(p.name for p in DATA.rglob("Track*") if p.is_dir())[:3], "...")


def tracks():
    """Yield (name, sr, mix excerpt (C,N), true stem excerpts by CleanSplit stem, truth notes)."""
    import pretty_midi
    import soundfile as sf
    import yaml

    for d in sorted(p for p in DATA.rglob("Track*") if p.is_dir()):
        meta = yaml.safe_load((d / "metadata.yaml").read_text())
        mix, sr = sf.read(str(d / "mix.wav"), dtype="float32", always_2d=True)
        mix = mix.T
        n = mix.shape[-1]
        dur = n / sr
        a_s = max(0.0, dur / 2 - EXCERPT_S / 2)
        a, b = int(a_s * sr), int(min(dur, a_s + EXCERPT_S) * sr)
        stems, truth = {}, []
        for sid, sm in meta["stems"].items():
            wav, midi = d / "stems" / f"{sid}.wav", d / "MIDI" / f"{sid}.mid"
            # BabySlakh's `audio_rendered` flag is False for all 226 stems although 209 have audio, so existence
            # of both files decides. 4 stems have audio but no MIDI: they are in the mix without truth, so notes
            # transcribed from them count as false positives for routes A, C and C2 alike; route D never sees them.
            if not (wav.is_file() and midi.is_file()):
                continue
            is_drum, prog = bool(sm.get("is_drum")), int(sm.get("program_num", 0))
            x, _ = sf.read(str(wav), dtype="float32", always_2d=True)
            key = cleansplit_stem(prog, is_drum)
            seg = x.T[:, a:b]
            stems[key] = stems.get(key, 0) + seg
            for inst in pretty_midi.PrettyMIDI(str(midi)).instruments:
                for nt in inst.notes:
                    if a_s <= nt.start < a_s + EXCERPT_S:
                        truth.append((nt.start - a_s, min(nt.end, a_s + EXCERPT_S) - a_s, nt.pitch,
                                      "drums" if is_drum else PROGRAM_GROUP.get(prog, "other"), is_drum))
        yield d.name, sr, mix[:, a:b], stems, truth


def _separate_all(items, cache: Path):
    """Route C input: the shipped `ensemble` separator on each 16 kHz mix, upsampled to 44.1 kHz."""
    import soxr

    from cleansplit.separation import registry

    out = {}
    todo = [(name, sr, mix) for name, sr, mix in items if not (cache / f"{name}.npz").is_file()]
    if todo:
        sep = registry.create("ensemble")
        for name, sr, mix in todo:
            x44 = soxr.resample(mix.T, sr, 44100).T.astype(np.float32)
            if x44.shape[0] == 1:
                x44 = np.repeat(x44, 2, axis=0)
            res = sep.separate(x44, 44100)
            cache.mkdir(parents=True, exist_ok=True)
            np.savez(cache / f"{name}.npz", **{k: v for k, v in res.stems.items()})
            print(f"separated {name}", flush=True)
    for name, _, _ in items:
        out[name] = dict(np.load(cache / f"{name}.npz"))
    return out


def _muscriptor_route(route: str, jobs, model: str = "medium"):
    """jobs: list of (track, stem, audio, sr, instruments). Cached per route as JSON."""
    from cleansplit.transcription.muscriptor_backend import transcribe_many

    path = CACHE / f"route_{route}_{model}.json"
    if path.is_file():
        return json.loads(path.read_text())
    notes, info = transcribe_many([(a, sr, inst, f"{t}|{s}") for t, s, a, sr, inst in jobs], model=model)
    res = {"info": info, "notes": {}}
    for (t, s, *_), ns in zip(jobs, notes, strict=True):
        res["notes"].setdefault(t, []).extend(
            [[n.onset_s, n.offset_s, n.pitch, n.instrument, n.is_drum, s] for n in ns])
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res))
    print(f"route {route}: {sum(len(v) for v in res['notes'].values())} notes, "
          f"peak VRAM {info['peak_vram_mb']:.0f} MB", flush=True)
    return res


def run() -> None:
    from cleansplit.transcription.muscriptor_backend import STEM_INSTRUMENTS

    data = list(tracks())
    print(f"{len(data)} tracks x {EXCERPT_S:.0f} s", flush=True)
    items = [(name, sr, mix) for name, sr, mix, _, _ in data]
    _muscriptor_route("A", [(name, "mix", mix, sr, None) for name, sr, mix in items])
    sep = _separate_all(items, CACHE / "ensemble_stems")
    jobs_c = [(name, s, x, 44100, STEM_INSTRUMENTS.get(s)) for name, *_ in data for s, x in sep[name].items()]
    _muscriptor_route("C", jobs_c)
    jobs_d = [(name, s, x, sr, STEM_INSTRUMENTS.get(s)) for name, sr, _, stems, _ in data for s, x in stems.items()]
    _muscriptor_route("D", jobs_d)

    def norm(x, target_dbfs=-20.0):
        rms = float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))
        return x if rms < 10 ** (-60 / 20) else (x * (10 ** (target_dbfs / 20) / rms)).astype(np.float32)

    _muscriptor_route("C2n", [(n, s, norm(x), sr, inst) for n, s, x, sr, inst in jobs_c if s != "piano"])

    tk_path = CACHE / "route_C2_transkun_piano.json"
    if not tk_path.is_file():
        from cleansplit.transcription.transkun_backend import TranskunPiano

        tk = TranskunPiano("cuda")
        res = {n: [[x.onset_s, x.offset_s, x.pitch, x.instrument, False, "piano", x.velocity]
                   for x in tk.transcribe(sep[n]["piano"], 44100)] for n in sep}
        tk_path.write_text(json.dumps(res))
    score(data)


def _match(ref, est):
    """(tp, n_ref, n_est) for pitched notes: pitch + onset within ONSET_TOL, offsets ignored."""
    import mir_eval

    if not ref or not est:
        return 0, len(ref), len(est)
    ri = np.array([[r[0], max(r[1], r[0] + 1e-3)] for r in ref])
    ei = np.array([[e[0], max(e[1], e[0] + 1e-3)] for e in est])
    rp = mir_eval.util.midi_to_hz(np.array([r[2] for r in ref], dtype=float))
    ep = mir_eval.util.midi_to_hz(np.array([e[2] for e in est], dtype=float))
    matching = mir_eval.transcription.match_notes(ri, rp, ei, ep, onset_tolerance=ONSET_TOL, offset_ratio=None)
    return len(matching), len(ref), len(est)


def _grouped(ref, est, key):
    tp = nr = ne = 0
    for g in {key(x) for x in ref} | {key(x) for x in est}:
        t, r, e = _match([x for x in ref if key(x) == g], [x for x in est if key(x) == g])
        tp, nr, ne = tp + t, nr + r, ne + e
    return tp, nr, ne


def _f1(tp, nr, ne):
    return 2 * tp / (nr + ne) if nr + ne else float("nan")


def score(data=None) -> dict:
    data = data or list(tracks())
    routes = {r: json.loads((CACHE / f"route_{r}_medium.json").read_text())["notes"] for r in ("A", "C", "D")}
    tk = json.loads((CACHE / "route_C2_transkun_piano.json").read_text())
    c2 = {t: [n for n in routes["C"].get(t, []) if n[5] != "piano"] + tk.get(t, []) for t in routes["C"]}
    routes["C2"] = c2
    c2n_path = CACHE / "route_C2n_medium.json"
    if c2n_path.is_file():
        c2n = json.loads(c2n_path.read_text())["notes"]
        routes["C2n"] = {t: c2n.get(t, []) + tk.get(t, []) for t in routes["C"]}
    totals = {r: {m: [0, 0, 0] for m in ("multi", "onset", "drum", "piano")} for r in routes}
    per_track = {r: {} for r in routes}
    for name, _, _, _, truth in data:
        ref_p = [x for x in truth if not x[4]]
        ref_d = [(x[0], x[1], drum_class(x[2])) for x in truth if x[4]]
        ref_piano = [x for x in ref_p if x[3] in PIANO_GROUPS]
        for r, notes in routes.items():
            est = notes.get(name, [])
            est_p = [x for x in est if not x[4]]
            est_d = [(x[0], x[1], drum_class(x[2])) for x in est if x[4]]
            m = {"multi": _grouped(ref_p, est_p, key=lambda x: x[3]),
                 "onset": _match(ref_p, est_p),
                 "drum": _grouped([(a, b, 60, c) for a, b, c in ref_d], [(a, b, 60, c) for a, b, c in est_d],
                                  key=lambda x: x[3]),
                 "piano": _match(ref_piano, [x for x in est_p if x[3] in PIANO_GROUPS])}
            for k, v in m.items():
                totals[r][k] = [a + b for a, b in zip(totals[r][k], v, strict=True)]
            per_track[r][name] = _f1(*m["multi"])
    summary = {r: {k: _f1(*v) for k, v in t.items()} for r, t in totals.items()}
    wins = sum(per_track["C"][n] > per_track["A"][n] for n in per_track["A"])
    q1 = bool(summary["C"]["multi"] > summary["A"]["multi"] and wins > len(per_track["A"]) / 2)
    q2 = bool(summary["C2"]["piano"] > summary["C"]["piano"])
    q3 = None
    if "C2n" in summary:
        wins3 = sum(per_track["C2n"][n] > per_track["C2"][n] for n in per_track["C2"])
        q3 = bool(summary["C2n"]["multi"] > summary["C2"]["multi"] and wins3 > len(per_track["C2"]) / 2)
    print(f"\nBabySlakh, {len(data)} tracks x {EXCERPT_S:.0f} s. Pooled F1 (onset tolerance 50 ms, offsets ignored)")
    print(f"  {'route':32s} {'multi-F1':>9s} {'onset-F1':>9s} {'drum-F1':>9s} {'piano-F1':>9s}")
    labels = {"A": "A  full mix", "C": "C  CleanSplit stems", "C2": "C2 stems, piano->Transkun",
              "C2n": "C2n C2 + stems at -20 dBFS", "D": "D  oracle stems"}
    for r in [x for x in ("A", "C", "C2", "C2n", "D") if x in summary]:
        s = summary[r]
        print(f"  {labels[r]:32s} {s['multi']:9.3f} {s['onset']:9.3f} {s['drum']:9.3f} {s['piano']:9.3f}")
    print(f"\n  Q1 stems beat full mix: multi-F1 {summary['C']['multi'] - summary['A']['multi']:+.3f}, "
          f"wins {wins}/{len(per_track['A'])} -> {'YES' if q1 else 'no'}")
    print(f"  Q2 Transkun for piano stems: piano-F1 {summary['C2']['piano'] - summary['C']['piano']:+.3f} -> "
          f"{'YES' if q2 else 'no'}")
    if q3 is not None:
        print(f"  Q3 normalise stems to -20 dBFS: multi-F1 {summary['C2n']['multi'] - summary['C2']['multi']:+.3f}, "
              f"wins {wins3}/{len(per_track['C2'])} -> {'YES' if q3 else 'no'}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"protocol": __doc__, "summary": summary, "totals": totals, "per_track_multi_f1": per_track,
                               "q1_stems_beat_mix": q1, "q1_wins": wins, "q2_transkun_piano": q2,
                               "q3_normalise_stems": q3}, indent=1))
    print(f"written {OUT}")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["extract", "run", "score"])
    a = p.parse_args()
    {"extract": extract, "run": run, "score": score}[a.cmd]()
