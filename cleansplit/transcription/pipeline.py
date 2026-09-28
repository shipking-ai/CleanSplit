"""Split -> MIDI. Reads a CleanSplit song folder (original.wav + stems/*.wav), writes midi/.

    outputs/<separator>/<song>/midi/
        <song>.mid        every stem, one track per stem and instrument, at the detected tempo
        <stem>.mid        one file per stem, for dragging a single part into a DAW
        report.json       what ran, how long, peak VRAM, tempo, note counts, licences, caveats

mode "stems" transcribes each separated stem with a hard instrument constraint; mode "mix" transcribes the original
mix unconstrained. Defaults decided by tools/eval/transcription_eval.py against exact MIDI (docs/04 section 13): stems beat
the mix (pre-registered Q1: multi-F1 +0.048, 13/20 tracks) and Transkun beats MuScriptor on the piano stem (Q2:
piano F1 +0.186). Together: multi-F1 0.226 -> 0.332 pooled, better on 16/20 tracks.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..audio.io import load_audio
from . import muscriptor_backend as mus
from .notes import TranscribedNote, write_midi

SILENT_DBFS = -60.0  # a stem this quiet is skipped: nothing to transcribe, only ghost notes to invent
# A real part does not average more than this. In BabySlakh's exact MIDI the densest stem averaged 30.7 notes/s
# (strummed guitar, 6 notes per strum); a runaway transcription of separation leftovers ran at 70-117 notes/s on a
# real song. Above the line a stem keeps its own .mid but is left out of the combined file, and the report says so.
# On the benchmark this never fires for the shipped route (stems + Transkun) and lifts plain stems 0.274 -> 0.298.
SUSPECT_NOTES_PER_S = 40.0


def _rms_dbfs(x: np.ndarray) -> float:
    return float(10 * np.log10(np.mean(np.asarray(x, dtype=np.float64) ** 2) + 1e-20))


def transcribe_split(song_dir: str | Path, mode: str = "stems", model: str = "medium", piano: str = "transkun",
                     log=print) -> dict:
    song_dir = Path(song_dir)
    orig = load_audio(song_dir / "original.wav")
    sr = orig.sample_rate
    t0 = time.time()
    skipped = {}
    items = []
    if mode == "mix":
        items.append((orig.audio, sr, None, "mix"))
    elif mode == "stems":
        for p in sorted((song_dir / "stems").glob("*.wav")):
            x = load_audio(p).audio
            level = _rms_dbfs(x)
            if level < SILENT_DBFS:
                skipped[p.stem] = f"silent ({level:.1f} dBFS)"
                continue
            if p.stem == "piano" and piano == "transkun":
                continue  # handled below
            items.append((x, sr, mus.STEM_INSTRUMENTS.get(p.stem), p.stem))
    else:
        raise ValueError(f"mode must be 'stems' or 'mix', got {mode!r}")

    log(f"MuScriptor {model}: {len(items)} item(s) ({', '.join(i[3] for i in items)})")
    per_item, info = mus.transcribe_many(
        items, model=model, grid_from=orig.audio, grid_sr=sr,
        progress=lambda i, n, c, t, label: log(f"MIDI: {label} ({i}/{n}) {100 * c // max(t, 1)}%"))
    grid = info.get("grid") or {}
    delay = float(grid.get("onset_delay") or 0.0)
    notes: list[TranscribedNote] = []
    for ns in per_item:  # MuScriptor's measured lag against the beat grid (<= ~25 ms), subtracted as it recommends
        notes += [replace(n, onset_s=max(0.0, n.onset_s - delay), offset_s=max(0.0, n.offset_s - delay)) for n in ns]

    tk_info = None
    if mode == "stems" and piano == "transkun" and (song_dir / "stems" / "piano.wav").is_file() and "piano" not in skipped:
        from .transkun_backend import TranskunPiano

        t = time.time()
        pn = TranskunPiano().transcribe(load_audio(song_dir / "stems" / "piano.wav").audio, sr)
        notes += pn
        tk_info = {"backend": "transkun", "notes": len(pn), "seconds": round(time.time() - t, 1), "license": "MIT"}

    bpm = float(grid.get("bpm") or 120.0)
    out = song_dir / "midi"
    out.mkdir(parents=True, exist_ok=True)
    by_stem: dict[str, list[TranscribedNote]] = {}
    for n in notes:
        by_stem.setdefault(n.source.split(":")[0], []).append(n)
    duration = orig.audio.shape[-1] / sr
    suspect = {stem: round(len(ns) / duration, 1) for stem, ns in by_stem.items()
               if duration > 0 and len(ns) / duration > SUSPECT_NOTES_PER_S}
    if suspect:
        log(f"left out of the combined MIDI (implausibly dense, notes/s): {suspect}")
    full = write_midi([n for n in notes if n.source.split(":")[0] not in suspect], out / f"{song_dir.name}.mid",
                      tempo_bpm=bpm)
    files = {stem: str(write_midi(ns, out / f"{stem}.mid", tempo_bpm=bpm)) for stem, ns in by_stem.items()}
    report = {
        "song": song_dir.name, "mode": mode, "model": model, "piano_backend": piano,
        "tempo_bpm": bpm, "grid": grid, "onset_delay_subtracted_s": delay,
        "notes_total": len(notes), "notes_by_stem": {k: len(v) for k, v in by_stem.items()},
        "instruments_by_stem": {k: sorted({n.instrument for n in v}) for k, v in by_stem.items()},
        "skipped_stems": skipped, "suspect_stems_notes_per_s": suspect, "files": {"all": str(full), **files},
        "muscriptor": {k: v for k, v in info.items() if k != "grid"}, "transkun": tk_info,
        "seconds_total": round(time.time() - t0, 1),
        "caveats": [
            "MuScriptor does not predict velocity: its notes are written at velocity 90.",
            "Tempo is one constant BPM fitted by beat_this; note times are absolute, so the file plays in time even "
            "where the song drifts, but bar lines may not match a song that changes tempo."
            + (" This song has no constant tempo, so the BPM is a median-beat fallback." if grid.get("fallback") else ""),
            "MuScriptor weights are CC BY-NC 4.0: this MIDI is fine for your own music, not for a paid product built on it.",
            "Sub-bass (808s, near-sine bass) is a known weak spot: few or no bass notes usually means the model could "
            "not pitch it, not that the bass is absent.",
        ] + ([f"Left out of {song_dir.name}.mid as implausibly dense (> {SUSPECT_NOTES_PER_S:.0f} notes/s, likely "
              f"separation leftovers): {', '.join(suspect)}. Their own .mid files are still written."] if suspect else []),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"{len(notes)} notes -> {full} ({bpm:.1f} BPM)")
    return report
