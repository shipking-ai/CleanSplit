"""Runs INSIDE .venv-transcribe (MuScriptor's own environment), never inside CleanSplit's.

    python _muscriptor_worker.py job.json result.json

MuScriptor pins packages that conflict with CleanSplit's (fastapi, and numpy<2 on some platforms), and its weights
are CC BY-NC 4.0 behind a HuggingFace gate, so it lives in a separate interpreter and is spoken to through files.
This script imports nothing from CleanSplit.

job.json:
  {"model": "medium"|"large"|"small", "dtype": "float32"|"float16",
   "items": [{"path": "...wav", "instruments": ["voice"] | null}, ...],
   "grid_from": "...full mix wav" | null}      # optional: detect tempo/bars on this file (beat_this, via MuScriptor)
result.json:
  {"model": ..., "items": [{"path": ..., "seconds": float,
                             "notes": [[onset_s, offset_s, pitch, instrument, is_drum], ...]}, ...]}
"""
import json
import sys
import time

import soundfile as sf
import torch


def main(job_path: str, out_path: str) -> None:
    from muscriptor import NoteEndEvent, NoteStartEvent, TranscriptionModel
    from muscriptor.events import ProgressEvent

    job = json.load(open(job_path, encoding="utf-8"))
    t0 = time.time()
    model = TranscriptionModel.load_model(job.get("model", "medium"), device="cuda" if torch.cuda.is_available() else "cpu",
                                          dtype=job.get("dtype", "float32"))
    load_s = time.time() - t0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    results = []
    n_items = len(job["items"])
    for idx, item in enumerate(job["items"], 1):
        audio, sr = sf.read(item["path"], dtype="float32", always_2d=True)
        wav = torch.from_numpy(audio.T.copy())
        t = time.time()
        open_notes, notes = {}, []  # open_notes: start-event index -> NoteStartEvent
        for ev in model.transcribe((wav, sr), instruments=item.get("instruments")):
            if isinstance(ev, ProgressEvent):
                if ev.total and (ev.completed % 5 == 0 or ev.completed == ev.total):
                    # One parseable line per ~25 s of audio, so the caller can show progress instead of silence.
                    print(f"[progress] {idx}/{n_items} {ev.completed}/{ev.total}", file=sys.stderr, flush=True)
                continue
            if isinstance(ev, NoteStartEvent):
                open_notes[ev.index] = ev
            elif isinstance(ev, NoteEndEvent):
                st = open_notes.pop(ev.start_event_index, ev.start_event)
                notes.append([st.start_time, ev.end_time, st.pitch, st.instrument, st.instrument == "drums"])
        dur = audio.shape[0] / sr
        for st in open_notes.values():  # notes still sounding at the end of the file
            notes.append([st.start_time, dur, st.pitch, st.instrument, st.instrument == "drums"])
        notes.sort()
        results.append({"path": item["path"], "seconds": time.time() - t, "notes": notes})
        print(f"[worker] {item['path']}: {len(notes)} notes in {time.time() - t:.1f}s", file=sys.stderr, flush=True)
    grid = None
    if job.get("grid_from"):
        # Tempo and bar lines from the full mix, then MuScriptor's own measurement of how late its onsets sit
        # against those beats (up to ~25 ms), measured over every non-drum onset of every item.
        from muscriptor.utils.beats import BeatDetectionError, detect_grid

        mix, msr = sf.read(job["grid_from"], dtype="float32", always_2d=True)
        try:
            g = detect_grid(torch.from_numpy(mix.T.copy()), msr)
            onsets = sorted(n[0] for r in results for n in r["notes"])
            g = g.with_onset_delay(onsets) if onsets else g
            grid = {"bpm": float(g.bpm), "beats_per_bar": g.beats_per_bar, "first_downbeat": float(g.first_downbeat),
                    "onset_delay": float(g.onset_delay or 0.0)}
        except BeatDetectionError as e:
            grid = {"error": str(e)}
            # No constant tempo (swing, rubato, a tracker flipping between half and double time). A DAW still wants
            # *a* tempo, and 120 BPM is certainly wrong: fall back to the median beat interval, which ignores the
            # occasional irregular beat, and say that it is a fallback. Note times stay absolute either way.
            try:
                import numpy as np
                from beat_this.inference import Audio2Beats

                beats, _ = Audio2Beats(checkpoint_path="final0", device="cpu", dbn=False)(mix.mean(axis=1), msr)
                ibi = np.diff(np.asarray(beats, dtype=float))
                ibi = ibi[(ibi > 0.25) & (ibi < 1.5)]  # 40-240 BPM
                if ibi.size >= 8:
                    grid.update(bpm=float(60.0 / np.median(ibi)), fallback="median beat interval (tempo not constant)",
                                onset_delay=0.0, beats_per_bar=None)
            except Exception as e2:  # the fallback must never cost the transcription
                grid["fallback_error"] = f"{type(e2).__name__}: {e2}"
    peak = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None
    json.dump({"model": job.get("model", "medium"), "dtype": job.get("dtype", "float32"), "load_seconds": load_s,
               "peak_vram_mb": peak, "items": results, "grid": grid}, open(out_path, "w", encoding="utf-8"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
