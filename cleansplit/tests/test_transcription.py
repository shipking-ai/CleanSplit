"""Transcription plumbing: note format, MIDI writing, stem routing, the song pipeline (backends mocked).

The models themselves are measured in tools/eval/transcription_eval.py against exact MIDI, not here.
"""

import json

import numpy as np
import pytest

from cleansplit.transcription import muscriptor_backend as mus
from cleansplit.transcription.notes import DEFAULT_VELOCITY, TranscribedNote, write_midi


def test_write_midi_round_trips_times_pitches_tracks_and_drums(tmp_path):
    pretty_midi = pytest.importorskip("pretty_midi")
    notes = [
        TranscribedNote(0.50, 1.00, 60, "voice", False, None, "vocals:muscriptor"),
        TranscribedNote(1.25, 1.50, 64, "voice", False, None, "vocals:muscriptor"),
        TranscribedNote(0.00, 0.10, 36, "drums", True, None, "drums:muscriptor"),
        TranscribedNote(2.00, 2.50, 48, "acoustic_piano", False, 37, "piano:transkun"),
        TranscribedNote(3.00, 3.00, 50, "acoustic_piano", False, 200, "piano:transkun"),  # zero length, velocity > 127
    ]
    pm = pretty_midi.PrettyMIDI(str(write_midi(notes, tmp_path / "x.mid", tempo_bpm=93.0)))
    assert pm.get_tempo_changes()[1][0] == pytest.approx(93.0, abs=0.01)
    by_name = {i.name: i for i in pm.instruments}
    assert set(by_name) == {"vocals - voice", "drums - drums", "piano - acoustic piano"}
    assert by_name["drums - drums"].is_drum and not by_name["vocals - voice"].is_drum
    voc = sorted(by_name["vocals - voice"].notes, key=lambda n: n.start)
    assert [n.pitch for n in voc] == [60, 64]
    assert voc[0].start == pytest.approx(0.50, abs=2e-3) and voc[0].end == pytest.approx(1.00, abs=2e-3)
    assert all(n.velocity == DEFAULT_VELOCITY for n in voc)  # MuScriptor has no velocity: the fixed default
    piano = sorted(by_name["piano - acoustic piano"].notes, key=lambda n: n.start)
    assert piano[0].velocity == 37 and piano[1].velocity == 127  # clamped
    assert piano[1].end > piano[1].start  # zero-length notes get a minimum length instead of vanishing


def test_stem_constraints_are_valid_muscriptor_group_names():
    """A typo here would make MuScriptor raise at run time, after a long separation."""
    valid = {"acoustic_piano", "electric_piano", "acoustic_guitar", "clean_electric_guitar",
             "distorted_electric_guitar", "acoustic_bass", "electric_bass", "voice", "synth_lead", "drums"}
    for stem, names in mus.STEM_INSTRUMENTS.items():
        assert names is None or set(names) <= valid, stem
    assert mus.STEM_INSTRUMENTS["other"] is None  # 'other' is whatever the separator could not name


def _fake_split(tmp_path, silent=("guitar",)):
    from cleansplit.audio.io import save_audio

    sr, n = 44100, 44100 * 2
    rng = np.random.default_rng(0)
    d = tmp_path / "ensemble" / "Song"
    save_audio(d / "original.wav", (rng.standard_normal((2, n)) * 0.1).astype(np.float32), sr)
    for s in ("vocals", "drums", "bass", "guitar", "piano", "other"):
        x = np.zeros((2, n), np.float32) if s in silent else (rng.standard_normal((2, n)) * 0.05).astype(np.float32)
        save_audio(d / "stems" / f"{s}.wav", x, sr)
    return d


def test_pipeline_skips_silent_stems_constrains_each_stem_and_subtracts_the_onset_lag(tmp_path, monkeypatch):
    pytest.importorskip("pretty_midi")
    from cleansplit.transcription import pipeline

    seen = {}

    def fake_transcribe_many(items, model="medium", grid_from=None, grid_sr=None, **kw):
        seen["items"] = [(src, inst) for _, _, inst, src in items]
        seen["grid"] = grid_from is not None
        notes = [[TranscribedNote(1.0, 1.5, 60, "drums" if src == "drums" else (inst or ["organ"])[0],
                                  src == "drums", None, f"{src}:muscriptor")] for _, _, inst, src in items]
        return notes, {"model": model, "dtype": "float32", "load_seconds": 0, "peak_vram_mb": 0, "seconds": [],
                       "grid": {"bpm": 97.0, "beats_per_bar": 4, "first_downbeat": 0.1, "onset_delay": 0.02}}

    monkeypatch.setattr(pipeline.mus, "transcribe_many", fake_transcribe_many)
    rep = pipeline.transcribe_split(_fake_split(tmp_path), mode="stems", piano="muscriptor", log=lambda m: None)
    assert "guitar" in rep["skipped_stems"]
    assert dict(seen["items"]) == {"vocals": ["voice"], "drums": ["drums"], "bass": mus.STEM_INSTRUMENTS["bass"],
                                   "piano": mus.STEM_INSTRUMENTS["piano"], "other": None}
    assert seen["grid"] and rep["tempo_bpm"] == 97.0 and rep["onset_delay_subtracted_s"] == 0.02
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(rep["files"]["all"])
    starts = [n.start for i in pm.instruments for n in i.notes]
    assert starts and all(abs(t - 0.98) < 1e-3 for t in starts)  # 1.0 s minus the 20 ms lag; 960 ppq keeps it < 1 ms
    assert set(rep["files"]) == {"all", "vocals", "drums", "bass", "piano", "other"}
    assert json.loads((tmp_path / "ensemble" / "Song" / "midi" / "report.json").read_text())["mode"] == "stems"


def test_default_routes_the_piano_stem_to_transkun_not_muscriptor(tmp_path, monkeypatch):
    """Measured default (docs/04 section 13, Q2): piano stems go to Transkun, which also brings velocities."""
    pytest.importorskip("pretty_midi")
    from cleansplit.transcription import pipeline, transkun_backend

    sent = []
    monkeypatch.setattr(pipeline.mus, "transcribe_many", lambda items, **kw: (
        sent.extend(src for *_, src in items) or [[] for _ in items],
        {"model": "medium", "dtype": "float32", "load_seconds": 0, "peak_vram_mb": 0, "seconds": [], "grid": None}))

    class FakeTranskun:
        def transcribe(self, audio, sr, source="piano"):
            return [TranscribedNote(0.5, 1.0, 60, "acoustic_piano", False, 71, "piano:transkun")]

    monkeypatch.setattr(transkun_backend, "TranskunPiano", FakeTranskun)
    rep = pipeline.transcribe_split(_fake_split(tmp_path), mode="stems", log=lambda m: None)
    assert "piano" not in sent and rep["piano_backend"] == "transkun"
    assert rep["transkun"]["notes"] == 1 and rep["notes_by_stem"]["piano"] == 1
    import pretty_midi

    (piano,) = pretty_midi.PrettyMIDI(rep["files"]["piano"]).instruments
    assert piano.notes[0].velocity == 71  # Transkun's velocity survives, not the fixed default


def test_pipeline_mix_mode_sends_one_unconstrained_item(tmp_path, monkeypatch):
    pytest.importorskip("pretty_midi")
    from cleansplit.transcription import pipeline

    captured = []
    monkeypatch.setattr(pipeline.mus, "transcribe_many", lambda items, **kw: (
        captured.extend(items) or [[]], {"model": "medium", "dtype": "float32", "load_seconds": 0,
                                         "peak_vram_mb": 0, "seconds": [], "grid": None}))
    rep = pipeline.transcribe_split(_fake_split(tmp_path), mode="mix", log=lambda m: None)
    assert len(captured) == 1 and captured[0][2] is None and captured[0][3] == "mix"
    assert rep["tempo_bpm"] == 120.0  # no grid -> the MIDI default, and the report says so via grid == {}


def test_note_matching_uses_onset_and_pitch_only():
    """The eval's matcher: 50 ms onset tolerance, offsets ignored, pitch exact."""
    import sys
    from pathlib import Path

    pytest.importorskip("mir_eval")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
    import transcription_eval as T

    ref = [(1.00, 1.5, 60), (2.00, 2.5, 62), (3.00, 3.5, 64)]
    est = [(1.04, 9.9, 60), (2.00, 2.5, 63), (3.20, 3.5, 64), (5.0, 5.1, 70)]  # hit, wrong pitch, late, spurious
    assert T._match(ref, est) == (1, 3, 4)
    assert T._f1(1, 3, 4) == pytest.approx(2 / 7)
    assert T.drum_class(36) == "kick" and T.drum_class(42) == "hihat" and T.drum_class(81) == "other"
    assert T.PROGRAM_GROUP[33] == "electric_bass" and T.PROGRAM_GROUP[0] == "acoustic_piano"


def test_implausibly_dense_stem_is_kept_alone_but_left_out_of_the_combined_file(tmp_path, monkeypatch):
    pytest.importorskip("pretty_midi")
    import pretty_midi

    from cleansplit.transcription import pipeline

    def fake(items, **kw):
        out = []
        for _, _, inst, src in items:
            k = 200 if src == "other" else 2  # 200 notes in 2 s = 100 notes/s for 'other'
            out.append([TranscribedNote(i * 0.01, i * 0.01 + 0.05, 60, "french_horn" if src == "other" else "voice",
                                        False, None, f"{src}:muscriptor") for i in range(k)])
        return out, {"model": "medium", "dtype": "float32", "load_seconds": 0, "peak_vram_mb": 0, "seconds": [],
                     "grid": None}

    monkeypatch.setattr(pipeline.mus, "transcribe_many", fake)
    rep = pipeline.transcribe_split(_fake_split(tmp_path), piano="muscriptor", log=lambda m: None)
    assert set(rep["suspect_stems_notes_per_s"]) == {"other"}
    names = {i.name for i in pretty_midi.PrettyMIDI(rep["files"]["all"]).instruments}
    assert not any(n.startswith("other") for n in names) and any(n.startswith("vocals") for n in names)
    assert len(pretty_midi.PrettyMIDI(rep["files"]["other"]).instruments[0].notes) == 200  # still written on its own
    assert any("implausibly dense" in c for c in rep["caveats"])
