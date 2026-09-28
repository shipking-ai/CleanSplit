"""UI layer: what the app shows must come from the files on disk, and nothing outside them."""

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from cleansplit.artifacts.region import ArtifactMap, ArtifactRegion
from cleansplit.audio.io import save_audio
from cleansplit.ui.server import create_app
from cleansplit.ui.service import Service


@pytest.fixture
def out_root(tmp_path, song):
    """A split on disk in the same layout the CLI writes, one variant, one song."""
    mixture, stems = song
    d = tmp_path / "ensemble" / "Test_Song"
    save_audio(d / "original.wav", mixture, 44100)
    for name, s in stems.items():
        save_audio(d / "stems" / f"{name}.wav", s, 44100)
    (d / "stems" / "manifest.json").write_text(json.dumps(
        {"separator": "ensemble", "source": str(tmp_path / "Test Song.wav")}), encoding="utf-8")
    return tmp_path


def test_songs_lists_what_is_on_disk(out_root, song):
    (songs,) = Service(out_root).songs()
    assert songs["id"] == "ensemble/Test_Song"
    assert songs["title"] == "Test Song"
    assert songs["stems"][:3] == ["vocals", "drums", "bass"]  # display order, not glob order
    assert songs["duration"] == pytest.approx(song[0].shape[-1] / 44100, abs=1e-6)
    assert songs["analysed"] is False


def test_peaks_are_bounded_by_the_audio_and_cached(out_root, song):
    svc = Service(out_root)
    pk = svc.peaks("ensemble", "Test_Song", "vocals", buckets=200)
    x = song[1]["vocals"].mean(axis=0)
    assert len(pk["max"]) == len(pk["min"]) == len(pk["rms"]) <= 200
    assert max(pk["max"]) <= float(x.max()) + 1e-4
    assert min(pk["min"]) >= float(x.min()) - 1e-4
    assert (out_root / "ensemble" / "Test_Song" / "peaks" / "vocals.200.json").is_file()
    assert svc.peaks("ensemble", "Test_Song", "vocals", buckets=200) == pk  # served from cache


def test_stem_levels_are_relative_to_the_song(out_root):
    levels = Service(out_root).stem_levels("ensemble", "Test_Song")
    assert set(levels) == {"vocals", "drums", "bass", "guitar", "piano", "other"}
    for lvl in levels.values():
        assert lvl["rel_song_db"] == pytest.approx(lvl["rms_dbfs"] - Service(out_root).peaks("ensemble", "Test_Song", "original")["rms_dbfs"])
        assert lvl["rel_song_db"] < 0.1  # a single stem is never louder than the whole song


def test_song_dir_refuses_paths_outside_the_output_root(out_root):
    svc = Service(out_root)
    with pytest.raises(FileNotFoundError):
        svc.song_dir("..", "..")
    with pytest.raises(FileNotFoundError):
        svc.audio_path("ensemble", "Test_Song", "../../original")


def test_a_stem_name_cannot_reach_a_file_that_actually_exists_outside_the_song(out_root, tmp_path):
    """The traversal test above passed for the wrong reason, which is worth a test of its own.

    It asserted FileNotFoundError for `../../original` -- and got one, because no such file existed. `audio_path`
    checked `is_file()` but never checked WHERE the path landed, so a stem name that pointed at a real file outside the
    song folder would have been served. This plants such a file first, so the assertion can only pass if the name is
    rejected. CodeQL's py/path-injection is what prompted the second look.
    """
    secret = out_root.parent / "secret.wav"
    save_audio(secret, np.zeros((2, 1000), dtype=np.float32), 44100)
    assert secret.is_file()                     # the target is real, so is_file() alone would let it through
    # FOUR levels, counted rather than guessed: from <out_root>/<variant>/<slug>/stems/ it takes stems -> slug ->
    # variant -> out_root to get out. A shallower `../../` only reaches the variant folder and would make this test
    # pass against the vulnerable code too -- which is the exact mistake the test above made.
    svc = Service(out_root)
    for bad in ("../../../../secret", "..\..\..\..\secret", "..", ".", ""):
        with pytest.raises(FileNotFoundError):
            svc.audio_path("ensemble", "Test_Song", bad)
        with pytest.raises(FileNotFoundError):
            svc.peaks("ensemble", "Test_Song", bad)      # this one would otherwise WRITE a cache file outside
    assert svc.audio_path("ensemble", "Test_Song", "vocals").is_file()   # the legitimate name still works


def test_analysis_is_absent_until_it_has_been_measured(out_root):
    svc = Service(out_root)
    assert svc.analysis("ensemble", "Test_Song") == {"available": False, "regions": [], "metrics": {}}
    amap = ArtifactMap(regions=[
        ArtifactRegion(stem="vocals", start_s=1.0, end_s=1.5, freq_low_hz=2000.0, freq_high_hz=4000.0,
                       confidence=0.8, artifact_types=("residual_energy",))])
    amap.save(out_root / "ensemble" / "Test_Song" / "analysis" / "artifact_map.json")
    a = svc.analysis("ensemble", "Test_Song")
    assert a["available"] and len(a["regions"]) == 1 and a["regions"][0]["stem"] == "vocals"


def test_api_serves_state_song_and_audio(out_root):
    with TestClient(create_app(str(out_root))) as client:
        state = client.get("/api/state").json()
        assert [s["id"] for s in state["songs"]] == ["ensemble/Test_Song"]
        assert {s["id"] for s in state["separators"]} >= {"ensemble", "bs_roformer_sw"}

        song = client.get("/api/song/ensemble/Test_Song").json()
        assert set(song["peaks"]) == {"original", "vocals", "drums", "bass", "guitar", "piano", "other"}
        assert song["analysis"]["available"] is False

        wav = client.get("/api/audio/ensemble/Test_Song/vocals.wav")
        assert wav.status_code == 200 and wav.content[:4] == b"RIFF"
        assert client.get("/api/song/ensemble/Nope").status_code == 404
        assert client.get("/api/audio/ensemble/Test_Song/nosuch.wav").status_code == 404


def test_api_rejects_unknown_separators_and_missing_files(out_root, tmp_path):
    with TestClient(create_app(str(out_root))) as client:
        assert client.post("/api/jobs", json={"input": str(tmp_path / "missing.wav"), "separator": "ensemble"}).status_code == 400
        real = tmp_path / "x.wav"
        save_audio(real, np.zeros((2, 4410), dtype=np.float32), 44100)
        assert client.post("/api/jobs", json={"input": str(real), "separator": "magic_ai"}).status_code == 400


def test_midi_endpoint_queues_a_job_for_an_existing_split_only(out_root, monkeypatch):
    ran = []
    monkeypatch.setattr("cleansplit.transcription.pipeline.transcribe_split",
                        lambda song_dir, log=print, **kw: ran.append(str(song_dir)) or {})
    with TestClient(create_app(str(out_root))) as client:
        assert client.post("/api/midi", json={"variant": "ensemble", "slug": "Nope"}).status_code == 404
        job = client.post("/api/midi", json={"variant": "ensemble", "slug": "Test_Song"}).json()
        assert job["kind"] == "midi" and job["slug"] == "Test_Song"
        svc = client.app.state.service
        import time

        for _ in range(100):
            if svc.jobs[job["id"]].state in ("done", "error"):
                break
            time.sleep(0.05)
        assert svc.jobs[job["id"]].state == "done", svc.jobs[job["id"]].error
        assert ran and ran[0].endswith("Test_Song")
        assert client.get("/api/state").json()["songs"][0]["midi"] is False  # the mock wrote no .mid
