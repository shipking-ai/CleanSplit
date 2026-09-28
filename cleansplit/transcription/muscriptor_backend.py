"""MuScriptor through its own interpreter.

MuScriptor pins packages that conflict with CleanSplit's environment, and its weights are CC BY-NC 4.0 behind a
HuggingFace gate the user has to accept themselves. So it runs in ``.venv-transcribe`` (created with
``uv venv .venv-transcribe --python 3.11`` and ``uv pip install muscriptor==0.3.0 --torch-backend=cu128``) and is
driven through JSON files by ``_muscriptor_worker.py``. One worker call loads the model once and transcribes every
item, because loading the medium model costs far more than transcribing a stem.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .notes import TranscribedNote

ROOT = Path(__file__).resolve().parents[2]
VENV_PY = ROOT / ".venv-transcribe" / "Scripts" / "python.exe"
WORKER = Path(__file__).with_name("_muscriptor_worker.py")

# Hard constraints per CleanSplit stem. 'other' stays unconstrained: it is everything the separator could not name.
STEM_INSTRUMENTS: dict[str, list[str] | None] = {
    "vocals": ["voice"],
    "drums": ["drums"],
    "bass": ["electric_bass", "acoustic_bass", "synth_lead"],
    "piano": ["acoustic_piano", "electric_piano"],
    "guitar": ["acoustic_guitar", "clean_electric_guitar", "distorted_electric_guitar"],
    "other": None,
}


def available() -> bool:
    return VENV_PY.is_file()


def transcribe_many(items: list[tuple[np.ndarray | str | Path, int | None, list[str] | None, str]],
                    model: str = "medium", dtype: str = "float32", timeout_s: float = 7200,
                    grid_from: np.ndarray | None = None, grid_sr: int | None = None,
                    progress=None) -> tuple[list[list[TranscribedNote]], dict]:
    """items: (audio (C, N) array or wav path, sample_rate or None for paths, instrument constraint or None, source label).
    Returns one note list per item and the worker's run info (load time, peak VRAM, seconds per item)."""
    if not available():
        raise FileNotFoundError(f"MuScriptor environment not found at {VENV_PY}; see this module's docstring")
    import soundfile as sf

    with tempfile.TemporaryDirectory(prefix="cleansplit_mus_") as td:
        td = Path(td)
        job_items = []
        for i, (audio, sr, instruments, _) in enumerate(items):
            if isinstance(audio, (str, Path)):
                path = str(audio)
            else:
                path = str(td / f"item_{i}.wav")
                sf.write(path, np.asarray(audio, dtype=np.float32).T, int(sr), subtype="FLOAT")
            job_items.append({"path": path, "instruments": instruments})
        job, out = td / "job.json", td / "result.json"
        grid_path = None
        if grid_from is not None:
            grid_path = str(td / "grid_mix.wav")
            sf.write(grid_path, np.asarray(grid_from, dtype=np.float32).T, int(grid_sr), subtype="FLOAT")
        job.write_text(json.dumps({"model": model, "dtype": dtype, "items": job_items, "grid_from": grid_path}),
                       encoding="utf-8")
        # Stream the worker's stderr: "[progress] item/items chunk/chunks" lines go to `progress` as they arrive,
        # so a long song shows movement instead of looking hung; the tail is kept for the error message.
        import collections
        import threading

        proc = subprocess.Popen([str(VENV_PY), str(WORKER), str(job), str(out)], stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
        tail = collections.deque(maxlen=15)

        def pump():
            for line in proc.stderr:
                line = line.rstrip()
                tail.append(line)
                if progress and line.startswith("[progress] "):
                    try:
                        items_part, chunks_part = line.split()[1:3]
                        i, n = map(int, items_part.split("/"))
                        c, t = map(int, chunks_part.split("/"))
                        progress(i, n, c, t, items[i - 1][3] if 0 < i <= len(items) else "?")
                    except ValueError:
                        pass

        reader = threading.Thread(target=pump, daemon=True)
        reader.start()
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            raise
        reader.join(timeout=5)
        if proc.returncode != 0 or not out.is_file():
            raise RuntimeError(f"MuScriptor worker failed (exit {proc.returncode}):\n" + "\n".join(tail))
        res = json.loads(out.read_text(encoding="utf-8"))
    notes = []
    for (_, _, _, source), r in zip(items, res["items"], strict=True):
        notes.append([TranscribedNote(float(a), float(b), int(p), str(inst), bool(drum), None, f"{source}:muscriptor")
                      for a, b, p, inst, drum in r["notes"]])
    info = {"backend": "muscriptor", "model": res["model"], "dtype": res["dtype"], "load_seconds": res["load_seconds"],
            "peak_vram_mb": res["peak_vram_mb"], "seconds": [r["seconds"] for r in res["items"]],
            "grid": res.get("grid"),
            "license": "weights CC BY-NC 4.0 (non-commercial); code MIT"}
    return notes, info
