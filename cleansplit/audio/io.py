"""Audio decoding / encoding.

All audio inside CleanSplit is ``np.ndarray`` shaped ``(channels, samples)``, float32 unless
explicitly promoted to float64 for arithmetic (see ``reconstruction``).

Decoding order: libsndfile (wav/flac/ogg/aiff and mp3 on libsndfile>=1.1) then ffmpeg.
Note: MP3/AAC decoders differ in how they handle encoder delay/padding, so stems produced by
another tool from a lossy file can be offset by some samples. ``reconstruction.align``
measures and reports that offset rather than assuming zero.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


class AudioDecodeError(RuntimeError):
    pass


@dataclass
class DecodedAudio:
    audio: np.ndarray  # (C, N) float32
    sample_rate: int
    source_path: str
    decoder: str
    source_subtype: str | None = None


def _load_soundfile(path: Path) -> DecodedAudio:
    info = sf.info(str(path))
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)  # (N, C)
    return DecodedAudio(
        audio=np.ascontiguousarray(data.T),
        sample_rate=int(sr),
        source_path=str(path),
        decoder="soundfile",
        source_subtype=info.subtype,
    )


def _ffprobe(path: Path) -> tuple[int, int]:
    exe = shutil.which("ffprobe")
    if exe is None:
        raise AudioDecodeError("ffprobe not found on PATH")
    out = subprocess.run(
        [exe, "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=sample_rate,channels", "-of", "json", str(path)],
        capture_output=True,
        check=False,
    )
    if out.returncode != 0:
        raise AudioDecodeError(f"ffprobe failed: {out.stderr.decode(errors='replace')[:500]}")
    streams = json.loads(out.stdout).get("streams", [])
    if not streams:
        raise AudioDecodeError(f"no audio stream in {path}")
    return int(streams[0]["sample_rate"]), int(streams[0]["channels"])


def _load_ffmpeg(path: Path) -> DecodedAudio:
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise AudioDecodeError("ffmpeg not found on PATH")
    sr, ch = _ffprobe(path)
    out = subprocess.run(
        [exe, "-v", "error", "-nostdin", "-i", str(path), "-map", "0:a:0", "-f", "f32le", "-acodec", "pcm_f32le", "-"],
        capture_output=True,
        check=False,
    )
    if out.returncode != 0:
        raise AudioDecodeError(f"ffmpeg failed: {out.stderr.decode(errors='replace')[:500]}")
    flat = np.frombuffer(out.stdout, dtype="<f4")
    if flat.size % ch:
        raise AudioDecodeError("ffmpeg output not divisible by channel count")
    audio = flat.reshape(-1, ch).T.copy()
    return DecodedAudio(audio=audio, sample_rate=sr, source_path=str(path), decoder="ffmpeg")


def load_audio(path: str | Path) -> DecodedAudio:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        return _load_soundfile(path)
    except Exception as sf_err:  # libsndfile cannot decode this container/codec
        try:
            return _load_ffmpeg(path)
        except AudioDecodeError as ff_err:
            raise AudioDecodeError(f"could not decode {path}: soundfile: {sf_err}; ffmpeg: {ff_err}") from ff_err


def save_audio(path: str | Path, audio: np.ndarray, sample_rate: int, subtype: str = "FLOAT") -> None:
    """Write (C, N) audio. Default is 32-bit float WAV: no clipping, no dither, no rescaling."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if audio.ndim != 2:
        raise ValueError(f"expected (C, N) audio, got shape {audio.shape}")
    if not np.all(np.isfinite(audio)):
        raise ValueError(f"refusing to write non-finite samples to {path}")
    if subtype != "FLOAT":
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 1.0:
            raise ValueError(f"{path}: peak {peak:.4f} > 1.0 would clip in integer PCM; write FLOAT instead")
    sf.write(str(path), audio.T.astype(np.float32, copy=False), sample_rate, subtype=subtype)
