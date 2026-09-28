"""Separators that do not run a model.

* ``StemFolderSeparator``: analyze stems produced elsewhere (e.g. exported from the UVR GUI).
  File names are matched case-insensitively to canonical stem names, including UVR's
  ``<song>_(Vocals).wav`` naming.
* ``OracleSeparator``: returns given arrays. Used for synthetic ground-truth tests.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from ..audio.conform import conform_channels, resample
from ..audio.io import load_audio
from .base import CANONICAL_STEMS, SeparationResult, Separator

_ALIASES = {
    "vocals": ("vocals", "vocal", "voice"),
    "drums": ("drums", "drum"),
    "bass": ("bass",),
    "guitar": ("guitar", "guitars"),
    "piano": ("piano", "keys", "keyboard", "keyboards"),
    "other": ("other", "others"),
}
_AUDIO_EXT = {".wav", ".flac", ".aif", ".aiff", ".ogg", ".mp3"}


def match_stem_files(folder: Path, stems=CANONICAL_STEMS) -> dict[str, Path]:
    files = [p for p in folder.iterdir() if p.suffix.lower() in _AUDIO_EXT]
    found: dict[str, Path] = {}
    for stem in stems:
        hits = []
        for p in files:
            tokens = set(re.split(r"[^a-z]+", p.stem.lower()))
            if tokens & set(_ALIASES.get(stem, (stem,))):
                hits.append(p)
        if len(hits) > 1:
            raise ValueError(f"ambiguous files for stem '{stem}': {[h.name for h in hits]}")
        if hits:
            found[stem] = hits[0]
    return found


class StemFolderSeparator(Separator):
    name = "stem_folder"

    def __init__(self, folder: str | Path, sample_rate: int = 44100, channels: int = 2, require_all: bool = True):
        self.folder = Path(folder)
        self.sample_rate = sample_rate
        self.channels = channels
        self.files = match_stem_files(self.folder)
        missing = [s for s in CANONICAL_STEMS if s not in self.files]
        if missing and require_all:
            raise FileNotFoundError(f"missing stems in {self.folder}: {missing}")
        self.stems = tuple(self.files)

    def cache_key(self) -> dict:
        return {**self.describe(), "files": {k: [str(v), v.stat().st_size, v.stat().st_mtime_ns] for k, v in self.files.items()}}

    def separate(self, mixture: np.ndarray, sample_rate: int) -> SeparationResult:
        stems, notes = {}, {}
        n = mixture.shape[-1]
        for name, path in self.files.items():
            d = load_audio(path)
            a, conv = conform_channels(d.audio, self.channels)
            note = []
            if conv:
                note.append(conv)
            if d.sample_rate != sample_rate:
                a = resample(a, d.sample_rate, sample_rate)
                note.append(f"resampled {d.sample_rate}->{sample_rate}")
            if a.shape[-1] != n:
                note.append(f"length {a.shape[-1]} vs mixture {n} (handled by alignment)")
            stems[name] = a.astype(np.float32)
            notes[name] = {"path": str(path), "decoder": d.decoder, "subtype": d.source_subtype, "notes": note}
        return SeparationResult(stems, sample_rate, self.name, {"files": notes})


class OracleSeparator(Separator):
    name = "oracle"

    def __init__(self, stems: dict[str, np.ndarray], sample_rate: int = 44100, chunk_starts=None, chunk_size=None):
        self._stems = stems
        self.sample_rate = sample_rate
        self.channels = next(iter(stems.values())).shape[0]
        self.stems = tuple(stems)
        self._chunk_starts = list(chunk_starts or [])
        self._chunk_size = chunk_size

    def separate(self, mixture: np.ndarray, sample_rate: int) -> SeparationResult:
        return SeparationResult(
            {k: v.astype(np.float32) for k, v in self._stems.items()},
            sample_rate,
            self.name,
            {},
            self._chunk_starts,
            self._chunk_size,
        )
