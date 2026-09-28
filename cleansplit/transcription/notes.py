"""The one note format every transcription backend returns, and the MIDI writer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

# MuScriptor's instrument groups -> a representative General MIDI program for the written file.
GM_PROGRAM = {
    "acoustic_piano": 0, "electric_piano": 4, "chromatic_percussion": 11, "organ": 19, "acoustic_guitar": 25,
    "clean_electric_guitar": 27, "distorted_electric_guitar": 30, "acoustic_bass": 32, "electric_bass": 33,
    "violin": 40, "viola": 41, "cello": 42, "contrabass": 43, "orchestral_harp": 46, "timpani": 47,
    "string_ensemble": 48, "synth_strings": 50, "voice": 52, "orchestra_hit": 55, "trumpet": 56, "trombone": 57,
    "tuba": 58, "french_horn": 60, "brass_section": 61, "soprano_and_alto_sax": 65, "tenor_sax": 66,
    "baritone_sax": 67, "oboe": 68, "english_horn": 69, "bassoon": 70, "clarinet": 71, "flutes": 73,
    "synth_lead": 80, "synth_pad": 88,
}
DEFAULT_VELOCITY = 90  # MuScriptor does not predict velocity; one fixed value is written and flagged in the report


@dataclass(frozen=True)
class TranscribedNote:
    onset_s: float
    offset_s: float
    pitch: int  # MIDI pitch; for drums the General MIDI percussion key
    instrument: str  # MuScriptor group name, e.g. "voice", "drums", "acoustic_piano"
    is_drum: bool
    velocity: int | None = None  # None when the backend does not predict it
    source: str = ""  # which stem / backend produced it, e.g. "vocals:muscriptor"

    def to_dict(self) -> dict:
        return asdict(self)


def write_midi(notes: list[TranscribedNote], path: str | Path, tempo_bpm: float = 120.0) -> Path:
    """One MIDI track per (source, instrument). Times are absolute seconds, so the file plays in time whatever
    tempo a DAW assumes; ``tempo_bpm`` only sets the grid the DAW shows."""
    import pretty_midi

    # 960 ticks per beat: pretty_midi's default of 220 snaps notes to a ~3 ms grid at 100 BPM, which is a
    # measurable fraction of the 50 ms onset tolerance transcription is scored with. 960 is ~0.6 ms.
    pm = pretty_midi.PrettyMIDI(resolution=960, initial_tempo=float(tempo_bpm))
    tracks: dict[tuple[str, str], pretty_midi.Instrument] = {}
    for n in sorted(notes, key=lambda n: (n.source, n.instrument, n.onset_s)):
        key = (n.source, n.instrument)
        if key not in tracks:
            stem = n.source.split(":")[0] if n.source else ""
            name = f"{stem} - {n.instrument.replace('_', ' ')}" if stem else n.instrument.replace("_", " ")
            tracks[key] = pretty_midi.Instrument(program=0 if n.is_drum else GM_PROGRAM.get(n.instrument, 0),
                                                 is_drum=n.is_drum, name=name)
        end = max(n.offset_s, n.onset_s + 0.01)  # a zero-length note is dropped by most DAWs
        vel = int(n.velocity) if n.velocity is not None else DEFAULT_VELOCITY
        tracks[key].notes.append(pretty_midi.Note(velocity=max(1, min(127, vel)), pitch=int(n.pitch),
                                                  start=float(n.onset_s), end=float(end)))
    pm.instruments.extend(tracks.values())
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(path))
    return path
