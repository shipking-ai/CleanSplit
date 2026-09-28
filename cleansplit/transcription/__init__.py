"""Audio -> MIDI transcription, stem-aware.

CleanSplit already separates a song into stems, so each stem can be transcribed by the model that suits it, with a
hard instrument constraint (a drum stem can only produce drums, a vocal stem only voice). Whether that actually
beats transcribing the full mix is measured, not assumed: tools/eval/transcription_eval.py, docs/04 section 13.

Backends:
  muscriptor  MuScriptor (Kyutai + Mirelo, 2026), 36 instrument groups incl. drums and voice. Code MIT, weights
              CC BY-NC 4.0 behind a HuggingFace gate. Runs in its own interpreter (.venv-transcribe), see
              muscriptor_backend.py. No velocities.
  transkun    Transkun v2 (Yan & Duan), piano specialist with velocities. MIT. In-process.
"""

from .notes import TranscribedNote, write_midi  # noqa: F401
