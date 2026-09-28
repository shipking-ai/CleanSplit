"""Dev probe: why does the `other` stem of After 2 turn into 16,000 French-horn notes, and are drums recoverable?

60 s excerpt (1:00-2:00), MuScriptor medium, per stem. Compares the `ensemble` split (drums missed) with the
`ensemble_demucs` split (drums present), and `other` unconstrained vs constrained away from drums.
Prints notes per second per stem, the instrument mix, and seconds taken.
"""
import collections
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cleansplit.audio.io import load_audio
from cleansplit.transcription import muscriptor_backend as mus

A, B = 60 * 44100, 120 * 44100
items = []
for variant in ("ensemble", "ensemble_demucs"):
    d = ROOT / "outputs" / variant / "After_2" / "stems"
    for stem in ("drums", "bass", "other"):
        x = load_audio(d / f"{stem}.wav").audio[:, A:B]
        items.append((x, 44100, mus.STEM_INSTRUMENTS.get(stem), f"{variant}/{stem}"))
# `other` with every group except drums allowed: does the horn flood come from drum hits it cannot label as drums?
x = load_audio(ROOT / "outputs" / "ensemble" / "After_2" / "stems" / "other.wav").audio[:, A:B]
no_drums = [g for g in ("acoustic_piano", "electric_piano", "chromatic_percussion", "organ", "acoustic_guitar",
                        "clean_electric_guitar", "distorted_electric_guitar", "electric_bass", "string_ensemble",
                        "synth_strings", "voice", "brass_section", "synth_lead", "synth_pad")]
items.append((x, 44100, no_drums, "ensemble/other_constrained"))

notes, info = mus.transcribe_many(items, progress=lambda i, n, c, t, l: print(f"  {l} {c}/{t}", flush=True)
                                  if c == t else None)
for (_, _, _, label), ns, sec in zip(items, notes, info["seconds"], strict=True):
    c = collections.Counter(n.instrument for n in ns)
    print(f"{label:34s} {len(ns):5d} notes = {len(ns) / 60:5.1f}/s in {sec:6.1f}s  {c.most_common(4)}")
