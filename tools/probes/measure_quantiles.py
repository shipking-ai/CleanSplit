"""Dev: distribution of each per-stem detector's physical measure on an analyzed song (how close to threshold?)."""
import sys
from pathlib import Path

import numpy as np

from cleansplit.analysis.context import AnalysisContext
from cleansplit.artifacts import detectors
from cleansplit.audio.io import load_audio
from cleansplit.audio.stft import STFTGrid
from cleansplit.config.settings import AnalysisConfig
from cleansplit.reconstruction.core import model_matched_reference

song = Path(sys.argv[1])
seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
cfg = AnalysisConfig()
n = int(seconds * 44100)
O = load_audio(song / "original.wav").audio[:, :n]
stems = {p.stem: load_audio(p).audio[:, :n] for p in (song / "stems").glob("*.wav")}
ref = model_matched_reference(O, STFTGrid(), True)
ctx = AnalysisContext(ref, stems, 44100, cfg)
thr = {"cancellation": cfg.cancellation.center, "modulation": cfg.modulation.center, "transient": cfg.transient.center}
for name in ("cancellation", "modulation", "transient", "leakage"):
    for em in detectors.create([name])[0].detect(ctx):
        gated = em.evidence > 0
        vals = em.measure[gated] if gated.any() else np.array([np.nan])
        if name == "transient":
            vals = em.measure[em.measure != 0] if np.any(em.measure != 0) else np.array([np.nan])
        q = np.nanpercentile(vals, [50, 90, 99, 99.9]) if np.isfinite(vals).any() else [np.nan] * 4
        print(f"{name:13s} {em.stem:7s} gated_cells={int(gated.sum()):8d} p50={q[0]:7.2f} p90={q[1]:7.2f} p99={q[2]:7.2f} p99.9={q[3]:7.2f} "
              f"max_ev={em.evidence.max():.2f} center={thr.get(name, 'r-std<1.5')}")
