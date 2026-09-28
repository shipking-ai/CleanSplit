"""Dev: why does the modulation detector miss randomized warble boxes?"""
import numpy as np

from cleansplit.analysis.context import TINY, AnalysisContext, to_db
from cleansplit.artifacts import corruptions as C
from cleansplit.artifacts.detectors.modulation import ModulationDetector, _modulation_energy
from cleansplit.audio.synthetic import make_song
from cleansplit.audio.tf_edit import TFBox
from cleansplit.config.settings import AnalysisConfig

mix, stems = make_song(8.0, 0)
cfg = AnalysisConfig()
for box in [TFBox(3.0, 4.2, 800, 3000), TFBox(3.96, 5.13, 1024, 2464), TFBox(2.34, 3.27, 1278, 3967)]:
    st, gt = C.warble(stems, "vocals", box, conserve=True, sink="other", seed=5)
    ctx = AnalysisContext(mix.astype(np.float64), st, 44100, cfg)
    em = [e for e in ModulationDetector().detect(ctx) if e.stem == "vocals"][0]
    rows = (em.row_bounds_hz[:, 0] < box.freq_high_hz) & (em.row_bounds_hz[:, 1] > box.freq_low_hz)
    cols = (em.col_bounds_s[:, 0] < box.end_s) & (em.col_bounds_s[:, 1] > box.start_s)
    m = em.measure[np.ix_(rows, cols)]
    ev = em.evidence[np.ix_(rows, cols)]
    # recompute gates
    fr = 44100 / cfg.hop
    win, hop = int(round(0.5 * fr)), int(round(0.1 * fr))
    Es, Ms = _modulation_energy(ctx.B_stems["vocals"], win, hop, fr, 3, 20)
    Em, Mm = _modulation_energy(ctx.B_mix, win, hop, fr, 3, 20)
    depth = (np.sqrt(Es) / np.maximum(Ms, TINY))[np.ix_(rows, cols)]
    share = (to_db(Ms + TINY) - to_db(Mm + TINY))[np.ix_(rows, cols)]
    print(box, "bands", rows.sum(), "windows", cols.sum())
    print("  measure p50/p90/max", np.round(np.percentile(m, [50, 90, 100]), 1), " ev max", round(float(ev.max()), 2), " cells ev>=.6", int((ev >= 0.6).sum()))
    print("  depth p50/max", np.round(np.percentile(depth, [50, 100]), 2), " share p50/min", np.round([np.median(share), share.min()], 1))
