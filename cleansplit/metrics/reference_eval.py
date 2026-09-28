"""Separated stems vs ground-truth reference stems (e.g. a multitrack dataset). Per-stem SDR (plain SNR form),
SI-SDR, and energy ratios. Note: SW may have been trained on public multitrack sets; results on those can be
optimistic (possible train/test contamination)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..audio.conform import conform_channels, resample
from ..audio.io import load_audio
from ..separation.precomputed import match_stem_files
from . import signal as M


def evaluate_against_reference(estimate_dir, reference_dir) -> dict:
    est = match_stem_files(Path(estimate_dir))
    ref = match_stem_files(Path(reference_dir))
    out = {"estimate_dir": str(estimate_dir), "reference_dir": str(reference_dir), "stems": {}, "missing_reference": sorted(set(est) - set(ref))}
    for name in sorted(set(est) & set(ref)):
        e, r = load_audio(est[name]), load_audio(ref[name])
        ea, ra = e.audio, r.audio
        if r.sample_rate != e.sample_rate:
            ra = resample(ra, r.sample_rate, e.sample_rate)
        ra, _ = conform_channels(ra, ea.shape[0])
        n = min(ea.shape[-1], ra.shape[-1])
        ea, ra = ea[..., :n].astype(np.float64), ra[..., :n].astype(np.float64)
        silent_ref = float(np.mean(ra**2)) < 1e-10
        out["stems"][name] = {
            "reference_silent": silent_ref,
            "sdr_db": None if silent_ref else M.snr_db(ra, ea),
            "si_sdr_db": None if silent_ref else M.si_sdr_db(ra, ea),
            "estimate_energy_db": M.energy_db(ea),
            "reference_energy_db": M.energy_db(ra),
        }
    return out
