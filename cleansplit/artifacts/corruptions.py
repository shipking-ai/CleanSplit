"""Controlled, ground-truth-labelled corruptions that mimic separator failure modes.

``conserve=True`` moves the change into a sink stem so that sum(stems) is unchanged: this is the hard case
where the mixture residual is blind and only per-stem detectors can succeed. ``conserve=False`` removes or adds
energy, which the residual must see.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfiltfilt

from ..audio.stft import STFTGrid
from ..audio.tf_edit import TFBox, apply_delta, tf_region_delta


@dataclass
class GroundTruth:
    kind: str
    stem: str
    start_s: float
    end_s: float
    freq_low_hz: float
    freq_high_hz: float
    conserve: bool
    sink: str | None
    expected_types: tuple[str, ...]

    def to_dict(self):
        return asdict(self)


GRID = STFTGrid()


def _apply(stems, stem, box, edit, conserve, sink, grid=GRID):
    out = dict(stems)
    delta, a, b = tf_region_delta(out[stem], grid, box, edit)
    out[stem] = apply_delta(out[stem], delta, a, b)
    if conserve:
        out[sink] = apply_delta(out[sink], -delta, a, b)
    return out


def warble(stems, stem, box: TFBox, depth_db=12.0, rate_hz=(4.0, 14.0), conserve=True, sink="other", seed=0):
    """Band-local random attenuation flutter at 4-14 Hz (attenuation only; lost energy -> sink if conserve)."""
    rng = np.random.default_rng(seed)
    fr = GRID.sample_rate / GRID.hop

    def edit(S, mask):
        F, T = S.shape[-2:]
        group = 12  # independent flutter per ~260 Hz group => band-local
        g = np.zeros((F, T))
        sos = butter(2, [rate_hz[0], rate_hz[1]], "bandpass", fs=fr, output="sos")
        for f in range(0, F, group):
            n = sosfiltfilt(sos, rng.standard_normal(T + 64))[32:-32]
            n = n / (np.max(np.abs(n)) + 1e-12)
            g[f : f + group] = 0.5 * (n + 1.0)  # in [0, 1]
        gain = 10 ** (-depth_db * g / 20)
        return S * gain

    gt = GroundTruth("warble", stem, box.start_s, box.end_s, box.freq_low_hz, box.freq_high_hz, conserve, sink if conserve else None, ("unnatural_modulation",))
    return _apply(stems, stem, box, edit, conserve, sink), gt


def cancellation(stems, stem, partner, box: TFBox, level_db=0.0, seed=0):
    """Add band-limited noise to ``stem`` and subtract it from ``partner`` (mixture unchanged)."""
    rng = np.random.default_rng(seed)

    def edit(S, mask):
        ref = np.sqrt(np.mean(np.abs(S * mask) ** 2) + 1e-20)
        noise = (rng.standard_normal(S.shape) + 1j * rng.standard_normal(S.shape)) / np.sqrt(2)
        return S + noise * ref * 10 ** (level_db / 20) * 3.0

    gt = GroundTruth("cancellation", stem, box.start_s, box.end_s, box.freq_low_hz, box.freq_high_hz, True, partner, ("phase_cancellation",))
    return _apply(stems, stem, box, edit, True, partner), gt


def smear(stems, stem, box: TFBox, frames=9, conserve=True, sink="other"):
    """Temporal blur of magnitude (phase kept) -> pre-echo / soft attacks."""

    def edit(S, mask):
        mag = np.abs(S)
        blurred = np.maximum(uniform_filter1d(mag**2, size=frames, axis=-1, mode="nearest"), 0.0) ** 0.5
        return blurred * np.exp(1j * np.angle(S))

    gt = GroundTruth("smear", stem, box.start_s, box.end_s, box.freq_low_hz, box.freq_high_hz, conserve, sink if conserve else None, ("transient_smearing",))
    return _apply(stems, stem, box, edit, conserve, sink), gt


def hf_noise(stems, stem, box: TFBox, level_db=-20.0, conserve=False, sink="other", seed=0):
    """Add noise in an HF box at ``level_db`` relative to the stem's broadband RMS spectrum level."""
    rng = np.random.default_rng(seed)
    ref = np.sqrt(np.mean(stems[stem].astype(np.float64) ** 2)) * np.sqrt(GRID.n_fft * 0.375)  # hann power gain

    def edit(S, mask):
        noise = (rng.standard_normal(S.shape) + 1j * rng.standard_normal(S.shape)) / np.sqrt(2)
        return S + noise * ref * 10 ** (level_db / 20)

    gt = GroundTruth("hf_noise", stem, box.start_s, box.end_s, box.freq_low_hz, box.freq_high_hz, conserve, sink if conserve else None, ("high_frequency_noise",))
    return _apply(stems, stem, box, edit, conserve, sink), gt


def leakage(stems, source, receiver, box: TFBox, gain_db=-15.0):
    """Copy a TF box of ``source`` into ``receiver`` at gain_db and remove it from source (mixture unchanged)."""
    out = dict(stems)
    g = 10 ** (gain_db / 20)
    delta, a, b = tf_region_delta(out[source], GRID, box, lambda S, m: S * (1 - g))  # delta = -g * source_box
    out[source] = apply_delta(out[source], delta, a, b)
    out[receiver] = apply_delta(out[receiver], -delta, a, b)
    gt = GroundTruth("leakage", receiver, box.start_s, box.end_s, box.freq_low_hz, box.freq_high_hz, True, source, ("possible_leakage",))
    return out, gt


def musical_noise(stems, stem, source, box: TFBox, density=0.05, seed=0):
    """Move ``source``'s content in randomly chosen isolated TF cells into ``stem`` (mixture unchanged).
    Equivalent to a mask that randomly spikes to 1 for ``stem``: the classic origin of musical noise."""
    rng = np.random.default_rng(seed)
    out = dict(stems)

    def edit(S, mask):
        cells = (rng.random(S.shape[-2:]) < density).astype(np.float64)
        return S * (1.0 - cells)

    delta, a, b = tf_region_delta(out[source], GRID, box, edit)  # delta = -(source content in spike cells)
    out[source] = apply_delta(out[source], delta, a, b)
    out[stem] = apply_delta(out[stem], -delta, a, b)
    gt = GroundTruth("musical_noise", stem, box.start_s, box.end_s, box.freq_low_hz, box.freq_high_hz, True, source, ("musical_noise",))
    return out, gt


def dropout(stems, stem, box: TFBox, conserve=False, sink="other"):
    """Remove a TF box (missing information)."""
    gt = GroundTruth("dropout", stem, box.start_s, box.end_s, box.freq_low_hz, box.freq_high_hz, conserve, sink if conserve else None, ("reconstruction_discrepancy",))
    return _apply(stems, stem, box, lambda S, m: np.zeros_like(S), conserve, sink), gt
