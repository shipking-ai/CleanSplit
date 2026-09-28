"""Signal fidelity metrics (float64 throughout).

References:
  SI-SDR: Le Roux, Wisdom, Erdogan, Hershey, "SDR - half-baked or well done?", ICASSP 2019 (arXiv:1811.02508)
  Multi-Mel-SNR: metric used in the ICASSP 2026 Music Source Restoration challenge (arXiv:2601.04343);
                 implemented here as mean over mel resolutions of 10log10(||M_ref||^2/||M_ref - M_est||^2).
"""

from __future__ import annotations

import numpy as np

from ..audio.stft import STFTGrid, stft

EPS = 1e-12


def _flat(x):
    return np.asarray(x, dtype=np.float64).reshape(-1)


def energy_db(x) -> float:
    e = float(np.sum(_flat(x) ** 2))
    return 10 * np.log10(e + EPS)


def snr_db(reference, estimate) -> float:
    """10log10(||ref||^2 / ||ref - est||^2). With ref = O and est = R this is the mixture SDR (BSS-eval-free SDR)."""
    r, e = _flat(reference), _flat(estimate)
    num = float(np.dot(r, r))
    den = float(np.dot(r - e, r - e))
    if den <= 0:
        return float("inf")
    return 10 * np.log10((num + EPS) / den)


def si_sdr_db(reference, estimate) -> float:
    r, e = _flat(reference), _flat(estimate)
    r = r - r.mean()
    e = e - e.mean()
    rr = float(np.dot(r, r))
    if rr <= 0:
        return float("nan")
    alpha = float(np.dot(e, r)) / rr
    target = alpha * r
    noise = e - target
    nn = float(np.dot(noise, noise))
    if nn <= 0:
        return float("inf")
    return 10 * np.log10((float(np.dot(target, target)) + EPS) / nn)


def residual_energy_rel_db(reference, residual) -> float:
    """Residual energy relative to reference energy (dB). -inf when bit-exact."""
    num = float(np.sum(_flat(residual) ** 2))
    den = float(np.sum(_flat(reference) ** 2))
    if num <= 0:
        return float("-inf")
    return 10 * np.log10(num / (den + EPS))


def mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float = 0.0, fmax: float | None = None) -> np.ndarray:
    """Slaney-style triangular mel filters (area-normalized), shape (n_mels, n_fft//2+1)."""
    fmax = fmax or sr / 2

    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)

    def mel2hz(m):
        return 700.0 * (10 ** (np.asarray(m) / 2595.0) - 1.0)

    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    pts = mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2))
    fb = np.zeros((n_mels, freqs.size))
    for i in range(n_mels):
        lo, c, hi = pts[i], pts[i + 1], pts[i + 2]
        up = (freqs - lo) / max(c - lo, EPS)
        down = (hi - freqs) / max(hi - c, EPS)
        fb[i] = np.maximum(0, np.minimum(up, down))
        fb[i] *= 2.0 / max(hi - lo, EPS)
    return fb


def multi_mel_snr_db(reference, estimate, sr: int = 44100, resolutions=((512, 128, 80), (1024, 256, 128), (2048, 512, 160))) -> float:
    vals = []
    for n_fft, hop, n_mels in resolutions:
        g = STFTGrid(sr, n_fft, hop)
        fb = mel_filterbank(sr, n_fft, n_mels)
        Mr = np.tensordot(fb, np.abs(stft(reference, g, np.complex128)), axes=([1], [-2]))
        Me = np.tensordot(fb, np.abs(stft(estimate, g, np.complex128)), axes=([1], [-2]))
        num = float(np.sum(Mr**2))
        den = float(np.sum((Mr - Me) ** 2))
        vals.append(float("inf") if den <= 0 else 10 * np.log10((num + EPS) / den))
    return float(np.mean(vals))


def log_spectral_distance(reference, estimate, grid: STFTGrid = STFTGrid()) -> float:
    """Mean over frames of RMS (over bins) difference of log10 power spectra (in 'LSD' units, not dB)."""
    Pr = np.abs(stft(reference, grid, np.complex128)) ** 2
    Pe = np.abs(stft(estimate, grid, np.complex128)) ** 2
    d = np.log10(Pr + 1e-10) - np.log10(Pe + 1e-10)
    d = d.reshape(-1, d.shape[-2], d.shape[-1])
    return float(np.mean(np.sqrt(np.mean(d**2, axis=-2))))


def multires_stft_distance(reference, estimate, sr: int = 44100, sizes=(512, 1024, 2048)) -> float:
    """Spectral-convergence + log-magnitude L1, averaged over resolutions (lower is better)."""
    tot = 0.0
    for n in sizes:
        g = STFTGrid(sr, n, n // 4)
        R = np.abs(stft(reference, g, np.complex128))
        E = np.abs(stft(estimate, g, np.complex128))
        sc = np.linalg.norm(R - E) / (np.linalg.norm(R) + EPS)
        mag = np.mean(np.abs(np.log(R + 1e-7) - np.log(E + 1e-7)))
        tot += sc + mag
    return float(tot / len(sizes))


def band_residual_db(reference, residual, grid: STFTGrid = STFTGrid(), edges=(0, 20, 60, 250, 500, 2000, 4000, 8000, 16000, 22050)) -> dict:
    """Residual-to-reference energy (dB) per frequency band."""
    Pr = (np.abs(stft(reference, grid, np.complex128)) ** 2).reshape(-1, grid.n_bins, grid.n_frames(reference.shape[-1])).sum(axis=(0, 2))
    Pe = (np.abs(stft(residual, grid, np.complex128)) ** 2).reshape(-1, grid.n_bins, grid.n_frames(residual.shape[-1])).sum(axis=(0, 2))
    f = grid.freqs()
    out = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (f >= lo) & (f < hi)
        num, den = Pe[m].sum(), Pr[m].sum()
        out[f"{lo}-{hi}Hz"] = None if den <= 0 else (float("-inf") if num <= 0 else float(10 * np.log10(num / den)))
    return out


def finite_or_none(x):
    if x is None:
        return None
    if isinstance(x, float) and not np.isfinite(x):
        return "inf" if x > 0 else ("-inf" if x < 0 else "nan")
    return x
