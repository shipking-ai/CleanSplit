"""Mixture reconstruction and residual.

    R = sum_i S_i          E = O - R

Done in float64. Before summing, stems are checked (not assumed) for:
  * equal sample rate and channel count (enforced upstream),
  * equal length (pad/trim, recorded),
  * a common integer-sample lag vs O (cross-correlation, compensated, recorded),
  * polarity (a negative optimal gain is reported; flipped only if requested),
  * a global gain mismatch (e.g. a tool that normalized its exports; reported, applied only if requested).

``model_matched_reference`` applies the separator's known deterministic processing to O (BS-RoFormer
zeroes STFT bin 0 in every stem). Residual against that reference isolates mask errors from a design
property that would otherwise always show up as sub-20 Hz residual.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import scipy.fft

from ..audio.stft import STFTGrid, istft, stft


@dataclass
class AlignmentReport:
    lag_samples: int
    lag_applied: bool
    correlation_peak: float
    polarity_inverted: bool
    polarity_fixed: bool
    ls_gain: float
    gain_applied: bool
    length_adjustments: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


@dataclass
class Reconstruction:
    mixture: np.ndarray  # O (C, N) float64
    stems: dict[str, np.ndarray]  # aligned stems (C, N) float32 (as written to disk)
    reconstructed: np.ndarray  # R (C, N) float64
    residual: np.ndarray  # E = O - R (C, N) float64
    reference: np.ndarray  # O_ref: model-matched reference (C, N) float64 (== O when not applicable)
    residual_matched: np.ndarray  # O_ref - R
    alignment: AlignmentReport


def estimate_lag(reference: np.ndarray, signal: np.ndarray, max_lag: int) -> tuple[int, float]:
    """Integer lag L such that signal[n] ~ reference[n - L]; normalized correlation at the peak (signed)."""
    a = reference.astype(np.float64).mean(axis=0)
    b = signal.astype(np.float64).mean(axis=0)
    n = max(a.size, b.size)
    nfft = scipy.fft.next_fast_len(2 * n)
    A = scipy.fft.rfft(a, nfft)
    B = scipy.fft.rfft(b, nfft)
    xc = scipy.fft.irfft(B * np.conj(A), nfft)  # xc[k] = sum_n b[n] a[n-k]
    lags = np.concatenate((np.arange(0, max_lag + 1), np.arange(-max_lag, 0)))
    vals = np.concatenate((xc[: max_lag + 1], xc[nfft - max_lag :]))
    k = int(np.argmax(np.abs(vals)))
    denom = np.sqrt(np.dot(a, a) * np.dot(b, b))
    return int(lags[k]), float(vals[k] / denom) if denom > 0 else 0.0


def _shift(x: np.ndarray, lag: int) -> np.ndarray:
    """Undo a lag: returns y with y[n] = x[n + lag], zero-filled."""
    if lag == 0:
        return x
    y = np.zeros_like(x)
    if lag > 0:
        y[..., : x.shape[-1] - lag] = x[..., lag:]
    else:
        y[..., -lag:] = x[..., : x.shape[-1] + lag]
    return y


def model_matched_reference(mixture: np.ndarray, grid: STFTGrid, zero_dc: bool) -> np.ndarray:
    if not zero_dc:
        return mixture.astype(np.float64)
    spec = stft(mixture, grid, dtype=np.complex128)
    spec[..., 0, :] = 0.0
    return istft(spec, grid, mixture.shape[-1])


def reconstruct(
    mixture: np.ndarray,
    stems: dict[str, np.ndarray],
    *,
    max_lag_s: float = 0.5,
    sample_rate: int = 44100,
    fix_lag: bool = True,
    fix_polarity: bool = False,
    fix_gain: bool = False,
    zero_dc_grid: STFTGrid | None = None,
) -> Reconstruction:
    O = mixture.astype(np.float64)
    n = O.shape[-1]
    warnings = []
    adj = {}
    aligned = {}
    for name, s in stems.items():
        if s.shape[0] != O.shape[0]:
            raise ValueError(f"stem '{name}' has {s.shape[0]} channels, mixture has {O.shape[0]}")
        if not np.all(np.isfinite(s)):
            raise ValueError(f"stem '{name}' contains non-finite samples")
        d = s.shape[-1] - n
        if d:
            adj[name] = int(d)
            s = s[..., :n] if d > 0 else np.pad(s, ((0, 0), (0, -d)))
        aligned[name] = s.astype(np.float64)
    if adj:
        warnings.append(f"stem lengths differed from mixture (samples, +longer/-shorter): {adj}")

    R = np.zeros_like(O)
    for s in aligned.values():
        R += s

    max_lag = int(max_lag_s * sample_rate)
    lag, corr = estimate_lag(O, R, min(max_lag, n // 2))
    lag_applied = False
    if lag != 0:
        warnings.append(f"reconstructed mix is offset by {lag} samples relative to the mixture")
        if fix_lag:
            aligned = {k: _shift(v, lag) for k, v in aligned.items()}
            R = _shift(R, lag)
            lag_applied = True
            warnings.append(f"offset compensated; the first/last {abs(lag)} samples of stems are zero-filled")

    rr = float(np.sum(R * R))
    gain = float(np.sum(O * R) / rr) if rr > 0 else 0.0
    inverted = gain < 0
    pol_fixed = False
    if inverted:
        warnings.append("stems appear polarity-inverted relative to the mixture")
        if fix_polarity:
            aligned = {k: -v for k, v in aligned.items()}
            R = -R
            gain = -gain
            pol_fixed = True
    gain_applied = False
    if rr > 0 and abs(abs(gain) - 1.0) > 0.01:  # > ~0.09 dB
        warnings.append(f"least-squares gain between stems and mixture is {gain:.4f} ({20*np.log10(abs(gain)):+.2f} dB)")
        if fix_gain:
            aligned = {k: v * abs(gain) for k, v in aligned.items()}
            R = R * abs(gain)
            gain_applied = True

    E = O - R
    ref = model_matched_reference(O, zero_dc_grid, True) if zero_dc_grid is not None else O
    Em = ref - R
    report = AlignmentReport(lag, lag_applied, corr, inverted, pol_fixed, gain, gain_applied, adj, warnings)
    return Reconstruction(O, {k: v.astype(np.float32) for k, v in aligned.items()}, R, E, ref, Em, report)


def peak_report(signals: dict[str, np.ndarray]) -> dict[str, dict]:
    out = {}
    for k, v in signals.items():
        peak = float(np.max(np.abs(v))) if v.size else 0.0
        out[k] = {
            "peak": peak,
            "peak_dbfs": float(20 * np.log10(peak)) if peak > 0 else None,
            "exceeds_full_scale": peak > 1.0,
            "samples_over_full_scale": int(np.count_nonzero(np.abs(v) > 1.0)),
        }
    return out
