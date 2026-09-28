import json

import numpy as np
import pytest

from cleansplit.artifacts.region import ArtifactMap, ArtifactRegion, DetectorEvidence
from cleansplit.audio.stft import STFTGrid
from cleansplit.metrics import signal as M
from cleansplit.reconstruction.core import estimate_lag, model_matched_reference, peak_report, reconstruct


def test_oracle_stems_reconstruct_exactly(song):
    mix, stems = song
    rec = reconstruct(mix, stems)
    # float32 mixture vs float64 sum of float32 stems: only float32 rounding remains
    assert M.residual_energy_rel_db(rec.mixture, rec.residual) < -120
    assert rec.alignment.lag_samples == 0 and not rec.alignment.polarity_inverted
    assert abs(rec.alignment.ls_gain - 1) < 1e-6
    assert rec.residual.dtype == np.float64


def test_sum_is_done_in_float64_not_float32(rng):
    # values chosen so float32 accumulation would lose the small stems
    big = np.full((2, 5000), 1.0e4, dtype=np.float32)
    small = {f"s{i}": np.full((2, 5000), 3.0e-4, dtype=np.float32) for i in range(5)}
    stems = {"big": big, **small}
    mix = (big.astype(np.float64) + 5 * 3.0e-4).astype(np.float64)
    rec = reconstruct(mix, stems)
    np.testing.assert_allclose(rec.reconstructed, 1.0e4 + 5 * np.float64(np.float32(3.0e-4)), rtol=0, atol=1e-9)


@pytest.mark.parametrize("lag", [37, -250, 1000])
def test_lag_detected_and_compensated(song, lag):
    mix, stems = song
    shifted = {k: np.roll(v, lag, axis=-1) for k, v in stems.items()}
    est, corr = estimate_lag(mix, sum(s.astype(np.float64) for s in shifted.values()), 4000)
    assert est == lag and corr > 0.99
    rec = reconstruct(mix, shifted)
    assert rec.alignment.lag_applied and rec.alignment.lag_samples == lag
    edge = abs(lag) + 10
    inner = slice(edge, mix.shape[-1] - edge)
    assert M.residual_energy_rel_db(rec.mixture[:, inner], rec.residual[:, inner]) < -100


def test_polarity_and_gain_reported_not_silently_fixed(song):
    mix, stems = song
    inv = {k: -v for k, v in stems.items()}
    rec = reconstruct(mix, inv)
    assert rec.alignment.polarity_inverted and not rec.alignment.polarity_fixed
    rec2 = reconstruct(mix, inv, fix_polarity=True)
    assert rec2.alignment.polarity_fixed and M.residual_energy_rel_db(mix, rec2.residual) < -120
    quiet = {k: v * 0.5 for k, v in stems.items()}
    rec3 = reconstruct(mix, quiet)
    assert abs(rec3.alignment.ls_gain - 2.0) < 1e-4 and not rec3.alignment.gain_applied
    assert any("gain" in w for w in rec3.alignment.warnings)


def test_length_mismatch_padded_and_reported(song):
    mix, stems = song
    ragged = dict(stems)
    ragged["bass"] = stems["bass"][:, :-100]
    ragged["piano"] = np.pad(stems["piano"], ((0, 0), (0, 64)))
    rec = reconstruct(mix, ragged)
    assert rec.alignment.length_adjustments == {"bass": -100, "piano": 64}
    assert all(s.shape == mix.shape for s in rec.stems.values())


def test_channel_mismatch_and_nonfinite_rejected(song):
    mix, stems = song
    bad = dict(stems)
    bad["bass"] = stems["bass"][:1]
    with pytest.raises(ValueError, match="channels"):
        reconstruct(mix, bad)
    bad = dict(stems)
    bad["bass"] = stems["bass"].copy()
    bad["bass"][0, 3] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        reconstruct(mix, bad)


def test_residual_detects_missing_stem_energy(song):
    mix, stems = song
    partial = dict(stems)
    partial["vocals"] = stems["vocals"] * 0.0
    rec = reconstruct(mix, partial)
    # residual must equal the missing vocals
    np.testing.assert_allclose(rec.residual, stems["vocals"].astype(np.float64), atol=1e-6)


def test_model_matched_reference_equals_model_zero_dc_processing(rng):
    """Must equal what BS-RoFormer does (torch.stft -> bin0 = 0 -> torch.istft), NOT an ideal DC removal:
    with a Hann window DC also leaks into bin 1, so part of the DC survives in the model's output."""
    torch = pytest.importorskip("torch")
    g = STFTGrid()
    t = np.arange(44100 * 2) / 44100
    x = np.stack([0.3 + 0.2 * np.sin(2 * np.pi * 1000 * t), 0.05 * rng.standard_normal(t.size)])
    ref = model_matched_reference(x, g, zero_dc=True)
    w = torch.hann_window(2048, dtype=torch.float64)
    S = torch.stft(torch.from_numpy(x), 2048, 512, 2048, w, center=True, return_complex=True)
    S[:, 0] = 0
    expected = torch.istft(S, 2048, 512, 2048, w, center=True, length=x.shape[-1]).numpy()
    np.testing.assert_allclose(ref, expected, atol=1e-9)
    inner = slice(4096, -4096)
    tone = np.sin(2 * np.pi * 1000 * t[inner])
    assert abs(np.dot(ref[0, inner], tone) / np.dot(tone, tone) - 0.2) < 1e-3  # 1 kHz untouched
    assert abs(ref[0, inner].mean()) < 0.3 * 0.5  # DC substantially reduced (not eliminated)


def test_peak_report_flags_overs():
    rep = peak_report({"a": np.array([[0.5, -1.25]]), "b": np.zeros((1, 3))})
    assert rep["a"]["exceeds_full_scale"] and rep["a"]["samples_over_full_scale"] == 1
    assert not rep["b"]["exceeds_full_scale"] and rep["b"]["peak_dbfs"] is None


# ---- metrics ----

def test_metrics_known_values(rng):
    s = rng.standard_normal(100000)
    n = rng.standard_normal(100000)
    n = n / np.linalg.norm(n) * np.linalg.norm(s) * 10 ** (-20 / 20)
    assert abs(M.snr_db(s, s + n) - 20.0) < 1e-9
    assert abs(M.residual_energy_rel_db(s, n) + 20.0) < 1e-9
    assert M.si_sdr_db(s, 3.7 * s) > 200  # scale invariant
    assert abs(M.si_sdr_db(s, s + n) - 20.0) < 0.1
    assert M.snr_db(s, s) == float("inf")
    assert M.residual_energy_rel_db(s, np.zeros_like(s)) == float("-inf")


def test_spectral_metrics_zero_for_identity_and_positive_otherwise(song):
    mix, stems = song
    x = mix[:, : 44100 * 2].astype(np.float64)
    assert M.log_spectral_distance(x, x) == 0.0
    assert M.multires_stft_distance(x, x) == 0.0
    assert M.multi_mel_snr_db(x, x) == float("inf")
    y = x * 0.9
    assert M.log_spectral_distance(x, y) > 0 and 15 < M.multi_mel_snr_db(x, y) < 25  # 20log10(1/0.1)=20 dB


def test_band_residual(song):
    mix, stems = song
    x = mix[:, : 44100 * 2].astype(np.float64)
    bands = M.band_residual_db(x, 0.1 * x)
    vals = [v for v in bands.values() if isinstance(v, float)]
    assert all(abs(v + 20) < 0.01 for v in vals)


# ---- artifact map serialization ----

def test_artifact_map_roundtrip(tmp_path):
    r = ArtifactRegion(
        stem="vocals", start_s=81.42, end_s=81.73, freq_low_hz=1700, freq_high_hz=4300, confidence=0.87,
        artifact_types=["spectral_instability", "possible_separation_artifact"],
        detectors={"modulation": DetectorEvidence(0.87, "dB", 4.1, 3.2, {"cells": 5})}, id="vocals-00000",
    )
    amap = ArtifactMap([r], source={"input": "x.wav"}, stem_summaries={"vocals": {}})
    p = tmp_path / "m.json"
    amap.save(p)
    raw = json.loads(p.read_text())
    assert raw["schema"] == "cleansplit.artifact_map/1"
    back = ArtifactMap.load(p)
    assert back.regions[0] == ArtifactRegion.from_dict(r.to_dict())
    assert back.regions[0].detectors["modulation"].measure_peak == 4.1


def test_artifact_region_validation_and_schema_check():
    with pytest.raises(ValueError):
        ArtifactRegion("vocals", 2.0, 1.0, 0, 10, 0.5, [])
    with pytest.raises(ValueError):
        ArtifactRegion("vocals", 1.0, 2.0, 0, 10, 1.5, [])
    with pytest.raises(ValueError, match="schema"):
        ArtifactMap.from_dict({"schema": "other/9", "regions": []})
