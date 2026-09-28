import shutil

import numpy as np
import pytest
import soundfile as sf

from cleansplit.audio.conform import conform_channels, count_clipped, resample, validate_and_conform
from cleansplit.audio.io import load_audio, save_audio
from cleansplit.audio.stft import STFTGrid, istft, stft
from cleansplit.audio.tf_edit import TFBox, apply_delta, changed_support, tf_region_delta


def test_save_load_float_roundtrip_is_bit_exact(tmp_path, rng):
    x = (rng.standard_normal((2, 44100)) * 0.3).astype(np.float32)
    x[0, 10] = 1.7  # over full scale must survive in FLOAT
    save_audio(tmp_path / "a.wav", x, 44100)
    d = load_audio(tmp_path / "a.wav")
    assert d.sample_rate == 44100 and d.audio.shape == (2, 44100)
    assert np.array_equal(d.audio, x)


def test_save_refuses_clipping_integer_pcm_and_nonfinite(tmp_path):
    x = np.zeros((2, 1000), dtype=np.float32)
    x[0, 5] = 1.2
    with pytest.raises(ValueError, match="clip"):
        save_audio(tmp_path / "b.wav", x, 44100, subtype="PCM_16")
    x[0, 5] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        save_audio(tmp_path / "c.wav", x, 44100)


def test_mono_file_loads_as_one_channel(tmp_path, rng):
    x = (rng.standard_normal(22050) * 0.1).astype(np.float32)
    sf.write(tmp_path / "m.wav", x, 22050, subtype="FLOAT")
    d = load_audio(tmp_path / "m.wav")
    assert d.audio.shape == (1, 22050)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_ffmpeg_fallback_decodes_m4a(tmp_path, rng):
    import subprocess

    src = tmp_path / "s.wav"
    sf.write(src, (rng.standard_normal((44100, 2)) * 0.1).astype(np.float32), 44100)
    dst = tmp_path / "s.m4a"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-c:a", "aac", str(dst)], check=True)
    d = load_audio(dst)
    assert d.decoder == "ffmpeg" and d.audio.shape[0] == 2 and d.sample_rate == 44100
    assert abs(d.audio.shape[1] - 44100) < 4096  # AAC priming/padding


def test_channel_conform():
    m = np.arange(10, dtype=np.float32)[None]
    s, note = conform_channels(m, 2)
    assert s.shape == (2, 10) and np.array_equal(s[0], s[1]) and "mono" in note
    back, _ = conform_channels(s, 1)
    assert np.allclose(back, m)
    with pytest.raises(ValueError):
        conform_channels(np.zeros((6, 10)), 2)


def test_resample_preserves_tone_and_is_linear(rng):
    sr_in, sr_out = 48000, 44100
    t = np.arange(sr_in) / sr_in
    a = np.stack([np.sin(2 * np.pi * 1000 * t), np.sin(2 * np.pi * 3000 * t)]) * 0.5
    b = rng.standard_normal((2, sr_in)) * 0.1
    ra, rb, rab = resample(a, sr_in, sr_out), resample(b, sr_in, sr_out), resample(a + b, sr_in, sr_out)
    assert ra.shape == (2, sr_out)
    np.testing.assert_allclose(ra + rb, rab, atol=1e-12)  # linear: resample(sum) == sum(resample)
    tt = np.arange(sr_out) / sr_out
    mid = slice(2000, sr_out - 2000)
    np.testing.assert_allclose(ra[0, mid], 0.5 * np.sin(2 * np.pi * 1000 * tt[mid]), atol=2e-3)


def test_validate_conform_reports_and_does_not_normalize_loudness(rng):
    x = (rng.standard_normal((1, 48000)) * 0.01).astype(np.float32)
    y, rep = validate_and_conform(x, 48000, 44100, 2)
    assert y.shape == (2, 44100) and rep.resampled and rep.channel_conversion
    assert abs(np.sqrt(np.mean(y.astype(np.float64) ** 2)) - 0.01) < 0.001  # level unchanged
    with pytest.raises(ValueError):
        validate_and_conform(np.full((2, 44100), np.inf, dtype=np.float32), 44100)


def test_clipping_detection():
    x = np.zeros((2, 1000), dtype=np.float32)
    x[0, 100:110] = 1.0
    x[1, 500] = -1.0  # single full-scale sample is not a clipping run
    assert count_clipped(x) == 10


@pytest.mark.parametrize("n_fft,hop", [(2048, 512), (512, 128)])
def test_stft_istft_perfect_reconstruction(rng, n_fft, hop):
    g = STFTGrid(44100, n_fft, hop)
    x = rng.standard_normal((2, 30011))
    S = stft(x, g, dtype=np.complex128)
    assert S.shape == (2, n_fft // 2 + 1, 1 + 30011 // hop)
    np.testing.assert_allclose(istft(S, g, x.shape[-1]), x, atol=1e-10)


def test_stft_matches_torch_convention(rng):
    torch = pytest.importorskip("torch")
    g = STFTGrid()
    x = rng.standard_normal(20000)
    ours = stft(x, g, dtype=np.complex128)
    theirs = torch.stft(torch.from_numpy(x), 2048, 512, 2048, torch.hann_window(2048, dtype=torch.float64), center=True, return_complex=True).numpy()
    np.testing.assert_allclose(ours, theirs, atol=1e-9)


def test_tf_edit_is_local_and_bit_identical_outside(rng):
    g = STFTGrid()
    x = (rng.standard_normal((2, 44100 * 4)) * 0.1).astype(np.float32)
    box = TFBox(1.5, 2.0, 1000.0, 3000.0)
    delta, a, b = tf_region_delta(x, g, box, lambda S, m: S * 0.0)
    y = apply_delta(x, delta, a, b)
    sup = changed_support(x, y)
    assert sup is not None
    lo, hi = sup
    # support limited to frames touching the box: +- one window around the box
    assert lo >= int(1.5 * 44100) - 2048 - 512 and hi <= int(2.0 * 44100) + 2048 + 512
    assert np.array_equal(x[:, :lo], y[:, :lo]) and np.array_equal(x[:, hi:], y[:, hi:])
