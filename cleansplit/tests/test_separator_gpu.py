"""Real BS-RoFormer SW integration (needs torch + the local UVR checkpoint; CUDA preferred)."""

import numpy as np
import pytest

from cleansplit.metrics import signal as M


def _separator(**kw):
    torch = pytest.importorskip("torch")
    from cleansplit.models import checkpoints
    from cleansplit.separation.roformer import BSRoformerSeparator

    try:
        checkpoints.find_checkpoint(checkpoints.BS_ROFO_SW_FIXED)
    except FileNotFoundError:
        pytest.skip("BS-Rofo-SW-Fixed checkpoint not installed")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available (CPU path works but is too slow for the test suite)")
    return BSRoformerSeparator(**kw)


@pytest.mark.gpu
def test_sw_separates_six_aligned_stems(song):
    mix, _ = song
    sep = _separator(chunk_size=44100 * 5, num_overlap=2)  # short chunks so several overlap-add seams are exercised
    res = sep.separate(mix, 44100)
    assert set(res.stems) == {"vocals", "drums", "bass", "guitar", "piano", "other"}
    assert all(s.shape == mix.shape and s.dtype == np.float32 for s in res.stems.values())
    assert res.metadata["checkpoint_sha256"] == "24e7d35ee9c64415673d3fd33e06a67cac2c103c5df6267ba1576459c775916e"
    assert len(res.chunk_starts) >= 3
    R = sum(s.astype(np.float64) for s in res.stems.values())
    from cleansplit.reconstruction.core import estimate_lag

    lag, corr = estimate_lag(mix, R, 4096)
    assert lag == 0 and corr > 0.95  # sample-aligned output
    assert M.snr_db(mix, R) > 10  # masks roughly sum to one even on out-of-domain synthetic audio


@pytest.mark.gpu
def test_sw_tta_is_average_of_mapped_passes(song):
    mix, _ = song
    short = mix[:, : 44100 * 6]
    plain = _separator(chunk_size=44100 * 4)
    tta = _separator(chunk_size=44100 * 4, tta=True)
    a = plain.separate(short, 44100).stems
    b = plain.separate(short[::-1].copy(), 44100).stems
    c = plain.separate(-short, 44100).stems
    res = tta.separate(short, 44100)
    assert res.metadata["tta"] == ["original", "channel_swap", "polarity_invert"]
    assert tta.cache_key()["tta"] is True and plain.cache_key()["tta"] is False
    for k in a:
        expected = (a[k].astype(np.float64) + b[k][::-1] - c[k]) / 3
        np.testing.assert_allclose(res.stems[k], expected, atol=1e-5)


@pytest.mark.gpu
def test_ensemble_is_mixture_consistent_and_follows_recipe(song):
    _separator()  # skips when SW/CUDA unavailable
    from cleansplit.separation import registry

    try:
        ens = registry.create("ensemble", tta=False)
    except FileNotFoundError:
        pytest.skip("ep317 checkpoint not installed")
    mix = song[0][:, : 44100 * 6]
    res = ens.separate(mix, 44100)
    assert set(res.stems) == {"vocals", "drums", "bass", "guitar", "piano", "other"}
    total = sum(v.astype(np.float64) for v in res.stems.values())
    assert np.max(np.abs(total - mix)) < 1e-5  # exact up to float32 storage
    sw = ens.sw.separate(mix, 44100).stems
    voc = ens.voc.separate(mix, 44100).stems["vocals"].astype(np.float64)
    np.testing.assert_allclose(res.stems["vocals"], 0.5 * (sw["vocals"].astype(np.float64) + voc), atol=1e-5)
    np.testing.assert_allclose(res.stems["drums"], sw["drums"], atol=1e-6)


@pytest.mark.gpu
def test_sw_overlap_add_preserves_per_chunk_consistency(song):
    """Stems are blended with identical weights, so chunking must not change sum(stems) beyond what the model does:
    compare two different chunkings; the reconstruction error level should be similar."""
    mix, _ = song
    a = _separator(chunk_size=44100 * 4, num_overlap=2).separate(mix, 44100)
    b = _separator(chunk_size=44100 * 6, num_overlap=2).separate(mix, 44100)
    ra = sum(s.astype(np.float64) for s in a.stems.values())
    rb = sum(s.astype(np.float64) for s in b.stems.values())
    assert abs(M.snr_db(mix, ra) - M.snr_db(mix, rb)) < 6


def _mdx23c(**kw):
    torch = pytest.importorskip("torch")
    from cleansplit.models import checkpoints
    from cleansplit.separation.mdx23c_sep import MDX23CSeparator

    try:
        checkpoints.find_checkpoint(checkpoints.MDX23C_INSTVOC_HQ)
    except FileNotFoundError:
        pytest.skip("MDX23C InstVoc HQ checkpoint not installed")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return MDX23CSeparator(**kw)


@pytest.mark.gpu
def test_mdx23c_separates_aligned_vocals_and_instrumental(song):
    """The native chunk (5.9 s) on an 8 s song with overlap 2 exercises several overlap-add seams."""
    mix, _ = song
    res = _mdx23c(num_overlap=2).separate(mix, 44100)
    assert set(res.stems) == {"vocals", "instrumental"}
    assert all(s.shape == mix.shape and s.dtype == np.float32 for s in res.stems.values())
    assert res.metadata["checkpoint_sha256"] == "49d51472769e34a2501cd1da782346a3212555c3a5619fc2c53507445528d816"
    assert len(res.chunk_starts) >= 3
    R = res.stems["vocals"].astype(np.float64) + res.stems["instrumental"].astype(np.float64)
    from cleansplit.reconstruction.core import estimate_lag

    lag, corr = estimate_lag(mix, R, 4096)
    assert lag == 0 and corr > 0.95
    assert M.snr_db(mix, R) > 10


def test_overlap_reaches_both_roformer_members_of_the_ensemble_and_the_cache_key():
    """`--overlap 4` was silently dropped for `--separator ensemble`, the default recipe, so the one artifact lever
    that passed both gates (docs/04 §14.5) could not reach the code path that ships. No GPU: constructors only."""
    from cleansplit.separation import registry

    ens = registry.create("ensemble", num_overlap=2)
    assert ens.sw.num_overlap == 2 and ens.voc.num_overlap == 2  # an explicit request must reach both members
    d = registry.create("ensemble")
    assert d.sw.num_overlap == 4 and d.voc.num_overlap == 4  # default is the best measured, not the fastest
    assert ens.cache_key()["sw"]["num_overlap"] == 2  # a re-run at another overlap must not reuse the cache
