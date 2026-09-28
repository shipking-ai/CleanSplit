"""Apollo restorer integration (needs CUDA and the weights UVR installs; otherwise skipped).

These check plumbing, locality and the gate — not restoration quality. Quality is measured in
docs/04_results.md section 9, where Apollo lost 24 of 24 measurements against doing nothing.
"""

import numpy as np
import pytest

from cleansplit.restoration.baselines import create


def test_apollo_is_registered_and_marked_generative():
    from cleansplit.restoration.apollo import ApolloRestorer

    r = create("apollo")
    assert isinstance(r, ApolloRestorer) and r.name == "apollo" and r.generative
    assert create("apollo:variant=vocal_restore").variant == "vocal_restore"


def test_folder_name_survives_restorer_options():
    """'apollo:variant=vocal_restore' must not become a path with ':' and '=' in it (illegal on Windows)."""
    from cleansplit.restoration.pipeline import _folder_name

    assert _folder_name("apollo:variant=vocal_restore") == "apollo_variant_vocal_restore"
    assert _folder_name("residual_reallocation") == "residual_reallocation"


def _apollo_available():
    torch = pytest.importorskip("torch")
    from cleansplit.models.checkpoints import find_apollo

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    try:
        find_apollo("mp3_enhancer")
    except FileNotFoundError:
        pytest.skip("Apollo weights not installed (UVR models/Apollo_Models)")


@pytest.mark.gpu
def test_apollo_model_preserves_shape_and_level(song):
    _apollo_available()
    from cleansplit.restoration.apollo import ApolloModel

    mix, _ = song
    x = np.asarray(mix[:, : 3 * 44100], dtype=np.float64)
    model = ApolloModel(variant="mp3_enhancer", device="cuda")
    assert model.sha256 and model.params_m == pytest.approx(16.539, abs=0.01)
    y = model.enhance(x)
    assert y.shape == x.shape and np.isfinite(y).all()
    # An enhancer changes the signal but must not rescale it: within 6 dB of the input's RMS.
    rms = lambda v: 10 * np.log10(np.mean(v.astype(np.float64) ** 2) + 1e-20)
    assert abs(rms(y) - rms(x)) < 6.0
    assert not np.array_equal(y, x.astype(np.float32))


@pytest.mark.gpu
def test_apollo_chunking_matches_a_single_pass(song):
    """Overlap-add must not introduce seams: chunked output ~= one forward pass over the same audio."""
    _apollo_available()
    from cleansplit.restoration.apollo import ApolloModel

    mix, _ = song
    x = np.asarray(mix[:, : 8 * 44100], dtype=np.float64)
    model = ApolloModel(variant="mp3_enhancer", device="cuda")
    whole = model.enhance(x, chunk_s=20.0)  # longer than the signal -> single pass
    chunked = model.enhance(x, chunk_s=2.0, overlap_s=0.5, pad_s=0.25)
    err = 10 * np.log10(np.mean((whole - chunked) ** 2) / (np.mean(whole**2) + 1e-20) + 1e-20)
    assert err < -25.0, f"chunk seams at {err:.1f} dB relative error"


@pytest.mark.gpu
def test_apollo_proposals_stay_inside_the_region_or_are_rejected(song):
    _apollo_available()
    from cleansplit.analysis.pipeline import analyze_separation
    from cleansplit.artifacts import corruptions as C
    from cleansplit.audio.tf_edit import TFBox
    from cleansplit.restoration.pipeline import restore_in_memory
    from cleansplit.separation.base import SeparationResult

    mix, stems = song
    corrupted, _ = C.dropout(stems, "piano", TFBox(4.0, 4.4, 300, 1500))
    amap = analyze_separation(mix, SeparationResult(corrupted, 44100, "oracle", {"zero_dc": False})).artifact_map
    out = restore_in_memory(mix, corrupted, amap, "apollo", max_regions=1)
    (d,) = out["summary"]["passes"][0]["decisions"]
    changed = {k for k in stems if not np.array_equal(out["stems"][k], corrupted[k])}
    if d["accepted"]:
        lo, hi = d["support_samples"]
        for k in changed:
            diff = np.flatnonzero(np.any(out["stems"][k] != corrupted[k], axis=0))
            assert diff.min() >= lo and diff.max() < hi
    else:
        assert not changed  # a rejected generative proposal leaves every stem bit-identical
