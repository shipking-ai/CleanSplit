"""A2SB restorer integration (needs CUDA, third_party code and hash-verified weights; otherwise skipped).
Uses 3 sampling steps: these tests check plumbing and constraints, not restoration quality."""

import numpy as np
import pytest

from cleansplit.restoration.baselines import create


def test_restorer_option_parsing():
    r = create("residual_reallocation")
    assert r.name == "residual_reallocation"
    with pytest.raises(KeyError):
        create("nope")


def _a2sb_available():
    torch = pytest.importorskip("torch")
    from cleansplit.restoration.a2sb import DEFAULT_CKPT_DIR, DEFAULT_REPO

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not (DEFAULT_REPO / "networks.py").is_file() or not list(DEFAULT_CKPT_DIR.glob("A2SB_twosplit_*.ckpt")):
        pytest.skip("A2SB code/weights not installed")


@pytest.mark.gpu
@pytest.mark.parametrize("restorer", ["a2sb", "a2sb_consistent"])
def test_a2sb_proposals_respect_region_and_gate(song, restorer):
    _a2sb_available()
    from cleansplit.analysis.pipeline import analyze_separation
    from cleansplit.artifacts import corruptions as C
    from cleansplit.audio.tf_edit import TFBox
    from cleansplit.restoration.pipeline import restore_in_memory
    from cleansplit.separation.base import SeparationResult

    mix, stems = song
    corrupted, gt = C.dropout(stems, "piano", TFBox(4.0, 4.4, 300, 1500))
    amap = analyze_separation(mix, SeparationResult(corrupted, 44100, "oracle", {"zero_dc": False})).artifact_map
    r = create(f"{restorer}:n_steps=3")
    assert r.generative
    out = restore_in_memory(mix, corrupted, amap, f"{restorer}:n_steps=3", max_regions=1)
    decisions = out["summary"]["passes"][0]["decisions"]
    assert len(decisions) == 1
    d = decisions[0]
    changed = {k for k in stems if not np.array_equal(out["stems"][k], corrupted[k])}
    if d["accepted"]:
        lo, hi = d["support_samples"]
        for k in changed:
            diff = np.flatnonzero(np.any(out["stems"][k] != corrupted[k], axis=0))
            assert diff.min() >= lo and diff.max() < hi
    else:
        assert not changed  # a rejected generative proposal leaves every stem bit-identical
