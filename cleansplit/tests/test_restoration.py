import numpy as np
import pytest

from cleansplit.analysis.pipeline import analyze_separation
from cleansplit.artifacts import corruptions as C
from cleansplit.audio.stft import STFTGrid
from cleansplit.audio.tf_edit import TFBox, excerpt_bounds
from cleansplit.config.settings import AnalysisConfig
from cleansplit.metrics import signal as M
from cleansplit.restoration.base import RestorationContext, RestorationProposal, Restorer
from cleansplit.restoration.engine import GateConfig, evaluate_and_apply
from cleansplit.restoration.pipeline import restore_in_memory
from cleansplit.separation.base import SeparationResult


def analyze(mix, stems):
    return analyze_separation(mix, SeparationResult(stems, 44100, "oracle", {"zero_dc": False}))


@pytest.fixture(scope="module")
def dropout_case(song):
    mix, stems = song
    corrupted, gt = C.dropout(stems, "piano", TFBox(6.0, 6.5, 200, 2000))
    return mix, stems, corrupted, gt, analyze(mix, corrupted)


def test_residual_reallocation_is_accepted_and_improves_consistency(dropout_case):
    mix, clean, corrupted, gt, res = dropout_case
    out = restore_in_memory(mix, corrupted, res.artifact_map, "residual_reallocation")
    s = out["summary"]
    decisions = s["passes"][0]["decisions"]
    assert any(d["accepted"] for d in decisions)
    assert s["reconstruction"]["after"]["residual_rel_db"] < s["reconstruction"]["before"]["residual_rel_db"] - 10
    for d in decisions:
        if d["accepted"]:
            assert d["err_after"] <= d["err_before"] * 10 ** (0.05 / 10) + 1e-12
    # the restored piano moved toward the true piano in the damaged region
    a, b = int(5.8 * 44100), int(6.7 * 44100)
    err_before = M.residual_energy_rel_db(clean["piano"][:, a:b], corrupted["piano"][:, a:b].astype(np.float64) - clean["piano"][:, a:b])
    err_after = M.residual_energy_rel_db(clean["piano"][:, a:b], out["stems"]["piano"][:, a:b].astype(np.float64) - clean["piano"][:, a:b])
    assert err_after < err_before


def test_untouched_audio_is_bit_identical(dropout_case):
    mix, clean, corrupted, gt, res = dropout_case
    out = restore_in_memory(mix, corrupted, res.artifact_map, "residual_reallocation")
    allowed = {k: np.zeros(mix.shape[-1], dtype=bool) for k in corrupted}
    for d in out["summary"]["passes"][0]["decisions"]:
        if d["accepted"]:
            lo, hi = d["support_samples"]
            for k in d["stems_changed"]:
                allowed[k][lo:hi] = True
    for k in corrupted:
        outside = ~allowed[k]
        assert np.array_equal(out["stems"][k][:, outside], corrupted[k][:, outside])
    # stems whose regions were not restored stay entirely bit-identical
    changed = {k for d in out["summary"]["passes"][0]["decisions"] if d["accepted"] for k in d["stems_changed"]}
    for k in set(corrupted) - changed:
        assert np.array_equal(out["stems"][k], corrupted[k])


def test_identity_restorer_changes_nothing(dropout_case):
    mix, clean, corrupted, gt, res = dropout_case
    out = restore_in_memory(mix, corrupted, res.artifact_map, "identity")
    assert all(v["bit_identical"] for v in out["summary"]["stem_changes"].values())
    assert not any(d["accepted"] for d in out["summary"]["passes"][0]["decisions"])


class _HallucinatingRestorer(Restorer):
    """Adds plausible-sounding but absent content (a tone) to the stem in the region."""

    name = "hallucinate"

    def restore(self, stem, regions, context):
        out = []
        for r in regions:
            a, b = excerpt_bounds(context.mixture.shape[-1], context.grid, TFBox(r.start_s, r.end_s, r.freq_low_hz, r.freq_high_hz))
            t = np.arange(b - a) / context.sample_rate
            f = 0.5 * (r.freq_low_hz + r.freq_high_hz)
            tgt = "piano" if stem == "mixture" else stem
            new = context.stems[tgt][..., a:b].astype(np.float64) + 0.2 * np.sin(2 * np.pi * f * t)
            out.append(RestorationProposal(r, {tgt: new}, (a, b), self.name))
        return out


def test_gate_rejects_hallucinated_content(dropout_case):
    mix, clean, corrupted, gt, res = dropout_case
    grid = STFTGrid()
    ctx = RestorationContext(mix.astype(np.float64), {k: v.copy() for k, v in corrupted.items()}, 44100, grid, res.artifact_map)
    regions = [r for r in res.artifact_map.regions if r.stem == "mixture" and r.confidence >= 0.6]
    assert regions
    before = {k: v.copy() for k, v in ctx.stems.items()}
    for r in regions:
        for prop in _HallucinatingRestorer().restore(r.stem, [r], ctx):
            d = evaluate_and_apply(prop, ctx, GateConfig(), AnalysisConfig())
            assert not d.accepted
            assert any("increased" in reason or "not explained" in reason for reason in d.reasons)
    for k in before:
        assert np.array_equal(ctx.stems[k], before[k])  # rejected proposals leave audio untouched


class _SpillingRestorer(Restorer):
    """Proposes changes far outside the region; the projection must discard them."""

    name = "spill"

    def restore(self, stem, regions, context):
        out = []
        for r in regions:
            a, b = excerpt_bounds(context.mixture.shape[-1], context.grid, TFBox(r.start_s, r.end_s, r.freq_low_hz, r.freq_high_hz))
            E = context.residual[..., a:b]
            tgt = "piano"
            noise = np.random.default_rng(0).standard_normal((2, b - a)) * 0.05  # broadband, whole excerpt
            out.append(RestorationProposal(r, {tgt: context.stems[tgt][..., a:b] + E + noise}, (a, b), self.name))
        return out


def test_projection_confines_changes_to_region(dropout_case):
    mix, clean, corrupted, gt, res = dropout_case
    grid = STFTGrid()
    ctx = RestorationContext(mix.astype(np.float64), {k: v.copy() for k, v in corrupted.items()}, 44100, grid, res.artifact_map)
    r = max((r for r in res.artifact_map.regions if r.stem == "mixture"), key=lambda r: r.confidence)
    prop = _SpillingRestorer().restore(r.stem, [r], ctx)[0]
    d = evaluate_and_apply(prop, ctx, GateConfig(require_evidence_decrease=False, max_new_energy_ratio=10.0, tolerance_db=60), AnalysisConfig())
    assert d.projection_discarded_rel_db is not None and d.projection_discarded_rel_db > -3  # most of the spill discarded
    lo, hi = d.support_samples
    margin = 2048 + 512
    assert lo >= int(r.start_s * 44100) - margin and hi <= int(r.end_s * 44100) + margin
