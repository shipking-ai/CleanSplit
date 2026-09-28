"""Ground-truth tests: inject known corruptions into synthetic stems and check the analysis finds them.

These encode behaviour demonstrated on the synthetic benchmark (tools/probes/explore_detectors.py), including known
blind spots, which are asserted explicitly so a future change that alters them is noticed.
"""

import pytest

from cleansplit.analysis.pipeline import analyze_separation
from cleansplit.artifacts import corruptions as C
from cleansplit.audio.tf_edit import TFBox
from cleansplit.separation.base import SeparationResult


def analyze(mix, stems):
    return analyze_separation(mix, SeparationResult(stems, 44100, "oracle", {"zero_dc": False}))


def overlapping(amap, stem, gt, detector=None):
    out = []
    for r in amap.regions:
        if r.stem != stem:
            continue
        if not (r.start_s < gt.end_s and gt.start_s < r.end_s):
            continue
        if stem != "mixture" and not (r.freq_low_hz < gt.freq_high_hz and gt.freq_low_hz < r.freq_high_hz):
            continue
        if detector and detector not in r.detectors:
            continue
        out.append(r)
    return out


@pytest.fixture(scope="module")
def clean_result(song):
    mix, stems = song
    return analyze(mix, stems)


def test_clean_stems_physical_detectors_silent(clean_result):
    """Residual, cancellation and transient detectors must report nothing on exact ground-truth stems."""
    for r in clean_result.artifact_map.regions:
        assert not ({"residual", "cancellation", "transient"} & set(r.detectors)), r


AMBIGUOUS = {"modulation", "leakage"}  # confidence-capped detectors (0.7), see docs/03_artifact_detectors.md


def test_clean_stems_few_false_positives(clean_result):
    regions = clean_result.artifact_map.regions
    unambiguous = [r for r in regions if set(r.detectors) - AMBIGUOUS]
    assert not unambiguous, [(r.stem, r.start_s, list(r.detectors)) for r in unambiguous]
    # measured: 4 capped modulation regions on this 8 s song (benchmark: ~19.5 regions/min over 5 seeds)
    assert len(regions) <= 6, [(r.stem, r.start_s, list(r.detectors)) for r in regions]
    assert all(r.confidence <= 0.7 for r in regions)


def test_clean_reconstruction_metrics(clean_result):
    mm = clean_result.metrics["mixture"]["full_band"]
    assert mm["residual_energy_rel_db"] < -120


CASES = {
    # name: (corruption factory, [(stem that must be flagged, detector)])
    "warble_conserve": (lambda s: C.warble(s, "vocals", TFBox(3.0, 4.2, 800, 3000), conserve=True, sink="other"), [("vocals", "modulation")]),
    "warble_lossy": (lambda s: C.warble(s, "vocals", TFBox(3.0, 4.2, 800, 3000), conserve=False), [("vocals", "modulation"), ("mixture", "residual")]),
    "cancellation": (lambda s: C.cancellation(s, "guitar", "piano", TFBox(4.5, 5.2, 1500, 5000)), [("guitar", "cancellation"), ("piano", "cancellation")]),
    "smear_conserve": (lambda s: C.smear(s, "drums", TFBox(2.0, 3.5, 1000, 16000), conserve=True, sink="other"), [("drums", "transient")]),
    "smear_lossy": (lambda s: C.smear(s, "drums", TFBox(2.0, 3.5, 1000, 16000), conserve=False), [("drums", "transient"), ("mixture", "residual")]),
    "hf_noise_lossy": (lambda s: C.hf_noise(s, "vocals", TFBox(5.0, 6.0, 9000, 13000), level_db=-10), [("mixture", "residual")]),
    "hf_noise_conserve": (lambda s: C.hf_noise(s, "vocals", TFBox(5.0, 6.0, 9000, 13000), level_db=-10, conserve=True, sink="drums"), [("vocals", "cancellation")]),
    "leakage_hats_into_vocals": (lambda s: C.leakage(s, "drums", "vocals", TFBox(1.0, 3.0, 2000, 12000), gain_db=-12), [("vocals", "leakage")]),
    "dropout_lossy": (lambda s: C.dropout(s, "piano", TFBox(6.0, 6.5, 200, 2000)), [("mixture", "residual")]),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_injected_corruption_is_detected(song, name):
    mix, stems = song
    factory, expectations = CASES[name]
    corrupted, gt = factory(stems)
    # the corruption did not touch audio outside its region's support (tf_edit guarantee)
    res = analyze(mix, corrupted)
    for stem, det in expectations:
        hits = overlapping(res.artifact_map, stem, gt, det)
        assert hits, f"{name}: no {det} region on {stem} overlapping {gt.to_dict()}"
        # regions only exist when peak evidence >= 0.6; capped detectors report confidence in [~0.5, 0.7]
        assert max(r.detectors[det].confidence for r in hits) >= (0.5 if det in AMBIGUOUS else 0.6)
    if gt.conserve:
        assert res.metrics["mixture"]["full_band"]["residual_energy_rel_db"] < -100  # mixture really unchanged
        assert not any("residual" in r.detectors for r in res.artifact_map.regions)


def test_hf_noise_regions_are_labelled(song):
    mix, stems = song
    corrupted, gt = C.hf_noise(stems, "vocals", TFBox(5.0, 6.0, 9000, 13000), level_db=-10, conserve=True, sink="drums")
    res = analyze(mix, corrupted)
    hits = overlapping(res.artifact_map, "vocals", gt, "cancellation")
    assert any("high_frequency_noise" in r.artifact_types for r in hits)


def test_leakage_names_its_source(song):
    mix, stems = song
    corrupted, gt = C.leakage(stems, "drums", "vocals", TFBox(1.0, 3.0, 2000, 12000), gain_db=-12)
    hits = overlapping(analyze(mix, corrupted).artifact_map, "vocals", gt, "leakage")
    assert any("leakage_from:drums" in r.artifact_types for r in hits)
    assert all(r.confidence <= 0.7 for r in hits)  # capped: leakage is inherently ambiguous


# ---- documented blind spots (asserted so changes are noticed, not hidden) ----

def test_blind_spot_leakage_under_receivers_own_content(song):
    mix, stems = song
    corrupted, gt = C.leakage(stems, "vocals", "guitar", TFBox(4.0, 6.0, 500, 3000), gain_db=-15)
    assert not overlapping(analyze(mix, corrupted).artifact_map, "guitar", gt, "leakage")


def test_blind_spot_conserving_dropout_is_invisible_on_the_source_stem(song):
    """Removing content from one stem and adding it to another keeps the mixture intact; nothing on the
    *source* stem distinguishes 'missing' from 'never there'. This is why ground-truth evaluation is needed."""
    mix, stems = song
    corrupted, gt = C.dropout(stems, "piano", TFBox(6.0, 6.5, 200, 2000), conserve=True, sink="guitar")
    res = analyze(mix, corrupted)
    assert not overlapping(res.artifact_map, "piano", gt, "residual")
