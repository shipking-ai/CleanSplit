"""Every dB figure quoted to a user must still be in the committed measurement JSON that produced it.

This is the cheapest guard the project has against its worst failure mode. docs/04 and the `--quality` help text quote
about forty numbers; the benchmark JSON they came from is committed beside them; and nothing but care has been keeping
the two in step. Care has already failed twice here -- docs/04 sections 14.6 and 14.7 record a delta that was restated
three times before anyone recomputed it -- and a stale number in a README is indistinguishable, to a reader, from a
fabricated one.

So these tests re-read the JSON and fail if a quoted figure has drifted. No GPU, no model weights, no dataset and no
network: the arrays were scored once and the verdicts were written down, so re-checking the arithmetic is free and runs
on every push.

What this deliberately does NOT do is re-derive the numbers from audio. That would need the 20-song MUSDB cache and a
GPU. The claim under test is narrower and still worth asserting: **the prose agrees with the committed data.**
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "outputs" / "_benchmarks"
DOCS = ROOT / "docs" / "04_results.md"


def _verdict(fname: str, arm: str) -> dict:
    p = BENCH / fname
    if not p.is_file():
        pytest.skip(f"{p.relative_to(ROOT)} not present")
    d = json.loads(p.read_text(encoding="utf-8"))
    if arm not in d.get("verdict", {}):
        pytest.skip(f"{fname} has no verdict for {arm}")
    return d["verdict"][arm]


# (stem, quoted dB, quoted win count) exactly as docs/04 section 14.15 and cleansplit/cli/main.py state them.
TTA_IN_ENSEMBLE = [("vocals", 0.008, 18), ("drums", 0.015, 16), ("bass", 0.033, 17)]
TTA_ON_SW_ALONE = [("vocals", 0.027, 19), ("drums", 0.015, 16), ("bass", 0.033, 17)]


@pytest.mark.parametrize(("fname", "arm", "quoted"), [
    ("musdb18hq_tta_in_ensemble.json", "ens_tta_ov4", TTA_IN_ENSEMBLE),
    ("musdb18hq_tta_vs_overlap.json", "sw_tta_ov4", TTA_ON_SW_ALONE),
])
def test_quoted_tta_deltas_match_the_committed_json(fname: str, arm: str, quoted: list) -> None:
    """The measurement that removed TTA from the default recipe (docs/04 section 14.15).

    Rounding, not equality: the docs quote three decimal places, so a figure is correct if it rounds to what is written.
    """
    v = _verdict(fname, arm)
    for stem, db, won in quoted:
        actual = v["paired_delta_db"][stem]
        # Within half of the last quoted decimal place: the docs round to three, so +0.0615 may legitimately be
        # written +0.062. Tighter than this would be asserting Python's float rounding, not the measurement.
        assert actual == pytest.approx(db, abs=6e-4), f"{arm} {stem}: docs say {db:+.3f}, JSON has {actual:+.5f}"
        assert v["won"][stem] == won, f"{arm} {stem}: docs say {won} wins, JSON has {v['won'][stem]}"


def test_the_same_arm_passes_against_the_weak_baseline_and_fails_against_the_right_one() -> None:
    """The nuance that decided the default, and the reason a gate result is meaningless without its baseline.

    `ens_tta_ov4` PASSES the gate when measured against `fast` -- of course it does, it is four levers better -- and
    FAILS when measured against `ens_ov4`, the recipe that differs from it by TTA alone. Only the second comparison
    answers "is TTA worth 2x compute", and quoting the first as though it did would be the §14.4 mistake in a new
    costume. Both files are committed precisely so the pair can be checked.
    """
    assert _verdict("musdb18hq_fullband.json", "ens_tta_ov4")["fullband_better_on_vocals_drums_bass"] is True
    assert _verdict("musdb18hq_tta_in_ensemble.json", "ens_tta_ov4")["fullband_better_on_vocals_drums_bass"] is False
    # and the tier that shipped clears the gate against fast on its own merits
    assert _verdict("musdb18hq_fullband.json", "ens_ov4")["fullband_better_on_vocals_drums_bass"] is True


def test_tta_never_lost_a_song_which_is_the_whole_point_of_the_effect_size_floor() -> None:
    """The claim in docs/04 section 14.15 that carries the argument: 0 regressions across 80 song-stem comparisons.

    If TTA ever did make a song worse, the section's strongest sentence would be false, so it is asserted directly.
    """
    total = 0
    for fname, arm in (("musdb18hq_tta_in_ensemble.json", "ens_tta_ov4"), ("musdb18hq_tta_vs_overlap.json", "sw_tta_ov4")):
        v = _verdict(fname, arm)
        for stem, short in v["songs_short_of_baseline"].items():
            assert short == 0, f"{arm} {stem}: {short} songs worse than baseline, docs/04 section 14.15 claims none"
            total += v["n"][stem]
    assert total == 160, f"the section says 80 song-stem comparisons per arm; the JSON covers {total} in total"


def test_the_adoption_floor_quoted_everywhere_is_the_one_the_gate_actually_applied() -> None:
    """+0.02 dB appears in the README, the CLI help and four docs sections. It is a number in the JSON too."""
    for fname, arm in (("musdb18hq_tta_in_ensemble.json", "ens_tta_ov4"), ("musdb18hq_tta_vs_overlap.json", "sw_tta_ov4")):
        assert _verdict(fname, arm)["min_delta_db"] == 0.02


def test_the_cli_quality_notes_quote_the_measured_ladder() -> None:
    """quality_note() is what a user reads before spending an hour of GPU time; its figures come from the same JSON."""
    from cleansplit.cli.main import QUALITY_TIERS, quality_note

    v = _verdict("musdb18hq_fullband.json", "ens_ov4")   # the `best` tier, measured against `fast`
    best = quality_note("best")
    for stem in ("vocals", "drums", "bass"):
        quoted = re.search(rf"\+([0-9.]+) dB {stem}|\+([0-9.]+) {stem}", best)
        assert quoted, f"quality_note('best') no longer quotes a {stem} figure"
        got = float(next(g for g in quoted.groups() if g))
        assert v["paired_delta_db"][stem] == pytest.approx(got, abs=6e-4), stem
    # and the advertised compute ratios are the pass counts, not prose
    assert QUALITY_TIERS["best"][3] == 4 * QUALITY_TIERS["fast"][3]
    assert QUALITY_TIERS["best"][3] == 2 * QUALITY_TIERS["balanced"][3]


def test_docs_section_14_15_exists_and_states_the_removal() -> None:
    """The CLI help, the README and this test file all cite section 14.15 by number. It has to be there."""
    if not DOCS.is_file():
        pytest.skip("docs/04_results.md not present")
    text = DOCS.read_text(encoding="utf-8")
    assert "### 14.15" in text, "docs/04 section 14.15 is cited by the CLI and the README but is missing"
    body = text.split("### 14.15", 1)[1]
    assert "8 forward passes" in body, "section 14.15 no longer states the new default cost"
