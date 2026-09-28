"""The comparison statistic every quality verdict in this project rests on.

This file exists because the unpaired form of it has already manufactured two false results here: it claimed ep317
alone beat the shipped ensemble by +0.32 dB when ep317 actually loses on 17/20 songs (docs/04 section 14.5), and it
made overlap 8 look indistinguishable from overlap 4 when overlap 8 is in fact consistently +0.01 dB better and merely
not worth 2x the render time (section 14.6). A median-of-A minus median-of-B is not a per-song difference, and a
by-position pairing is not a pairing at all once two arms have scored different songs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "eval"))
from fullband_check import paired


def test_pairs_by_song_name_not_by_position():
    # The candidate is missing the song the baseline is WORST on. Pairing by position would line "b" up against the
    # baseline's "a" and report a fictitious win; pairing by name reports the truth, that the candidate lost on both.
    base = {"a": 1.0, "b": 10.0, "c": 11.0}
    cand = {"b": 9.0, "c": 10.0}
    d, won, n = paired(cand, base)
    assert (n, won) == (2, 0)
    assert d == pytest.approx(-1.0)


def test_paired_delta_is_not_the_difference_of_medians():
    # Both arms have median 2.0, so the unpaired form reports exactly 0.0 dB and calls this a tie -- while the
    # candidate is in fact +1 dB better on two songs out of three. (Winning EVERY song with an equal median is
    # impossible: elementwise domination forces every order statistic up, the median included. The reportable hazard
    # is exactly this one, a majority win that the difference of medians hides.)
    base = {"a": 0.0, "b": 2.0, "c": 4.0}
    cand = {"a": 1.0, "b": 3.0, "c": 2.0}
    assert float(np.median(list(cand.values()))) == float(np.median(list(base.values())))
    d, won, n = paired(cand, base)
    assert (n, won) == (3, 2)
    assert d == pytest.approx(1.0)


def test_no_shared_songs_reports_nan_rather_than_a_number():
    d, won, n = paired({"x": 1.0}, {"y": 1.0})
    assert n == 0 and won == 0 and np.isnan(d)


def test_ties_do_not_count_as_wins():
    d, won, n = paired({"a": 1.0, "b": 2.0}, {"a": 1.0, "b": 2.0})
    assert (n, won) == (2, 0)
    assert d == pytest.approx(0.0)
