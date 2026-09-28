import numpy as np

from cleansplit.artifacts import corruptions as C
from cleansplit.audio.tf_edit import TFBox
from cleansplit.metrics.restoration_experiment import _box_total_changes, run_experiment
from cleansplit.separation.base import SeparationResult


class _FakeSeparator:
    """Returns fixed (corrupted) stems: lets the harness be tested without a GPU model."""

    name = "fake"
    sample_rate = 44100

    def __init__(self, stems):
        self.stems = stems

    def separate(self, mix, sr):
        return SeparationResult({k: v.copy() for k, v in self.stems.items()}, sr, self.name, {"zero_dc": False})


def test_experiment_harness_with_groups(song):
    mix, truth6 = song
    corrupted, gt = C.dropout(truth6, "piano", TFBox(6.0, 6.5, 200, 2000))
    # coarser truth: 'keys' = guitar + piano, the others identity
    truth = {k: v for k, v in truth6.items() if k not in ("guitar", "piano")}
    truth["keys"] = (truth6["guitar"].astype(np.float64) + truth6["piano"]).astype(np.float32)
    groups = {"vocals": ["vocals"], "drums": ["drums"], "bass": ["bass"], "other": ["other"], "keys": ["guitar", "piano"]}
    res = run_experiment(
        _FakeSeparator(corrupted), restorers=("identity", "residual_reallocation"), songs=[("s", mix, truth)],
        groups=groups, max_regions=5, log=lambda m: None,
    )
    e = res["results"][0]
    assert set(e["separation_sdr_db"]) == set(truth)
    ident = e["restorers"]["identity"]
    assert ident["accepted"] == 0 and ident["box_truth_changes"] == []
    rr = e["restorers"]["residual_reallocation"]
    assert rr["accepted"] >= 1
    assert all(x["truth_stem"] in truth for x in rr["box_truth_changes"])
    # the dropout removed keys content: giving the residual back must reduce pooled stem error in those boxes
    assert rr["pooled_change_db_accepted"] < 0
    assert all(abs(x["total_err_change_db"]) < 60 for x in rr["box_total_changes"])  # floored: no runaway ratios


def test_box_total_is_energy_weighted():
    rows = [
        {"region_id": "r", "accepted": True, "err_before": 1e-12, "err_after": 1e-9, "truth_energy": 1e-12},  # silent group: +30 dB
        {"region_id": "r", "accepted": True, "err_before": 1.0, "err_after": 0.5, "truth_energy": 4.0},  # loud group: -3 dB
    ]
    (t,) = _box_total_changes(rows)
    floor = 1e-3 * (4.0 + 1e-12)
    assert abs(t["total_err_change_db"] - 10 * np.log10((0.5 + 1e-9 + floor) / (1.0 + 1e-12 + floor))) < 1e-6
