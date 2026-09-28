"""End-to-end through the CLI with files on disk (no GPU): stems exported 'UVR-style' next to a mixture."""

import json

import numpy as np

from cleansplit.artifacts import corruptions as C
from cleansplit.artifacts.region import ArtifactMap
from cleansplit.audio.io import load_audio, save_audio
from cleansplit.audio.tf_edit import TFBox
from cleansplit.cli.main import main


def test_analyze_restore_compare_e2e(tmp_path, song, capsys):
    mix, stems = song
    corrupted, gt = C.dropout(stems, "piano", TFBox(6.0, 6.5, 200, 2000))
    save_audio(tmp_path / "My Song.wav", mix, 44100)
    stem_dir = tmp_path / "uvr"
    for k, v in corrupted.items():
        save_audio(stem_dir / f"My Song_({k.capitalize()}).wav", v, 44100)
    out = tmp_path / "outputs"

    assert main(["analyze", str(tmp_path / "My Song.wav"), "--stems-dir", str(stem_dir), "--out", str(out)]) == 0
    summary = json.loads(capsys.readouterr().out)
    song_dir = out / "My_Song"
    for rel in [
        "original.wav", "stems/vocals.wav", "stems/other.wav", "reconstruction/reconstructed.wav",
        "reconstruction/residual.wav", "analysis/artifact_map.json", "analysis/metrics.json", "analysis/report.json",
    ]:
        assert (song_dir / rel).is_file(), rel
    amap = ArtifactMap.load(song_dir / "analysis" / "artifact_map.json")
    assert any(r.stem == "mixture" and "residual" in r.detectors and r.start_s < gt.end_s and gt.start_s < r.end_s for r in amap.regions)
    assert summary["regions_total"] == len(amap.regions)
    residual = load_audio(song_dir / "reconstruction" / "residual.wav").audio
    # residual file equals the removed piano content (float32 WAV)
    np.testing.assert_allclose(residual, stems["piano"] - corrupted["piano"], atol=1e-5)

    assert main(["restore", str(tmp_path / "My Song.wav"), "--out", str(out)]) == 0
    rsum = json.loads(capsys.readouterr().out)
    assert rsum["reconstruction"]["after"]["residual_rel_db"] < rsum["reconstruction"]["before"]["residual_rel_db"]
    rdir = song_dir / "restoration" / "residual_reallocation"
    assert (rdir / "restoration_report.json").is_file() and (rdir / "stems" / "piano.wav").is_file()

    assert main(["compare", str(song_dir / "original.wav"), str(rdir / "reconstructed.wav")]) == 0
    cmp = json.loads(capsys.readouterr().out)
    assert cmp["lag_samples"] == 0 and cmp["snr_db"] > 30


def test_cli_parser_accepts_separator_choices():
    from cleansplit.cli.main import build_parser

    for sep in ("bs_roformer_sw", "ensemble", "ensemble_demucs", "bs_roformer_ep317", "htdemucs_ft", "mdx23c_instvoc_hq"):
        args = build_parser().parse_args(["separate", "x.wav", "--separator", sep])
        assert args.separator == sep
    args = build_parser().parse_args(["evaluate", "--restoration-experiment", "--restorers", "identity", "--seeds", "0"])
    assert args.restoration_experiment and args.separator == "bs_roformer_sw"


def test_defaults_are_the_best_measured_setting_not_the_fastest():
    """The user's rule: the default split mode is whichever measures best, and speed is the opt-out.

    `ensemble` + overlap 4 + TTA is the best measured combination (docs/04 §11, §14.1, §14.5). These asserts exist
    because each of them was once silently the fast-but-worse option: the CLI defaulted to single-pass SW with no TTA,
    and `midi` built its own Namespace that pinned overlap 2 / tta False.
    """
    from cleansplit.cli.main import build_parser

    for cmd in ("separate", "analyze"):
        a = build_parser().parse_args([cmd, "x.wav"])
        assert (a.separator, a.overlap, a.tta, a.fp16) == ("ensemble", 4, True, False), cmd
    fast = build_parser().parse_args(["separate", "x.wav", "--no-tta", "--overlap", "2", "--separator", "bs_roformer_sw"])
    assert (fast.separator, fast.overlap, fast.tta) == ("bs_roformer_sw", 2, False)  # opting out still works
    assert build_parser().parse_args(["midi", "x.wav"]).separator == "ensemble"
    # `evaluate` stays pinned to single-pass SW: docs/04's experiment numbers were produced with it, and moving this
    # default would silently change what a published number means.
    ev = build_parser().parse_args(["evaluate", "--restoration-experiment"])
    assert ev.separator == "bs_roformer_sw"


def test_compare_identical(tmp_path, song, capsys):
    mix, _ = song
    save_audio(tmp_path / "a.wav", mix, 44100)
    save_audio(tmp_path / "b.wav", mix, 44100)
    assert main(["compare", str(tmp_path / "a.wav"), str(tmp_path / "b.wav")]) == 0
    res = json.loads(capsys.readouterr().out)
    assert res["identical"] is True and res["snr_db"] == "inf"
