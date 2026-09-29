"""End-to-end through the CLI with files on disk (no GPU): stems exported 'UVR-style' next to a mixture."""

import json

import numpy as np
import pytest

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

    `ensemble` + overlap 4 is the best measured combination (docs/04 §11, §14.5). These asserts exist because each of
    them was once silently the fast-but-worse option: the CLI defaulted to single-pass SW with no TTA, and `midi` built
    its own Namespace that pinned overlap 2 / tta False.

    TTA is NOT part of it as of 2026-09-28, and the distinction this test guards matters: it was dropped because it
    FAILED the measurement gate (+0.008 vocals / +0.015 drums at overlap 4, two of three gated stems under the +0.02 dB
    floor -- docs/04 §14.15), not because it was slow. Speed is still never the reason the default changes.
    """
    from cleansplit.cli.main import QUALITY_DEFAULT, QUALITY_TIERS, build_parser

    for cmd in ("separate", "analyze"):
        a = build_parser().parse_args([cmd, "x.wav"])
        assert (a.separator, a.overlap, a.tta, a.fp16) == ("ensemble", 4, False, False), cmd
        # and tied to the tier table, so the two cannot drift apart silently
        sep, ov, tta, _, _ = QUALITY_TIERS[QUALITY_DEFAULT]
        assert (a.separator, a.overlap, a.tta) == (sep, ov, tta), cmd
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


def test_every_registered_separator_is_selectable_from_the_cli():
    """SCNet XL IHF was registered and integrated but hardcoded out of the CLI's choices, so nobody could name it.

    Deriving the choices from the registry means registering a separator is enough to expose it. `stem_folder` is the
    one deliberate exclusion: it reads pre-separated stems from a directory, for evaluation, and takes no audio input.
    """
    from cleansplit.cli.main import build_parser, separator_choices
    from cleansplit.separation import registry

    names = registry.names() if hasattr(registry, "names") else list(registry._FACTORIES)
    assert set(separator_choices()) == {n for n in names if n != "stem_folder"}
    assert "scnet_xl_ihf" in separator_choices()

    p = build_parser()
    for name in separator_choices():
        args = p.parse_args(["separate", "song.wav", "--separator", name])
        assert args.separator == name


def test_quality_tiers_are_the_measured_recipes_and_explicit_flags_win():
    """--quality is sugar over measured flag combinations; it must never silently beat what the user typed.

    The three defaults asserted here are the ones docs/04 sections 14.5 and 14.8 measured, and `evaluate` must keep its
    pin to single-pass SW (changing it would silently change what a published number in docs/04 means), which is why
    it does not accept --quality at all.
    """
    from cleansplit.cli.main import QUALITY_TIERS, build_parser, quality_note, resolve_quality

    def r(argv):
        a = build_parser().parse_args(argv)
        resolve_quality(a)
        return a.separator, a.overlap, a.tta

    assert r(["separate", "x.wav"]) == ("ensemble", 4, False)
    assert r(["separate", "x.wav", "--quality", "best"]) == ("ensemble", 4, False)
    assert r(["separate", "x.wav", "--quality", "fast"]) == ("bs_roformer_sw", 2, False)
    assert r(["separate", "x.wav", "--quality", "balanced"]) == ("ensemble", 2, False)
    # an explicit flag beats the tier, in both directions
    assert r(["separate", "x.wav", "--quality", "fast", "--overlap", "8"]) == ("bs_roformer_sw", 8, False)
    assert r(["separate", "x.wav", "--quality", "fast", "--tta"]) == ("bs_roformer_sw", 2, True)
    assert r(["separate", "x.wav", "--quality", "best", "--tta"]) == ("ensemble", 4, True)
    # evaluate keeps its pinned separator and refuses the tier flag
    assert r(["evaluate", "x.wav"])[0] == "bs_roformer_sw"
    with pytest.raises(SystemExit):
        build_parser().parse_args(["evaluate", "x.wav", "--quality", "fast"])
    # the advertised compute cost is the pass count: best/fast is exactly 4x, best/balanced exactly 2x. Both halved
    # when TTA left the default (docs/04 section 14.15); quality_note() quotes these ratios, so they are asserted.
    assert QUALITY_TIERS["best"][3] == 4 * QUALITY_TIERS["fast"][3]
    assert QUALITY_TIERS["best"][3] == 2 * QUALITY_TIERS["balanced"][3]
    assert "4x cheaper than best" in quality_note("fast")
    assert "2x cheaper than best" in quality_note("balanced")


def test_tta_reaches_the_ensemble_so_the_balanced_tier_is_real():
    """--no-tta was silently dropped for `ensemble`, exactly as --overlap once was (docs/04 section 14.5).

    That bug would have made the `balanced` tier a false advertisement: it would have reported 4 units of compute while
    actually running SW three times, at 16. This asserts the flag reaches the constructed separator.
    """
    import argparse

    from cleansplit.cli.main import _make_separator

    seen = {}

    def fake_create(name, **kw):
        seen.update({"name": name, **kw})
        return object()

    from cleansplit.separation import registry

    real = registry.create
    registry.create = fake_create
    try:
        for tta in (True, False):
            seen.clear()
            _make_separator(argparse.Namespace(
                separator="ensemble", device="cpu", overlap=2, tta=tta, checkpoint=None,
                chunk_size=None, fp16=False, stems_dir=None))
            assert seen["name"] == "ensemble"
            assert seen["num_overlap"] == 2
            assert seen["tta"] is tta, f"tta={tta} did not reach the ensemble"
    finally:
        registry.create = real


def test_the_quality_flag_help_matches_the_tier_table() -> None:
    """argparse help is prose about measurements, so it drifts like any other prose.

    It did: after TTA left the default on 2026-09-28 the --quality help still advertised "ensemble + TTA at overlap 4,
    16 units" and "8x cheaper", and --separator still described vocals=mean(SW+TTA, ep317). quality_note() was updated
    and tested; this text was not, and nothing checked it. The unit counts below are read from QUALITY_TIERS rather
    than written out, so the table stays the single source of truth.
    """
    from cleansplit.cli.main import QUALITY_TIERS, build_parser

    help_text = _quality_help(build_parser())
    best, fast = QUALITY_TIERS["best"][3], QUALITY_TIERS["fast"][3]
    assert f"{best} units of compute" in help_text
    assert f"{fast} units" in help_text
    assert f"{best // fast}x cheaper" in help_text
    # The default recipe has TTA off, so the help must not advertise it as part of the default.
    assert QUALITY_TIERS["best"][2] is False
    assert "+ TTA" not in help_text


def _quality_help(parser) -> str:
    """The --quality and --separator help strings from the `separate` subparser, concatenated."""
    sub = next(a for a in parser._actions if hasattr(a, "choices") and isinstance(a.choices, dict))
    sep_parser = sub.choices["separate"]
    return " ".join(a.help or "" for a in sep_parser._actions if a.dest in ("quality", "separator"))


def test_quality_note_states_a_range_not_a_single_db_number():
    """docs/04 section 14.11: best-vs-fast runs from about -0.09 to +5.12 dB per song, so a fixed figure would mislead."""
    from cleansplit.cli.main import quality_note

    best, fast = quality_note("best"), quality_note("fast")
    assert "5.12" in best and "-0.09" in best and "median" in best.lower()
    assert "14.11" in fast
