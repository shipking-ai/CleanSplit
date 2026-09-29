"""CleanSplit command line.

  cleansplit doctor
  cleansplit separate song.wav
  cleansplit analyze  song.wav [--stems-dir DIR]
  cleansplit restore  song.wav [--restorer residual_reallocation]
  cleansplit evaluate --synthetic | song.wav --reference-stems DIR
  cleansplit compare  original.wav restored.wav
  cleansplit midi     song.wav [--mode stems|mix]
  cleansplit ui
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path


def _print_json(obj):
    from ..analysis.pipeline import _json_safe

    print(json.dumps(_json_safe(obj), indent=2))


def _load_config(path):
    from ..config.settings import AnalysisConfig

    if not path:
        return AnalysisConfig()
    return AnalysisConfig.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _make_separator(args):
    from ..separation import registry

    if getattr(args, "stems_dir", None):
        return registry.create("stem_folder", folder=args.stems_dir)
    if args.separator in ("ensemble", "ensemble_demucs"):
        # tta must be forwarded: it controls whether the ensemble's SW member runs 3 passes or 1, which is the whole
        # difference between the `best` and `balanced` quality tiers. It was omitted here, so --no-tta was silently
        # ignored for the default separator -- the same bug --overlap had (docs/04 section 14.5).
        return registry.create(args.separator, device=args.device, num_overlap=args.overlap, tta=args.tta)
    if args.separator == "htdemucs_ft":
        return registry.create("htdemucs_ft", device=args.device)
    if args.separator == "mdx23c_instvoc_hq":  # vocals + instrumental only; SW's chunk/fp16/tta options do not apply
        return registry.create("mdx23c_instvoc_hq", device=args.device)
    return registry.create(
        args.separator,
        checkpoint=args.checkpoint,
        device=args.device,
        chunk_size=args.chunk_size,
        num_overlap=args.overlap,
        fp16=args.fp16,
        tta=args.tta,
    )


def cmd_doctor(args):
    from ..models import checkpoints
    from ..models.device import probe

    info = probe().to_dict()
    report = {"device": info, "ffmpeg": shutil.which("ffmpeg"), "ffprobe": shutil.which("ffprobe")}
    try:
        ck, cfg = checkpoints.find_checkpoint(checkpoints.BS_ROFO_SW_FIXED, args.checkpoint)
        report["bs_roformer_sw"] = {"checkpoint": str(ck), "config": str(cfg)}
        try:
            report["bs_roformer_sw"]["sha256_verified"] = checkpoints.verify(checkpoints.BS_ROFO_SW_FIXED, ck)
        except ValueError as e:
            report["bs_roformer_sw"]["sha256_error"] = str(e)
    except FileNotFoundError as e:
        report["bs_roformer_sw"] = {"error": str(e)}
    if info["cuda_available"] and info["vram_total_mb"] and info["vram_total_mb"] < 6000:
        report["note"] = "less than 6 GB VRAM: use --fp16 or a smaller --chunk-size if separation runs out of memory"
    _print_json(report)
    return 0


def variant_root(args) -> Path:
    """The folder holding one separator's songs: `<out>/<separator>`.

    The app has always used this layout -- `ui/service.py` passes `out_root / separator` into `run_analyze`, and
    `songs()` scans `<out>/<variant>/<slug>` -- and `cleansplit midi` already assumed it. `separate`, `analyze` and
    `evaluate` wrote `<out>/<slug>` instead, so a split made from the command line was invisible in the app and the two
    halves of the project disagreed about where a song lives. Nested is now the single layout everywhere.
    """
    return Path(args.out) / variant_name(args)


def variant_name(args) -> str:
    """The separator sub-folder's name: the separator that ACTUALLY ran, not merely the one named on the flag.

    `--stems-dir` replaces the separator entirely -- `_make_separator` returns the `stem_folder` reader and ignores
    `--separator` -- so filing such a run under `ensemble` would attribute someone else's stems to a model that never
    ran, in the folder name the app displays as the variant.
    """
    return "stem_folder" if getattr(args, "stems_dir", None) else args.separator


def song_out_dir(args) -> Path:
    """`<out>/<separator>/<slug>` for this invocation's input."""
    from ..analysis.pipeline import song_slug

    return variant_root(args) / song_slug(args.input)


def _existing_song_dir(args) -> Path:
    """`song_out_dir`, but name the pre-2026-09-29 flat folder explicitly if that is where the song actually is.

    Silently falling back would leave two layouts alive forever and make "where is my song" unanswerable; an error that
    prints the exact move command is more useful than either that or a bare not-found.
    """
    from ..analysis.pipeline import song_slug

    d = song_out_dir(args)
    if (d / "stems").is_dir():
        return d
    legacy = Path(args.out) / song_slug(args.input)
    if (legacy / "stems").is_dir():
        raise SystemExit(
            f"{d} not found, but {legacy} exists.\n"
            f"Songs now live under <out>/<separator>/<slug> so the CLI and the app agree. Move it with:\n"
            f'  mv "{legacy}" "{d}"')
    return d


def cmd_separate(args):
    from ..analysis.pipeline import audio_digest, write_json
    from ..audio.conform import validate_and_conform
    from ..audio.io import load_audio, save_audio

    sep = _make_separator(args)
    d = load_audio(args.input)
    O, rep = validate_and_conform(d.audio, d.sample_rate, sep.sample_rate, sep.channels)
    res = sep.separate(O, sep.sample_rate)
    out = song_out_dir(args)
    save_audio(out / "original.wav", O, sep.sample_rate)
    for name, s in res.stems.items():
        save_audio(out / "stems" / f"{name}.wav", s, sep.sample_rate)
    write_json(
        out / "stems" / "manifest.json",
        {"key": {"input_sha256": audio_digest(O), "separator": sep.cache_key()}, "separator": res.separator, "metadata": res.metadata, "chunk_starts": res.chunk_starts, "chunk_size": res.chunk_size},
    )
    _print_json({"output_dir": str(out), "validation": rep.to_dict(), "separator": res.metadata})
    return 0


def cmd_analyze(args):
    from ..analysis.pipeline import run_analyze

    sep = _make_separator(args)
    cfg = _load_config(args.config)
    if args.detectors:
        cfg.enabled_detectors = tuple(args.detectors.split(","))
    result = run_analyze(args.input, variant_root(args), sep, cfg, reuse_stems=not args.no_reuse)
    rep = result["report"]
    summary = {
        "output_dir": result["output_dir"],
        "reconstruction": rep["reconstruction_headline"],
        "regions_total": rep["artifacts"]["regions_total"],
        "regions_by_stem": rep["artifacts"]["regions_by_stem"],
        "regions_by_type": rep["artifacts"]["regions_by_type"],
        "warnings": rep["warnings"],
        "seconds_total": rep["seconds_total"],
    }
    _print_json(summary)
    return 0


def cmd_restore(args):
    from ..restoration.pipeline import run_restore

    result = run_restore(
        args.input, variant_root(args), restorer_name=args.restorer, min_confidence=args.min_confidence,
        tolerance_db=args.tolerance_db, max_iterations=args.max_iterations, max_regions=args.max_regions,
    )
    _print_json(result)
    return 0


def cmd_evaluate(args):
    from ..analysis.pipeline import write_json

    if args.synthetic:
        from ..metrics.detection_benchmark import run_benchmark

        cfg = _load_config(args.config)
        if args.detectors:
            cfg.enabled_detectors = tuple(args.detectors.split(","))
        res = run_benchmark(seeds=tuple(args.seeds), config=cfg, log=lambda m: print(m, file=sys.stderr))
        out = Path(args.out) / "_benchmarks" / "synthetic_detection.json"
        write_json(out, res)
        _print_json({"written": str(out), "clean": res["clean"], "recall": {k: v["recall"] for k, v in res["scenarios"].items()}})
        return 0
    if args.restoration_experiment:
        from ..metrics.restoration_experiment import run_experiment

        sep = _make_separator(args)
        res = run_experiment(
            sep, restorers=tuple(args.restorers.split(",")), seeds=tuple(args.seeds), duration_s=args.duration,
            max_regions=args.max_regions, log=lambda m: print(m, file=sys.stderr, flush=True),
        )
        tag = "_".join(args.restorers.split(",")) + "_seeds" + "-".join(map(str, args.seeds))
        out = Path(args.out) / "_benchmarks" / f"restoration_experiment_{sep.name}_{tag}.json"
        write_json(out, res)
        summary = {
            e["song"]: {r: {"accepted": v["accepted"], "rejected": v["rejected"],
                            "pooled_change_db_accepted": v["pooled_change_db_accepted"],
                            "pooled_change_db_all_proposals": v["pooled_change_db_all_proposals"]}
                        for r, v in e["restorers"].items()}
            for e in res["results"]
        }
        _print_json({"written": str(out), "summary": summary})
        return 0
    if not args.input or not args.reference_stems:
        print("evaluate needs --synthetic, or INPUT with --reference-stems DIR", file=sys.stderr)
        return 2
    from ..metrics.reference_eval import evaluate_against_reference

    out = _existing_song_dir(args)
    res = evaluate_against_reference(out / "stems", args.reference_stems)
    write_json(out / "analysis" / "reference_evaluation.json", res)
    _print_json(res)
    return 0


def cmd_ui(args):
    if args.server:
        from ..ui.server import serve

        print(f"CleanSplit UI on http://127.0.0.1:{args.port} (Ctrl+C to stop)")
        serve(port=args.port, out_root=args.out)
        return 0
    from ..ui.desktop import launch

    launch(out_root=args.out, port=args.port or None, dev=args.dev)
    return 0


def cmd_midi(args):
    """Separate (or reuse the cached split), then transcribe to MIDI."""
    from ..transcription.pipeline import transcribe_split

    song_dir = song_out_dir(args)
    if not (song_dir / "stems").is_dir() or not (song_dir / "original.wav").is_file():
        print(f"no split at {song_dir}; separating with {args.separator} first", file=sys.stderr)
        # `out` is passed through unchanged now that cmd_separate nests by separator itself; the old call rewrote it to
        # `<out>/<separator>` to compensate for the flat layout, which would double-nest today.
        # overlap and tta come from the tier table rather than being written out: this used to hardcode tta=True, so
        # `midi` silently separated with the 16-unit recipe that section 14.15 rejected as the default.
        _, ov, tta, _, _ = QUALITY_TIERS[QUALITY_DEFAULT]
        sep_args = argparse.Namespace(**{**vars(args), "stems_dir": None, "checkpoint": None,
                                         "chunk_size": None, "overlap": ov, "fp16": False, "tta": tta})
        cmd_separate(sep_args)
    report = transcribe_split(song_dir, mode=args.mode, model=args.model, piano=args.piano,
                              log=lambda m: print(m, file=sys.stderr, flush=True))
    _print_json({k: report[k] for k in ("files", "tempo_bpm", "notes_by_stem", "instruments_by_stem", "skipped_stems",
                                        "seconds_total", "caveats")})
    return 0


def cmd_compare(args):
    from ..metrics.compare import compare_files

    _print_json(compare_files(args.a, args.b))
    return 0


QUALITY_TIERS = {
    # name: (separator, num_overlap, tta, passes_per_chunk * overlap, one-line description)
    # `passes` is the exact relative cost from docs/04 section 14.8: forward passes per chunk times the overlap factor,
    # since roformer.py sets step = chunk // num_overlap. It is countable, not timed, so it holds on any machine.
    "fast": ("bs_roformer_sw", 2, False, 2, "single-pass SW at overlap 2"),
    "balanced": ("ensemble", 2, False, 4, "SW averaged with ep317 at overlap 2, no TTA"),
    # TTA was removed from `best` on 2026-09-28: at overlap 4 inside this ensemble it buys +0.008 dB vocals,
    # +0.015 drums and +0.033 bass for 2x the compute, so two of the three gated stems sit under the +0.02 dB
    # adoption floor and the gate fails (docs/04 section 14.15). Pass --tta to get the old 16-unit recipe back.
    "best": ("ensemble", 4, False, 8, "SW averaged with ep317 at overlap 4, no TTA"),
}
QUALITY_DEFAULT = "best"

# Every command that reads or writes a song uses `<out>/<separator>/<slug>`, which is the layout the app scans. Said
# once here because six subparsers repeat the flag.
OUT_HELP = "root output folder; songs are written to <out>/<separator>/<slug>, the layout the app reads"


def quality_note(tier: str) -> str:
    """What each tier actually costs and actually buys, in measured terms only.

    Deliberately does NOT promise a fixed dB difference. docs/04 section 14.11: best-vs-fast on vocals ranges from
    about -0.09 dB to +5.12 dB across 20 MUSDB songs, so a single number on a tier label would be misleading on most
    songs in both directions. The median is given as a median and the range is given as a range.
    """
    sep, ov, tta, passes, what = QUALITY_TIERS[tier]
    if tier == "fast":
        return (f"quality=fast: {what}, {passes} units of compute (4x cheaper than best). "
                f"Typically close to best, but on some songs several dB worse -- see docs/04 section 14.11.")
    if tier == "balanced":
        return (f"quality=balanced: {what}, {passes} units -- 2x cheaper than best. Recovers 96% of best's vocal gain "
                f"(+0.442 of +0.460 dB over fast, 18/20 songs; only 0.018 dB behind best). Drums and bass are "
                f"IDENTICAL to fast -- measured at +0.0000 dB, 0/20 songs -- because the second model is a vocal "
                f"model: best buys those with overlap alone (docs/04 sections 14.12, 14.15).")
    return (f"quality=best: {what}, {passes} units of compute. Median +0.460 dB vocals over fast on 20 MUSDB songs "
            f"(18/20), range about -0.09 to +5.12 dB per song. Also +0.095 dB drums (18/20) and +0.061 dB bass "
            f"(16/20), which balanced does not get. Adding --tta doubles this to 16 units for +0.008 vocals / "
            f"+0.015 drums / +0.033 bass -- rejected as the default by the adoption floor (docs/04 section 14.15).")


def resolve_quality(args) -> None:
    """Apply the --quality tier, but only where the user did not say otherwise.

    `--separator`, `--overlap` and `--tta/--no-tta` all default to None so that "not given" is distinguishable from
    "given the same value as the tier". An explicit flag therefore always wins over the tier, and a subcommand that
    pins a separator for protocol reasons (evaluate) keeps its pin when no tier is requested.
    """
    tier = getattr(args, "quality", None)
    base_sep = getattr(args, "_default_separator", QUALITY_TIERS[QUALITY_DEFAULT][0])
    if tier:
        sep, ov, tta, _passes, _what = QUALITY_TIERS[tier]
    else:
        _, ov, tta, _, _ = QUALITY_TIERS[QUALITY_DEFAULT]
        sep = base_sep
    if getattr(args, "separator", None) is None:
        args.separator = sep
    if getattr(args, "overlap", None) is None:
        args.overlap = ov
    if getattr(args, "tta", None) is None:
        args.tta = tta


class _Parser(argparse.ArgumentParser):
    """Resolves --quality as part of parsing, so a parsed Namespace always carries the EFFECTIVE settings.

    Without this, `--separator/--overlap/--tta` would read back as None to every caller that parses without going
    through main() -- including cmd_analyze, which rebuilds a Namespace to run separation itself, and every test. The
    sentinel Nones exist only to tell "not given" from "given the tier's value"; nobody downstream should ever see one.
    """

    def parse_args(self, args=None, namespace=None):
        ns = super().parse_args(args, namespace)
        if hasattr(ns, "_default_separator"):
            resolve_quality(ns)
        return ns


def separator_choices() -> list[str]:
    """Every registered separator a user can actually name, so registering one is enough to expose it.

    This was hardcoded, and SCNet XL IHF was consequently unreachable from the command line for as long as it had been
    registered. `stem_folder` is excluded because it reads pre-separated stems from a directory and is an evaluation
    helper, not a separator.
    """
    from cleansplit.separation import registry

    names = registry.names() if hasattr(registry, "names") else list(registry._FACTORIES)
    return [n for n in sorted(names) if n != "stem_folder"]


def build_parser() -> argparse.ArgumentParser:
    p = _Parser(prog="cleansplit", description="Stem separation + artifact-aware, mixture-consistent analysis")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    def sep_opts(sp, default_separator="ensemble", allow_quality=True):
        sp.set_defaults(_default_separator=default_separator)
        if allow_quality:
            sp.add_argument("--quality", choices=sorted(QUALITY_TIERS), default=None,
                            help="pick a measured speed/quality point instead of setting the flags by hand. "
                                 "best (the default): the ensemble at overlap 4, no TTA, 8 units of compute. "
                                 "fast: single-pass SW at overlap 2, 2 units, so 4x cheaper. Compute is counted in "
                                 "forward passes, not timed (docs/04 section 14.8). No tier advertises a fixed dB "
                                 "cost, because best-vs-fast ranges from -0.09 to +5.12 dB per song (section 14.11). "
                                 "Any explicit --separator/--overlap/--tta overrides the tier")
        sp.add_argument("--separator", default=None, choices=separator_choices(),
                        help="ensemble (default): vocals=mean(SW, ep317), other=remainder. The best measured "
                             "quality (MUSDB18-HQ, 20 songs: +0.45 dB vocals paired median vs SW, better on 18/20; "
                             "+1.12 dB vocal SAR, the largest artifact gain measured -- docs/04 sections 11 and 14). "
                             "Cost, counting forward passes per chunk (docs/04 section 14.8): the ensemble is 4 "
                             "passes (SW three times for TTA, plus ep317 once), so at the same --overlap and --tta it "
                             "is about 1.3x bs_roformer_sw, and about 8x the fastest setting "
                             "(--separator bs_roformer_sw --no-tta --overlap 2). Per-song quality differs far more "
                             "than the median suggests: usually near-identical to the fast path, up to 5 dB better on "
                             "some songs (section 14.11). ensemble_demucs also averages drums/bass with HTDemucs_ft "
                             "and is measurably WORSE on both axes (drums -0.55 dB SNR on 20/20 songs, -1.08 dB SAR) "
                             "except as insurance when SW misses an instrument outright; scnet_xl_ihf is a four-stem "
                             "convolutional model kept available for comparison")
        sp.add_argument("--checkpoint", help="path to BS-Rofo-SW-Fixed.ckpt (default: search UVR install)")
        sp.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N")
        sp.add_argument("--chunk-size", type=int, default=None, help="samples per inference chunk (default: model config, 588800)")
        sp.add_argument("--overlap", type=int, default=None,
                        help="chunk overlap factor (step = chunk/overlap); default 4, the best measured. Cleaner on "
                             "every stem than the old default of 2 (MUSDB18-HQ 20 songs: +0.05..+0.09 dB full-band "
                             "SNR, better on 14-19/20 songs, and +0.03..+0.09 dB SAR -- docs/04 section 14.5) for "
                             "about 2x the GPU time. Pass --overlap 2 to halve the time for a sub-0.1 dB loss. "
                             "Applies to the RoFormer models, including inside the ensembles")
        sp.add_argument("--fp16", action="store_true", help="half-precision autocast (less VRAM; adds numerical noise to residuals)")
        sp.add_argument("--tta", dest="tta", action="store_true", default=None,
                        help="average original / channel-swapped / polarity-inverted passes, at 3x the GPU time. OFF by "
                             "default since 2026-09-28: it is better on all four stems and never worse on any of 20 "
                             "MUSDB songs, but only by +0.008 vocals / +0.015 drums / +0.033 bass at overlap 4, so two "
                             "of the three gated stems fall under the +0.02 dB adoption floor (docs/04 section 14.15). "
                             "Applies to SW inside the ensembles too, taking the default recipe from 8 units to 16")
        sp.add_argument("--no-tta", dest="tta", action="store_false",
                        help="single pass instead of the 3-pass average: 3x faster, slightly worse on every stem")

    sp = sub.add_parser("doctor", help="check GPU, ffmpeg and model files")
    sp.add_argument("--checkpoint")
    sp.set_defaults(fn=cmd_doctor)

    sp = sub.add_parser("separate", help="six-stem separation only")
    sp.add_argument("input")
    sp.add_argument("--out", default="outputs", help=OUT_HELP)
    sep_opts(sp)
    sp.set_defaults(fn=cmd_separate)

    sp = sub.add_parser("analyze", help="separate (or load stems), reconstruct, residual, artifact map, report")
    sp.add_argument("input")
    sp.add_argument("--out", default="outputs", help=OUT_HELP)
    sp.add_argument("--stems-dir", help="analyze existing stems (e.g. exported from UVR) instead of separating")
    sp.add_argument("--no-reuse", action="store_true", help="re-run separation even if cached stems match")
    sp.add_argument("--config", help="JSON file with AnalysisConfig overrides")
    sp.add_argument("--detectors", help="comma-separated detector list (overrides config)")
    sep_opts(sp)
    sp.set_defaults(fn=cmd_analyze)

    sp = sub.add_parser("restore", help="region-restricted restoration with mixture-consistency acceptance test")
    sp.add_argument("input")
    sp.add_argument("--out", default="outputs", help=OUT_HELP)
    # restore reads what `analyze` wrote, so it has to be told which separator's folder to look in. It does no
    # separation of its own, which is why it takes only this one flag from the separation set.
    # `stem_folder` is in these choices although it is not in separator_choices(): for restore the flag names a
    # FOLDER to work in, not a model to run, and `analyze --stems-dir` files its output under exactly that name.
    # Without it, a run started from pre-separated stems could not be restored at all.
    sp.add_argument("--separator", default=QUALITY_TIERS[QUALITY_DEFAULT][0],
                    choices=sorted([*separator_choices(), "stem_folder"]),
                    help="which separator's output folder to restore in; must match the `analyze` run")
    sp.add_argument("--restorer", default="residual_reallocation")
    sp.add_argument("--min-confidence", type=float, default=0.6)
    sp.add_argument("--tolerance-db", type=float, default=0.05, help="max allowed increase of local mixture error (dB)")
    sp.add_argument("--max-iterations", type=int, default=1, help="restore -> re-analyze passes (stops early if nothing is accepted)")
    sp.add_argument("--max-regions", type=int, default=None, help="only the N highest-confidence regions per pass")
    sp.set_defaults(fn=cmd_restore)

    sp = sub.add_parser("evaluate", help="synthetic detection benchmark, or separation vs reference stems")
    sp.add_argument("input", nargs="?")
    sp.add_argument("--synthetic", action="store_true", help="detection benchmark on injected corruptions")
    sp.add_argument("--restoration-experiment", action="store_true", help="known stems -> real separator -> restore -> score vs truth")
    sp.add_argument("--restorers", default="identity,residual_reallocation,region_wiener")
    sp.add_argument("--duration", type=float, default=20.0)
    sp.add_argument("--max-regions", type=int, default=200)
    sp.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    sp.add_argument("--reference-stems")
    sp.add_argument("--out", default="outputs", help=OUT_HELP)
    sp.add_argument("--config")
    sp.add_argument("--detectors")
    sep_opts(sp, default_separator="bs_roformer_sw", allow_quality=False)  # the experiment protocols in docs/04 were run on single-pass SW; changing this default would silently change what a published number means
    sp.set_defaults(fn=cmd_evaluate)

    sp = sub.add_parser("ui", help="open the CleanSplit desktop app")
    sp.add_argument("--out", default="outputs", help="folder the app reads and writes (one sub-folder per separator)")
    sp.add_argument("--port", type=int, default=8770, help="0 = pick a free port")
    sp.add_argument("--server", action="store_true", help="run the server only (open the URL in a browser yourself)")
    sp.add_argument("--dev", action="store_true", help="open with developer tools")
    sp.set_defaults(fn=cmd_ui)

    sp = sub.add_parser("midi", help="audio -> MIDI: separate, then transcribe each stem (MuScriptor; Transkun for piano)")
    sp.add_argument("input")
    sp.add_argument("--out", default="outputs", help=OUT_HELP)
    sp.add_argument("--separator", default="ensemble", choices=["ensemble", "ensemble_demucs", "bs_roformer_sw"])
    sp.add_argument("--device", default="auto")
    sp.add_argument("--mode", default="stems", choices=["stems", "mix"],
                    help="stems (default, measured better on 16/20 tracks): each separated stem with an instrument "
                         "constraint; mix: the whole song at once")
    sp.add_argument("--model", default="medium", choices=["small", "medium", "large"], help="MuScriptor size")
    sp.add_argument("--piano", default="transkun", choices=["transkun", "muscriptor"],
                    help="piano stem model; transkun measured +0.19 piano F1 over muscriptor and adds velocities")
    sp.set_defaults(fn=cmd_midi)

    sp = sub.add_parser("compare", help="objective comparison of two audio files")
    sp.add_argument("a")
    sp.add_argument("b")
    sp.set_defaults(fn=cmd_compare)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)   # _Parser has already applied any --quality tier
    if getattr(args, "quality", None):
        print(quality_note(args.quality), file=sys.stderr)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
