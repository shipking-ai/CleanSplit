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
        return registry.create(args.separator, device=args.device, num_overlap=args.overlap)
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


def cmd_separate(args):
    from ..analysis.pipeline import audio_digest, song_slug, write_json
    from ..audio.conform import validate_and_conform
    from ..audio.io import load_audio, save_audio

    sep = _make_separator(args)
    d = load_audio(args.input)
    O, rep = validate_and_conform(d.audio, d.sample_rate, sep.sample_rate, sep.channels)
    res = sep.separate(O, sep.sample_rate)
    out = Path(args.out) / song_slug(args.input)
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
    result = run_analyze(args.input, args.out, sep, cfg, reuse_stems=not args.no_reuse)
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
        args.input, args.out, restorer_name=args.restorer, min_confidence=args.min_confidence,
        tolerance_db=args.tolerance_db, max_iterations=args.max_iterations, max_regions=args.max_regions,
    )
    _print_json(result)
    return 0


def cmd_evaluate(args):
    from ..analysis.pipeline import song_slug, write_json

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

    out = Path(args.out) / song_slug(args.input)
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
    from ..analysis.pipeline import song_slug
    from ..transcription.pipeline import transcribe_split

    song_dir = Path(args.out) / args.separator / song_slug(args.input)
    if not (song_dir / "stems").is_dir() or not (song_dir / "original.wav").is_file():
        print(f"no split at {song_dir}; separating with {args.separator} first", file=sys.stderr)
        sep_args = argparse.Namespace(**{**vars(args), "out": str(Path(args.out) / args.separator), "stems_dir": None,
                                         "checkpoint": None, "chunk_size": None, "overlap": 4, "fp16": False, "tta": True})
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cleansplit", description="Stem separation + artifact-aware, mixture-consistent analysis")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    def sep_opts(sp, default_separator="ensemble"):
        sp.add_argument("--separator", default=default_separator,
                        choices=["bs_roformer_sw", "ensemble", "ensemble_demucs", "bs_roformer_ep317", "htdemucs_ft", "mdx23c_instvoc_hq"],
                        help="ensemble (default): vocals=mean(SW+TTA, ep317), other=remainder. The best measured "
                             "quality (MUSDB18-HQ, 20 songs: +0.45 dB vocals paired median vs SW, better on 18/20; "
                             "+1.12 dB vocal SAR, the largest artifact gain measured -- docs/04 sections 11 and 14). "
                             "About 4 model passes, so roughly 4x slower than bs_roformer_sw, which is the fast "
                             "single-pass option; ensemble_demucs also averages drums/bass with HTDemucs_ft and is "
                             "measurably WORSE on both axes (drums -0.55 dB SNR on 20/20 songs, -1.08 dB SAR) except "
                             "as insurance when SW misses an instrument outright")
        sp.add_argument("--checkpoint", help="path to BS-Rofo-SW-Fixed.ckpt (default: search UVR install)")
        sp.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N")
        sp.add_argument("--chunk-size", type=int, default=None, help="samples per inference chunk (default: model config, 588800)")
        sp.add_argument("--overlap", type=int, default=4,
                        help="chunk overlap factor (step = chunk/overlap); default 4, the best measured. Cleaner on "
                             "every stem than the old default of 2 (MUSDB18-HQ 20 songs: +0.05..+0.09 dB full-band "
                             "SNR, better on 14-19/20 songs, and +0.03..+0.09 dB SAR -- docs/04 section 14.5) for "
                             "about 2x the GPU time. Pass --overlap 2 to halve the time for a sub-0.1 dB loss. "
                             "Applies to the RoFormer models, including inside the ensembles")
        sp.add_argument("--fp16", action="store_true", help="half-precision autocast (less VRAM; adds numerical noise to residuals)")
        sp.add_argument("--tta", dest="tta", action="store_true", default=True,
                        help="average original / channel-swapped / polarity-inverted passes. ON by default: better on "
                             "all four stems (docs/04 sections 6 and 14.1), at 3x the GPU time. Already used for SW "
                             "inside the ensembles, where this flag does not change the recipe")
        sp.add_argument("--no-tta", dest="tta", action="store_false",
                        help="single pass instead of the 3-pass average: 3x faster, slightly worse on every stem")

    sp = sub.add_parser("doctor", help="check GPU, ffmpeg and model files")
    sp.add_argument("--checkpoint")
    sp.set_defaults(fn=cmd_doctor)

    sp = sub.add_parser("separate", help="six-stem separation only")
    sp.add_argument("input")
    sp.add_argument("--out", default="outputs")
    sep_opts(sp)
    sp.set_defaults(fn=cmd_separate)

    sp = sub.add_parser("analyze", help="separate (or load stems), reconstruct, residual, artifact map, report")
    sp.add_argument("input")
    sp.add_argument("--out", default="outputs")
    sp.add_argument("--stems-dir", help="analyze existing stems (e.g. exported from UVR) instead of separating")
    sp.add_argument("--no-reuse", action="store_true", help="re-run separation even if cached stems match")
    sp.add_argument("--config", help="JSON file with AnalysisConfig overrides")
    sp.add_argument("--detectors", help="comma-separated detector list (overrides config)")
    sep_opts(sp)
    sp.set_defaults(fn=cmd_analyze)

    sp = sub.add_parser("restore", help="region-restricted restoration with mixture-consistency acceptance test")
    sp.add_argument("input")
    sp.add_argument("--out", default="outputs")
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
    sp.add_argument("--out", default="outputs")
    sp.add_argument("--config")
    sp.add_argument("--detectors")
    sep_opts(sp, default_separator="bs_roformer_sw")  # the experiment protocols in docs/04 were run on single-pass SW; changing this default would silently change what a published number means
    sp.set_defaults(fn=cmd_evaluate)

    sp = sub.add_parser("ui", help="open the CleanSplit desktop app")
    sp.add_argument("--out", default="outputs", help="folder the app reads and writes (one sub-folder per separator)")
    sp.add_argument("--port", type=int, default=8770, help="0 = pick a free port")
    sp.add_argument("--server", action="store_true", help="run the server only (open the URL in a browser yourself)")
    sp.add_argument("--dev", action="store_true", help="open with developer tools")
    sp.set_defaults(fn=cmd_ui)

    sp = sub.add_parser("midi", help="audio -> MIDI: separate, then transcribe each stem (MuScriptor; Transkun for piano)")
    sp.add_argument("input")
    sp.add_argument("--out", default="outputs")
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
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
