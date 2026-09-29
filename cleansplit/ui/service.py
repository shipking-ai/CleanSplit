"""State the UI needs: the songs on disk, a one-at-a-time job queue, and cached waveform peaks.

The GPU fits one separation at a time, so jobs are serialised in a single worker thread. Everything is derived from
the same ``outputs/`` layout the CLI writes, so the UI and the CLI stay interchangeable.
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..analysis.pipeline import audio_digest, run_analyze, song_slug, write_json
from ..artifacts.region import ArtifactMap
from ..audio.conform import validate_and_conform
from ..audio.io import load_audio, save_audio
from ..separation import registry

STEM_ORDER = ["vocals", "drums", "bass", "guitar", "piano", "other"]
# Order is the order the app offers them; the first is the default. "best measured" is not a slogan: see
# docs/04_results.md sections 11 and 14. Speeds assume overlap 4, the default since section 14.5 (about 2x the
# overlap-2 times these used to quote).
SEPARATORS = [
    {"id": "ensemble", "label": "Ensemble", "detail": "SW×3 + ep317, overlap 4 · best measured quality",
     "speed": "~7× song length"},
    {"id": "ensemble_demucs", "label": "Ensemble + Demucs",
     "detail": "measurably worse on drums and bass — only when a lane comes out empty", "speed": "~8× song length"},
    {"id": "bs_roformer_sw", "label": "SW (fast)", "detail": "single pass, no TTA, 6 stems · lowest quality",
     "speed": "~1× song length"},
]


@dataclass
class Job:
    id: str
    kind: str  # "separate" | "analyze"
    input_path: str
    separator: str
    out_root: str
    slug: str = ""
    state: str = "queued"  # queued | running | done | error
    stage: str = ""
    started: float | None = None
    finished: float | None = None
    error: str = ""
    log: list[str] = field(default_factory=list)

    def to_dict(self):
        d = asdict(self)
        d["elapsed"] = (self.finished or time.time()) - self.started if self.started else 0.0
        return d


class Service:
    def __init__(self, out_root: Path | str = "outputs"):
        self.out_root = Path(out_root)
        self.out_root.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, Job] = {}
        self.order: list[str] = []
        self._q: queue.Queue[str] = queue.Queue()
        self._lock = threading.Lock()
        self._listeners: list[queue.Queue] = []
        self._worker = threading.Thread(target=self._run_worker, daemon=True)
        self._worker.start()
        # Splits made by the CLI have no cached waveforms; compute them in the background so opening a song is instant.
        threading.Thread(target=self._warm_peaks, daemon=True).start()

    def _warm_peaks(self) -> None:
        for song in self.songs():
            for stem in ["original"] + song["stems"]:
                with contextlib.suppress(Exception):
                    self.peaks(song["variant"], song["slug"], stem)
            self._emit({"type": "peaks", "song": song["id"]})

    # ---- events (server-sent to the UI) ----
    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue()
        with self._lock:
            self._listeners.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._listeners:
                self._listeners.remove(q)

    def _emit(self, event: dict) -> None:
        with self._lock:
            listeners = list(self._listeners)
        for q in listeners:
            q.put(event)

    def _progress(self, job: Job, stage: str) -> None:
        job.stage = stage
        job.log.append(stage)
        self._emit({"type": "job", "job": job.to_dict()})

    # ---- jobs ----
    def submit(self, kind: str, input_path: str, separator: str) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, input_path=str(input_path), separator=separator,
                  out_root=str(self.out_root / separator), slug=song_slug(input_path))
        self.jobs[job.id] = job
        self.order.append(job.id)
        self._q.put(job.id)
        self._emit({"type": "job", "job": job.to_dict()})
        return job

    def submit_midi(self, variant: str, slug: str) -> Job:
        """Transcribe an existing split to MIDI (docs/04 section 13: stems + Transkun piano, the measured default)."""
        d = self.song_dir(variant, slug)
        job = Job(id=uuid.uuid4().hex[:12], kind="midi", input_path=str(d), separator=variant,
                  out_root=str(d.parent), slug=slug)
        self.jobs[job.id] = job
        self.order.append(job.id)
        self._q.put(job.id)
        self._emit({"type": "job", "job": job.to_dict()})
        return job

    def _run_worker(self) -> None:
        while True:
            job = self.jobs[self._q.get()]
            job.state, job.started = "running", time.time()
            try:
                if job.kind == "midi":
                    from ..transcription.pipeline import transcribe_split

                    self._progress(job, "transcribing to MIDI")
                    transcribe_split(job.input_path, log=lambda m: self._progress(job, m[:60]))
                    job.state = "done"
                    continue
                self._progress(job, "loading model")
                sep = registry.create(job.separator, device="auto") if job.separator in ("ensemble", "ensemble_demucs", "htdemucs_ft") \
                    else registry.create(job.separator, device="auto", tta=False)
                out = Path(job.out_root)
                if job.kind == "separate":
                    self._separate(job, sep, out)
                else:
                    self._progress(job, "analysing")
                    run_analyze(job.input_path, out, sep)
                job.state = "done"
            except Exception as exc:  # surfaced in the UI, never swallowed
                job.state, job.error = "error", f"{type(exc).__name__}: {exc}"
            finally:
                job.finished = time.time()
                self._emit({"type": "job", "job": job.to_dict()})
                self._emit({"type": "songs"})

    def _separate(self, job: Job, sep, out: Path) -> None:
        d = load_audio(job.input_path)
        self._progress(job, "preparing audio")
        O, rep = validate_and_conform(d.audio, d.sample_rate, sep.sample_rate, sep.channels)
        song_dir = out / job.slug
        save_audio(song_dir / "original.wav", O, sep.sample_rate)
        self._progress(job, f"separating ({sep.name})")
        res = sep.separate(O, sep.sample_rate)
        self._progress(job, "writing stems")
        for name, s in res.stems.items():
            save_audio(song_dir / "stems" / f"{name}.wav", s, sep.sample_rate)
        write_json(song_dir / "stems" / "manifest.json", {
            "key": {"input_sha256": audio_digest(O), "separator": sep.cache_key()},
            "separator": res.separator, "metadata": res.metadata,
            "chunk_starts": res.chunk_starts, "chunk_size": res.chunk_size,
            "source": str(Path(job.input_path).resolve()), "validation": rep.to_dict(),
        })
        self._progress(job, "computing waveforms")
        for name in [*list(res.stems), "original"]:
            self.peaks(job.separator, job.slug, name)

    # ---- songs on disk ----
    def songs(self) -> list[dict]:
        out = []
        for variant_dir in sorted(p for p in self.out_root.iterdir() if p.is_dir()):
            for song_dir in sorted(p for p in variant_dir.iterdir() if p.is_dir()):
                man = song_dir / "stems" / "manifest.json"
                if not man.is_file():
                    continue
                try:
                    meta = json.loads(man.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    continue
                stems = sorted((p.stem for p in (song_dir / "stems").glob("*.wav")), key=lambda s: (STEM_ORDER.index(s) if s in STEM_ORDER else 99, s))
                orig = song_dir / "original.wav"
                info = load_audio(orig).audio.shape[-1] / 44100 if orig.is_file() else 0
                out.append({
                    "id": f"{variant_dir.name}/{song_dir.name}",
                    "variant": variant_dir.name,
                    "slug": song_dir.name,
                    "title": song_dir.name.replace("_", " "),
                    "duration": info,
                    "stems": stems,
                    "source": meta.get("source", ""),
                    "separator": meta.get("separator", variant_dir.name),
                    "analysed": (song_dir / "analysis" / "artifact_map.json").is_file(),
                    "midi": (song_dir / "midi" / f"{song_dir.name}.mid").is_file(),
                    "created": man.stat().st_mtime,
                })
        return sorted(out, key=lambda s: -s["created"])

    # ---- the one path helper: every request-derived filesystem path in this class is built here ----
    #
    # Deliberately written with os.path rather than pathlib, to the shape CodeQL's py/path-injection query actually
    # recognises. That query is a two-state flow (PathInjectionQuery.qll): a tainted path starts NotNormalized, ONLY a
    # call to os.path.normpath / abspath / realpath moves it to NormalizedUnchecked, and ONLY a `.startswith(...)`
    # guard on its true branch clears it. `Path.resolve()` and `Path.is_relative_to()` model neither step, so the
    # guard that used to live here was invisible to the analyser however correct it was -- 18 alerts dismissed by hand,
    # re-filed every time these lines moved. One helper means one flow to reason about instead of a dozen.
    #
    # The prefix is `root + os.sep`, NOT `root`. A bare `startswith(root)` would satisfy CodeQL while accepting
    # `<root>-evil` as a child of `<root>`; the trailing separator is what makes this mean "strictly inside".
    #
    # The joined path is normalised with `normpath`, NOT `realpath`, and that distinction is load bearing. realpath
    # touches the filesystem, and on Windows it resolves an existing path by a different route than a missing one, so
    # racing it against a concurrent `os.replace` made the windows/py3.10 CI leg reject `peaks/vocals.300.json` as
    # outside a directory it was plainly inside (run 36588347468, caught by the 8-thread test below). normpath is pure
    # string arithmetic: it collapses `..` without a syscall, so it cannot race and cannot vary by platform or version.
    # Containment must never depend on a call that can fail transiently. `root` itself still gets realpath, once --
    # it always exists, so that call is stable.
    #
    # What this therefore enforces is LEXICAL containment, which is what defeats a `..` traversal arriving in a
    # request. It does not resolve symlinks, so a link planted inside the output tree that points out of it is not
    # caught here; doing that needs local write access to `outputs/`, which is the operator who already owns the disk.
    # The old `is_relative_to` line nominally covered that case, and dropping it is a deliberate trade against a
    # race that broke real runs -- stated plainly rather than quietly lost.
    @staticmethod
    def _confine(root: Path, *parts: str) -> Path:
        """`root` joined with `parts`, guaranteed to land lexically inside `root`, or FileNotFoundError.

        Every part is treated as untrusted. Callers that additionally need a single path COMPONENT -- no separators at
        all, because the value is interpolated into a filename -- call `_stem_name` first; this promises containment
        only.
        """
        root_s = os.path.realpath(root)
        full = os.path.normpath(os.path.join(root_s, *parts))
        if not full.startswith(root_s + os.sep):
            raise FileNotFoundError(f"{'/'.join(parts)!r} resolves outside {root_s}")
        return Path(full)

    def song_dir(self, variant: str, slug: str) -> Path:
        p = self._confine(self.out_root, variant, slug)
        if not p.is_dir():
            raise FileNotFoundError(f"unknown song {variant}/{slug}")
        return p

    @staticmethod
    def _stem_name(stem: str) -> str:
        """A stem name that is safe to interpolate into a path, or an error.

        `song_dir` already confines variant and slug, but `stem` reached `d / "stems" / f"{stem}.wav"` unchecked, so
        a value containing `..` escaped the song folder and `is_file()` happily confirmed whatever it landed on.
        Reaching it needs a literal `/` or `\\` to survive routing, which Starlette's path parameters do not pass
        through -- so this was a latent hole rather than a live one, and it is closed here rather than left depending
        on the router's escaping. `service.py` is a plain Python API; something other than that one route will call it.
        Found by CodeQL's py/path-injection on 2026-09-28, the first day code scanning could run on this repository.
        """
        if not stem or stem != Path(stem).name or stem in (".", "..") or any(c in stem for c in '/\\:'):
            raise FileNotFoundError(f"invalid stem name {stem!r}")
        return stem

    def audio_path(self, variant: str, slug: str, stem: str) -> Path:
        d = self.song_dir(variant, slug)
        stem = self._stem_name(stem)
        p = self._confine(d, "original.wav") if stem == "original" else self._confine(d, "stems", f"{stem}.wav")
        if not p.is_file():
            raise FileNotFoundError(p)
        return p

    # ---- waveform peaks (min/max per bucket), cached on disk ----
    def peaks(self, variant: str, slug: str, stem: str, buckets: int = 2400) -> dict:
        d = self.song_dir(variant, slug)
        stem = self._stem_name(stem)   # this one WRITES a cache file, so an unchecked name would create it anywhere
        cache = self._confine(d, "peaks", f"{stem}.{buckets}.json")
        if cache.is_file():
            try:
                return json.loads(cache.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass   # truncated or half-written: fall through and recompute, then replace it atomically below
        x = load_audio(self.audio_path(variant, slug, stem)).audio.astype(np.float32)
        mono = x.mean(axis=0)
        n = mono.size
        step = max(1, n // buckets)
        usable = (n // step) * step
        frames = mono[:usable].reshape(-1, step)
        mx, mn = frames.max(axis=1), frames.min(axis=1)
        rms = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1))
        data = {
            "duration": n / 44100,
            "peak": float(np.max(np.abs(mono))) if n else 0.0,
            "rms_dbfs": float(10 * np.log10(np.mean(mono.astype(np.float64) ** 2) + 1e-20)),
            "max": [round(float(v), 4) for v in mx],
            "min": [round(float(v), 4) for v in mn],
            "rms": [round(float(v), 4) for v in rms],
        }
        cache.parent.mkdir(parents=True, exist_ok=True)
        # Atomically, via a temp file in the same directory and os.replace. `write_text` creates the file and then
        # fills it, so a concurrent reader that got past `is_file()` above could read nothing or half of the ~58 kB
        # this produces -- and FastAPI serves these handlers from a threadpool, so two requests for the same song do
        # overlap. A flaky `json.decoder.JSONDecodeError: Expecting value` on the ubuntu/py3.10 CI leg (run
        # 36480845648, green on rerun) is what pointed here. os.replace is atomic on both POSIX and Windows, so a
        # reader now sees either the previous complete file or the new one.
        # mkstemp rather than a name built from pid and thread id: the OS guarantees uniqueness, so no collision
        # theory is needed to explain a stray file.
        fd, tmp_name = tempfile.mkstemp(dir=str(cache.parent), prefix=f"{cache.name}.", suffix=".tmp")
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(json.dumps(data))
            # Windows refuses os.replace onto a path any other handle has open (WinError 5), which a concurrent
            # reader, an antivirus scan or the search indexer can all cause. Losing this race costs nothing: these
            # peaks are a pure function of (audio, buckets), so whatever is or ends up in the cache equals `data`.
            # The caller still gets the freshly computed value, which is the part that has to be correct.
            with contextlib.suppress(OSError):
                os.replace(tmp, cache)
        finally:
            # Also suppressed, and for the same Windows reason: a scanner holding the temp file makes the delete fail,
            # and failing the whole request over a stray cache temp file would be a worse outcome than leaving it.
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)
        return data

    def analysis(self, variant: str, slug: str) -> dict:
        d = self.song_dir(variant, slug)
        amap_path = self._confine(d, "analysis", "artifact_map.json")
        if not amap_path.is_file():
            return {"available": False, "regions": [], "metrics": {}}
        amap = ArtifactMap.load(amap_path)
        metrics_path = self._confine(d, "analysis", "metrics.json")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
        return {
            "available": True,
            "regions": [r.to_dict() for r in sorted(amap.regions, key=lambda r: r.start_s)],
            "summaries": amap.stem_summaries,
            "metrics": metrics,
        }

    def stem_names(self, variant: str, slug: str) -> list[str]:
        """The stems on disk for a song, display order first.

        Exists so `server.py` never builds a path of its own -- the route used to glob
        `svc.song_dir(...) / "stems"` itself, which put a second path-construction site outside `_confine`.
        """
        d = self.song_dir(variant, slug)
        names = [p.stem for p in self._confine(d, "stems").glob("*.wav")]
        return sorted(names, key=lambda s: (STEM_ORDER.index(s) if s in STEM_ORDER else 99, s))

    def reveal_dir(self, variant: str, slug: str, sub: str | None = None) -> Path:
        """The folder to open in the file manager: a song, or one of three named subfolders of it."""
        d = self.song_dir(variant, slug)
        if sub in ("midi", "stems", "analysis"):
            p = self._confine(d, sub)
            if p.is_dir():
                return p
        return d

    def stem_levels(self, variant: str, slug: str) -> dict:
        d = self.song_dir(variant, slug)
        levels = {}
        orig = self.peaks(variant, slug, "original")
        for p in sorted(self._confine(d, "stems").glob("*.wav")):
            pk = self.peaks(variant, slug, p.stem)
            levels[p.stem] = {"rms_dbfs": pk["rms_dbfs"], "peak": pk["peak"], "rel_song_db": pk["rms_dbfs"] - orig["rms_dbfs"]}
        return levels
