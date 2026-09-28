"""State the UI needs: the songs on disk, a one-at-a-time job queue, and cached waveform peaks.

The GPU fits one separation at a time, so jobs are serialised in a single worker thread. Everything is derived from
the same ``outputs/`` layout the CLI writes, so the UI and the CLI stay interchangeable.
"""

from __future__ import annotations

import contextlib
import json
import queue
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

    def song_dir(self, variant: str, slug: str) -> Path:
        p = (self.out_root / variant / slug).resolve()
        if not p.is_relative_to(self.out_root.resolve()) or not p.is_dir():
            raise FileNotFoundError(f"unknown song {variant}/{slug}")
        return p

    def audio_path(self, variant: str, slug: str, stem: str) -> Path:
        d = self.song_dir(variant, slug)
        p = (d / "original.wav") if stem == "original" else (d / "stems" / f"{stem}.wav")
        if not p.is_file():
            raise FileNotFoundError(p)
        return p

    # ---- waveform peaks (min/max per bucket), cached on disk ----
    def peaks(self, variant: str, slug: str, stem: str, buckets: int = 2400) -> dict:
        d = self.song_dir(variant, slug)
        cache = d / "peaks" / f"{stem}.{buckets}.json"
        if cache.is_file():
            return json.loads(cache.read_text())
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
        cache.write_text(json.dumps(data))
        return data

    def analysis(self, variant: str, slug: str) -> dict:
        d = self.song_dir(variant, slug)
        amap_path = d / "analysis" / "artifact_map.json"
        if not amap_path.is_file():
            return {"available": False, "regions": [], "metrics": {}}
        amap = ArtifactMap.load(amap_path)
        metrics = json.loads((d / "analysis" / "metrics.json").read_text(encoding="utf-8")) if (d / "analysis" / "metrics.json").is_file() else {}
        return {
            "available": True,
            "regions": [r.to_dict() for r in sorted(amap.regions, key=lambda r: r.start_s)],
            "summaries": amap.stem_summaries,
            "metrics": metrics,
        }

    def stem_levels(self, variant: str, slug: str) -> dict:
        d = self.song_dir(variant, slug)
        levels = {}
        orig = self.peaks(variant, slug, "original")
        for p in sorted((d / "stems").glob("*.wav")):
            pk = self.peaks(variant, slug, p.stem)
            levels[p.stem] = {"rms_dbfs": pk["rms_dbfs"], "peak": pk["peak"], "rel_song_db": pk["rms_dbfs"] - orig["rms_dbfs"]}
        return levels
