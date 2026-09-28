"""Local HTTP API for the CleanSplit desktop UI. Binds to 127.0.0.1 only; no telemetry, no external requests."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .service import SEPARATORS, Service

STATIC = Path(__file__).parent / "static"


def create_app(out_root: str = "outputs") -> FastAPI:
    app = FastAPI(title="CleanSplit", docs_url=None, redoc_url=None)
    svc = Service(out_root)
    app.state.service = svc
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (STATIC / "index.html").read_text(encoding="utf-8")

    @app.get("/api/state")
    def state():
        return {
            "separators": SEPARATORS,
            "songs": svc.songs(),
            "jobs": [svc.jobs[i].to_dict() for i in svc.order[-20:]],
            "out_root": str(svc.out_root.resolve()),
        }

    @app.post("/api/jobs")
    async def create_job(req: Request):
        body = await req.json()
        path = Path(body.get("input", ""))
        if not path.is_file():
            raise HTTPException(400, f"file not found: {path}")
        separator = body.get("separator", "ensemble")
        if separator not in {s["id"] for s in SEPARATORS}:
            raise HTTPException(400, f"unknown separator {separator}")
        return svc.submit(body.get("kind", "separate"), str(path), separator).to_dict()

    @app.post("/api/midi")
    async def create_midi_job(req: Request):
        body = await req.json()
        try:
            return svc.submit_midi(body["variant"], body["slug"]).to_dict()
        except FileNotFoundError as e:
            raise HTTPException(404, str(e)) from e

    @app.get("/api/events")
    async def events(request: Request):
        q = svc.subscribe()

        async def stream():
            try:
                yield ": connected\n\n"
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        event = await asyncio.get_event_loop().run_in_executor(None, q.get, True, 15)
                    except Exception:
                        yield ": ping\n\n"
                        continue
                    yield f"data: {json.dumps(event)}\n\n"
            finally:
                svc.unsubscribe(q)

        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/song/{variant}/{slug}")
    def song(variant: str, slug: str):
        try:
            return {
                "levels": svc.stem_levels(variant, slug),
                "analysis": svc.analysis(variant, slug),
                "peaks": {stem: svc.peaks(variant, slug, stem) for stem in
                          ["original"] + [p.stem for p in (svc.song_dir(variant, slug) / "stems").glob("*.wav")]},
            }
        except FileNotFoundError as e:
            raise HTTPException(404, str(e)) from e

    @app.get("/api/audio/{variant}/{slug}/{stem}.wav")
    def audio(variant: str, slug: str, stem: str):
        try:
            return FileResponse(svc.audio_path(variant, slug, stem), media_type="audio/wav")
        except FileNotFoundError as e:
            raise HTTPException(404, str(e)) from e

    @app.post("/api/reveal")
    async def reveal(req: Request):
        """Open a song's folder in Explorer (the one thing the web layer cannot do itself)."""
        import subprocess

        body = await req.json()
        d = svc.song_dir(body["variant"], body["slug"])
        if body.get("sub") in ("midi", "stems", "analysis") and (d / body["sub"]).is_dir():
            d = d / body["sub"]
        subprocess.Popen(["explorer", str(d)])
        return {"opened": str(d)}

    return app


def serve(host: str = "127.0.0.1", port: int = 8770, out_root: str = "outputs", log_level: str = "warning"):
    import uvicorn

    uvicorn.run(create_app(out_root), host=host, port=port, log_level=log_level)
