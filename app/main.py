from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, asdict
from typing import Any, AsyncGenerator, Dict, Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .runners import bench_runner, load_runner, BenchConfig, LoadConfig

app = FastAPI(title="vLLM Perf Web", version="0.1.0")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# ---------------------------------------------------------------------------
# In-memory job store (good enough for a single-node lab app).
# If you ever scale this, swap for Redis.
# ---------------------------------------------------------------------------

@dataclass
class JobState:
    job_id: str
    kind: str  # "bench" | "load"
    created_at: float
    done: bool = False
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


_jobs: Dict[str, JobState] = {}
_queues: Dict[str, "asyncio.Queue[dict]"] = {}


def _new_job(kind: str) -> JobState:
    jid = str(uuid.uuid4())
    st = JobState(job_id=jid, kind=kind, created_at=time.time())
    _jobs[jid] = st
    _queues[jid] = asyncio.Queue()
    return st


async def _push(job_id: str, event: dict) -> None:
    q = _queues.get(job_id)
    if q is not None:
        await q.put(event)


async def _run_job(job: JobState, cfg: Any) -> None:
    try:
        await _push(job.job_id, {"type": "started", "job": asdict(job)})
        if job.kind == "bench":
            result = await bench_runner(cfg, lambda ev: _push(job.job_id, ev))
        else:
            result = await load_runner(cfg, lambda ev: _push(job.job_id, ev))

        job.result = result
        job.done = True
        await _push(job.job_id, {"type": "done", "result": result})
    except Exception as e:
        job.error = str(e)
        job.done = True
        await _push(job.job_id, {"type": "error", "error": str(e)})


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    with open("app/templates/index.html", "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# API: create jobs
# ---------------------------------------------------------------------------

@app.post("/api/bench")
async def start_bench(req: Request) -> JSONResponse:
    payload = await req.json()
    cfg = BenchConfig.from_payload(payload)
    job = _new_job("bench")
    asyncio.create_task(_run_job(job, cfg))
    return JSONResponse({"job_id": job.job_id})


@app.post("/api/load")
async def start_load(req: Request) -> JSONResponse:
    payload = await req.json()
    cfg = LoadConfig.from_payload(payload)
    job = _new_job("load")
    asyncio.create_task(_run_job(job, cfg))
    return JSONResponse({"job_id": job.job_id})


# ---------------------------------------------------------------------------
# API: events + results
# ---------------------------------------------------------------------------

@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str) -> StreamingResponse:
    if job_id not in _jobs:
        return StreamingResponse(iter([b"event: error\ndata: {\"error\":\"unknown job\"}\n\n"]), media_type="text/event-stream")

    async def gen() -> AsyncGenerator[bytes, None]:
        q = _queues[job_id]
        # Immediately send a "hello" so the UI can flip to running state fast.
        yield b"event: hello\ndata: {}\n\n"
        while True:
            ev = await q.get()
            data = json.dumps(ev, ensure_ascii=False)
            yield f"data: {data}\n\n".encode("utf-8")
            if ev.get("type") in ("done", "error"):
                break

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/api/jobs/{job_id}/result")
async def job_result(job_id: str) -> JSONResponse:
    job = _jobs.get(job_id)
    if not job:
        return JSONResponse({"error": "unknown job"}, status_code=404)
    return JSONResponse({"job": asdict(job)})
