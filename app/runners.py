from __future__ import annotations

import asyncio
import json
import math
import re
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx

EventSink = Callable[[dict], Awaitable[None]]

# ---------------------------------------------------------------------------
# Small helpers (keep in sync with CLI semantics)
# ---------------------------------------------------------------------------

def _host(server: str) -> str:
    try:
        u = urlparse(server)
        return u.netloc or server
    except Exception:
        return server

def _pct(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    v = sorted(values)
    k = int(round((p / 100.0) * (len(v) - 1)))
    return v[max(0, min(k, len(v) - 1))]

def _nan(x: float) -> bool:
    return x != x

def build_urls(server: str) -> Tuple[str, str]:
    server = server.rstrip("/")
    return f"{server}/v1/chat/completions", f"{server}/v1/models"

async def get_loaded_model(client: httpx.AsyncClient, models_endpoint: str) -> str:
    try:
        r = await client.get(models_endpoint)
        r.raise_for_status()
        data = r.json()
        return data["data"][0]["id"]
    except Exception:
        return "Unknown Model"

def _auth_headers(token: Optional[str]) -> Dict[str, str]:
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}

# ---------------------------------------------------------------------------
# Config objects
# ---------------------------------------------------------------------------

@dataclass
class BenchConfig:
    server: str
    prompt: str
    runs: int
    max_tokens: int
    temperature: float
    timeout: float
    stream: bool
    model: Optional[str]
    insecure: bool
    token: Optional[str]

    @staticmethod
    def from_payload(p: Dict[str, Any]) -> "BenchConfig":
        return BenchConfig(
            server=str(p.get("server", "")).strip(),
            prompt=str(p.get("prompt", "")).strip(),
            runs=int(p.get("runs", 5)),
            max_tokens=int(p.get("max_tokens", 512)),
            temperature=float(p.get("temperature", 0.2)),
            timeout=float(p.get("timeout", 120.0)),
            stream=bool(p.get("stream", True)),
            model=(str(p.get("model")).strip() if p.get("model") else None),
            insecure=bool(p.get("insecure", False)),
            token=(str(p.get("token")).strip() if p.get("token") else None),
        )

@dataclass
class LoadConfig:
    server: str
    model: Optional[str]
    prompt: str
    requests: int
    concurrency: int
    max_tokens: int
    temperature: float
    timeout: float
    stream: bool
    insecure: bool
    token: Optional[str]

    @staticmethod
    def from_payload(p: Dict[str, Any]) -> "LoadConfig":
        return LoadConfig(
            server=str(p.get("server", "")).strip(),
            model=(str(p.get("model")).strip() if p.get("model") else None),
            prompt=str(p.get("prompt", "")).strip(),
            requests=int(p.get("requests", 20)),
            concurrency=int(p.get("concurrency", 20)),
            max_tokens=int(p.get("max_tokens", 512)),
            temperature=float(p.get("temperature", 0.8)),
            timeout=float(p.get("timeout", 120.0)),
            stream=bool(p.get("stream", True)),
            insecure=bool(p.get("insecure", False)),
            token=(str(p.get("token")).strip() if p.get("token") else None),
        )

# ---------------------------------------------------------------------------
# Core request logic (bench + load share this)
# ---------------------------------------------------------------------------

async def _stream_chat(
    client: httpx.AsyncClient,
    endpoint: str,
    payload: dict,
) -> Dict[str, Any]:
    """
    Stream a /v1/chat/completions request and compute:
      - total_s
      - ttfb_s  (first parsed event)
      - ttft_s  (first content token)
      - usage prompt_tokens / completion_tokens (if included)
    """
    t0 = time.perf_counter()

    ttfb_s: Optional[float] = None
    ttft_s: Optional[float] = None
    p_tok: Optional[int] = None
    c_tok: Optional[int] = None
    http_status: Optional[int] = None

    async with client.stream("POST", endpoint, json=payload) as resp:
        http_status = resp.status_code
        resp.raise_for_status()

        async for raw_line in resp.aiter_lines():
            if not raw_line:
                continue
            line = raw_line.strip()
            if not line.startswith("data:"):
                continue
            chunk = line[len("data:"):].strip()
            if chunk == "[DONE]":
                break

            try:
                evt = json.loads(chunk)
            except json.JSONDecodeError:
                continue

            if ttfb_s is None:
                ttfb_s = time.perf_counter() - t0

            # TTFT: first time we see actual content
            try:
                delta = evt["choices"][0].get("delta", {})
                content = delta.get("content")
                if content and ttft_s is None:
                    ttft_s = time.perf_counter() - t0
            except Exception:
                pass

            u = evt.get("usage")
            if isinstance(u, dict):
                if isinstance(u.get("prompt_tokens"), int):
                    p_tok = u["prompt_tokens"]
                if isinstance(u.get("completion_tokens"), int):
                    c_tok = u["completion_tokens"]

    t1 = time.perf_counter()
    total_s = t1 - t0

    # fallbacks
    if ttfb_s is None:
        ttfb_s = float("nan")
    if ttft_s is None:
        ttft_s = ttfb_s

    e2e = (c_tok / total_s) if isinstance(c_tok, int) and total_s > 0 else None
    gen = None
    if isinstance(c_tok, int) and not _nan(ttft_s):
        gen_time = total_s - ttft_s
        if gen_time > 0.001:
            gen = c_tok / gen_time

    return {
        "total_s": total_s,
        "ttfb_s": ttfb_s,
        "ttft_s": ttft_s,
        "pTok": p_tok,
        "cTok": c_tok,
        "e2e_tps": e2e,
        "gen_tps": gen,
        "http": http_status,
    }

async def _nonstream_chat(
    client: httpx.AsyncClient,
    endpoint: str,
    payload: dict,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    r = await client.post(endpoint, json=payload)
    status = r.status_code
    r.raise_for_status()
    data = r.json()
    t1 = time.perf_counter()
    total_s = t1 - t0

    usage = data.get("usage", {}) if isinstance(data, dict) else {}
    p_tok = usage.get("prompt_tokens") if isinstance(usage, dict) else None
    c_tok = usage.get("completion_tokens") if isinstance(usage, dict) else None

    e2e = (c_tok / total_s) if isinstance(c_tok, int) and total_s > 0 else None
    return {
        "total_s": total_s,
        "ttfb_s": float("nan"),
        "ttft_s": float("nan"),
        "pTok": p_tok if isinstance(p_tok, int) else None,
        "cTok": c_tok if isinstance(c_tok, int) else None,
        "e2e_tps": e2e,
        "gen_tps": None,
        "http": status,
    }

# ---------------------------------------------------------------------------
# Bench
# ---------------------------------------------------------------------------

async def bench_runner(cfg: BenchConfig, emit: EventSink) -> Dict[str, Any]:
    if not cfg.server:
        raise ValueError("server is required")
    endpoint, models = build_urls(cfg.server)

    limits = httpx.Limits(max_connections=10, max_keepalive_connections=10)
    timeout = httpx.Timeout(cfg.timeout)

    async with httpx.AsyncClient(
        verify=not cfg.insecure,
        headers=_auth_headers(cfg.token),
        timeout=timeout,
        limits=limits,
    ) as client:
        model = cfg.model or await get_loaded_model(client, models)

        await emit({
            "type": "header",
            "kind": "bench",
            "model": model,
            "host": _host(cfg.server),
            "api": "/v1/chat/completions",
            "models": "/v1/models",
            "stream": cfg.stream,
            "runs": cfg.runs,
            "max_tokens": cfg.max_tokens,
            "temperature": cfg.temperature,
            "timeout": cfg.timeout,
            "insecure": cfg.insecure,
            "has_token": bool(cfg.token),
        })

        rows: List[dict] = []
        totals: List[float] = []
        ttfts: List[float] = []
        ttfbs: List[float] = []
        e2e: List[float] = []
        gen: List[float] = []

        for i in range(cfg.runs):
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": cfg.prompt}],
                "max_tokens": cfg.max_tokens,
                "temperature": cfg.temperature,
            }

            if cfg.stream:
                payload["stream"] = True
                payload["stream_options"] = {"include_usage": True}
                r = await _stream_chat(client, endpoint, payload)
            else:
                payload["stream"] = False
                r = await _nonstream_chat(client, endpoint, payload)

            r["i"] = i + 1
            rows.append(r)

            totals.append(r["total_s"])
            if not _nan(r["ttfb_s"]):
                ttfbs.append(r["ttfb_s"])
            if not _nan(r["ttft_s"]):
                ttfts.append(r["ttft_s"])
            if isinstance(r.get("e2e_tps"), (int, float)) and not _nan(float(r["e2e_tps"])):
                e2e.append(float(r["e2e_tps"]))
            if isinstance(r.get("gen_tps"), (int, float)) and not _nan(float(r["gen_tps"])):
                gen.append(float(r["gen_tps"]))

            await emit({"type": "row", "row": r})

        summary = {
            "total_s": {"avg": sum(totals)/len(totals), "p50": _pct(totals, 50), "p95": _pct(totals, 95)},
            "ttfb_s": {"avg": (sum(ttfbs)/len(ttfbs)) if ttfbs else float("nan"),
                       "p50": _pct(ttfbs, 50), "p95": _pct(ttfbs, 95)} if cfg.stream else None,
            "ttft_s": {"avg": (sum(ttfts)/len(ttfts)) if ttfts else float("nan"),
                       "p50": _pct(ttfts, 50), "p95": _pct(ttfts, 95)} if cfg.stream else None,
            "e2e_tps": {"avg": (sum(e2e)/len(e2e)) if e2e else float("nan"),
                        "p50": _pct(e2e, 50), "p95": _pct(e2e, 95)} if e2e else None,
            "gen_tps": {"avg": (sum(gen)/len(gen)) if gen else float("nan"),
                        "p50": _pct(gen, 50), "p95": _pct(gen, 95)} if gen else None,
            "tokens": {
                "pTok": next((r["pTok"] for r in reversed(rows) if isinstance(r.get("pTok"), int)), None),
                "cTok": next((r["cTok"] for r in reversed(rows) if isinstance(r.get("cTok"), int)), None),
            }
        }

        return {"kind": "bench", "model": model, "config": cfg.__dict__, "rows": rows, "summary": summary}

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

async def load_runner(cfg: LoadConfig, emit: EventSink) -> Dict[str, Any]:
    if not cfg.server:
        raise ValueError("server is required")
    endpoint, models = build_urls(cfg.server)

    limits = httpx.Limits(max_connections=max(10, cfg.concurrency), max_keepalive_connections=max(10, cfg.concurrency))
    timeout = httpx.Timeout(cfg.timeout)

    async with httpx.AsyncClient(
        verify=not cfg.insecure,
        headers=_auth_headers(cfg.token),
        timeout=timeout,
        limits=limits,
    ) as client:
        model = cfg.model or await get_loaded_model(client, models)

        await emit({
            "type": "header",
            "kind": "load",
            "model": model,
            "host": _host(cfg.server),
            "api": "/v1/chat/completions",
            "models": "/v1/models",
            "stream": cfg.stream,
            "requests": cfg.requests,
            "concurrency": cfg.concurrency,
            "max_tokens": cfg.max_tokens,
            "temperature": cfg.temperature,
            "timeout": cfg.timeout,
            "insecure": cfg.insecure,
            "has_token": bool(cfg.token),
        })

        sem = asyncio.Semaphore(cfg.concurrency)

        rows: List[dict] = []
        totals: List[float] = []
        ttfts: List[float] = []
        ttfbs: List[float] = []
        e2e: List[float] = []
        gen: List[float] = []

        start_wall = time.perf_counter()

        async def one(i: int) -> None:
            async with sem:
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": cfg.prompt.format(i=i)}],
                    "max_tokens": cfg.max_tokens,
                    "temperature": cfg.temperature,
                }
                try:
                    if cfg.stream:
                        payload["stream"] = True
                        payload["stream_options"] = {"include_usage": True}
                        r = await _stream_chat(client, endpoint, payload)
                    else:
                        payload["stream"] = False
                        r = await _nonstream_chat(client, endpoint, payload)

                    r["i"] = i
                    rows.append(r)

                    totals.append(r["total_s"])
                    if not _nan(r["ttfb_s"]):
                        ttfbs.append(r["ttfb_s"])
                    if not _nan(r["ttft_s"]):
                        ttfts.append(r["ttft_s"])
                    if isinstance(r.get("e2e_tps"), (int, float)):
                        e2e.append(float(r["e2e_tps"]))
                    if isinstance(r.get("gen_tps"), (int, float)):
                        gen.append(float(r["gen_tps"]))

                    await emit({"type": "row", "row": r})
                except Exception as e:
                    err = {"i": i, "error": str(e), "http": None}
                    rows.append(err)
                    await emit({"type": "row", "row": err})

        tasks = [asyncio.create_task(one(i)) for i in range(1, cfg.requests + 1)]
        await asyncio.gather(*tasks)

        wall = time.perf_counter() - start_wall
        ok = sum(1 for r in rows if r.get("http") == 200 and "error" not in r)
        failed = len(rows) - ok
        achieved_rps = (len(rows) / wall) if wall > 0 else float("nan")

        summary = {
            "wall_time_s": wall,
            "achieved_rps": achieved_rps,
            "ok": ok,
            "failed": failed,
            "total_s": {"avg": sum(totals)/len(totals), "p50": _pct(totals, 50), "p95": _pct(totals, 95)} if totals else None,
            "ttfb_s": {"avg": (sum(ttfbs)/len(ttfbs)) if ttfbs else float("nan"),
                       "p50": _pct(ttfbs, 50), "p95": _pct(ttfbs, 95)} if cfg.stream else None,
            "ttft_s": {"avg": (sum(ttfts)/len(ttfts)) if ttfts else float("nan"),
                       "p50": _pct(ttfts, 50), "p95": _pct(ttfts, 95)} if cfg.stream else None,
            "e2e_tps": {"avg": (sum(e2e)/len(e2e)) if e2e else float("nan"),
                        "p50": _pct(e2e, 50), "p95": _pct(e2e, 95)} if e2e else None,
            "gen_tps": {"avg": (sum(gen)/len(gen)) if gen else float("nan"),
                        "p50": _pct(gen, 50), "p95": _pct(gen, 95)} if gen else None,
        }

        return {"kind": "load", "model": model, "config": cfg.__dict__, "rows": rows, "summary": summary}
