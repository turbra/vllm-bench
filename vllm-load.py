#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from time import perf_counter
from typing import Optional, Tuple
from urllib.parse import urlparse

import requests

# Optional: suppress warnings when using --insecure (skip TLS verify)
try:
    import urllib3
except Exception:  # pragma: no cover
    urllib3 = None

# -----------------------------------------------------------------------------
# Defaults (edit here, or override with CLI args)
# -----------------------------------------------------------------------------
SERVER = "https://vllm-endpoint.example.com"
DEFAULT_TIMEOUT = 120.0
DEFAULT_TEMPERATURE = 0.8
DEFAULT_REQUESTS = 10
DEFAULT_CONCURRENCY = 10
DEFAULT_MAX_TOKENS = 512
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# ANSI color (auto-off when not a TTY)
# -----------------------------------------------------------------------------
USE_COLOR = sys.stdout.isatty()
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _vis_len(s: str) -> int:
    return len(_ANSI_RE.sub("", s))


def _pad_to(s: str, width: int) -> str:
    return s + (" " * max(0, width - _vis_len(s)))


def c(s: str, code: str) -> str:
    if not USE_COLOR:
        return s
    return f"\x1b[{code}m{s}\x1b[0m"


def bold(s: str) -> str:
    return c(s, "1")


def dim(s: str) -> str:
    return c(s, "2")


def cyan(s: str) -> str:
    return c(s, "36")


def green(s: str) -> str:
    return c(s, "32")


def yellow(s: str) -> str:
    return c(s, "33")


def red(s: str) -> str:
    return c(s, "31")


# -----------------------------------------------------------------------------
# Formatting helpers (match vllm-bench style)
# -----------------------------------------------------------------------------
def _term_width(default: int = 96) -> int:
    try:
        return shutil.get_terminal_size((default, 20)).columns
    except Exception:
        return default


def _hr(char: str = "─", width: int = 72) -> str:
    return char * width


def _wrap(s: str, width: int) -> list[str]:
    return textwrap.wrap(s, width=width, break_long_words=False, break_on_hyphens=False) or [s]


def _box(title: str, lines: list[str], width: int = 72) -> str:
    top = f"┌{_hr('─', width-2)}┐"
    mid = (
        f"│ {bold(title[:width-4]).ljust(width-4 + (len(bold('')) - len('')))} │"
        if USE_COLOR
        else f"│ {title[:width-4].ljust(width-4)} │"
    )
    sep = f"├{_hr('─', width-2)}┤"

    body_lines: list[str] = []
    for ln in lines:
        for sub in _wrap(ln, width - 4):
            body_lines.append(f"│ {_pad_to(sub, width-4)} │")

    bot = f"└{_hr('─', width-2)}┘"
    return "\n".join([top, mid, sep, "\n".join(body_lines), bot])


def _pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    v = sorted(values)
    k = int(round((p / 100.0) * (len(v) - 1)))
    return v[max(0, min(k, len(v) - 1))]


def _fmt_ms(seconds: float) -> str:
    return "N/A" if seconds != seconds else f"{seconds * 1000:.2f} ms"


def _fmt_num(x: Optional[float], fmt: str = "{:.2f}") -> str:
    if x is None:
        return "N/A"
    try:
        if x != x:
            return "N/A"
        return fmt.format(x)
    except Exception:
        return "N/A"


def _host(url: str) -> str:
    try:
        u = urlparse(url)
        return u.netloc or url
    except Exception:
        return url


def build_headers(token: Optional[str]) -> dict:
    """Return request headers (Authorization if token is provided)."""
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}

# -----------------------------------------------------------------------------
# API helpers
# -----------------------------------------------------------------------------
def build_urls(server: str) -> Tuple[str, str]:
    server = server.rstrip("/")
    return f"{server}/v1/chat/completions", f"{server}/v1/models"


def get_loaded_model(models_endpoint: str, timeout: float, verify: bool, headers: dict) -> str:
    try:
        resp = requests.get(models_endpoint, timeout=timeout, verify=verify, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return data["data"][0]["id"]
    except requests.exceptions.RequestException as e:
        print(f"{yellow('WARN')}: Error connecting to server models endpoint: {e}")
        return "Unknown Model"
    except (IndexError, KeyError, ValueError):
        print(f"{yellow('WARN')}: No models listed on the server (or unexpected /v1/models response).")
        return "Unknown Model"


def post_chat_completions(endpoint: str, payload: dict, timeout: float, stream: bool, verify: bool, headers: dict) -> requests.Response:
    """
    POST to /v1/chat/completions.
    If it returns 404, try a common fallback (/v1/chat/completion).
    """
    try_endpoints = [endpoint]
    if endpoint.endswith("/v1/chat/completions"):
        try_endpoints.append(endpoint[:-1])  # .../completion

    last_err: Optional[Exception] = None
    for ep in try_endpoints:
        try:
            r = requests.post(ep, json=payload, timeout=timeout, stream=stream, verify=verify, headers=headers)
            if r.status_code == 404:
                last_err = requests.exceptions.HTTPError(f"404 Not Found for url: {ep}")
                continue
            r.raise_for_status()
            return r
        except Exception as e:
            last_err = e

    raise last_err if last_err else RuntimeError("Unknown error posting to chat completions")


@dataclass
class Result:
    idx: int
    ok: bool
    status: Optional[int]
    total_s: float
    ttfb_s: float
    ttft_s: float
    p_tok: Optional[int]
    c_tok: Optional[int]
    tps_e2e: Optional[float]
    tps_gen: Optional[float]
    error: Optional[str] = None


def run_one(
    idx: int,
    endpoint: str,
    model: str,
    prompt_template: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
    stream: bool,
    verify: bool,
    headers: dict,
) -> Result:
    prompt = prompt_template.format(i=idx) if "{i}" in prompt_template else f"{prompt_template} (Request {idx})"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": bool(stream),
    }
    if stream:
        payload["stream_options"] = {"include_usage": True}

    t0 = perf_counter()
    try:
        r = post_chat_completions(endpoint, payload, timeout=timeout, stream=stream, verify=verify, headers=headers)
        status = r.status_code

        # Non-stream: end-to-end only
        if not stream:
            t1 = perf_counter()
            total = t1 - t0
            data = r.json()
            usage = data.get("usage", {}) if isinstance(data, dict) else {}
            p_tok = usage.get("prompt_tokens")
            c_tok = usage.get("completion_tokens")
            tps_e2e = (c_tok / total) if isinstance(c_tok, int) and total > 0 else None
            return Result(
                idx=idx,
                ok=True,
                status=status,
                total_s=total,
                ttfb_s=float("nan"),
                ttft_s=float("nan"),
                p_tok=p_tok if isinstance(p_tok, int) else None,
                c_tok=c_tok if isinstance(c_tok, int) else None,
                tps_e2e=tps_e2e,
                tps_gen=None,
            )

        # Stream: parse SSE events
        ttfb_s: Optional[float] = None
        ttft_s: Optional[float] = None
        p_tok: Optional[int] = None
        c_tok: Optional[int] = None

        for raw_line in r.iter_lines(decode_unicode=True):
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
                ttfb_s = perf_counter() - t0

            try:
                delta = evt["choices"][0].get("delta", {})
                content = delta.get("content")
                if content and ttft_s is None:
                    ttft_s = perf_counter() - t0
            except Exception:
                pass

            u = evt.get("usage")
            if isinstance(u, dict):
                if isinstance(u.get("prompt_tokens"), int):
                    p_tok = u["prompt_tokens"]
                if isinstance(u.get("completion_tokens"), int):
                    c_tok = u["completion_tokens"]

        t1 = perf_counter()
        total = t1 - t0

        if ttfb_s is None:
            ttfb_s = float("nan")
        if ttft_s is None:
            ttft_s = ttfb_s  # fallback

        tps_e2e = (c_tok / total) if isinstance(c_tok, int) and total > 0 else None

        tps_gen = None
        if isinstance(c_tok, int) and ttft_s == ttft_s:
            gen_time = total - ttft_s
            if gen_time > 0.001:  # clamp to avoid silly spikes
                tps_gen = c_tok / gen_time

        return Result(
            idx=idx,
            ok=True,
            status=status,
            total_s=total,
            ttfb_s=ttfb_s,
            ttft_s=ttft_s,
            p_tok=p_tok,
            c_tok=c_tok,
            tps_e2e=tps_e2e,
            tps_gen=tps_gen,
        )

    except Exception as e:
        t1 = perf_counter()
        return Result(
            idx=idx,
            ok=False,
            status=None,
            total_s=t1 - t0,
            ttfb_s=float("nan"),
            ttft_s=float("nan"),
            p_tok=None,
            c_tok=None,
            tps_e2e=None,
            tps_gen=None,
            error=str(e),
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="vLLM concurrent load test (OpenAI Chat Completions).")
    parser.add_argument("--server", type=str, default=SERVER, help="Base server URL (defaults to SERVER constant).")
    parser.add_argument("--requests", type=int, default=DEFAULT_REQUESTS, help="Total number of requests to send.")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="Max in-flight requests.")
    parser.add_argument(
        "--prompt",
        type=str,
        default=(
            "Design a small web service that exposes a REST API for managing tasks. "
            "Requirements: plain English (no code); describe endpoints, request/response shapes, and error handling; "
            "include exactly 5 endpoints; keep the answer under 400 words. (Request {i})"
        ),
        help="Prompt template. Use {i} to inject request number.",
    )
    parser.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Max tokens per response.")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Sampling temperature.")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="HTTP timeout seconds.")
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Bearer token for Authorization header (optional). If omitted, checks --token-env.",
    )
    parser.add_argument(
        "--token-env",
        type=str,
        default="VLLM_TOKEN",
        help="Environment variable name to read token from (default: VLLM_TOKEN).",
    )
    parser.add_argument("--insecure", action="store_true", help="Skip TLS certificate verification (like curl -k).")
    parser.add_argument("--stream", action="store_true", help="Enable streaming to compute TTFB/TTFT.")
    parser.add_argument("--model", type=str, default=None, help="Override model name (else use /v1/models).")

    args = parser.parse_args()

    server = args.server.strip()
    if (
        server in ("REPLACE_ME", "")
        or "your-vllm-host.example.com" in server
        or "<your-route-hostname>" in server
        or "vllm-endpoint.example.com" in server
        or server == SERVER  # still default placeholder
    ):
        print(
            "SERVER is still a placeholder. Set SERVER at top of file "
            "(e.g. https://your-vllm-host.example.com) or pass --server https://<your-route-hostname>."
        )
        return 2

    verify = not args.insecure
    if args.insecure and urllib3 is not None:
        try:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        except Exception:
            pass

    token = args.token or (os.getenv(args.token_env) if args.token_env else None)
    headers = build_headers(token)

    width = min(_term_width(), 110)
    chat_endpoint, models_endpoint = build_urls(server)
    model_name = args.model or get_loaded_model(models_endpoint, timeout=args.timeout, verify=verify, headers=headers)

    header_lines = [
        f"Model       : {cyan(model_name)}",
        f"Host        : {_host(server)}",
        f"API         : /v1/chat/completions",
        f"Models      : /v1/models",
        f"Mode        : {'stream (TTFB/TTFT enabled)' if args.stream else 'non-stream'}",
        f"Auth        : {'Bearer token' if token else 'none'}",
        f"Requests    : {args.requests}",
        f"Concurrency : {args.concurrency}",
        f"Max tokens  : {args.max_tokens}",
        f"Temp        : {args.temperature}",
        f"Timeout     : {args.timeout:.1f}s",
    ]
    print(_box("vLLM Load Test", header_lines, width=width))

    # Fire concurrent requests
    start_all = perf_counter()
    results: list[Result] = []
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
        futures = [
            ex.submit(
                run_one,
                i,
                chat_endpoint,
                model_name,
                args.prompt,
                args.max_tokens,
                args.temperature,
                args.timeout,
                args.stream,
                verify,
                headers,
            )
            for i in range(1, args.requests + 1)
        ]
        for fut in as_completed(futures):
            results.append(fut.result())
    wall_s = perf_counter() - start_all

    results.sort(key=lambda r: r.idx)

    # Table
    table_w = width
    print(dim(_hr("─", table_w)))
    if args.stream:
        print(f"{'#':>2}  {'Total(s)':>8}  {'TTFB':>10}  {'TTFT':>10}  {'pTok':>5}  {'cTok':>5}  {'e2e(tok/s)':>11}  {'gen(tok/s)':>11}  {'HTTP':>4}")
    else:
        print(f"{'#':>2}  {'Total(s)':>8}  {'pTok':>5}  {'cTok':>5}  {'e2e(tok/s)':>11}  {'HTTP':>4}")
    print(dim(_hr("─", table_w)))

    totals: list[float] = []
    ttfts: list[float] = []
    speeds_e2e: list[float] = []
    speeds_gen: list[float] = []
    ok_count = 0

    for r in results:
        totals.append(r.total_s)
        if r.ok:
            ok_count += 1

        if args.stream and r.ttft_s == r.ttft_s:
            ttfts.append(r.ttft_s)

        if isinstance(r.tps_e2e, (int, float)) and r.tps_e2e == r.tps_e2e:
            speeds_e2e.append(float(r.tps_e2e))

        if isinstance(r.tps_gen, (int, float)) and r.tps_gen == r.tps_gen:
            speeds_gen.append(float(r.tps_gen))

        http = str(r.status) if isinstance(r.status, int) else ("ERR" if not r.ok else "?")
        http_disp = green(http) if r.ok and http.startswith("2") else (red(http) if http == "ERR" or (http.isdigit() and not http.startswith("2")) else http)

        if args.stream:
            ttfb_str = _fmt_ms(r.ttfb_s)
            ttft_str = _fmt_ms(r.ttft_s)
            e2e_str = _fmt_num(r.tps_e2e, "{:.2f}")
            gen_str = _fmt_num(r.tps_gen, "{:.2f}")

            print(
                f"{r.idx:>2}  "
                f"{r.total_s:>8.3f}  "
                f"{cyan(f'{ttfb_str:>10}') if ttfb_str!='N/A' else f'{ttfb_str:>10}'}  "
                f"{cyan(f'{ttft_str:>10}') if ttft_str!='N/A' else f'{ttft_str:>10}'}  "
                f"{(str(r.p_tok) if isinstance(r.p_tok, int) else '?'):>5}  "
                f"{(str(r.c_tok) if isinstance(r.c_tok, int) else '?'):>5}  "
                f"{green(f'{e2e_str:>11}') if e2e_str!='N/A' else f'{e2e_str:>11}'}  "
                f"{green(f'{gen_str:>11}') if gen_str!='N/A' else f'{gen_str:>11}'}  "
                f"{http_disp:>4}"
            )
        else:
            e2e_str = _fmt_num(r.tps_e2e, "{:.2f}")
            print(
                f"{r.idx:>2}  "
                f"{r.total_s:>8.3f}  "
                f"{(str(r.p_tok) if isinstance(r.p_tok, int) else '?'):>5}  "
                f"{(str(r.c_tok) if isinstance(r.c_tok, int) else '?'):>5}  "
                f"{green(f'{e2e_str:>11}') if e2e_str!='N/A' else f'{e2e_str:>11}'}  "
                f"{http_disp:>4}"
            )

        if (not r.ok) and r.error:
            msg = f"{yellow('WARN')}: req {r.idx}: {r.error}"
            for sub in _wrap(msg, table_w):
                print(dim(sub))

    print(dim(_hr("─", table_w)))

    # Summary
    fail_count = len(results) - ok_count

    avg_total = sum(totals) / len(totals) if totals else float("nan")
    p50_total = _pct(totals, 50)
    p95_total = _pct(totals, 95)

    avg_ttft = (sum(ttfts) / len(ttfts)) if ttfts else float("nan")
    p50_ttft = _pct(ttfts, 50)
    p95_ttft = _pct(ttfts, 95)

    avg_e2e = (sum(speeds_e2e) / len(speeds_e2e)) if speeds_e2e else float("nan")
    p50_e2e = _pct(speeds_e2e, 50)
    p95_e2e = _pct(speeds_e2e, 95)

    avg_gen = (sum(speeds_gen) / len(speeds_gen)) if speeds_gen else float("nan")
    p50_gen = _pct(speeds_gen, 50)
    p95_gen = _pct(speeds_gen, 95)

    rps = (len(results) / wall_s) if wall_s > 0 else float("nan")

    summary_lines = [
        f"Wall time   : {bold(_fmt_num(wall_s, '{:.3f}'))}s | Achieved RPS {bold(_fmt_num(rps, '{:.2f}'))}",
        f"Results     : {green(str(ok_count))} ok | {red(str(fail_count))} failed",
        f"Total time  : avg {bold(_fmt_num(avg_total, '{:.3f}'))}s | p50 {_fmt_num(p50_total, '{:.3f}')}s | p95 {_fmt_num(p95_total, '{:.3f}')}s",
    ]
    if args.stream:
        summary_lines.append(
            f"TTFT        : avg {cyan(_fmt_ms(avg_ttft))} | p50 {_fmt_ms(p50_ttft)} | p95 {_fmt_ms(p95_ttft)}"
        )
        summary_lines.append(
            f"Speed(e2e)  : avg {green(_fmt_num(avg_e2e))} | p50 {_fmt_num(p50_e2e)} | p95 {_fmt_num(p95_e2e)}"
        )
        summary_lines.append(
            f"Speed(gen)  : avg {green(_fmt_num(avg_gen))} | p50 {_fmt_num(p50_gen)} | p95 {_fmt_num(p95_gen)}"
        )
    else:
        summary_lines.append(
            f"Speed(e2e)  : avg {green(_fmt_num(avg_e2e))} | p50 {_fmt_num(p50_e2e)} | p95 {_fmt_num(p95_e2e)}"
        )

    print(_box("Summary", summary_lines, width=width))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
