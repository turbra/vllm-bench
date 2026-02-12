import argparse
import json
import re
import time
import sys
import shutil
import textwrap
from urllib.parse import urlparse
from time import perf_counter
from typing import Optional, Tuple

import requests

# -----------------------------------------------------------------------------
# Defaults (edit here, or override with CLI args)
# -----------------------------------------------------------------------------
SERVER = "https://your-vllm-host.example.com"
DEFAULT_TIMEOUT = 120.0
DEFAULT_TEMPERATURE = 0.2
DEFAULT_RUNS = 5
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
    # pad based on visible length, not raw length
    return s + (" " * max(0, width - _vis_len(s)))

def c(s: str, code: str) -> str:
    """Wrap string in ANSI code if output is a TTY."""
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


# -----------------------------------------------------------------------------
# Formatting helpers
# -----------------------------------------------------------------------------
def _term_width(default: int = 88) -> int:
    try:
        return shutil.get_terminal_size((default, 20)).columns
    except Exception:
        return default


def _host(url: str) -> str:
    try:
        u = urlparse(url)
        return u.netloc or url
    except Exception:
        return url


def _wrap(s: str, width: int) -> list[str]:
    return textwrap.wrap(s, width=width, break_long_words=False, break_on_hyphens=False) or [s]


def _hr(char: str = "─", width: int = 72) -> str:
    return char * width


def _box(title: str, lines: list[str], width: int = 72) -> str:
    top = f"┌{_hr('─', width-2)}┐"
    mid = f"│ {bold(title[:width-4]).ljust(width-4 + (len(bold('')) - len('')))} │" if USE_COLOR else f"│ {title[:width-4].ljust(width-4)} │"
    sep = f"├{_hr('─', width-2)}┤"

    body_lines = []
    for ln in lines:
        # Ensure hard wrap for long lines (urls, etc.)
        for sub in _wrap(ln, width - 4):
            body_lines.append(f"│ {_pad_to(sub, width-4)} │")

    bot = f"└{_hr('─', width-2)}┘"
    return "\n".join([top, mid, sep, "\n".join(body_lines), bot])


def _pct(values: list[float], p: float) -> float:
    """Nearest-rank percentile, p in [0,100]."""
    if not values:
        return float("nan")
    v = sorted(values)
    k = int(round((p / 100.0) * (len(v) - 1)))
    return v[max(0, min(k, len(v) - 1))]


def _fmt_ms(seconds: float) -> str:
    return "N/A" if seconds != seconds else f"{seconds * 1000:.2f} ms"  # NaN check


def _fmt_num(x: Optional[float], fmt: str = "{:.2f}") -> str:
    if x is None:
        return "N/A"
    try:
        if x != x:  # NaN
            return "N/A"
        return fmt.format(x)
    except Exception:
        return "N/A"


def build_urls(server: str) -> Tuple[str, str]:
    server = server.rstrip("/")
    return f"{server}/v1/chat/completions", f"{server}/v1/models"


def get_loaded_model(models_endpoint: str, timeout: float) -> str:
    try:
        resp = requests.get(models_endpoint, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return data["data"][0]["id"]
    except requests.exceptions.RequestException as e:
        print(f"{yellow('WARN')}: Error connecting to server models endpoint: {e}")
        return "Unknown Model"
    except (IndexError, KeyError, ValueError):
        print(f"{yellow('WARN')}: No models listed on the server (or unexpected /v1/models response).")
        return "Unknown Model"


def post_chat_completions(endpoint: str, payload: dict, timeout: float, stream: bool) -> requests.Response:
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
            r = requests.post(ep, json=payload, timeout=timeout, stream=stream)
            if r.status_code == 404:
                last_err = requests.exceptions.HTTPError(f"404 Not Found for url: {ep}")
                continue
            r.raise_for_status()
            return r
        except Exception as e:
            last_err = e

    raise last_err if last_err else RuntimeError("Unknown error posting to chat completions")


def run_once_non_stream(
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> Tuple[float, Optional[int], Optional[int], Optional[float]]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }

    t0 = perf_counter()
    r = post_chat_completions(endpoint, payload, timeout=timeout, stream=False)
    t1 = perf_counter()

    data = r.json()
    usage = data.get("usage", {}) if isinstance(data, dict) else {}
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")

    tokens_per_second = None
    if isinstance(completion_tokens, int) and (t1 - t0) > 0:
        tokens_per_second = completion_tokens / (t1 - t0)

    return (t1 - t0, prompt_tokens, completion_tokens, tokens_per_second)


def run_once_stream(
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> Tuple[float, float, Optional[int], Optional[int], Optional[float]]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    t0 = perf_counter()
    r = post_chat_completions(endpoint, payload, timeout=timeout, stream=True)

    ttft_s: Optional[float] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None

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

        # TTFT: first time we see actual content in delta
        try:
            delta = evt["choices"][0].get("delta", {})
            content = delta.get("content")
            if content and ttft_s is None:
                ttft_s = perf_counter() - t0
        except Exception:
            pass

        # usage may appear on a chunk if include_usage is supported
        u = evt.get("usage")
        if isinstance(u, dict):
            if isinstance(u.get("prompt_tokens"), int):
                prompt_tokens = u["prompt_tokens"]
            if isinstance(u.get("completion_tokens"), int):
                completion_tokens = u["completion_tokens"]

    t1 = perf_counter()
    total_time_s = t1 - t0

    if ttft_s is None:
        ttft_s = float("nan")

    tokens_per_second = None
    if isinstance(completion_tokens, int) and total_time_s > 0:
        gen_time = total_time_s - (ttft_s if ttft_s == ttft_s else 0.0)  # handle NaN
        if gen_time > 0:
            tokens_per_second = completion_tokens / gen_time

    return total_time_s, ttft_s, prompt_tokens, completion_tokens, tokens_per_second


def benchmark_vllm(
    server: str,
    prompt: str,
    runs: int,
    max_tokens: int,
    temperature: float,
    timeout: float,
    stream: bool,
    model_override: Optional[str],
):
    width = min(_term_width(), 96)

    chat_endpoint, models_endpoint = build_urls(server)
    model_name = model_override or get_loaded_model(models_endpoint, timeout=timeout)

    # "Pretty" URL presentation
    host = _host(server)
    api_path = "/v1/chat/completions"
    models_path = "/v1/models"

    header_lines = [
        f"Model       : {cyan(model_name)}",
        f"Host        : {host}",
        f"API         : {api_path}",
        f"Models      : {models_path}",
        f"Mode        : {'stream (TTFT enabled)' if stream else 'non-stream'}",
        f"Runs        : {runs}",
        f"Max tokens  : {max_tokens}",
        f"Temp        : {temperature}",
        f"Timeout     : {timeout:.1f}s",
    ]

    print(_box("vLLM Benchmark", header_lines, width=width))

    totals: list[float] = []
    ttfts: list[float] = []
    speeds: list[float] = []
    prompt_tok_seen: Optional[int] = None
    completion_tok_seen: Optional[int] = None

    table_w = width
    print(dim(_hr("─", table_w)))
    print(f"{'#':>2}  {'Total(s)':>8}  {'TTFT':>10}  {'pTok':>5}  {'cTok':>5}  {'Speed(tok/s)':>13}")
    print(dim(_hr("─", table_w)))

    for i in range(runs):
        if stream:
            total_s, ttft_s, p_tok, c_tok, tps = run_once_stream(
                chat_endpoint, model_name, prompt, max_tokens, temperature, timeout
            )
            totals.append(total_s)
            if ttft_s == ttft_s:  # not NaN
                ttfts.append(ttft_s)
        else:
            total_s, p_tok, c_tok, tps = run_once_non_stream(
                chat_endpoint, model_name, prompt, max_tokens, temperature, timeout
            )
            totals.append(total_s)
            ttft_s = float("nan")

        if isinstance(tps, (int, float)) and tps == tps:
            speeds.append(float(tps))

        if isinstance(p_tok, int):
            prompt_tok_seen = p_tok
        if isinstance(c_tok, int):
            completion_tok_seen = c_tok

        ttft_str = _fmt_ms(ttft_s) if stream else "N/A"
        speed_str = _fmt_num(tps, "{:.2f}")

        # Light color accents:
        total_disp = f"{total_s:>8.3f}"
        ttft_disp = cyan(f"{ttft_str:>10}") if stream and ttft_str != "N/A" else f"{ttft_str:>10}"
        speed_disp = green(f"{speed_str:>13}") if speed_str != "N/A" else f"{speed_str:>13}"

        print(
            f"{i+1:>2}  "
            f"{total_disp}  "
            f"{ttft_disp}  "
            f"{(str(p_tok) if isinstance(p_tok, int) else '?'):>5}  "
            f"{(str(c_tok) if isinstance(c_tok, int) else '?'):>5}  "
            f"{speed_disp}"
        )

    print(dim(_hr("─", table_w)))

    # Summary stats
    avg_total = sum(totals) / len(totals) if totals else float("nan")
    p50_total = _pct(totals, 50)
    p95_total = _pct(totals, 95)

    avg_ttft = (sum(ttfts) / len(ttfts)) if ttfts else float("nan")
    p50_ttft = _pct(ttfts, 50)
    p95_ttft = _pct(ttfts, 95)

    avg_speed = (sum(speeds) / len(speeds)) if speeds else float("nan")
    p50_speed = _pct(speeds, 50)
    p95_speed = _pct(speeds, 95)

    summary_lines = [
        f"Total time  : avg {bold(_fmt_num(avg_total, '{:.3f}'))}s | p50 {_fmt_num(p50_total, '{:.3f}')}s | p95 {_fmt_num(p95_total, '{:.3f}')}s",
        f"TTFT        : avg {cyan(_fmt_ms(avg_ttft))} | p50 {_fmt_ms(p50_ttft)} | p95 {_fmt_ms(p95_ttft)}"
        if stream
        else "TTFT        : N/A (run with --stream)",
        f"Speed       : avg {green(_fmt_num(avg_speed))} tok/s | p50 {_fmt_num(p50_speed)} | p95 {_fmt_num(p95_speed)}"
        if speeds
        else "Speed       : N/A (usage not returned by server)",
        f"Tokens      : prompt {prompt_tok_seen if prompt_tok_seen is not None else '?'} | completion {completion_tok_seen if completion_tok_seen is not None else '?'}",
    ]
    print(_box("Summary", summary_lines, width=width))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vLLM OpenAI Chat Completions Benchmark.")

    # Default comes from top-of-file SERVER
    parser.add_argument(
        "--server",
        type=str,
        default=SERVER,
        help="Base server URL. Defaults to SERVER constant at top of file.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=(
            "Explain the fundamental principles of quantum computing, including superposition and entanglement, and discuss "
            "the primary challenges in building a large-scale quantum computer."
        ),
        help="Prompt to use.",
    )
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="Number of benchmark iterations.")
    parser.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Max tokens per run.")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Sampling temperature.")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="HTTP timeout seconds.")
    parser.add_argument("--stream", action="store_true", help="Enable streaming to measure TTFT.")
    parser.add_argument("--model", type=str, default=None, help="Override model name (else use /v1/models).")

    args = parser.parse_args()

    # Guard: fail fast if placeholder wasn't replaced (but allow CLI override)
    server = args.server.strip()
    if server in ("REPLACE_ME", "") or "your-vllm-host.example.com" in server or "<your-route-hostname>" in server:
        raise SystemExit(
            "SERVER is still a placeholder. Set SERVER at top of file (e.g. https://your-vllm-host.example.com) "
            "or run with --server https://<your-route-hostname>."
        )

    benchmark_vllm(
        server=server,
        prompt=args.prompt,
        runs=args.runs,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout=args.timeout,
        stream=args.stream,
        model_override=args.model,
    )
