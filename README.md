Here’s an updated **single README** that clearly supports **both options**:

* **Option 1:** Run the scripts directly (CLI)
* **Option 2:** Run the web UI via Docker

It also **strongly recommends `--server`** (no editing code), keeps your existing sections, and adds a small “Web UI (Docker)” quick start. Based on your current README content. 

---

# vLLM Perf Tools (CLI + Web UI)

Two small, dependency-light tools for testing **vLLM** deployments that expose the **OpenAI-compatible Chat Completions API**.

* **`vllm-bench.py`**: single-request benchmarking (repeat N times)
* **`vllm-load.py`**: concurrent load testing (N total requests, limited by concurrency)
* **Web UI** (Docker): run benchmarks/load tests from your browser

Both scripts support:

* ✅ Pretty table/box output + ANSI color (auto-disabled when stdout is not a TTY)
* ✅ `--server` flag (**recommended** way to point at your endpoint — no code edits required)
* ✅ `--insecure` to skip TLS verification (curl `-k` equivalent)
* ✅ Optional Bearer token auth (`--token` or `--token-env`)

## Inspired by

* Country Boy Computers blog: [https://countryboycomputersbg.com/blog/](https://countryboycomputersbg.com/blog/)
* Country Boy Computers on YouTube: [https://www.youtube.com/@CountryBoyComputers](https://www.youtube.com/@CountryBoyComputers)

---

## Choose how you want to run it

### Option 1: CLI scripts (no Docker)

Use this if you want quick terminal testing, automation, or CI.

### Option 2: Web UI (Docker)

Use this if you want a browser form + clickable test execution.

---

## Requirements

### CLI scripts

* Python 3.x
* `requests`

Install:

```bash
pip install requests
```

### Web UI (Docker)

* Podman or Docker
* Uses the included `Dockerfile` + `requirements.txt`

---

## Quick start (Option 1: CLI)

### 1) Benchmark a server (single request repeated)

**Streaming mode (recommended):**

```bash
python3 vllm-bench.py \
  --server https://my-vllm.apps.example.net \
  --stream \
  --runs 5 \
  --max_tokens 512
```

**Non-stream mode:**

```bash
python3 vllm-bench.py \
  --server https://my-vllm.apps.example.net \
  --runs 5 \
  --max_tokens 512
```
<img width="775" height="667" alt="image" src="https://github.com/user-attachments/assets/8cfd25d7-cee6-4bbb-9299-a1dbb1951c80" />
---

### 2) Load test a server (concurrent requests)

Example: **20 total requests**, up to **20 in flight** at a time:

```bash
python3 vllm-load.py \
  --server https://my-vllm.apps.example.net \
  --model gpt-oss-20b \
  --requests 20 \
  --concurrency 20 \
  --stream \
  --max_tokens 512
```

If you want multiple “waves” of requests (more samples without increasing pressure), do `requests > concurrency`:

```bash
python3 vllm-load.py \
  --server https://my-vllm.apps.example.net \
  --model gpt-oss-20b \
  --requests 200 \
  --concurrency 20 \
  --stream
```
<img width="889" height="1026" alt="image" src="https://github.com/user-attachments/assets/88863388-8951-4a2e-8660-5591f53d0c75" />

### Do `--requests` and `--concurrency` need to match?

No.

* `--concurrency` = max simultaneous in-flight requests (pressure)
* `--requests` = total number of requests you want to send (sample size)

So:

* `--requests 20 --concurrency 20` = one “wave” of 20 at once
* `--requests 200 --concurrency 20` = 10 waves (still only 20 at a time)

---

## Quick start (Option 2: Web UI in Docker)

Build:

```bash
podman build -t vllm-perf-web:latest .
```

Run:

```bash
podman run --rm -p 8080:8080 vllm-perf-web:latest
```

Open:

* [http://localhost:8080](http://localhost:8080)

In the UI you can enter:

* Server URL
* Optional token
* Optional `--insecure` / skip TLS verify
* Bench vs Load settings (runs, requests, concurrency, stream, max_tokens, etc.)

---

## TLS: self-signed / lab endpoints

Skip TLS verification (curl `-k` equivalent):

```bash
python3 vllm-bench.py --server https://my-vllm.apps.example.net --stream --insecure
python3 vllm-load.py  --server https://my-vllm.apps.example.net --stream --insecure --requests 50 --concurrency 10
```

---

## Auth: endpoints requiring a token

### Recommended: env var

```bash
export VLLM_TOKEN="REDACTED"

python3 vllm-bench.py --server https://my-vllm.apps.example.net --stream
python3 vllm-load.py  --server https://my-vllm.apps.example.net --stream --requests 50 --concurrency 10
```

### Or pass directly

```bash
python3 vllm-bench.py --server https://my-vllm.apps.example.net --stream --token "REDACTED"
python3 vllm-load.py  --server https://my-vllm.apps.example.net --stream --token "REDACTED" --requests 50 --concurrency 10
```

### Use a different env var name

```bash
export MY_TOKEN="REDACTED"
python3 vllm-bench.py --server https://my-vllm.apps.example.net --token-env MY_TOKEN --stream
```

> The tools never print your token.

---

## What the columns mean

Both tools use the same column names and formatting.

* **Total(s)**: End-to-end time for the request (seconds).
* **TTFB**: *Time To First Byte / first stream event* (milliseconds).
  Time until the first streaming `data:` event is received and successfully parsed. Only available with `--stream`.
* **TTFT**: *Time To First Token* (milliseconds).
  Time until the first generated **content token** is observed in the stream (`delta.content`). Only available with `--stream`.
  If the server never emits `delta.content`, TTFT may fall back to TTFB.
* **pTok**: Prompt tokens (`usage.prompt_tokens`).
* **cTok**: Completion tokens (`usage.completion_tokens`).
* **e2e(tok/s)**: End-to-end throughput (tokens/sec), computed as: `cTok / Total`.
  Most stable speed metric for comparing runs/models.
* **gen(tok/s)**: Generation throughput after first token (tokens/sec), computed as: `cTok / (Total - TTFT)`.
  Can look inflated if output is buffered then streamed in bursts.
* **HTTP** (load test only): HTTP status code returned by the server.

> Note: If your vLLM deployment does not return `usage`, `pTok/cTok` and speed columns may show as `?` / `N/A`.

---

## Flags

Always refer to built-in help for the authoritative list:

```bash
python3 vllm-bench.py --help
python3 vllm-load.py --help
```

### Common flags (both tools)

* `--server SERVER`
  Base server URL (**recommended** way to set the endpoint).

* `--stream`
  Enable streaming to measure **TTFB/TTFT**.

* `--timeout SECONDS`
  HTTP timeout.

* `--temperature FLOAT`
  Sampling temperature.

* `--max_tokens N`
  Max tokens per response.

* `--model MODEL`
  Override model name (otherwise use first model returned by `/v1/models`).

* `--insecure`
  Skip TLS certificate verification (curl `-k` equivalent).

* `--token TOKEN`
  Send `Authorization: Bearer <TOKEN>`.

* `--token-env ENV_VAR`
  Read bearer token from env var (default: `VLLM_TOKEN`).
  If `--token` is provided, it wins.

### Bench-only flags

* `--runs N` — Number of benchmark iterations.
* `--prompt TEXT` — Prompt to use.

### Load-only flags

* `--requests N` — Total number of requests to send.
* `--concurrency N` — Maximum number of requests in flight at once.
* `--prompt TEXT` — Prompt template. Use `{i}` to inject request number (keeps each request unique).

---

## API compatibility

Targets vLLM’s OpenAI-compatible endpoints:

* `GET  /v1/models`
* `POST /v1/chat/completions`

Streaming mode expects SSE lines like:

* `data: {...json...}`
* `data: [DONE]`

---

## Notes

* The first request is often slower due to warm-up effects.
* If you set `--max_tokens 512` and always see `cTok = 512`, you’re consistently hitting the cap.
* Tokenization differs across models; `pTok` may vary for the same prompt.
* Under load (`vllm-load.py`), TTFT often balloons due to queueing; **e2e(tok/s)** is usually the best apples-to-apples metric.
