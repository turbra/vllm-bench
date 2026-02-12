# vllm-bench

Simple, dependency-light benchmark tool for **vLLM** deployments that expose the **OpenAI-compatible Chat Completions API**.

## Inspired by

- Country Boy Computers blog: https://countryboycomputersbg.com/blog/
- Country Boy Computers on YouTube: https://www.youtube.com/@CountryBoyComputers

It measures:
- **Total latency** per request
- **TTFT (Time To First Token)** when using `--stream`
- **Generation throughput (tokens/sec)** using `usage.completion_tokens`
- **Token counts**: `pTok` (prompt tokens) and `cTok` (completion tokens)

Output is formatted with box/table summaries and **ANSI color** (auto-disabled when stdout is not a TTY).

---

## Requirements

- Python 3.x
- `requests`

Install dependency:

```bash
pip install requests
````

---

## Quick start

1. Set your server at the top of `vllm-bench.py`:

```python
SERVER = "https://your-vllm-host.example.com"
```

2. Run a benchmark:

### Stream mode (recommended)

Measures TTFT and throughput:

```bash
python3 vllm-bench.py --stream --runs 5 --max_tokens 512
```

### Non-stream mode

Measures total latency and throughput (TTFT is not measurable without streaming):

```bash
python3 vllm-bench.py --runs 5 --max_tokens 512
```

---

## What the columns mean

* **Total(s)**: End-to-end time for the request (seconds)
* **TTFT**: Time to first generated token (milliseconds). Only available with `--stream`
* **pTok**: Prompt tokens (`usage.prompt_tokens`)
* **cTok**: Completion tokens (`usage.completion_tokens`)
* **Speed(tok/s)**: Approx. generation speed (tokens/sec).
  In stream mode, computed using roughly: `cTok / (Total - TTFT)`.

> Note: If your vLLM deployment does not return `usage`, `pTok/cTok/speed` may show as `?` / `N/A`.

---

## Flags

You can view all flags anytime:

```bash
python3 vllm-bench.py --help
```

Current options:

* `--server SERVER`
  Base server URL. Defaults to the `SERVER` constant at top of file.

* `--prompt PROMPT`
  Prompt to use.

* `--runs RUNS`
  Number of benchmark iterations.

* `--max_tokens MAX_TOKENS`
  Max tokens to generate per run.

* `--temperature TEMPERATURE`
  Sampling temperature.

* `--timeout TIMEOUT`
  HTTP timeout seconds.

* `--stream`
  Enable streaming to measure TTFT.

* `--model MODEL`
  Override model name (otherwise uses the first model returned by `/v1/models`).

---

## Examples

Benchmark a specific server without editing the script:

```bash
python3 vllm-bench.py --server https://my-vllm.apps.example.net --stream
```

Benchmark with a custom prompt:

```bash
python3 vllm-bench.py --stream --prompt "Summarize the plot of Hamlet in 5 bullet points."
```

Override the model (if your `/v1/models` returns multiple):

```bash
python3 vllm-bench.py --stream --model my-model-name
```

---

## API compatibility

This tool targets vLLM’s OpenAI-compatible endpoints:

* `GET  /v1/models`
* `POST /v1/chat/completions`

In streaming mode, it expects SSE responses like:

* `data: {...json...}`
* `data: [DONE]`

---

## Notes / gotchas

* **First request is often slower (TTFT)** due to warm caches, TLS session reuse, and model runtime warm-up.
* If you set `--max_tokens 512` and always see `cTok = 512`, you are consistently hitting the generation cap.
* Tokenization is model/tokenizer specific — `pTok` may differ between models for the same prompt.
