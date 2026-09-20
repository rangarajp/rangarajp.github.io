# Phase 1 observability — client benches against local labs

Benchmarks hit the **existing** FastAPI labs (Qwen / TinyLlama only). No OpenAI models, no OpenAI-compatible server.

| Lab | URL | Endpoint | Metrics |
|-----|-----|----------|---------|
| `single_model_llm_serving` | `http://127.0.0.1:8000` | `POST /generate_stream` | TTFT, ITL, TPOT, E2E, tok/s |
| `multi_model_serving` | `http://127.0.0.1:8001` | `POST /generate` | E2E, warm/cold, LRU |

GPU util/memory sampled via `nvidia-smi` during runs. KV-cache % is not exposed by the HF streaming path — omitted on purpose.

## Setup

1. Copy `../local_paths.example.json` → `../local_paths.json` if needed (URLs default to the ports above).
2. Start the lab you want to measure (from its folder / `main.ipynb` as in the blog).
3. Install client deps: `pip install -r requirements.txt`

## Experiments

```text
experiments/
  01_concurrency_sweep.ipynb   # Qwen SSE concurrency on :8000
  01_concurrency_sweep_notes.md  # how to read batches + plots from that run
  02_multi_model_e2e.ipynb     # Qwen + TinyLlama e2e on :8001
  03_prompt_length_sweep.ipynb # prefill → TTFT (concurrency=1)
```

Run notebooks with the working directory set to `observability/` (or `notebooks/llm-inference/` — `__init__.py` adds both to `sys.path`).

## Quick smoke (Python)

In a **plain script**:

```python
from config import load_urls
from runner import concurrency_sweep_sync, save_csv, smoke_stream

urls = load_urls()
print(smoke_stream(urls["SINGLE_MODEL_URL"]))

rows = concurrency_sweep_sync(
    urls["SINGLE_MODEL_URL"],
    "What is KV cache?",
    levels=(1, 2, 4),
)
print(rows)
save_csv(rows, "concurrency_sweep.csv")
```

In a **Jupyter notebook**, prefer `await concurrency_sweep(...)` (an event loop is already running). Sync helpers now fall back to a worker thread if needed.

## Method

```text
Predict → Measure → Explain → Change one variable → Measure again
```

Single-model stream currently caps at ~20 completion tokens — keep that in mind when reading TPOT.