# Inference optimization topics

Hands-on labs live under `experiments/`.

Numbering: topic `N` → labs `N.1`, `N.2`, …

| # | Topic | Status | Labs |
|---|--------|--------|------|
| 1 | LLM Performance Benchmarks | done | [`1.1`](./experiments/1.1_single_model_performance.md) / [notebook](./experiments/1.1_single_model_performance.ipynb) — TTFT/TPOT/e2e · concurrency · prompt length · static/dynamic/continuous · chunked prefill |
| 2 | Batching — static, dynamic & continuous | in 1.1 | covered inside `1.1` |
| 3 | KV cache / PagedAttention | done | [`3.1`](./experiments/3.1_kv_cache.md) / [notebook](./experiments/3.1_kv_cache.ipynb) — with vs without · layout/HBM · paged blocks · prefix caching |
| 4 | Speculative Decoding | pending | TBD |
| 5 | Prefill Decode Disaggregation | pending | TBD |
| 6 | Prefix Caching | in 3.1 | hit/miss demo inside `3.1` §6 |
| 7 | Inference Routing | pending | TBD |
| 8 | KV Cache Offloading | pending | TBD (teased in 3.1 table) |
| 9 | Parallelism — data, tensor, pipeline, expert, hybrid | pending | TBD |
| 10 | Offline Batch Inference | pending | TBD |

## Layout

```text
observability/
  PLAN.md
  experiments/
    1.1_single_model_performance.ipynb|.md
    3.1_kv_cache.ipynb|.md
    _chunked_prefill_worker.py
    images/
```

Labs are self-contained. They load `model_path("QWEN_MODEL")` from `../local_paths.json` (default: **Qwen2.5-7B**).

## Method

```text
Predict → Measure → Explain → Change one variable → Measure again
```
