"""Phase 1 inference observability — client-side benches against local labs."""
from __future__ import annotations

from config import load_urls
from gpu_metrics import GpuSampler
from metrics import (
    E2ERequestMetrics,
    RequestMetrics,
    aggregate_e2e,
    aggregate_stream,
)
from multi_client import clear_cache, generate, generate_async, list_models
from runner import (
    concurrency_sweep,
    concurrency_sweep_sync,
    prompt_length_sweep,
    prompt_length_sweep_sync,
    run_multi_e2e_batch,
    run_stream_concurrency,
    run_stream_concurrency_sync,
    save_csv,
    save_json,
    smoke_stream,
)
from stream_client import stream_generate, stream_generate_async
from prompts import count_tokens, make_prompt, prompt_length_table

__all__ = [
    "E2ERequestMetrics",
    "GpuSampler",
    "RequestMetrics",
    "aggregate_e2e",
    "aggregate_stream",
    "clear_cache",
    "concurrency_sweep",
    "concurrency_sweep_sync",
    "count_tokens",
    "generate",
    "generate_async",
    "list_models",
    "load_urls",
    "make_prompt",
    "prompt_length_sweep",
    "prompt_length_sweep_sync",
    "prompt_length_table",
    "run_multi_e2e_batch",
    "run_stream_concurrency",
    "run_stream_concurrency_sync",
    "save_csv",
    "save_json",
    "smoke_stream",
    "stream_generate",
    "stream_generate_async",
]
