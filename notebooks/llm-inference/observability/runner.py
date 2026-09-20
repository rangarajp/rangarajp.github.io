"""Concurrent load runners for Phase 1 benches."""
from __future__ import annotations

import asyncio
import concurrent.futures
import csv
import json
from pathlib import Path
from typing import Any, Coroutine, Sequence, TypeVar

import httpx

from gpu_metrics import GpuSampler
from metrics import RequestMetrics, aggregate_e2e, aggregate_stream
from multi_client import generate_async
from stream_client import stream_generate, stream_generate_async

T = TypeVar("T")


def _run_async(coro: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine from sync code, including inside Jupyter's event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    # Jupyter / notebooks already have a loop — asyncio.run() is illegal there.
    # Run the coroutine in a fresh thread with its own loop.
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _ensure_results_dir(path: Path | str | None = None) -> Path:
    root = Path(__file__).resolve().parent / "results"
    root.mkdir(parents=True, exist_ok=True)
    if path is None:
        return root
    out = Path(path)
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


async def run_stream_concurrency(
    base_url: str,
    prompts: Sequence[str],
    *,
    timeout: float = 180.0,
    sample_gpu: bool = True,
    gpu_interval_s: float = 0.5,
) -> dict[str, Any]:
    """Fire len(prompts) concurrent /generate_stream calls."""
    sampler = GpuSampler(interval_s=gpu_interval_s) if sample_gpu else None
    if sampler:
        sampler.start()

    async with httpx.AsyncClient(trust_env=False, proxy=None) as client:
        tasks = [
            stream_generate_async(base_url, p, timeout=timeout, client=client)
            for p in prompts
        ]
        results: list[RequestMetrics] = list(await asyncio.gather(*tasks))

    gpu = sampler.stop() if sampler else {}
    summary = aggregate_stream(results)
    summary["concurrency"] = len(prompts)
    summary["gpu"] = gpu
    summary["requests"] = [r.to_dict() for r in results]
    return summary


def run_stream_concurrency_sync(
    base_url: str,
    prompts: Sequence[str],
    **kwargs: Any,
) -> dict[str, Any]:
    return _run_async(run_stream_concurrency(base_url, prompts, **kwargs))


async def concurrency_sweep(
    base_url: str,
    prompt: str,
    levels: Sequence[int] = (1, 2, 4, 8),
    *,
    warmup: bool = True,
    timeout: float = 180.0,
    sample_gpu: bool = True,
) -> list[dict[str, Any]]:
    """Sweep concurrency; reuse the same prompt text at each level."""
    if warmup:
        await stream_generate_async(base_url, prompt, timeout=timeout)

    rows: list[dict[str, Any]] = []
    for n in levels:
        prompts = [f"{prompt} [{i}]" for i in range(n)]
        row = await run_stream_concurrency(
            base_url,
            prompts,
            timeout=timeout,
            sample_gpu=sample_gpu,
        )
        # Flatten GPU into top-level for CSV friendliness
        gpu = row.get("gpu") or {}
        flat = {
            "concurrency": n,
            "n_ok": row["n_ok"],
            "n_err": row["n_err"],
            "ttft_p50_s": row["ttft_p50_s"],
            "ttft_p95_s": row["ttft_p95_s"],
            "tpot_p50_s": row["tpot_p50_s"],
            "e2e_p50_s": row["e2e_p50_s"],
            "throughput_tok_s": row["throughput_tok_s"],
            "throughput_req_s": row["throughput_req_s"],
            "wall_s": row["wall_s"],
            "gpu_util_mean_pct": gpu.get("gpu_util_mean_pct"),
            "gpu_util_max_pct": gpu.get("gpu_util_max_pct"),
            "mem_used_mean_mib": gpu.get("mem_used_mean_mib"),
        }
        rows.append(flat)
    return rows


def concurrency_sweep_sync(
    base_url: str,
    prompt: str,
    levels: Sequence[int] = (1, 2, 4, 8),
    **kwargs: Any,
) -> list[dict[str, Any]]:
    return _run_async(concurrency_sweep(base_url, prompt, levels, **kwargs))


async def prompt_length_sweep(
    base_url: str,
    lengths: Sequence[int] = (10, 50, 100, 250, 500),
    *,
    concurrency: int = 1,
    repeats: int = 3,
    model_dir: str | None = None,
    warmup: bool = True,
    timeout: float = 300.0,
    sample_gpu: bool = True,
) -> list[dict[str, Any]]:
    """Vary prompt length; keep concurrency and decode length ~fixed (stream cap ~20).

    Default concurrency=1 isolates prefill → TTFT (no queue noise from oversubscribe).
    """
    from prompts import count_tokens, make_prompt

    if warmup:
        await stream_generate_async(
            base_url,
            make_prompt(32, model_dir=model_dir),
            timeout=timeout,
        )

    rows: list[dict[str, Any]] = []
    for n in lengths:
        prompt = make_prompt(n, model_dir=model_dir)
        measured = count_tokens(prompt, model_dir)
        # Same prompt text for each concurrent copy / repeat wave
        wave_prompts = [f"{prompt}\n[rep={r}]" for r in range(max(1, concurrency))]

        # Aggregate across repeats (median of wave medians would need more code;
        # here we average flat fields across repeats for stability)
        wave_rows: list[dict[str, Any]] = []
        for _ in range(max(1, repeats)):
            summary = await run_stream_concurrency(
                base_url,
                wave_prompts,
                timeout=timeout,
                sample_gpu=sample_gpu,
            )
            gpu = summary.get("gpu") or {}
            wave_rows.append(
                {
                    "ttft_p50_s": summary["ttft_p50_s"],
                    "ttft_p95_s": summary["ttft_p95_s"],
                    "tpot_p50_s": summary["tpot_p50_s"],
                    "e2e_p50_s": summary["e2e_p50_s"],
                    "throughput_tok_s": summary["throughput_tok_s"],
                    "wall_s": summary["wall_s"],
                    "n_ok": summary["n_ok"],
                    "n_err": summary["n_err"],
                    "gpu_util_mean_pct": gpu.get("gpu_util_mean_pct"),
                    "mem_used_mean_mib": gpu.get("mem_used_mean_mib"),
                }
            )

        def _avg(key: str) -> float | None:
            vals = [w[key] for w in wave_rows if w.get(key) is not None]
            return (sum(vals) / len(vals)) if vals else None

        rows.append(
            {
                "target_prompt_tokens": n,
                "measured_prompt_tokens": measured,
                "concurrency": concurrency,
                "repeats": repeats,
                "n_ok": sum(w["n_ok"] for w in wave_rows),
                "n_err": sum(w["n_err"] for w in wave_rows),
                "ttft_p50_s": _avg("ttft_p50_s"),
                "ttft_p95_s": _avg("ttft_p95_s"),
                "tpot_p50_s": _avg("tpot_p50_s"),
                "e2e_p50_s": _avg("e2e_p50_s"),
                "throughput_tok_s": _avg("throughput_tok_s"),
                "wall_s": _avg("wall_s"),
                "gpu_util_mean_pct": _avg("gpu_util_mean_pct"),
                "mem_used_mean_mib": _avg("mem_used_mean_mib"),
            }
        )
    return rows


def prompt_length_sweep_sync(
    base_url: str,
    lengths: Sequence[int] = (10, 50, 100, 250, 500),
    **kwargs: Any,
) -> list[dict[str, Any]]:
    return _run_async(prompt_length_sweep(base_url, lengths, **kwargs))


async def run_multi_e2e_batch(
    base_url: str,
    jobs: Sequence[tuple[str, str]],
    *,
    max_new_tokens: int | None = 64,
    timeout: float = 180.0,
    sample_gpu: bool = True,
) -> dict[str, Any]:
    """Concurrent multi-model /generate. jobs = [(model_id, prompt), ...]."""
    sampler = GpuSampler() if sample_gpu else None
    if sampler:
        sampler.start()

    async with httpx.AsyncClient(trust_env=False, proxy=None) as client:
        tasks = [
            generate_async(
                base_url,
                model_id,
                prompt,
                max_new_tokens=max_new_tokens,
                timeout=timeout,
                client=client,
            )
            for model_id, prompt in jobs
        ]
        results = list(await asyncio.gather(*tasks))

    gpu = sampler.stop() if sampler else {}
    summary = aggregate_e2e(results)
    summary["gpu"] = gpu
    summary["requests"] = [r.to_dict() for r in results]
    return summary


def save_csv(rows: Sequence[dict[str, Any]], path: Path | str) -> Path:
    out = _ensure_results_dir(path)
    if not rows:
        out.write_text("", encoding="utf-8")
        return out
    keys = list(rows[0].keys())
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    return out


def save_json(payload: Any, path: Path | str) -> Path:
    out = _ensure_results_dir(path)
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return out


def smoke_stream(base_url: str, prompt: str = "Say hello in one word.") -> dict[str, Any]:
    """Single-request sanity check (sync)."""
    m = stream_generate(base_url, prompt)
    return m.to_dict()
