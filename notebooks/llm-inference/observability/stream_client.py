"""SSE client for single_model_llm_serving POST /generate_stream."""
from __future__ import annotations

import json
import time
from typing import Any

import httpx

from metrics import RequestMetrics


def stream_generate(
    base_url: str,
    prompt: str,
    *,
    timeout: float = 120.0,
    client: httpx.Client | None = None,
) -> RequestMetrics:
    """POST /generate_stream and timestamp each SSE token."""
    url = base_url.rstrip("/") + "/generate_stream"
    own_client = client is None
    client = client or httpx.Client(trust_env=False, proxy=None)
    metrics = RequestMetrics(prompt=prompt, t_send=time.perf_counter())
    try:
        with client.stream(
            "POST",
            url,
            json={"prompt": prompt},
            timeout=timeout,
        ) as response:
            if response.status_code != 200:
                body = response.read().decode("utf-8", errors="replace")
                metrics.error = f"HTTP {response.status_code}: {body[:300]}"
                return metrics
            for line in response.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data: dict[str, Any] = json.loads(line[6:])
                metrics.token_times.append(time.perf_counter())
                metrics.tokens.append(data.get("token", ""))
                sid = data.get("sequence_id")
                if sid:
                    metrics.sequence_id = sid
        if not metrics.token_times:
            metrics.error = "no tokens received"
    except Exception as exc:  # noqa: BLE001 — surface to metrics
        metrics.error = str(exc)
    finally:
        if own_client:
            client.close()
    return metrics


async def stream_generate_async(
    base_url: str,
    prompt: str,
    *,
    timeout: float = 120.0,
    client: httpx.AsyncClient | None = None,
) -> RequestMetrics:
    """Async variant for concurrent sweeps."""
    url = base_url.rstrip("/") + "/generate_stream"
    own_client = client is None
    client = client or httpx.AsyncClient(trust_env=False, proxy=None)
    metrics = RequestMetrics(prompt=prompt, t_send=time.perf_counter())
    try:
        async with client.stream(
            "POST",
            url,
            json={"prompt": prompt},
            timeout=timeout,
        ) as response:
            if response.status_code != 200:
                body = (await response.aread()).decode("utf-8", errors="replace")
                metrics.error = f"HTTP {response.status_code}: {body[:300]}"
                return metrics
            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data: dict[str, Any] = json.loads(line[6:])
                metrics.token_times.append(time.perf_counter())
                metrics.tokens.append(data.get("token", ""))
                sid = data.get("sequence_id")
                if sid:
                    metrics.sequence_id = sid
        if not metrics.token_times:
            metrics.error = "no tokens received"
    except Exception as exc:  # noqa: BLE001
        metrics.error = str(exc)
    finally:
        if own_client:
            await client.aclose()
    return metrics
