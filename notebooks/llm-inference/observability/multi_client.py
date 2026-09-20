"""HTTP helpers for multi_model_serving (:8001)."""
from __future__ import annotations

import time
from typing import Any

import httpx

from metrics import E2ERequestMetrics


def list_models(
    base_url: str,
    *,
    timeout: float = 30.0,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/models"
    own = client is None
    client = client or httpx.Client(trust_env=False, proxy=None)
    try:
        r = client.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()
    finally:
        if own:
            client.close()


def clear_cache(
    base_url: str,
    *,
    timeout: float = 60.0,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/admin/cache/clear"
    own = client is None
    client = client or httpx.Client(trust_env=False, proxy=None)
    try:
        r = client.post(url, timeout=timeout)
        r.raise_for_status()
        return r.json() if r.content else {"ok": True}
    finally:
        if own:
            client.close()


def generate(
    base_url: str,
    model_id: str,
    prompt: str,
    *,
    max_new_tokens: int | None = None,
    timeout: float = 120.0,
    client: httpx.Client | None = None,
) -> E2ERequestMetrics:
    """Blocking POST /generate with wall-clock e2e timing."""
    url = base_url.rstrip("/") + "/generate"
    payload: dict[str, Any] = {"model_id": model_id, "prompt": prompt}
    if max_new_tokens is not None:
        payload["max_new_tokens"] = max_new_tokens

    own = client is None
    client = client or httpx.Client(trust_env=False, proxy=None)
    t_send = time.perf_counter()
    try:
        r = client.post(url, json=payload, timeout=timeout)
        t_done = time.perf_counter()
        if r.status_code != 200:
            return E2ERequestMetrics(
                prompt=prompt,
                model_id=model_id,
                t_send=t_send,
                t_done=t_done,
                error=f"HTTP {r.status_code}: {r.text[:300]}",
            )
        data = r.json()
        return E2ERequestMetrics(
            prompt=prompt,
            model_id=model_id,
            t_send=t_send,
            t_done=t_done,
            generated_text=data.get("generated_text", ""),
        )
    except Exception as exc:  # noqa: BLE001
        return E2ERequestMetrics(
            prompt=prompt,
            model_id=model_id,
            t_send=t_send,
            t_done=time.perf_counter(),
            error=str(exc),
        )
    finally:
        if own:
            client.close()


async def generate_async(
    base_url: str,
    model_id: str,
    prompt: str,
    *,
    max_new_tokens: int | None = None,
    timeout: float = 120.0,
    client: httpx.AsyncClient | None = None,
) -> E2ERequestMetrics:
    url = base_url.rstrip("/") + "/generate"
    payload: dict[str, Any] = {"model_id": model_id, "prompt": prompt}
    if max_new_tokens is not None:
        payload["max_new_tokens"] = max_new_tokens

    own = client is None
    client = client or httpx.AsyncClient(trust_env=False, proxy=None)
    t_send = time.perf_counter()
    try:
        r = await client.post(url, json=payload, timeout=timeout)
        t_done = time.perf_counter()
        if r.status_code != 200:
            return E2ERequestMetrics(
                prompt=prompt,
                model_id=model_id,
                t_send=t_send,
                t_done=t_done,
                error=f"HTTP {r.status_code}: {r.text[:300]}",
            )
        data = r.json()
        return E2ERequestMetrics(
            prompt=prompt,
            model_id=model_id,
            t_send=t_send,
            t_done=t_done,
            generated_text=data.get("generated_text", ""),
        )
    except Exception as exc:  # noqa: BLE001
        return E2ERequestMetrics(
            prompt=prompt,
            model_id=model_id,
            t_send=t_send,
            t_done=time.perf_counter(),
            error=str(exc),
        )
    finally:
        if own:
            await client.aclose()
