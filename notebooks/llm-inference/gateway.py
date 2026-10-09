from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from urllib.parse import urlparse
import httpx
import os
import secrets
import time
import uuid

from model_registry import MODEL_REGISTRY
from routing import choose_model, get_fallback, get_model_config, routing_state
from metrics import metrics_store


app = FastAPI(
    title="Multi-Model LLM Gateway"
)


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=10000)
    model: str | None = None
    max_tokens: int = Field(default=64, ge=1, le=512)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


async def is_healthy(client, model_cfg):
    try:
        response = await client.get(
            model_cfg["health"],
            timeout=3.0,
        )
        return response.status_code == 200
    except httpx.HTTPError:
        return False


def record_failure(model_name, latency_seconds=0.0, failure_reason="unknown"):
    metrics_store.record_request(
        model_name=model_name,
        success=False,
        latency_seconds=latency_seconds,
        output_tokens=0,
        failure_reason=failure_reason,
    )


def require_api_key(api_key):
    expected_api_key = os.environ.get("GATEWAY_API_KEY")
    if not expected_api_key:
        raise HTTPException(
            status_code=503,
            detail="API authentication is not configured",
        )

    if not api_key or not secrets.compare_digest(api_key, expected_api_key):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )


@app.get("/health")
async def health():
    result = {}

    async with httpx.AsyncClient(trust_env=False) as client:
        for model_name, cfg in MODEL_REGISTRY.items():
            result[model_name] = {
                "healthy": await is_healthy(client, cfg)
            }

    return result


@app.get("/metrics")
def metrics():
    return metrics_store.snapshot()


@app.get("/models")
def models():
    return {
        "models": list(MODEL_REGISTRY.keys())
    }


@app.get("/status")
async def status():
    result = {
        "gateway": {
            "healthy": True,
            "host": "127.0.0.1",
            "port": 8000,
            "endpoint": "http://127.0.0.1:8000",
        },
        "models": {},
        "routing": routing_state.snapshot(),
        "metrics": metrics_store.snapshot(),
    }

    async with httpx.AsyncClient(trust_env=False) as client:
        for model_name, cfg in MODEL_REGISTRY.items():
            parsed_endpoint = urlparse(cfg["endpoint"])
            result["models"][model_name] = {
                "healthy": await is_healthy(client, cfg),
                "host": parsed_endpoint.hostname,
                "port": parsed_endpoint.port,
                "endpoint": cfg["endpoint"],
                "health_endpoint": cfg["health"],
                "served_name": cfg["served_name"],
            }

    return result


@app.get("/ui", include_in_schema=False)
def ui():
    return FileResponse("dashboard.html")


@app.post("/generate")
async def generate(
    request: GenerateRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    require_api_key(x_api_key)
    request_id = str(uuid.uuid4())[:8]

    try:
        selected_model = choose_model(request.prompt, request.model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    routing_reason = (
        "explicit_model"
        if request.model
        else "capability_priority_load_latency"
    )
    candidates = [selected_model]
    fallback_model = get_fallback(selected_model)
    if fallback_model and fallback_model not in candidates:
        candidates.append(fallback_model)

    timeout = httpx.Timeout(
        connect=5.0,
        read=120.0,
        write=10.0,
        pool=5.0,
    )

    async with httpx.AsyncClient(
        trust_env=False,
        timeout=timeout,
    ) as client:
        for attempt, model_name in enumerate(candidates):
            model_cfg = get_model_config(model_name)
            if model_cfg is None or not await is_healthy(client, model_cfg):
                record_failure(model_name, failure_reason="unhealthy")
                continue

            routing_state.acquire(model_name)

            payload = {
                "model": model_cfg["served_name"],
                "prompt": request.prompt,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
            }
            start = time.perf_counter()

            try:
                response = await client.post(
                    model_cfg["endpoint"],
                    json=payload,
                )
            except httpx.HTTPError:
                latency = time.perf_counter() - start
                routing_state.release(model_name, latency)
                record_failure(model_name, latency, "request_error")
                continue

            latency = time.perf_counter() - start
            routing_state.release(model_name, latency)
            if response.status_code >= 500 and attempt == 0:
                record_failure(model_name, latency, "server_error")
                continue
            if response.status_code != 200:
                record_failure(model_name, latency, "client_error")
                raise HTTPException(
                    status_code=response.status_code,
                    detail="Model rejected the request",
                )

            try:
                data = response.json()
                usage = data.get("usage", {})
                text = data["choices"][0]["text"]
            except (KeyError, IndexError, TypeError, ValueError):
                record_failure(model_name, latency, "invalid_response")
                raise HTTPException(
                    status_code=502,
                    detail="Model returned an invalid response",
                ) from None

            completion_tokens = usage.get("completion_tokens", 0) or 0
            metrics_store.record_request(
                model_name=model_name,
                success=True,
                latency_seconds=latency,
                output_tokens=completion_tokens,
                fallback_used=model_name != selected_model,
            )
            tokens_per_second = (
                completion_tokens / latency
                if latency > 0
                else 0
            )

            print(
                f"[{request_id}] model={model_name} status={response.status_code} "
                f"latency={latency:.2f}s prompt_tokens={usage.get('prompt_tokens')} "
                f"completion_tokens={completion_tokens} "
                f"tokens_per_second={tokens_per_second:.2f}"
            )

            return {
                "request_id": request_id,
                "requested_model": request.model,
                "model": model_name,
                "routing_reason": routing_reason,
                "fallback_used": model_name != selected_model,
                "latency_seconds": round(latency, 3),
                "tokens_per_second": round(tokens_per_second, 2),
                "usage": usage,
                "text": text,
            }

    raise HTTPException(
        status_code=503,
        detail="No model is currently available",
    )
