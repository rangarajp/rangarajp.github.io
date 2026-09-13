from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from typing import Any, Dict
import traceback
import uvicorn
import os
from .store import ModelStore, get_models_dir
from .manager import ModelManager

app = FastAPI(title="Multi-Model LLM Serving Demo")

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "models.json"
model_store = ModelStore(str(CONFIG_PATH))
model_manager = ModelManager(model_store, max_models=2)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    input_data: Any


class GenerateRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    prompt: str
    max_new_tokens: int = 64


def _get_worker_or_500(model_id: str):
    try:
        worker = model_manager.get_model_worker(model_id)
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        raise HTTPException(
            status_code=500,
            detail={"error": str(e), "traceback": tb, "models_dir": str(get_models_dir())},
        )
    if not worker:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    return worker


@app.post("/predict")
def predict(request: PredictionRequest):
    worker = _get_worker_or_500(request.model_id)
    try:
        return worker.predict(request.input_data)
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        raise HTTPException(status_code=500, detail={"error": str(e), "traceback": tb})


@app.post("/generate")
def generate(request: GenerateRequest):
    worker = _get_worker_or_500(request.model_id)
    try:
        return worker.predict(
            {"prompt": request.prompt, "max_new_tokens": request.max_new_tokens}
        )
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        raise HTTPException(status_code=500, detail={"error": str(e), "traceback": tb})


@app.get("/models")
def list_models() -> Dict[str, Any]:
    return {
        "models_dir": str(get_models_dir()),
        "max_models": model_manager.max_models,
        "available_models": {
            mid: {**meta.model_dump(), "resolved_path": str(meta.resolved_path())}
            for mid, meta in model_store.list_models().items()
        },
        "loaded_models": model_manager.list_loaded_models(),
    }


class CacheConfigRequest(BaseModel):
    max_models: int = 2


@app.post("/admin/cache/clear")
def admin_clear_cache() -> Dict[str, Any]:
    """Unload all models — use before cold-start timing tests."""
    evicted = model_manager.clear_cache()
    return {"evicted": evicted, "loaded_models": model_manager.list_loaded_models()}


@app.post("/admin/cache/config")
def admin_cache_config(request: CacheConfigRequest) -> Dict[str, Any]:
    """Set LRU capacity (e.g. max_models=1 to demo eviction)."""
    try:
        model_manager.set_max_models(request.max_models)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "max_models": model_manager.max_models,
        "loaded_models": model_manager.list_loaded_models(),
    }


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port)
