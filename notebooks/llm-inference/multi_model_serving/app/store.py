import json
import os
from pathlib import Path
from typing import Dict, Optional
from pydantic import BaseModel


# app/store.py — default MODELS_DIR is <repo>/models unless env MODELS_DIR is set
REPO_ROOT = Path(__file__).resolve().parents[4]


def get_models_dir() -> Path:
    """Local checkpoints root. Override with env MODELS_DIR (set in main.ipynb)."""
    return Path(
        os.environ.get("MODELS_DIR", str(REPO_ROOT / "models"))
    ).expanduser().resolve()


MODELS_DIR = get_models_dir()


class ModelMetadata(BaseModel):
    id: str
    name: str
    path: str  # absolute, folder name under MODELS_DIR, or "models/<name>"
    type: str
    framework: str
    version: str
    description: str

    def resolved_path(self) -> Path:
        """
        Resolve checkpoint path. Tries several layouts so both
        "Qwen2.5-..." and "models/Qwen2.5-..." work with MODELS_DIR.
        """
        p = Path(self.path).expanduser()
        if p.is_absolute():
            return p.resolve()

        models_dir = get_models_dir()
        candidates = [
            models_dir / p,           # MODELS_DIR / "Qwen2.5-..."
            models_dir / p.name,      # MODELS_DIR / basename
            REPO_ROOT / p,            # repo / "models/Qwen2.5-..."
        ]
        # Strip leading "models/" if present to avoid MODELS_DIR/models/...
        if p.parts and p.parts[0] == "models" and len(p.parts) > 1:
            candidates.insert(0, models_dir / Path(*p.parts[1:]))

        for c in candidates:
            if c.exists():
                return c.resolve()

        # Prefer MODELS_DIR / folder-name even if missing (clearer error later)
        return (models_dir / p.name).resolve()


class ModelStore:
    def __init__(self, config_path: str):
        self.models: Dict[str, ModelMetadata] = {}
        self._load_config(config_path)

    def _load_config(self, config_path: str):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
            for model in config["models"]:
                self.models[model["id"]] = ModelMetadata(**model)

    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        return self.models.get(model_id)

    def list_models(self) -> Dict[str, ModelMetadata]:
        return self.models
