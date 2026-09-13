"""Load machine-local paths. Never commit local_paths.json — copy from local_paths.example.json."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _candidates() -> list[Path]:
    env = os.environ.get("LLM_INFERENCE_PATHS")
    out: list[Path] = []
    if env:
        out.append(Path(env))
    here = Path(__file__).resolve().parent
    out.append(here / "local_paths.json")
    # cwd variants (notebook may chdir)
    out.append(Path.cwd() / "local_paths.json")
    out.append(Path.cwd().parent / "local_paths.json")
    out.append(Path.cwd().parent.parent / "local_paths.json")
    return out


def load_local_paths() -> dict[str, Any]:
    for path in _candidates():
        if path.is_file():
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            data["_config_file"] = str(path.resolve())
            return data
    example = Path(__file__).resolve().parent / "local_paths.example.json"
    raise FileNotFoundError(
        "Missing local_paths.json. Copy local_paths.example.json → local_paths.json "
        f"and set your machine paths. Looked next to {example}"
    )


def models_dir(cfg: dict[str, Any] | None = None) -> Path:
    cfg = cfg or load_local_paths()
    return Path(cfg["MODELS_DIR"]).expanduser().resolve()


def model_path(key: str, cfg: dict[str, Any] | None = None) -> Path:
    """key: QWEN_MODEL or TINYLLAMA_MODEL (folder name under MODELS_DIR)."""
    cfg = cfg or load_local_paths()
    return models_dir(cfg) / cfg[key]
