"""Resolve lab base URLs from local_paths.json."""
from __future__ import annotations

import sys
from pathlib import Path


def _ensure_parent_on_path() -> None:
    parent = Path(__file__).resolve().parent.parent
    s = str(parent)
    if s not in sys.path:
        sys.path.insert(0, s)


def load_urls() -> dict[str, str]:
    _ensure_parent_on_path()
    try:
        from local_paths import load_local_paths

        cfg = load_local_paths()
    except Exception:
        cfg = {}
    return {
        "SINGLE_MODEL_URL": cfg.get("SINGLE_MODEL_URL", "http://127.0.0.1:8000"),
        "MULTI_MODEL_URL": cfg.get("MULTI_MODEL_URL", "http://127.0.0.1:8001"),
    }
