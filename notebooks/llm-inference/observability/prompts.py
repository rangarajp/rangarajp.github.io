"""Synthetic prompts sized by approximate (or tokenizer) token count."""
from __future__ import annotations

from functools import lru_cache
from typing import Any


_FILLER = (
    "The quick brown fox studies attention, KV cache, continuous batching, "
    "prefill cost, and decode latency on a single GPU serving stack. "
)


@lru_cache(maxsize=2)
def _tokenizer(model_dir: str | None):
    if not model_dir:
        return None
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
    except Exception:
        return None


def count_tokens(text: str, model_dir: str | None = None) -> int:
    tok = _tokenizer(model_dir)
    if tok is not None:
        return len(tok.encode(text, add_special_tokens=False))
    # Rough English fallback used only if tokenizer is unavailable
    return max(1, len(text.split()))


def make_prompt(
    target_tokens: int,
    *,
    instruction: str = "In one short sentence, say what topic this passage discusses.",
    model_dir: str | None = None,
) -> str:
    """Build a prompt whose *total* length is near target_tokens (incl. instruction)."""
    if target_tokens < 8:
        target_tokens = 8

    # Grow filler until total tokens reach the target
    body = ""
    guard = 0
    while count_tokens(f"{instruction}\n\n{body}", model_dir) < target_tokens and guard < 5000:
        body += _FILLER
        guard += 1

    # Trim words if we overshoot
    text = f"{instruction}\n\n{body}".strip()
    while count_tokens(text, model_dir) > target_tokens and len(text) > len(instruction) + 10:
        text = text.rsplit(" ", 1)[0]
    return text


def prompt_length_table(
    lengths: list[int] | tuple[int, ...],
    *,
    model_dir: str | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for n in lengths:
        p = make_prompt(n, model_dir=model_dir)
        rows.append(
            {
                "target_tokens": n,
                "measured_tokens": count_tokens(p, model_dir),
                "chars": len(p),
                "prompt_preview": p[:120].replace("\n", " ") + ("…" if len(p) > 120 else ""),
            }
        )
    return rows
