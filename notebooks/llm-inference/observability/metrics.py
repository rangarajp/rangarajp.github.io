"""Request-level latency metrics from token arrival timestamps."""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, median
from typing import Any, Sequence


@dataclass
class RequestMetrics:
    """One streaming (or timed) request."""

    prompt: str
    t_send: float
    token_times: list[float] = field(default_factory=list)
    tokens: list[str] = field(default_factory=list)
    sequence_id: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and len(self.token_times) > 0

    @property
    def completion_tokens(self) -> int:
        return len(self.token_times)

    @property
    def ttft_s(self) -> float | None:
        if not self.token_times:
            return None
        return self.token_times[0] - self.t_send

    @property
    def itl_s(self) -> list[float]:
        if len(self.token_times) < 2:
            return []
        return [
            self.token_times[i] - self.token_times[i - 1]
            for i in range(1, len(self.token_times))
        ]

    @property
    def tpot_s(self) -> float | None:
        gaps = self.itl_s
        if not gaps:
            return None
        return mean(gaps)

    @property
    def e2e_s(self) -> float | None:
        if not self.token_times:
            return None
        return self.token_times[-1] - self.t_send

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "ok": self.ok,
            "error": self.error,
            "sequence_id": self.sequence_id,
            "completion_tokens": self.completion_tokens,
            "ttft_s": self.ttft_s,
            "tpot_s": self.tpot_s,
            "e2e_s": self.e2e_s,
            "itl_mean_s": mean(self.itl_s) if self.itl_s else None,
            "itl_p50_s": median(self.itl_s) if self.itl_s else None,
            "text": "".join(self.tokens),
        }


@dataclass
class E2ERequestMetrics:
    """Blocking /generate style timing (no per-token stream)."""

    prompt: str
    model_id: str
    t_send: float
    t_done: float
    generated_text: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def e2e_s(self) -> float:
        return self.t_done - self.t_send

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "model_id": self.model_id,
            "ok": self.ok,
            "error": self.error,
            "e2e_s": self.e2e_s,
            "generated_text": self.generated_text,
        }


def _percentile(values: Sequence[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    k = (len(ordered) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(ordered) - 1)
    if f == c:
        return ordered[f]
    return ordered[f] + (ordered[c] - ordered[f]) * (k - f)


def aggregate_stream(results: Sequence[RequestMetrics]) -> dict[str, Any]:
    """Aggregate concurrent streaming requests into one summary row."""
    ok = [r for r in results if r.ok]
    ttfts = [r.ttft_s for r in ok if r.ttft_s is not None]
    tpots = [r.tpot_s for r in ok if r.tpot_s is not None]
    e2es = [r.e2e_s for r in ok if r.e2e_s is not None]
    tokens = sum(r.completion_tokens for r in ok)

    wall_start = min((r.t_send for r in results), default=None)
    wall_end = max((r.token_times[-1] for r in ok if r.token_times), default=None)
    wall_s = (wall_end - wall_start) if wall_start is not None and wall_end is not None else None

    return {
        "n_requests": len(results),
        "n_ok": len(ok),
        "n_err": len(results) - len(ok),
        "ttft_p50_s": _percentile(ttfts, 50),
        "ttft_p95_s": _percentile(ttfts, 95),
        "ttft_mean_s": mean(ttfts) if ttfts else None,
        "tpot_p50_s": _percentile(tpots, 50),
        "tpot_mean_s": mean(tpots) if tpots else None,
        "e2e_p50_s": _percentile(e2es, 50),
        "e2e_mean_s": mean(e2es) if e2es else None,
        "completion_tokens": tokens,
        "wall_s": wall_s,
        "throughput_tok_s": (tokens / wall_s) if wall_s and wall_s > 0 else None,
        "throughput_req_s": (len(ok) / wall_s) if wall_s and wall_s > 0 else None,
    }


def aggregate_e2e(results: Sequence[E2ERequestMetrics]) -> dict[str, Any]:
    ok = [r for r in results if r.ok]
    e2es = [r.e2e_s for r in ok]
    wall_start = min((r.t_send for r in results), default=None)
    wall_end = max((r.t_done for r in ok), default=None)
    wall_s = (wall_end - wall_start) if wall_start is not None and wall_end is not None else None
    return {
        "n_requests": len(results),
        "n_ok": len(ok),
        "n_err": len(results) - len(ok),
        "e2e_p50_s": _percentile(e2es, 50),
        "e2e_p95_s": _percentile(e2es, 95),
        "e2e_mean_s": mean(e2es) if e2es else None,
        "wall_s": wall_s,
        "throughput_req_s": (len(ok) / wall_s) if wall_s and wall_s > 0 else None,
    }
