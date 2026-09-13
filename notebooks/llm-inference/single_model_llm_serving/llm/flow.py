"""Lightweight request-flow tracing for the serving stack."""

def flow(layer: str, msg: str) -> None:
    print(f">>> [{layer}] {msg}", flush=True)


def flow_skip(layer: str, reason: str) -> None:
    print(f"--- SKIP [{layer}] {reason}", flush=True)


def flow_banner(endpoint: str) -> None:
    print(f"\n{'=' * 60}\n REQUEST  {endpoint}\n{'=' * 60}", flush=True)


def flow_done(endpoint: str) -> None:
    print(f"{'=' * 60}\n DONE     {endpoint}\n{'=' * 60}\n", flush=True)
