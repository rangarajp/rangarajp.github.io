---
title: "Observability and production engineering"
description: "Request metrics, traces, failure diagnosis, and goodput."
section: learn
pubDate: 2026-10-29
part: 6
chapter: 33
order: 33
tags: ["llm-inference", "systems", "hands-on"]
---

An engine benchmark cannot tell you whether users receive timely answers through the full application. Instrument arrival, queue admission, prefill, first delivered token, subsequent tokens, completion, and cancellation. A correlation ID connects those events without logging private prompts by default.

## 1. Key metrics

| Metric | Definition | Diagnostic use |
|---|---|---|
| TTFT | Arrival to first delivered token | Queueing, prefill, delivery |
| ITL | Gaps between delivered tokens | Streaming stalls |
| TPOT | Mean post-first-token time per token | Request decode pace |
| Throughput | Defined token count / wall time | Aggregate capacity |
| Goodput | Work meeting a stated SLO / wall time | Useful capacity |
| KV occupancy | Allocated blocks / usable blocks | Admission pressure |

```python
# Synthetic delivery times in seconds since request arrival.
times = [0.12, 0.14, 0.17, 0.19]
gaps = [b-a for a,b in zip(times,times[1:])]
tpot = (times[-1]-times[0]) / (len(times)-1)
print('TTFT:', times[0], 'TPOT:', round(tpot, 4), 'max gap:', round(max(gaps), 4))
```

TPOT is undefined for a one-token answer; do not divide by zero or substitute TTFT. Report request percentiles and output lengths together. Pooling every token gap weights long answers more heavily than short ones, so state your aggregation method.

## 2. Failure modes and debugging

High TTFT with low prefill time suggests queueing or upstream work. Good TTFT with intermittent streaming stalls suggests scheduler interference, delivery buffering, or contention. Out-of-memory failures late in answers suggest insufficient growth reservations or memory accounting. Inspect traces before changing hardware.

Cancellation must release request state and stop unnecessary GPU work. Readiness should mean the model can accept work, not merely that an HTTP process exists. Retry policies need bounded attempts: restarting a streamed answer can duplicate output and multiply load.

## 3. Benchmark and operate

Use realistic prompt/output distributions and arrival rates, plus a concurrency sweep. Warm the engine, separate cold-cache from warm-prefix results, and include failures and rejected requests. Record software, model revision, quantization, hardware, and scheduler settings. Pair performance changes with quality checks.

Production layers include authentication, quotas, bounded queues, routing, healthy model replicas, and telemetry. Containers or orchestration do not remove these requirements. Continue with [capacity and cost](./capacity-and-cost).
