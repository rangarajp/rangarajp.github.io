---
title: "Serving systems architecture"
description: "Continuous batching, chunked prefill, disaggregation, and SLO-aware scheduling."
section: learn
pubDate: 2026-10-29
part: 3
chapter: 16
order: 16
tags: ["llm-inference", "systems", "hands-on"]
---

An inference server combines request validation, tokenization, a waiting queue, a scheduler, KV allocation, model execution, sampling, and streamed delivery. The scheduler coordinates scarce token slots and cache blocks; it does not merely collect HTTP requests.

## 1. Static and continuous batching

A static batch starts together and often waits for its longest member before admitting another batch. Continuous batching revisits membership after an execution iteration. Finished requests leave; waiting requests enter when memory and token budgets permit.

```python
# Toy decode scheduler: one token per active request per tick.
from collections import deque
waiting = deque([['A', 2], ['B', 5], ['C', 1]])
active, tick = [], 0
while waiting or active:
    while waiting and len(active) < 2:
        active.append(waiting.popleft())
    tick += 1
    print(tick, [name for name, remaining in active])
    active = [[name, remaining-1] for name, remaining in active if remaining > 1]
```

C enters at tick 3, before B finishes. This simulation assumes prefill is already complete and ignores memory. A real admission decision must reserve capacity for growing output and handle cancellation and errors.

## 2. Chunked prefill

A long prompt can monopolize an iteration and delay existing streams. Chunked prefill divides prompt work across iterations and combines it with decode work under a token budget. Smaller chunks can improve stream responsiveness but add scheduling overhead and may reduce prefill efficiency. Track both TTFT and inter-token gaps.

## 3. Disaggregated prefill and decode

Separate worker pools allow different hardware or batch policies for the two phases. The cost is transferring KV state and coordinating ownership, routing, and failure recovery. If cache transfer is slower than the interference it avoids, disaggregation hurts latency. Prefix placement and network topology affect the result.

## 4. SLO-aware scheduling

A latency objective should specify a percentile and workload. Token budgets, aging of queued requests, admission limits, and reserved decode capacity help manage it. An unbounded queue converts overload into extreme tail latency. Explicit rejection or backpressure is often more predictable.

Primary reading: [Sarathi-Serve](https://arxiv.org/abs/2403.02310) and [PagedAttention](https://arxiv.org/abs/2309.06180).
