---
title: "Data parallelism and replica serving"
description: "Replicate a model across workers, route independent requests, and account for capacity and queue imbalance."
section: learn
pubDate: 2026-10-29
part: 4
chapter: 20
order: 20
---

After optimizing one worker, scale independent request streams with replicas. This is the request-distribution case; the following chapters split computation or state within a model replica.

Data parallel inference runs independent request streams against complete model replicas. Each replica can itself be a group of tensor-parallel GPUs. A router decides which replica receives each request; a replica's local scheduler then batches its assigned sequences.

## 1. Same model, different requests

Four replicas can serve requests A–D concurrently without sharing each request's KV cache. There are no gradient all-reduces during ordinary inference. This differs from data-parallel training, where replicas synchronize gradients to maintain a common parameter update.

A model that does not fit on one device still needs model sharding or lower precision within each replica. Adding independent copies cannot split a single replica's weight footprint. Total resident weights across the fleet scale with replica count.

## 2. Round-robin versus work-aware routing

```python
# CPU-only queue simulation. Costs are arbitrary work units, not latency predictions.
requests = [20, 2, 2, 20, 2, 2]
round_robin, least_work = [0, 0], [0, 0]
for i, cost in enumerate(requests):
    round_robin[i % 2] += cost
    worker = min(range(2), key=lambda j: least_work[j])
    least_work[worker] += cost
print('round robin:', round_robin, 'estimated work:', least_work)
```

Both policies see the same number of requests, but request costs differ. Production routing can consider prompt length, active output budgets, queue depth, free KV blocks, and prefix locality. Exact future answer length is unknown, so estimates need headroom and feedback.

## 3. Capacity and failure

Two workers do not guarantee twice the useful throughput: shared CPU, storage, network, and cold model loading can limit the fleet. Test one-replica loss at peak traffic. Decide whether to reject excess work, degrade output budgets, or preserve latency for a smaller admitted load.

Compare [tensor parallelism](./tensor-parallelism), [pipeline parallelism](./pipeline-parallelism), and [multi-model routing](./gateway-registry-routing). The original working notes' distinction remains useful: replicas split requests; tensor parallelism splits operations inside a layer; pipeline parallelism splits layer depth.

## Hands-on: compare routing policies before adding replicas

Extend the CPU queue simulation with measured service-time samples and uneven request lengths. Publish per-replica work and tail latency. A real two-replica GPU throughput/failure experiment is a later step that needs two devices or two independent worker allocations.
