---
title: "Parallelism for large model inference"
description: "Tensor sharding, communication, and a matrix equivalence experiment."
section: learn
pubDate: 2026-10-29
part: 4
chapter: 21
order: 21
tags: ["llm-inference", "systems", "hands-on"]
---

# Parallelism for large model inference

A model may need multiple devices because its weights and request state exceed one GPU's memory, or because a particular workload benefits from extra compute. Parallelism adds communication, so capacity gain and latency gain are separate claims.

## 1. Tensor parallelism

Split a layer's matrices across devices. Column-parallel projections produce output shards; row-parallel projections combine partial sums. Attention heads and MLP intermediate channels provide natural partitions. Exact placement depends on architecture and implementation.

```python
# CPU PyTorch: algebra behind two output shards, not distributed execution.
import torch
torch.manual_seed(7)
x, w = torch.randn(3, 8), torch.randn(8, 12)
left, right = w.chunk(2, dim=1)
y = torch.cat([x @ left, x @ right], dim=1)
assert torch.allclose(y, x @ w, atol=1e-6)
# Split the reduction dimension instead: partial outputs must be summed.
xa, xb = x.chunk(2, dim=1)
wa, wb = w.chunk(2, dim=0)
assert torch.allclose(xa @ wa + xb @ wb, x @ w, atol=1e-6)
```

The second case illustrates why a collective reduction is needed. Actual latency depends on message size, interconnect, synchronization, and work available between collectives.

## 2. Choose the parallel axis

| Strategy | What is split | Main cost |
|---|---|---|
| Replicas | Independent requests | Extra copies of weights |
| Tensor parallelism | Work within layers | Frequent collectives |
| Pipeline parallelism | Layer depth | Stage transfers and bubbles |
| Expert parallelism | Expert placement | Token dispatch and load imbalance |
| Context parallelism | Sequence positions | Attention-state communication |

Replicas improve aggregate capacity but do not make an oversized model fit per replica. Tensor parallelism is most attractive with a fast interconnect. Continue with [pipeline](./pipeline-parallelism), [MoE](./expert-parallelism), and [long context](./context-parallelism).
