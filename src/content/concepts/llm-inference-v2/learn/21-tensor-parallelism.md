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

Replicas split independent requests. Tensor parallelism splits operations inside a layer, making communication part of that layer’s critical path.

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

## Existing sharding notebook: separate algebra from distributed execution

The [sharding notebook](https://github.com/rangarajp/rangarajp.github.io/blob/main/notebooks/llm-inference/gpu_fundamentals/sharding.ipynb) provides the companion path from matrix partitioning to real multi-GPU work. Its distributed section explicitly requires at least two CUDA devices and a `torchrun` launch. A one-V100 algebra demonstration can verify reconstruction, but it cannot measure cross-device collectives or tensor-parallel scaling.

## Hands-on: verify sharding algebra, then communication

Use the existing sharding notebook to reconstruct a matrix result from row and column shards. Publish numerical error on one device first. With at least two GPUs, separately measure compute, collective communication, and complete step time; report the interconnect.
