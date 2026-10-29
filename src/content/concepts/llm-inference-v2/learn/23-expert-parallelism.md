---
title: "Mixture of experts inference"
description: "Active parameters, resident weights, routing, and load imbalance."
section: learn
pubDate: 2026-10-29
part: 4
chapter: 23
order: 23
tags: ["llm-inference", "systems", "hands-on"]
---

Dense-model parallelism splits work chosen in advance. MoE routing chooses experts per token, so placement must account for both communication and uneven expert demand.

A mixture-of-experts (MoE) layer routes each token to a subset of expert feed-forward networks. Attention and shared components still run. Fewer active parameters per token can reduce arithmetic relative to activating every expert, but all potentially selected experts must remain accessible somewhere.

## 1. Total versus active parameters

A model with many total parameters and a small active subset is not equivalent in memory to a dense model of the active size. It also pays routing, dispatch, and combination costs. Expert parallelism distributes experts across devices, often sending token activations through an all-to-all exchange.

For selected experts $\mathcal{K}(x)$ and their routing weights $g_k(x)$, the expert contribution can be expressed as

$$
y(x)=\sum_{k\in\mathcal{K}(x)}g_k(x)\,E_k(x).
$$

The router selects computation, not free memory. A rarely selected expert still needs accessible weights, and moving a token to its device adds dispatch and combination traffic. Offloading experts to host memory or storage trades resident device capacity for loading latency and bandwidth. Report resident state and cold expert access separately from active arithmetic.

## 2. Inspect routing balance

```python
from collections import Counter
# Toy top-2 routes for six tokens, four experts.
routes = [(0, 1), (0, 2), (0, 1), (0, 3), (1, 2), (0, 1)]
loads = Counter(e for selected in routes for e in selected)
mean = sum(loads.values()) / 4
print(dict(loads), 'max/mean:', max(loads.values()) / mean)
```

A busy expert can delay the iteration while another device waits. In production, monitor dispatch bytes, expert load, padding, and any overflow policy. Dropping tokens to enforce capacity can change outputs and is not a transparent optimization.

## 3. Combine parallel strategies

Tensor parallelism can split work inside experts while expert parallelism distributes expert groups. The best layout depends on batch size, routing balance, and topology. Report total weight footprint and active computation separately when comparing MoE with dense models.

Primary reading: [DeepSeek-V2](https://arxiv.org/abs/2405.04434).

## Hands-on: stress uneven routing

Extend the top-2 routing example with balanced and skewed token assignments. Publish expert loads, maximum-to-mean load, and estimated dispatch bytes. This CPU simulation demonstrates imbalance; distributed all-to-all timing requires multiple devices.
