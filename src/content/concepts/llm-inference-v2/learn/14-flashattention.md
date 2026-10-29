---
title: "Kernel engineering and FlashAttention"
description: "Exact tiled attention, online softmax, and a CPU numerical demonstration."
section: learn
pubDate: 2026-10-29
part: 3
chapter: 14
order: 14
tags: ["llm-inference", "systems", "hands-on"]
---

# Kernel engineering and FlashAttention

Naive attention writes a score matrix and probability matrix proportional to $T^2$ into device memory. FlashAttention tiles the computation so those full intermediates need not live in HBM. It preserves dense attention semantics up to floating-point differences; it does not make dense attention linear in arithmetic work.

## 1. IO-aware exact attention

A tile of queries processes tiles of keys and values. Instead of saving all scores, the kernel maintains each row's running maximum, normalization sum, and weighted value sum. SRAM holds the working tiles. The result is fewer HBM transfers and less intermediate storage.

## 2. Online softmax

When a new tile changes the maximum from $m$ to $m'$, rescale previous sums by $\exp(m-m')$ before adding the new contributions. This keeps exponentials numerically stable while preserving the normalization across tiles.

```python
import math
scores, values = [2.0, -1.0, 4.0, 0.5], [10.0, 20.0, 30.0, 40.0]
m, denom, numerator = -math.inf, 0.0, 0.0
for start in range(0, len(scores), 2):
    tile_s, tile_v = scores[start:start+2], values[start:start+2]
    new_m = max(m, max(tile_s))
    correction = math.exp(m-new_m)
    probs = [math.exp(s-new_m) for s in tile_s]
    denom = correction * denom + sum(probs)
    numerator = correction * numerator + sum(p*v for p,v in zip(probs,tile_v))
    m = new_m
online = numerator / denom
weights = [math.exp(s-max(scores)) for s in scores]
reference = sum(w*v for w,v in zip(weights, values)) / sum(weights)
assert math.isclose(online, reference, rel_tol=1e-12)
print(online)
```

This scalar-value CPU experiment demonstrates the recurrence, not a fast kernel. A GPU implementation vectorizes value accumulation, partitions work, and handles masks and boundaries.

## 3. Use a supported attention backend

PyTorch's scaled dot-product attention API can select an optimized backend when shape, dtype, hardware, and masks permit it. Dispatch is not a guarantee of FlashAttention on every device. For a cached one-token decode, blindly setting `is_causal=True` with unequal query/key lengths can apply the wrong alignment. Supply the mask appropriate to the actual query positions, or use the engine's cache-aware attention path.

## 4. Triton and custom kernels

Triton expresses tiled GPU computation in Python-like code. Choosing tile sizes, warp counts, and layouts still requires understanding registers, memory coalescing, and occupancy. Start from a correct reference and profile actual inputs before replacing a library kernel.

Primary reading: [FlashAttention](https://arxiv.org/abs/2205.14135) and [FlashAttention-2](https://arxiv.org/abs/2307.08691).
