---
title: "Kernel engineering and FlashAttention"
description: "Cache management determines which state remains available."
section: learn
pubDate: 2026-10-29
chapter: 7
order: 7
---

Cache management determines which state remains available. Kernel engineering determines how efficiently the GPU computes with that state. The key is to reduce unnecessary memory traffic and dispatch overhead while preserving the operation’s mathematical meaning.

## 7.1 Why custom CUDA kernels matter

Decode consists of many operations on relatively small activation tensors. When kernels are short, launch and dispatch overhead can become a material part of latency.

### Fusion

Consider `y = silu(x + residual)`. Separate kernels can write and reread the addition result. A fused kernel can compute the addition and activation while the intermediate stays in registers. Fewer memory transfers and launches help; excessive fusion can raise register pressure and reduce occupancy.

### Compilation and CUDA graphs

A compiler can specialize operations and fuse compatible regions. CUDA graphs record a compatible GPU work sequence for replay with lower dispatch overhead. They do not remove the underlying matrix multiplications. Dynamic shapes, changing addresses, and control flow complicate capture; engines often maintain graph variants or padded execution sizes.

### Time GPU work correctly

```python
# Requires a CUDA-capable PyTorch installation and GPU.
import torch
if torch.cuda.is_available():
    x = torch.randn(1024, 1024, device='cuda')
    with torch.inference_mode():
        for _ in range(10):
            y = x @ x
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(50):
            y = x @ x
        end.record()
        end.synchronize()
        print(start.elapsed_time(end) / 50, 'ms per matmul')
else:
    print('CUDA required for this measurement')
```

Warmup avoids counting first-use initialization. Events measure device work between the events; request wall time includes queueing and CPU work too. Record model, GPU, dtype, shapes, software versions, and concurrency alongside results. The example measures repeated matrix multiplication, not LLM serving throughput.

## 7.2 FlashAttention: IO-aware exact attention

Naive attention writes a score matrix and probability matrix proportional to $T^2$ into device memory. FlashAttention tiles the computation so those full intermediates need not live in HBM. It preserves dense attention semantics up to floating-point differences; it does not make dense attention linear in arithmetic work.

A high-level expression can launch separate operations for scores, normalization, and the value-weighted result. Writing each intermediate to device memory and reading it back costs traffic even when the arithmetic is fast. Custom kernels can combine and tile those operations so more intermediate state remains on chip. Library kernels are often the right implementation; the point is to choose an execution path whose data movement matches the problem.

A tile of queries processes tiles of keys and values. Instead of saving all scores, the kernel maintains each row's running maximum, normalization sum, and weighted value sum. SRAM holds the working tiles. The result is fewer HBM transfers and less intermediate storage.

## 7.3 The online softmax trick

When a new tile changes the maximum from $m$ to $m'$, rescale previous sums by $\exp(m-m')$ before adding the new contributions. This keeps exponentials numerically stable while preserving the normalization across tiles.

For one query row, let $\mathcal{J}_i$ index the next key tile and $V_j$ be its value vectors. Maintain a maximum $m$, an unnormalized sum $\ell$, and a weighted-value accumulator $z$:

$$
\begin{aligned}
m' &= \max\!\left(m,\max_{j\in\mathcal{J}_i}s_j\right),\\
\ell' &= e^{m-m'}\ell+\sum_{j\in\mathcal{J}_i}e^{s_j-m'},\\
z' &= e^{m-m'}z+\sum_{j\in\mathcal{J}_i}e^{s_j-m'}V_j,\\
o &= z/\ell.
\end{aligned}
$$

The same rescaling must be applied to the normalization sum and the value accumulator when the maximum changes. The output division is performed after all relevant key tiles have contributed. There is one scalar maximum and sum per query row, plus a vector accumulator of value width; the state is independent of the number of keys, not independent of head width.

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

## 7.4 Triton: Pythonic kernel programming

### Use a supported attention backend

PyTorch's scaled dot-product attention API can select an optimized backend when shape, dtype, hardware, and masks permit it. Dispatch is not a guarantee of FlashAttention on every device. For a cached one-token decode, blindly setting `is_causal=True` with unequal query/key lengths can apply the wrong alignment. Supply the mask appropriate to the actual query positions, or use the engine's cache-aware attention path.

### Triton and custom kernels

Triton expresses tiled GPU computation in Python-like code. Choosing tile sizes, warp counts, and layouts still requires understanding registers, memory coalescing, and occupancy. Start from a correct reference and profile actual inputs before replacing a library kernel.

Primary reading: [FlashAttention](https://arxiv.org/abs/2205.14135) and [FlashAttention-2](https://arxiv.org/abs/2307.08691).

## Recorded notebook observations

These observations come from the saved notebook runs linked below. Their model, timing boundary, and hardware assumptions are part of the result.

### V100 lab: keep the backend claim precise

The cache and engine notebooks establish V100 runs, not a FlashAttention-2 or FlashAttention-3 benchmark. The [FlashAttention repository](https://github.com/Dao-AILab/flash-attention) lists newer architectures for its modern CUDA implementations; Volta is not a target for those FA2/FA3 comparisons. On V100, record the actual supported attention backend and validate numerical agreement with the reference. An SDPA API call alone does not prove that a FlashAttention kernel ran.

### V100 lab: eager execution was part of the recorded setup

The saved [server initialization](https://github.com/rangarajp/rangarajp.github.io/blob/main/notebooks/llm-inference/single_model_llm_serving/main.ipynb) records `enforce_eager=True` and a vLLM warning that CUDA graphs were disabled. That run cannot establish a graph-replay speedup. Use it as an eager baseline, then measure a separately supported graph configuration with matched shapes and output checks before making a comparison.

## Hands-on extensions

Use these experiments to connect the mechanisms above to measurements. They are proposed exercises unless a recorded result is explicitly identified.

### Hands-on: validate IO savings on supported backends

On the V100, first compare the explicit attention reference with an available SDPA backend and record which backend actually ran. Publish numerical error and peak memory as context grows. A FlashAttention-2/3 performance comparison requires compatible newer hardware; do not label a fallback measurement as FlashAttention.

### Hands-on: isolate dispatch overhead

Compare eager, supported compiled/fused, and graph-replayed versions of the same fixed-shape operation. Verify output tolerance first, then measure after warmup. Publish device time and host wall time separately; record graph-capture exclusions and buffer lifetime.
