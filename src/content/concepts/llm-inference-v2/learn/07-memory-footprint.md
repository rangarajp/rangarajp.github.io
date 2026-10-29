---
title: "Parameters, activations, and KV memory"
description: "Size weights, cache state, workspace, and concurrency using complete worked memory budgets."
section: learn
pubDate: 2026-10-29
part: 2
chapter: 9
order: 9
tags: ["llm-inference", "systems", "hands-on"]
---

GPU fundamentals explain the storage hierarchy. Here the question is capacity: what must fit while requests are active? The next chapter treats speed separately through arithmetic intensity and roofline estimates.

This chapter budgets weights, retained KV state, and temporary workspace. Use actual checkpoint head counts and consistent byte units; nominal device capacity alone does not determine the number of safe concurrent requests.

## 1. Sizing the pantry — weights, KV cache, misc

GPU HBM capacity (24 GB, 48 GB, 80 GB, …) is the first hard wall. Budget it as:

$$
\text{GPU memory} \approx \text{model weights} + \text{KV cache} + \text{misc}
$$

A model that “fits” at idle can still OOM under load. Memory decides *whether* you can serve; bandwidth and FLOPS decide *how fast*.

### Weights

$$
\text{Weight memory (bytes)} \approx \text{parameters} \times \text{bytes per parameter}
$$

| Format | Bits | Bytes |
|--------|------|-------|
| FP32 | 32 | 4 |
| FP16 / BF16 | 16 | 2 |
| INT8 / FP8 | 8 | 1 |
| INT4 | 4 | 0.5 |

| Parameters | FP16 / BF16 | FP8 / INT8 | INT4 |
|------------|-------------|------------|------|
| 7B–8B | ~14–16 GB | ~7–8 GB | ~3.5–4 GB |
| 70B | ~140 GB | ~70 GB | ~35 GB |

### KV cache — from transformer shapes

Keys and values are what get cached. Use a consistent shape from the dimension diagram:

<figure>

![Transformer architecture with expanded multi-head attention and tensor dimensions](../../llm-inference/images/transformer-complete-dimensions.png)

<figcaption><span class="figure-label">Figure 1.</span> Transformer dimensions used for KV sizing (batch 128, sequence 100, d_model 512, heads 8, d_k 64)</figcaption>
</figure>

| Symbol | Value in the figure | Meaning |
|--------|---------------------|---------|
| batch $B$ | 128 | parallel sequences |
| sequence $S$ | 100 | tokens in context |
| $d_{\text{model}}$ | 512 | model width |
| heads $H$ | 8 | attention heads |
| $d_k$ | 64 | per-head size ($d_{\text{model}} / H$) |

Projected K and V each have shape $(B,\ H,\ S,\ d_k) = (128,\ 8,\ 100,\ 64)$.

For one attention layer, store K and V:

$$
\text{KV elements per layer} = 2 \times B \times H \times S \times d_k
$$

$$
\begin{align*}
&= 2 \times 128 \times 8 \times 100 \times 64 \\
&= 13{,}107{,}200 \text{ elements}
\end{align*}
$$

At FP16 (2 bytes):

$$
13{,}107{,}200 \times 2 \approx 26.2\ \text{MB per layer}
$$

Check: $H \times d_k = d_{\text{model}}$, so the same number is $2 \times B \times S \times d_{\text{model}} \times \text{bytes}$.

For a stack of $L$ layers (decoder self-attention — the usual LLM case):

$$
\text{KV cache} = 2 \times L \times B \times H \times S \times d_k \times \text{bytes}
$$

Classic small transformer with $L = 6$, FP16, same $B,S$:

$$
6 \times 26.2\ \text{MB} \approx 157\ \text{MB}
$$

Tiny versus the weights — because $d_{\text{model}}=512$ and $S=100$. Real models are wider, deeper, and use much longer contexts.

Per-token form (handy for serving math):

$$
\text{bytes per token (all layers)} = 2 \times L \times H \times d_k \times \text{bytes}
$$

Diagram model, $L=6$, FP16: $2 \times 6 \times 8 \times 64 \times 2 = 12{,}288$ bytes ≈ 12 KB per token.  
At $S=100$: $12\ \text{KB} \times 100 = 1.2\ \text{MB per sequence}$; $\times B=128$ ≈ 157 MB — same as above.

### Rough estimate for a modern Llama-class 8B

Take a Llama-3-class 8B decoder-only model (GQA):

| Spec | Typical value |
|------|----------------|
| Parameters | ~8B |
| Layers $L$ | 32 |
| KV heads $H_{\text{kv}}$ | 8 (not 32 — grouped-query attention) |
| Head dim $d_k$ | 128 |
| Weight dtype | BF16 / FP16 → ~16 GB weights |

Per token (K+V, all layers, FP16):

$$
2 \times 32 \times 8 \times 128 \times 2 = 131{,}072\ \text{bytes} \approx 128\ \text{KB/token}
$$

| Context length $S$ | KV for 1 request | KV for batch $B$ |
|--------------------|------------------|------------------|
| 1k | ~128 MB | $B \times 128$ MB |
| 2k | ~256 MB | $B \times 256$ MB |
| 4k | ~512 MB | $B \times 512$ MB |
| 8k | ~1 GB | $B \times 1$ GB |

GQA (8 KV heads) already cuts KV versus full MHA (32 heads) by about 4×. Multi-query or more aggressive GQA shrinks it further.

### Misc

Reserve headroom beyond weights + KV:

| Item | Typical ballpark |
|------|------------------|
| CUDA context + cuBLAS / engine | ~0.5–2 GB |
| Activations / workspace (prefill) | grows with $B$ and $S$; often 1–4+ GB |
| Fragmentation / allocator slack | ~10% of free memory is a safe cushion |
| Torch / vLLM overhead | depends on engine; keep a fixed reserve |

Rule of thumb for planning:

$$
\text{Usable for KV} \approx \text{GPU GB} - \text{weights} - \text{misc reserve}
$$

Use about 2–4 GB misc on a 24 GB card and 3–5 GB on a 48 GB card unless you have measured otherwise.

### How many concurrent requests? — 24 GB vs 48 GB

Example: Llama-class 8B BF16 (~16 GB weights), FP16 KV (~128 KB/token), misc reserved as below.

*24 GB GPU* (for example A10-class / consumer 24 GB):

| | Estimate |
|---|----------|
| Weights | 16 GB |
| Misc reserve | 3 GB |
| Left for KV | ~5 GB |

$$
\text{Max total cached tokens} \approx \frac{5\ \text{GB}}{128\ \text{KB/token}} \approx 40{,}000\ \text{tokens}
$$

| Avg context per request | Max concurrent batch $B$ (approx.) |
|-------------------------|-------------------------------------|
| 1k tokens | ~40 |
| 2k | ~20 |
| 4k | ~10 |
| 8k | ~5 |

*48 GB GPU* (for example L40S):

| | Estimate |
|---|----------|
| Weights | 16 GB |
| Misc reserve | 4 GB |
| Left for KV | ~28 GB |

$$
\text{Max total cached tokens} \approx \frac{28\ \text{GB}}{128\ \text{KB/token}} \approx 224{,}000\ \text{tokens}
$$

| Avg context per request | Max concurrent batch $B$ (approx.) |
|-------------------------|-------------------------------------|
| 1k tokens | ~220 |
| 2k | ~110 |
| 4k | ~55 |
| 8k | ~28 |

Same model, same dtype: roughly 5–6× more concurrent sequences on 48 GB than on 24 GB, because almost all of the extra 24 GB becomes KV headroom.

What changes the batch you can actually run:

- Longer context → fewer parallel requests (linear in $S$)
- FP8/INT8 weights → more room for KV (for example ~8 GB weights → much higher $B$ on 24 GB)
- Larger model (70B) → may not fit in 24/48 GB without quantisation and multi-GPU
- Prefill activations → peak memory can exceed the steady decode KV budget; engines often limit max batch/context for that reason

So the 24 vs 48 GB question is rarely “does 8B fit?” (both can). It is *how much concurrent context* you can keep warm — and that is almost entirely a KV-cache sizing problem once weights are loaded.


## 2. Choose hardware by the scarce resource

Capacity determines whether weights, growing caches, and workspace fit. Bandwidth sets an ideal weight-streaming bound. Compute throughput matters when the matrix shapes expose enough reuse. Interconnect matters when work is sharded; topology, transfer direction, and synchronization matter as much as a link's headline rate.

For a concrete historical comparison, H100 SXM provides 80 GB and 3.35 TB/s; H100 NVL provides 94 GB per GPU and 3.9 TB/s. The NVIDIA BF16 figures of 1,979 and 1,671 TFLOP/s include structured sparsity. For dense roofline arithmetic use approximately half those rates. Do not mix sparse and dense figures in one comparison. These are specifications, not observed application throughput.

See [NVIDIA H100 specifications](https://www.nvidia.com/en-in/data-center/h100/), [GPU fundamentals](./gpu-fundamentals), and [inference mathematics](./inference-mathematics).

## 3. Recalculate a capacity budget without rounding units

```python
# Decimal GB capacity and actual bytes per cached token.
weights, bytes_per_token = 8e9 * 2, 2 * 32 * 8 * 128 * 2
for capacity, reserve in ((24e9, 3e9), (48e9, 4e9)):
    tokens = int((capacity - weights - reserve) // bytes_per_token)
    print(int(capacity / 1e9), 'GB:', tokens, 'cached tokens;', tokens // 4096, 'full 4096-token requests')
```

The earlier tables use rounded k-token and KB estimates. This calculation yields about 38,146 and 213,623 cached tokens, or 9 and 52 complete 4096-token requests. It still omits block padding and worst-case prefill workspace; reserve output growth before admitting those requests.

## V100 lab: predicted cache size matched the stored tensors

The [7B cache notebook](https://github.com/rangarajp/rangarajp.github.io/blob/main/notebooks/llm-inference/observability/experiments/3.1_kv_cache.ipynb) records 28 layers, 28 query heads, 4 KV heads, head width 128, and FP16 state. Applying the formula gives $2 \times 28 \times 4 \times 128 \times 2 = 57{,}344$ bytes, or 56 KiB per token across all layers.

| Retained tokens | Logical KV bytes per sequence |
|---:|---:|
| 512 | 28 MiB |
| 1024 | 56 MiB |
| 2048 | 112 MiB |
| 4096 | 224 MiB |
| 8192 | 448 MiB |

At a 10-token prompt, the recorded layer-zero key and value shapes were `[1, 4, 10, 128]`; total cache tensor storage was 573,440 bytes, matching the formula exactly. At 17 retained positions it was 974,848 bytes. Tensor storage is only the logical cache: allocator reserves, model weights, workspace, and engine block pools still contribute to process memory.

The table is for Qwen2.5-7B's configuration. The earlier Llama examples use different layer and KV-head counts, so their bytes per token differ even on the same GPU.

## Hands-on: compare predicted and allocated cache bytes

Use the 7B checkpoint configuration from the cache notebook. At several context lengths, compare the closed-form KV size with tensor storage and process peak allocation. Publish all three columns, since allocator/workspace memory is larger than logical KV storage.
