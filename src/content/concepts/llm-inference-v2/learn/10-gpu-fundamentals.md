---
title: "Hardware foundations for inference"
description: "SMs, tensor cores, memory hierarchy, mixed precision, and CPU constraints."
section: learn
pubDate: 2026-10-29
part: 2
chapter: 10
order: 10
tags: ["llm-inference", "systems", "hands-on"]
---

A GPU accelerates work by running many threads and reusing data. Its advertised FLOP rate is useful only when the kernel exposes enough parallel work and supplies operands quickly enough.

## 1. GPU architecture

Streaming multiprocessors (SMs) schedule groups of threads called warps. Registers hold per-thread values. Shared memory allows a thread block to reuse tiles. HBM holds weights and KV state; caches sit between HBM and execution units. Occupancy describes resident work, not useful throughput by itself. A kernel with high occupancy can still stall on memory.

## 2. Tensor cores and mixed precision

Tensor cores accelerate supported matrix operations. They require compatible formats, shapes, and kernels. FP16 and BF16 both use two bytes, but BF16 has a wider exponent range and fewer fraction bits. Accumulation may use a higher precision than multiplication. FP8 and integer formats require explicit scaling and supported kernels; changing a tensor dtype alone does not establish an efficient quantized execution path.

## 3. Memory hierarchy and bandwidth

Tiling loads a piece of each matrix once and reuses it across many products. Fusion keeps intermediate values close to execution rather than writing them to HBM. Neither creates more HBM capacity. A small-batch decode can remain bandwidth limited even while most tensor-core capacity is idle.

```python
# CPU-only arithmetic: reuse of one matrix over independent token rows.
rows, cols, element_bytes = 4096, 4096, 2
for tokens in (1, 8, 64, 512):
    flops = 2 * tokens * rows * cols
    moved = (rows * cols + tokens * (rows + cols)) * element_bytes
    print(tokens, round(flops / moved, 2), 'FLOP/byte')
```

The ratio rises with token rows because weights are amortized. This model ignores caches, launch overhead, and kernel efficiency. It explains a trend; it cannot replace a profile.

## 4. CPU and edge hardware

CPUs offer large system memory and flexible control flow, but their effective matrix throughput and memory bandwidth differ from a GPU's. Unified-memory devices share a memory pool with the operating system. A nominal capacity is not the amount an application can safely consume. Measure sustained performance, power, and thermal behavior for on-device use.

## 5. Emerging hardware trends

More HBM capacity, faster interconnects, and new low-precision instructions help different bottlenecks. An interconnect upgrade helps sharded execution only when communication matters. More tensor throughput helps compute-heavy prefill only when the kernels can use it. Start with a trace and a memory budget before choosing hardware.

Continue with [the prefill/decode byte estimates](./prefill-decode-kv-cache) and [kernel engineering](./kernel-fusion-cuda-graphs).
