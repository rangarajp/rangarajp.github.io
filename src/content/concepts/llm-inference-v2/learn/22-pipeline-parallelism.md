---
title: "Pipeline parallelism"
description: "Layer placement, microbatches, and pipeline bubbles."
section: learn
pubDate: 2026-10-29
part: 4
chapter: 22
order: 22
tags: ["llm-inference", "systems", "hands-on"]
---

Tensor parallelism divides layer operations. Pipeline parallelism instead assigns different layer ranges to stages; its central tradeoff is stage balance and idle pipeline time.

Pipeline parallelism assigns groups of layers to different devices. Activations move between stages while weights stay on their assigned device. It reduces per-device weight storage without requiring every layer to run across every GPU.

## 1. Why bubbles appear

A stage cannot process a microbatch until the previous stage has produced its input. Under an ideal equal-stage schedule with $p$ stages and $m$ independent microbatches, utilization is approximately $m/(m+p-1)$ for a forward pipeline.

```python
stages = 4
for microbatches in (1, 4, 16):
    utilization = microbatches / (microbatches + stages - 1)
    print(microbatches, round(utilization, 3))
```

One microbatch uses only a quarter of the pipeline's stage-time capacity. More independent work fills it, but a single autoregressive stream still needs its next token before beginning its next step. Pipeline throughput and single-user decode latency therefore need separate measurements.

## 2. Balance the stages

Equal layer counts do not guarantee equal work: embeddings, output projections, attention lengths, and MoE routing can differ. The slowest stage sets the steady-state pace. Account for activation transfer and runtime workspace as well as weights. The formula above ignores imbalance, communication, and scheduler interference.

## Hands-on: draw a stage utilization timeline

Simulate two stages with unequal service times and several microbatch counts. Publish occupied and idle stage intervals and compare utilization with the ideal formula. Claim a measured multi-GPU benefit only after running an actual distributed pipeline.
