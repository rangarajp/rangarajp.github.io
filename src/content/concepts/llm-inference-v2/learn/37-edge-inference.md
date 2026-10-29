---
title: "Edge and on-device inference"
description: "Memory budgeting, runtimes, thermal limits, and architecture choices."
section: learn
pubDate: 2026-10-29
part: 6
chapter: 30
order: 30
tags: ["llm-inference", "systems", "hands-on"]
---

The previous chapters distribute large models and long contexts across server resources. On-device inference changes the constraints to local memory, power, thermals, and offline behavior while keeping the same need for task-quality measurement.

On-device inference trades server dependence for local memory, battery, and thermal constraints. The relevant target is sustained useful response quality on the actual device.

## 1. Budget memory before loading

```python
# Hypothetical device budget in GiB; metadata/workspace are estimates.
parameters, bits = 1e9, 4
weights = parameters * bits / 8 / 2**30
kv = 2 * 24 * 8 * 128 * 4096 * 2 / 2**30
workspace, runtime = 0.5, 0.4
print(round(weights + kv + workspace + runtime, 2), 'GiB estimated process budget')
```

The estimate is about 1.74 GiB, before extra quantization metadata and other process allocations. A four-bit model still commonly has a higher-precision cache. Shared system memory also supports the OS and other applications. Reduce context or concurrency when cache state consumes the remaining budget.

## 2. Runtime frameworks

llama.cpp uses GGUF models and device-specific backends; MLX targets Apple hardware; ONNX Runtime and mobile runtimes support their own operators and execution providers. A model file, quantization scheme, and accelerator backend must match. Do not assume an NPU accelerates all operations just because the device advertises one.

## 3. Measure the whole experience

Track load time, first-token delay, sustained decode speed, peak memory, and energy over repeated sessions. Short bursts can look fast before thermal throttling. Test offline behavior and failure on insufficient memory as part of the application.

## 4. Architecture search for efficiency

Choosing fewer layers, narrower hidden states, GQA, or a smaller vocabulary changes quality and deployment cost. Architecture search evaluates such alternatives against device constraints, but it requires trained candidates or an appropriate proxy. An existing checkpoint cannot be arbitrarily reshaped without additional training.

Hardware-aware architecture search can vary depth, hidden width, head layout, supported precision, or early-exit policies against an actual device objective. Parameter count alone is a weak latency predictor: alignment, memory access, and supported operators can make a smaller candidate slower. Measure the chosen runtime on the target device and validate the selected subnet's quality; a supernet or pruning recipe does not remove that requirement.

Memory mapping a checkpoint can reduce eager copying and make loading practical, but it does not eliminate attention state or the need to fetch weight pages. Unified memory similarly changes where capacity comes from while leaving competition with the operating system and data movement in the execution budget.

## Hands-on: measure a real edge device

Use a small supported local model on the intended CPU or edge device and repeat prompts long enough to reveal thermal behavior. Publish load time, sustained output rate, process memory, and task quality. A V100 run is the server reference, not evidence of on-device energy or battery life.
