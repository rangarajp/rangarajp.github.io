---
title: "Edge and on-device inference"
description: "Memory budgeting, runtimes, thermal limits, and architecture choices."
section: learn
pubDate: 2026-10-29
part: 6
chapter: 37
order: 37
tags: ["llm-inference", "systems", "hands-on"]
---

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
