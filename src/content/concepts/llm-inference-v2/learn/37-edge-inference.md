---
title: "Edge and on-device inference"
description: "On-device inference brings computation closer to the user and can support offline use."
section: learn
pubDate: 2026-10-29
chapter: 11
order: 11
---

On-device inference brings computation closer to the user and can support offline use. The budget now includes memory shared with other applications, sustained power, and thermal limits. A configuration that loads successfully may still be too slow or unstable for a long session.

## 11.1 The edge inference landscape

Phones, laptops, and embedded devices place inference alongside other applications. Offline operation and local data processing can be valuable, but model quality must be assessed within a sustained resource budget. CPU, GPU, and NPU backends may support different operators and numeric formats.

Start with the task, allowable delay, available memory, and session duration. Then choose a model and runtime that satisfy those requirements together; a peak token-rate result from a short run does not establish a usable device experience.

## 11.2 Memory constraints and quantization for edge

```python
# Hypothetical device budget in GiB; metadata/workspace are estimates.
parameters, bits = 1e9, 4
weights = parameters * bits / 8 / 2**30
kv = 2 * 24 * 8 * 128 * 4096 * 2 / 2**30
workspace, runtime = 0.5, 0.4
print(round(weights + kv + workspace + runtime, 2), 'GiB estimated process budget')
```

The estimate is about 1.74 GiB, before extra quantization metadata and other process allocations. A four-bit model still commonly has a higher-precision cache. Shared system memory also supports the OS and other applications. Reduce context or concurrency when cache state consumes the remaining budget.

## 11.3 Runtime frameworks

### Runtime frameworks

llama.cpp uses GGUF models and device-specific backends; MLX targets Apple hardware; ONNX Runtime and mobile runtimes support their own operators and execution providers. A model file, quantization scheme, and accelerator backend must match. Do not assume an NPU accelerates all operations just because the device advertises one.

### Measure the whole experience

Track load time, first-token delay, sustained decode speed, peak memory, and energy over repeated sessions. Short bursts can look fast before thermal throttling. Test offline behavior and failure on insufficient memory as part of the application.

## 11.4 Neural architecture search for efficiency

Choosing fewer layers, narrower hidden states, GQA, or a smaller vocabulary changes quality and deployment cost. Architecture search evaluates such alternatives against device constraints, but it requires trained candidates or an appropriate proxy. An existing checkpoint cannot be arbitrarily reshaped without additional training.

Hardware-aware architecture search can vary depth, hidden width, head layout, supported precision, or early-exit policies against an actual device objective. Parameter count alone is a weak latency predictor: alignment, memory access, and supported operators can make a smaller candidate slower. Measure the chosen runtime on the target device and validate the selected subnet's quality; a supernet or pruning recipe does not remove that requirement.

Memory mapping a checkpoint can reduce eager copying and make loading practical, but it does not eliminate attention state or the need to fetch weight pages. Unified memory similarly changes where capacity comes from while leaving competition with the operating system and data movement in the execution budget.

## Hands-on extensions

Use these experiments to connect the mechanisms above to measurements. They are proposed exercises unless a recorded result is explicitly identified.

### Hands-on: measure a real edge device

Use a small supported local model on the intended CPU or edge device and repeat prompts long enough to reveal thermal behavior. Publish load time, sustained output rate, process memory, and task quality. A V100 run is the server reference, not evidence of on-device energy or battery life.
