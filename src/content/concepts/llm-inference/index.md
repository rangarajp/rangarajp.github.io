---
title: 'LLM Inference'
description: 'How large language models generate tokens efficiently at serving time — from GPU silicon to batching and quantization.'
pubDate: 'Aug 12 2026'
seriesOrder: 2
heroImage: '../../../assets/blog-placeholder-3.jpg'
---

LLM inference is what happens after training: turning a prompt into tokens, one step at a time, under latency and cost constraints. This book covers the ideas behind modern serving stacks — from GPU fundamentals and the generate loop to caching, batching, and decoding tricks.

Read the chapters in order. Each one builds on the last.

## Contents

1. [Inference engineering basics](./basics) — runtime, infrastructure, tooling; TTFT, ITL, TPS; arithmetic intensity
2. [GPU basics](./gpu-basics) — SMs, cores, cache hierarchy, and architecture generations
3. [GPU architecture for inference](./gpu-architecture) — FLOPS, memory, bandwidth; how to read specs and pick cards
4. [vLLM basics and KV cache](./vllm-basics-kv-cache) — why decode is expensive and what the KV cache fixes
5. [Serving engine internals](./serving-engine-internals) — the loop that batches, schedules, and streams tokens
6. [Serving multiple models](./serving-multi-models) — one API, many checkpoints, cache and eviction
7. [Batching](./inference-optimizations-batching) — continuous batching and chunked prefill
8. [Quantization](./inference-optimizations-quantization) — trading precision for memory and throughput

## Coming later

- *Speculative decoding* — drafting with a small model and verifying with a large one
- *Serving trade-offs* — latency vs throughput, context length, and memory walls

Local lab paths: use gitignored `notebooks/llm-inference/local_paths.json` (from `local_paths.example.json`). Absolute machine paths are not published in these chapters.
