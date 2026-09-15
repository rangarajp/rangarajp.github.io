---
title: 'LLM Inference Overview'
description: 'How large language models generate tokens efficiently at serving time.'
pubDate: 'Aug 12 2026'
heroImage: '../../../assets/blog-placeholder-3.jpg'
---

LLM inference is what happens after training: turning a prompt into tokens, one step at a time, under latency and cost constraints. This series covers the ideas behind modern serving stacks — from GPU fundamentals and the generate loop to caching, batching, and decoding tricks.

Read them in order for the full picture, or jump to any topic that interests you.

## 1. The pipeline

Topics to cover in this series:

1. **GPU architecture** — FLOPS, memory, and bandwidth; how to read specs and pick cards
2. **Autoregressive generation** — the prefill and decode loop that produces the next token
3. **KV cache** — storing past keys and values so decode does not recompute the full sequence
4. **Batching** — packing requests to keep GPUs busy without blowing up latency
5. **Quantization** — trading precision for memory and throughput
6. **Speculative decoding** — drafting with a small model and verifying with a large one
7. **Serving trade-offs** — latency vs throughput, context length, and memory walls

Together, these ideas explain how inference engines turn a trained transformer into a practical API.

## Related posts in this folder

- [GPU Architecture for LLM Inference](./gpu-architecture)
- [vLLM Basics and Why KV Cache Matters](./vllm-basics-kv-cache)
- [LLM Serving Engine Internals](./serving-engine-internals)
- [Serving Multiple Models](./serving-multi-models)

Local lab paths: use gitignored `notebooks/llm-inference/local_paths.json` (from `local_paths.example.json`). Absolute machine paths are not published in these posts.
