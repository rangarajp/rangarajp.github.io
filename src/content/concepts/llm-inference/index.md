---
title: 'LLM Inference Overview'
description: 'How large language models generate tokens efficiently at serving time.'
pubDate: 'Aug 12 2026'
heroImage: '../../../assets/blog-placeholder-3.jpg'
---

LLM inference is what happens after training: turning a prompt into tokens, one step at a time, under latency and cost constraints. This series covers the ideas behind modern serving stacks — from the basic generate loop to caching, batching, and decoding tricks.

Read them in order for the full picture, or jump to any topic that interests you.

## 1. The pipeline

Topics to cover in this series:

1. **Autoregressive generation** — the prefill and decode loop that produces the next token
2. **KV cache** — storing past keys and values so decode does not recompute the full sequence
3. **Batching** — packing requests to keep GPUs busy without blowing up latency
4. **Quantization** — trading precision for memory and throughput
5. **Speculative decoding** — drafting with a small model and verifying with a large one
6. **Serving trade-offs** — latency vs throughput, context length, and memory walls

Together, these ideas explain how inference engines turn a trained transformer into a practical API.
