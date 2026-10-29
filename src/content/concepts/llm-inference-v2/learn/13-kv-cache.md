---
title: "KV cache optimization"
description: "Prefix reuse, attention sinks, cache quantization, and the memory budget."
section: learn
pubDate: 2026-10-29
part: 3
chapter: 13
order: 13
tags: ["llm-inference", "systems", "hands-on"]
---

The [foundation cache example](./prefill-decode-kv-cache) appends tensors with `torch.cat`. That is easy to inspect, but repeated concatenation reallocates and copies state. A serving engine reserves or pages storage and writes new keys and values into available slots.

## 1. The memory budget

For ordinary MHA/GQA, cache bytes are $2BLH_{kv}Td_hs$, where $B$ is sequences, $L$ layers, $H_{kv}$ KV heads, $T$ retained tokens, $d_h$ head width, and $s$ bytes per element. Ragged batches use the sum of sequence lengths instead of $BT$.

```python
layers, heads, dim, element_bytes = 32, 8, 128, 2
bytes_per_token = 2 * layers * heads * dim * element_bytes
lengths = [512, 2048, 4096]
print(sum(lengths) * bytes_per_token / 2**20, 'MiB')  # 832 MiB
```

That excludes padding, metadata, and workspace. Output reservations also matter: admitting requests using prompt length alone can exhaust memory later.

## 2. Prefix caching

Requests with an identical prefix can reuse its computed KV state. Identity includes token IDs, model revision, adapter, position configuration, and other execution settings. Identical text under a different chat template may not produce identical tokens. Reuse must be scoped to the appropriate tenant or trust boundary.

A prefix cache avoids prefill for the reused positions. It still retains memory, and the next decode query still attends over that state. It is different from semantic response caching, which returns an earlier answer to a sufficiently similar query and changes application behavior.

## 3. StreamingLLM and attention sinks

StreamingLLM retains a few initial positions and a recent window. Some models assign substantial attention to initial tokens that act as sinks; keeping them can stabilize bounded-window streaming. It does not retain arbitrary facts from discarded middle tokens. Eviction changes the accessible context, so evaluate long-range task quality.

## 4. KV cache quantization

Lower-precision keys and values can reduce capacity and bandwidth requirements. Scales, outliers, and attention sensitivity make this a separate decision from weight quantization. Cache compression may introduce dequantization work and quality loss. Test at the intended context length, including retrieval of information far from the end.

Primary reading: [PagedAttention](https://arxiv.org/abs/2309.06180) and [StreamingLLM](https://arxiv.org/abs/2309.17453).
