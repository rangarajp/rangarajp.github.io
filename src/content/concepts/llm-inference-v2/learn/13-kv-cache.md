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

The baseline exposes the cost of recomputing a prefix. This chapter focuses on retaining, sharing, and compressing that state; the next chapter handles where the retained blocks live.

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

## 2. PagedAttention: allocate retained state in blocks

A request's final length is not known when it arrives. Reserving its maximum possible cache contiguously can leave large regions unused, while requiring one growing contiguous allocation can create allocation pressure. Paging separates logical sequence positions from physical storage blocks.

The engine allocates usable blocks from a pool and records where each request's positions live. An attention kernel follows that mapping when reading keys and values. A partially filled final block still has tail waste; shared blocks also need ownership and copy-on-write rules. Paging improves allocation efficiency, not the size of each uncompressed KV element or the dense attention equation.

The next chapter works through block addressing, sharing, and reclamation with code. Here the key budget distinction is between logical retained tokens, allocated block slots, and the engine's reserved pool.

## 3. Prefix caching

Requests with an identical prefix can reuse its computed KV state. Identity includes token IDs, model revision, adapter, position configuration, and other execution settings. Identical text under a different chat template may not produce identical tokens. Reuse must be scoped to the appropriate tenant or trust boundary.

A prefix cache avoids prefill for the reused positions. It still retains memory, and the next decode query still attends over that state. It is different from semantic response caching, which returns an earlier answer to a sufficiently similar query and changes application behavior.

## 4. StreamingLLM and attention sinks

StreamingLLM retains a few initial positions and a recent window. Some models assign substantial attention to initial tokens that act as sinks; keeping them can stabilize bounded-window streaming. It does not retain arbitrary facts from discarded middle tokens. Eviction changes the accessible context, so evaluate long-range task quality.

The retained state can be written as the union of initial sink positions and a recent window. That bounds storage, but it does not preserve the information content of a full unbounded conversation. If a required fact was evicted from the middle, the model needs another retrieval or memory mechanism to recover it.

## 5. KV cache quantization

Lower-precision keys and values can reduce capacity and bandwidth requirements. Scales, outliers, and attention sensitivity make this a separate decision from weight quantization. Cache compression may introduce dequantization work and quality loss. Its error is not confined to a disposable current step: retained approximations are reused by later queries and can affect future generated tokens. Test at the intended context length, including retrieval of information far from the end.

Primary reading: [PagedAttention](https://arxiv.org/abs/2309.06180) and [StreamingLLM](https://arxiv.org/abs/2309.17453).

## V100 lab: reuse a 481-token system prefix

The [7B cache notebook](https://github.com/rangarajp/rangarajp.github.io/blob/main/notebooks/llm-inference/observability/experiments/3.1_kv_cache.ipynb) compares full-prompt processing with reuse of an already prepared 481-token system prefix for three different questions.

| Question | Full prompt (ms) | Reused prefix (ms) |
|---|---:|---:|
| What is a KV cache? | 106.5 | 34.3 |
| What is PagedAttention? | 106.7 | 33.6 |
| What is prefill vs decode? | 106.9 | 33.5 |

The recorded medians were 106.7 ms and 33.6 ms, approximately 3.18× for this prompt-processing comparison. Preparing and retaining the prefix has a cost that the reused-prefix timing excludes. This is explicit Hugging Face cache reuse; the notebook's separate vLLM sample has prefix caching disabled. It therefore does not measure vLLM's automatic prefix-cache hit rate or end-to-end server latency.

## Hands-on: measure prefix reuse with cold and warm cases

Extend the 481-token system-prefix experiment with fully shared, partially shared, and changed prefixes. Record the exact common token count, cache-reset policy, and prompt-processing latency. Include a negative-control prefix that must not reuse state.
