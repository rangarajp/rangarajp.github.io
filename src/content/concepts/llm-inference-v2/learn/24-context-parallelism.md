---
title: "Long-context and memory management"
description: "Quadratic attention, ring attention, compression, and state space models."
section: learn
pubDate: 2026-10-29
part: 4
chapter: 24
order: 24
tags: ["llm-inference", "systems", "hands-on"]
---

# Long-context and memory management

Long context increases more than the input token bill. It grows KV state, prefill attention arithmetic, and the cache each decode query must read. Context capacity and reliable use of distant information are separate properties.

## 1. The quadratic attention problem

```python
# One hypothetical score tensor; FlashAttention avoids materializing it.
batch, heads, element_bytes = 1, 32, 2
for tokens in (4096, 32768, 131072):
    scores = batch * heads * tokens * tokens * element_bytes
    print(tokens, round(scores / 2**30, 1), 'GiB')
```

Eight times the length produces 64 times the score elements. FlashAttention reduces intermediate storage and IO, but full prefill attention still performs quadratic pairwise work. Single-token decode attends to the retained prefix with linear work per step.

## 2. Ring attention and context parallelism

Distribute sequence positions across devices. In ring attention, key/value blocks circulate so query shards can accumulate attention using an online normalization. This expands feasible context at the cost of transfers and synchronization. It does not make the total dense-attention arithmetic disappear.

## 3. Context compression

Retrieval selects relevant material; summarization rewrites it; sliding windows discard older positions; KV eviction removes stored state. These techniques have different semantics. A summary can omit an exact number needed later. Evaluate with tasks requiring distant facts, contradictions, and precise citations, rather than checking only fluent output.

## 4. State space models

SSMs such as Mamba maintain recurrent state whose decode storage does not grow in the same way as a full attention KV cache. They are a different model architecture, not an attention backend you can install under any transformer. SSMs should not be treated as synonymous with linear attention; hybrids combine these mechanisms differently.

Primary reading: [Ring Attention](https://arxiv.org/abs/2310.01889) and [Mamba](https://arxiv.org/abs/2312.00752).
