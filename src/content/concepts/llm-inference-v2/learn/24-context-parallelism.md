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

Expert placement distributes conditional computation; context parallelism distributes sequence state and attention work. This chapter also distinguishes that distribution from methods that discard or compress context.

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

Prompt compression can operate before the target model runs. LLMLingua uses a smaller model and a token budget to identify text that can be removed while attempting to preserve useful information. Its benefit must pay for the compressor's own work, and aggressive deletion can remove a number, qualifier, or dependency needed by the task. [The LLMLingua paper](https://arxiv.org/abs/2310.05736) provides the method and evaluated settings.

Learned compression takes another route: train a model to represent a long prefix using a smaller set of summary representations. That representation needs model support and appropriate training; it is not interchangeable with deleting arbitrary KV entries from a checkpoint. In both cases, evaluate exact recall and task success alongside saved tokens or bytes.

## 4. State space models

SSMs such as Mamba maintain recurrent state whose decode storage does not grow in the same way as a full attention KV cache. They are a different model architecture, not an attention backend you can install under any transformer. SSMs should not be treated as synonymous with linear attention; hybrids combine these mechanisms differently.

A simple recurrent state-space illustration is $h_t=Ah_{t-1}+Bx_t$, with output $y_t=Ch_t$. Selective models make parts of the state update input-dependent. During sequential decode, the retained recurrent state can have fixed size with respect to processed sequence length, unlike a growing attention KV cache.

This is a different access pattern: past information has been folded into state rather than retained as individually addressable keys and values. Fixed decode state does not mean zero workspace during prompt processing, nor does it guarantee the same exact-retrieval behavior as attention. Hybrid architectures can retain both recurrent state and attention caches.

Primary reading: [Ring Attention](https://arxiv.org/abs/2310.01889) and [Mamba](https://arxiv.org/abs/2312.00752).

## Hands-on: compare memory scaling and retained information

Plot predicted score/KV bytes and measured peak memory across context lengths on one V100. Add a retrieval task before and after windowing or summarization. A ring-attention communication experiment is separate and requires multiple devices.
