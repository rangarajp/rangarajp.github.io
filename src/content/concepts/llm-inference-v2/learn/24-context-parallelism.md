---
title: "Long-context and memory management"
description: "Long context increases both retained state and attention work."
section: learn
pubDate: 2026-10-29
chapter: 10
order: 10
---

Long context increases both retained state and attention work. Some techniques preserve full attention while changing its execution; others reduce or summarize the information retained. Distinguishing those choices is essential when exact recall matters.

## 10.1 The quadratic attention problem

```python
# One hypothetical score tensor; FlashAttention avoids materializing it.
batch, heads, element_bytes = 1, 32, 2
for tokens in (4096, 32768, 131072):
    scores = batch * heads * tokens * tokens * element_bytes
    print(tokens, round(scores / 2**30, 1), 'GiB')
```

Eight times the length produces 64 times the score elements. FlashAttention reduces intermediate storage and IO, but full prefill attention still performs quadratic pairwise work. Single-token decode attends to the retained prefix with linear work per step.

## 10.2 FlashAttention for long context

FlashAttention tiles attention and maintains an online normalization state, avoiding a full sequence-by-sequence score matrix in HBM. Dense prefill still performs quadratic arithmetic in sequence length; the gain is in intermediate storage and memory traffic. During decode, a new query still reads the retained keys and values.

For a long-context workload, measure peak allocated memory and prompt time separately. A lower memory footprint does not imply constant-time attention. The [kernel engineering chapter](./flashattention) derives the online softmax recurrence and provides the numerical example.

## 10.3 Ring attention

Distribute sequence positions across devices. In ring attention, key/value blocks circulate so query shards can accumulate attention using an online normalization. This expands feasible context at the cost of transfers and synchronization. It does not make the total dense-attention arithmetic disappear.

## 10.4 Context compression

Retrieval selects relevant material; summarization rewrites it; sliding windows discard older positions; KV eviction removes stored state. These techniques have different semantics. A summary can omit an exact number needed later. Evaluate with tasks requiring distant facts, contradictions, and precise citations, rather than checking only fluent output.

Prompt compression can operate before the target model runs. LLMLingua uses a smaller model and a token budget to identify text that can be removed while attempting to preserve useful information. Its benefit must pay for the compressor's own work, and aggressive deletion can remove a number, qualifier, or dependency needed by the task. [The LLMLingua paper](https://arxiv.org/abs/2310.05736) provides the method and evaluated settings.

Learned compression takes another route: train a model to represent a long prefix using a smaller set of summary representations. That representation needs model support and appropriate training; it is not interchangeable with deleting arbitrary KV entries from a checkpoint. In both cases, evaluate exact recall and task success alongside saved tokens or bytes.

## 10.5 State space models and recurrent memory

SSMs such as Mamba maintain recurrent state whose decode storage does not grow in the same way as a full attention KV cache. They are a different model architecture, not an attention backend you can install under any transformer. SSMs should not be treated as synonymous with linear attention; hybrids combine these mechanisms differently.

A simple recurrent state-space illustration is $h_t=Ah_{t-1}+Bx_t$, with output $y_t=Ch_t$. Selective models make parts of the state update input-dependent. During sequential decode, the retained recurrent state can have fixed size with respect to processed sequence length, unlike a growing attention KV cache.

This is a different access pattern: past information has been folded into state rather than retained as individually addressable keys and values. Fixed decode state does not mean zero workspace during prompt processing, nor does it guarantee the same exact-retrieval behavior as attention. Hybrid architectures can retain both recurrent state and attention caches.

Primary reading: [Ring Attention](https://arxiv.org/abs/2310.01889) and [Mamba](https://arxiv.org/abs/2312.00752).

## Hands-on extensions

Use these experiments to connect the mechanisms above to measurements. They are proposed exercises unless a recorded result is explicitly identified.

### Hands-on: compare memory scaling and retained information

Plot predicted score/KV bytes and measured peak memory across context lengths on one V100. Add a retrieval task before and after windowing or summarization. A ring-attention communication experiment is separate and requires multiple devices.
