---
title: "Attention Heads: MHA, MQA, GQA, and MLA"
description: "How MHA, MQA, GQA, and MLA change the key/value tensors you store, with a shape exercise and a byte comparison."
section: learn
pubDate: 2026-10-29
part: 1
chapter: 7
order: 7
tags: ["llm-inference", "fundamentals", "pytorch"]
---

The cache formula depends on KV heads, not just hidden width. Compare the attention layouts before using a checkpoint configuration to estimate memory.

The previous chapter's cache formula depends on $n_{\text{kv}}$, the number of key/value heads. This chapter is why that number is a design choice.

## 1. Multi-head attention

In *multi-head attention* (MHA) each query head has its own key head and value head. With 8 query heads:

```text
Queries:  Q0 Q1 Q2 Q3 Q4 Q5 Q6 Q7
Keys:     K0 K1 K2 K3 K4 K5 K6 K7
Values:   V0 V1 V2 V3 V4 V5 V6 V7
```

Heads can learn different comparisons. More heads do not by themselves raise quality. They do raise the cache: stored K and V scale with the head count. Llama-2-7B is this pattern (32 query heads and 32 KV heads).

## 2. Multi-query attention

*Multi-query attention* (MQA) keeps many query heads and shares one key head and one value head:

```text
Queries:  Q0 Q1 Q2 Q3 Q4 Q5 Q6 Q7
Keys:                 K0
Values:               V0
```

The cache shrinks by about the query-head count, because $n_{\text{kv}} = 1$. Every query head attends to that single K/V pair. The sharing is a capacity limit: one key space has to serve every query head. Models that use it are choosing a smaller cache, and they have to make up for that sharing in training. It is not a free speedup you can patch onto an MHA checkpoint without further work.

## 3. Grouped-query attention

*Grouped-query attention* (GQA) puts query heads into groups. Each group shares one K/V head.

```text
Hq = 8, Hkv = 2:
Q0 Q1 Q2 Q3 → K0, V0
Q4 Q5 Q6 Q7 → K1, V1
```

Llama-2-70B, Llama-3, Mistral, and Qwen2 use this pattern. Llama-2-70B uses 64 query heads and 8 KV heads, so each KV head serves 8 query heads. Against an MHA model with the same 64 query heads, the cache is 8 times smaller. Against MQA it is 8 times larger. The point of the group is that middle.

Holding batch, tokens, head width, and layers fixed, cache bytes are proportional to $n_{\text{kv}}$:

| Layout | $n_{\text{kv}}$ for 8 query heads | Cache versus MHA |
|---|---:|---:|
| MHA | 8 | 1× |
| GQA, groups of 4 | 2 | 1/4 |
| MQA | 1 | 1/8 |

That comparison is the cache only. It is not a claim about end-to-end tokens per second. The MLP weights, which often dominate the byte load during decode, do not shrink when you drop KV heads.

Attention scores need the query heads and the key heads to line up. A GQA kernel can broadcast each KV head across its group without writing a physical copy. The following expansion makes the shapes obvious. The final reshape may materialize copies; a production GQA kernel avoids storing these repeated heads:

```python
import torch

def repeat_kv(k, n_rep):
    """[B, Hkv, T, Dh] -> [B, Hkv * n_rep, T, Dh]."""
    b, h, t, d = k.shape
    k = k[:, :, None, :, :].expand(b, h, n_rep, t, d)
    return k.reshape(b, h * n_rep, t, d)

B, T, Hq, Hkv, Dh = 1, 1024, 8, 2, 128
q = torch.randn(B, Hq, T, Dh)
k = torch.randn(B, Hkv, T, Dh)
v = torch.randn(B, Hkv, T, Dh)

k_full = repeat_kv(k, n_rep=Hq // Hkv)
v_full = repeat_kv(v, n_rep=Hq // Hkv)
scores = q @ k_full.transpose(-2, -1) / (Dh ** 0.5)
print(tuple(q.shape), tuple(k.shape), tuple(scores.shape))
# [1, 8, 1024, 128]  [1, 2, 1024, 128]  [1, 8, 1024, 1024]
```

`scores` is still `[B, Hq, T, T]`. The savings are in `k` and `v`, which is what the cache keeps. The score matrix is not stored in the cache. It is recomputed each step, ideally without ever writing the full $T \times T$ block to HBM.


## 4. Multi-head latent attention

*Multi-head latent attention* (MLA), used by DeepSeek-V2 and later DeepSeek models, does not pick a smaller head count and stop. It compresses the key/value content of a token into a shorter latent vector of width $c$, with $c$ well below the usual $n_{\text{kv}} \times d_h$, and stores that latent in the cache. Queries are still multi-head. On the way into the score, the latent is mapped back up so the heads can attend. Position handling is part of that design (some dimensions are kept uncompressed so rotary position can be applied), which is why MLA is not "GQA with one more reduction."

The motive matches the earlier chapters: the cache grows with every token and every layer, and decode reads it on every step. A shorter stored vector is a shorter read. Published DeepSeek configurations cut that cached state several times versus MHA, on the order of 5–13× depending on the model, with the quality coming from training the compression jointly. Do not quote a single ratio as if it were a property of the name MLA.

A picture that is sometimes repeated, MHA : GQA : MLA ≈ 8 : 2 : 1, is one imagined stack (8 KV heads, then 2, then a latent near the size of 1 head). It is a mnemonic for that stack, not a measurement. Use the byte formula with the head counts or latent width you actually have.

## 5. Compare the formats

| Approach | Query heads | What the cache holds | Idea |
|---|---|---|---|
| MHA | Many | One K and V head per query head | Separate KV heads |
| MQA | Many | A single shared K and V head | Maximum sharing |
| GQA | Many | One K and V head per group | Sharing, with some separate KV spaces left |
| MLA | Many | A latent vector per token (plus a small positional piece) | Compress, then reconstruct for the score |

Per layer, ignoring padding, MHA, MQA, and GQA store:


$$
N = 2 \cdot B \cdot n_{\text{kv}} \cdot T \cdot d_h
$$


elements. Bytes are $N$ times the element size, times the layer count. The symbol in that product is $n_{\text{kv}}$, not the query-head count.

```python
def kv_bytes(layers, kv_heads, head_dim, tokens, bytes_per_elem=2, batch=1):
    return 2 * batch * layers * kv_heads * tokens * head_dim * bytes_per_elem

# Same toy depth, three layouts, 4096 tokens, FP16.
common = dict(layers=32, head_dim=128, tokens=4096)
for name, heads in (("MHA", 8), ("GQA", 2), ("MQA", 1)):
    gb = kv_bytes(kv_heads=heads, **common) / 1e9
    print(f"{name:4}  {gb:.3f} GB")
```

You should see a clean 8 : 2 : 1 ratio, because that is what `kv_heads` you passed in. Change `head_dim` or `layers` and all three rows scale together. Change only `kv_heads` and that row moves.


---


Continue with [the inference problem](./inference-problem) for the full reading map, then [hardware foundations](./gpu-fundamentals), [quantization](./quantization), and [KV cache optimization](./kv-cache).

**Papers worth the original wording:** Vaswani et al., *Attention Is All You Need* (the MHA formula). Shazeer, *Fast Transformer Decoding* (MQA). Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*. DeepSeek-AI, *DeepSeek-V2* (MLA). Su et al., *RoFormer* (the rotation introduced in [text to tokens](./from-text-to-tokens)).

## V100 lab: verify the checkpoint's head counts

The [Qwen2.5-7B cache inspection](https://github.com/rangarajp/rangarajp.github.io/blob/main/notebooks/llm-inference/observability/experiments/3.1_kv_cache.ipynb) prints `heads=28`, `kv_heads=4`, and `head_dim=128`. That is seven query heads per KV head. Its observed key shape `[1, 4, 10, 128]` stores four KV heads, not 28. Substituting query heads into the cache formula would overestimate this configuration's logical KV storage by seven times.

This is an observed configuration and tensor-shape check. It is not an experiment comparing independently trained MHA and GQA models at equal quality.

## Hands-on: measure KV storage rather than expanded tensors

Instantiate toy MHA, GQA, and MQA caches with matched batch, context, layers, and head width. Publish tensor-storage bytes and the predicted ratio. If benchmarking repeated heads, identify any materialized copies so the implementation does not erase the intended saving.
