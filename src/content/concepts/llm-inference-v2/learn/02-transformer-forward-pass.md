---
title: "One Forward Pass Through a Decoder Transformer"
description: "What one decoder layer does to hidden vectors: causal attention, a feed-forward network, residuals, and the shapes you should be able to predict."
section: learn
pubDate: 2026-10-29
part: 1
chapter: 2
order: 2
tags: ["llm-inference", "fundamentals", "pytorch"]
---

**Difficulty:** Beginner → Intermediate  
**Big question:** What transforms input embeddings into next-token scores?  
**You will build:** A shape-aware causal attention step, then a tiny feed-forward step beside it.

## 1. Anatomy of one forward pass

A decoder-only model (GPT-2, Llama, Qwen, Mistral) runs the same block \(L\) times. Weights differ per layer. The dataflow does not.

```text
Token IDs
  → embedding lookup
  → block 1 (attention, then MLP)
  → block 2
  → … block L
  → final norm
  → LM head
  → logits at every position
```

For a sequence of \(T\) tokens and width \(D\), the hidden state stays \(X \in \mathbb{R}^{B \times T \times D}\) from the embedding through the last block. Attention mixes tokens. The feed-forward network mixes features *inside* one token. Neither step changes \(B\), \(T\), or \(D\).

For teaching, follow a *pre-norm* block. Modern Llama-style models use this order. GPT-2 also uses pre-norm, with LayerNorm rather than the RMSNorm used by Llama.

```text
x ────────────────┐
│                 │
└─ Norm → Attention → (+) = h
                     │
      h ────────────┤
      └─ Norm → MLP → (+) = output
```

A residual adds the sublayer's update back onto its input, so each block learns a change rather than a full replacement. The norm (LayerNorm or RMSNorm) keeps the scale of that update stable. Exact norm placement varies. The residual is the part to remember: the output width equals the input width.

## 2. Attention, then a feed-forward network

One head builds queries, keys, and values with learned projections:

\[
Q = X W_Q, \qquad K = X W_K, \qquad V = X W_V
\]

\[
\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}} + M\right) V
\]

- \(X\) is the hidden state, shape `[B, T, D]`.
- \(Q\) asks which positions matter for this token.
- \(K\) is what each position offers for that comparison.
- \(V\) is the information that gets mixed.
- \(d_h\) is the head width. Dividing by \(\sqrt{d_h}\) keeps the dot products from saturating softmax when \(d_h\) is large.
- \(M\) is the causal mask. Future positions are set to \(-\infty\) before softmax, so token \(t\) cannot read tokens after \(t\).

Consider the sentence:

> The battery was overheating, so it needed cooling.

To represent *it*, attention can mix earlier positions such as *battery* and *overheating*. The weights are computed from the vectors, not from a grammar rule. They are not guaranteed to match the explanation a person would give.

After the heads are concatenated and projected, the feed-forward network runs independently at each position. GPT-2 uses a two-matrix MLP with GELU. Width expands to \(4D\), then returns to \(D\):

```python
import torch
import torch.nn.functional as F

def gpt2_mlp(x, w_up, w_down):
    # x: [B, T, D], w_up: [D, 4D], w_down: [4D, D]
    return F.gelu(x @ w_up) @ w_down
```

Llama-style models use *SwiGLU*, three matrices, with a gated product:

\[
\mathrm{FFN}(x) = \big(\mathrm{SiLU}(x W_1) \odot x W_3\big) W_2
\]

```python
def swiglu(x, w1, w3, w2):
    # w1, w3: [D, D_ff], w2: [D_ff, D]
    return (F.silu(x @ w1) * (x @ w3)) @ w2
```

In both designs the MLP does not look at other positions. Mixing across the sequence already happened in attention. The MLP's job is a position-wise feature transform, and it holds a large share of the layer's parameters (the \(D \times 4D\) style matrices). That is why, during decode, most of the bytes you load from memory are these weights, not the attention scores.

## 3. A minimal causal attention step

This snippet isolates attention. It omits the output projection, the norm, the MLP, and the residual. The tensors are random. This is not a language model.

```python
import math
import torch
import torch.nn.functional as F

torch.manual_seed(7)
B, T, D, H = 1, 4, 8, 2
dh = D // H
x = torch.randn(B, T, D)

wq = torch.randn(D, D) / math.sqrt(D)
wk = torch.randn(D, D) / math.sqrt(D)
wv = torch.randn(D, D) / math.sqrt(D)

def split_heads(t):
    # [B, T, D] -> [B, H, T, dh]
    return t.view(B, T, H, dh).transpose(1, 2)

q = split_heads(x @ wq)
k = split_heads(x @ wk)
v = split_heads(x @ wv)

scores = q @ k.transpose(-2, -1) / math.sqrt(dh)       # [B, H, T, T]
mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
scores = scores.masked_fill(mask, float("-inf"))
weights = F.softmax(scores, dim=-1)
context = weights @ v                                   # [B, H, T, dh]
out = context.transpose(1, 2).contiguous().view(B, T, D)

print("Input:", tuple(x.shape))
print("Attention weights:", tuple(weights.shape))
print("Attention output:", tuple(out.shape))
print(weights[0, 0])  # row t may only use columns 0..t
```

Elements above the diagonal of `weights` are zero. Row 0 puts all of its mass on column 0. Row 3 may use columns 0, 1, 2, and 3. If the query at position 3 matches the keys at positions 1 and 2, its output is a mixture of those value vectors, plus its own. It cannot read position 4.


## 4. Track the shapes

Standard multi-head attention, with `B = 1`, `T = 4`, `D = 8`, `H = 2`, so `dh = 4`:

| Tensor | Shape | Why it has that shape |
|---|---|---|
| Input `X` | `[1, 4, 8]` | One hidden vector per token |
| `Q`, `K`, `V` after the head split | `[1, 2, 4, 4]` | Two heads, each of width 4 |
| Scores \(QK^\top\) | `[1, 2, 4, 4]` | Every query against every key |
| Attention result | `[1, 2, 4, 4]` | Weighted values |
| Concatenated heads | `[1, 4, 8]` | Heads merged back to `D` |
| MLP hidden | `[1, 4, 32]` | Expansion, here `4D`, still one row per token |
| Block output | `[1, 4, 8]` | The residual restores width `D` |

The score tensor is the one that grows with \(T^2\). For `T = 4` that is nothing. For a 128k context it is the tensor you do not want to materialize in GPU memory. Later chapters on kernels come back to that. Here, notice only that `Q`, `K`, and `V` themselves are linear in \(T\), while the scores are quadratic.

Grouped-query attention and latent attention change the key and value shapes. The query path in this table is the multi-head case. [Attention variants](./attention-variants) redraws `K` and `V`.

## 5. Why the blocks are stacked

One block lets each token mix with earlier tokens, then transform its own features. The next block receives that result, not the original token IDs. Depth is repeated refinement of the same `[B, T, D]` tensor.

A useful split:

- *Across tokens:* self-attention, limited by the causal mask.
- *Within one token:* the MLP, shared weights, no sequence axis.
- *Residual path:* the block's update added back onto the incoming hidden state.

The attention output is not a sentence. It is another stack of hidden vectors. Only the vocabulary projection in the next chapter turns the last vector into token scores.


---

**Continue learning:** [← Previous chapter](./from-text-to-tokens) · [Next chapter →](./logits-and-sampling)
