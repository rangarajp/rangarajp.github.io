---
title: "Transformer inference mechanics"
description: "A request turns text into vectors, transforms those vectors through attention and feed-forward layers, and selects a token from vocabulary scores."
section: learn
pubDate: 2026-10-29
chapter: 3
order: 3
---

A request turns text into vectors, transforms those vectors through attention and feed-forward layers, and selects a token from vocabulary scores. Following that path makes the later optimizations concrete: we can identify exactly which tensors are computed, retained, or moved.

## 3.1 Anatomy of a transformer forward pass

The worked token IDs below come from the GPT-2 tokenizer in [`transformers_101.ipynb`](/build-from-scratch/gpt) and [`transformers_202.ipynb`](/build-from-scratch/gpt). A survey of tokenizer families is in [Tokenization](/concepts/transformers/tokenization).

### The model does not read words

Take the prompt:

> I am doing excellent

You see words. The model receives *token IDs*: integers from its tokenizer vocabulary. A token may be a word, a piece of a word, punctuation, or a whitespace pattern. One word is not one token.

GPT-2 byte-pair encoding (`tiktoken.get_encoding("gpt2")`) encodes that sentence as four IDs:

```text
Text:      I am doing excellent
Token IDs: [40, 716, 1804, 6275]
Pieces:    "I", " am", " doing", " excellent"
```

Decoding those IDs recovers the original string. The integers are not meanings. They are row indices in a vocabulary of size `50,257`.

The path into the network is:

1. *Text* — the prompt you type
2. *Tokenization* — BPE splits the text into vocabulary pieces
3. *Token IDs* — each piece maps to an integer
4. *Embedding lookup* — each ID selects a learned vector of width `d_model`
5. *Position* — the model injects order, either by adding a position vector (GPT-2) or by rotating queries and keys inside attention (Llama, Qwen, and most recent decoder models)

*Analogy:* a token ID is a dictionary index. The embedding row is a learned vector, not a definition written in English.

### Byte-pair encoding handles unknown spellings

BPE does not need a dictionary entry for every word. It merges frequent byte or character pairs until it reaches a target vocabulary, then spells rare or misspelled strings as a sequence of known pieces.

In the 101 notebook, a misspelling of *excellent* becomes seven tokens:

```text
Text:      I am doing excaellent
Token IDs: [40, 716, 1804, 2859, 64, 695, 298]
Pieces:    "I", " am", " doing", " exc", "a", "ell", "ent"
```

The familiar words stay intact. The unknown spelling splits into subwords the vocabulary already contains. Tokenization is not a spelling checker: the model can *encode* the string, then has to *learn* that those pieces usually do not belong together.

A second GPT-2 example from the 202 notebook shows a special token and an invented place name:

```text
Text: Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace.
```

That string is 20 tokens when you pass `allowed_special={"<|endoftext|>"}`. Then `<|endoftext|>` is one special ID (`50256`). If you omit `allowed_special`, `tiktoken` raises instead of silently splitting the marker. `someunknownPlace` is not one vocabulary word; BPE builds it from known fragments. Round-trip decode reconstructs the original text, including the marker.

*Rule of thumb* for this OpenAI-style BPE: about 1 token ≈ 4 characters ≈ ¾ of an English word. Code, technical terms, and other languages are often less efficient, so the same number of characters can cost more tokens and a longer prefill.

### From IDs to vectors

A token ID such as `40` is an index into `nn.Embedding(vocab_size, emb_dim)`. It is not a measure of meaning.

The 202 notebook uses GPT-2's vocabulary with a smaller demo width:

| Symbol | Notebook value | Role |
|---|---|---|
| `V` | `50,257` | GPT-2 vocabulary size |
| `B` | `2` | batch size |
| `T` | `4` | sequence length |
| `D` | `256` | embedding width in the demo (`768` in GPT-2 small) |

| Stage | Tensor shape | Meaning |
|---|---|---|
| Token IDs | `[B, T] = [2, 4]` | Integers in `0 … V−1` |
| Token embedding lookup | `[B, T, D] = [2, 4, 256]` | One vector per position |
| Position embeddings | `[T, D] = [4, 256]` | One vector per *index* `0 … T−1` |
| Input to the first block | `[B, T, D]` | Token embeddings plus position embeddings |

```python
import torch
import torch.nn as nn

vocab_size, emb_dim, max_length = 50257, 256, 4
inputs = torch.tensor([[15496, 11, 466, 345],
                       [11, 466, 345, 588]])  # [B, T]

token_embedding_layer = nn.Embedding(vocab_size, emb_dim)
pos_embedding_layer = nn.Embedding(max_length, emb_dim)

token_embeddings = token_embedding_layer(inputs)                 # [2, 4, 256]
pos_embeddings = pos_embedding_layer(torch.arange(max_length))   # [4, 256]
input_embeddings = token_embeddings + pos_embeddings             # [2, 4, 256]
```

Use `torch.arange(T)` for positions, not the token IDs. Position `0` always receives the same position vector, whichever word sits there.

Two prompts that tokenize to the same length still enter the first block as `[B, T, D]`. Changing the words changes which rows are read. Changing `T` changes how many rows you read. Changing `D` changes the width of every later matrix in the model.

### Anatomy of one forward pass

A decoder-only model (GPT-2, Llama, Qwen, Mistral) runs the same block $L$ times. Weights differ per layer. The dataflow does not.

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

For a sequence of $T$ tokens and width $D$, the hidden state stays $X \in \mathbb{R}^{B \times T \times D}$ from the embedding through the last block. Attention mixes tokens. The feed-forward network mixes features *inside* one token. Neither step changes $B$, $T$, or $D$.

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

### Attention, then a feed-forward network

One head builds queries, keys, and values with learned projections:


$$
Q = X W_Q, \qquad K = X W_K, \qquad V = X W_V
$$



$$
\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}} + M\right) V
$$


- $X$ is the hidden state, shape `[B, T, D]`.
- $Q$ asks which positions matter for this token.
- $K$ is what each position offers for that comparison.
- $V$ is the information that gets mixed.
- $d_h$ is the head width. Dividing by $\sqrt{d_h}$ keeps the dot products from saturating softmax when $d_h$ is large.
- $M$ is the causal mask. Future positions are set to $-\infty$ before softmax, so token $t$ cannot read tokens after $t$.

Consider the sentence:

> The battery was overheating, so it needed cooling.

To represent *it*, attention can mix earlier positions such as *battery* and *overheating*. The weights are computed from the vectors, not from a grammar rule. They are not guaranteed to match the explanation a person would give.

After the heads are concatenated and projected, the feed-forward network runs independently at each position. GPT-2 uses a two-matrix MLP with GELU. Width expands to $4D$, then returns to $D$:

```python
import torch
import torch.nn.functional as F

def gpt2_mlp(x, w_up, w_down):
    # x: [B, T, D], w_up: [D, 4D], w_down: [4D, D]
    return F.gelu(x @ w_up) @ w_down
```

Llama-style models use *SwiGLU*, three matrices, with a gated product:


$$
\mathrm{FFN}(x) = \big(\mathrm{SiLU}(x W_1) \odot x W_3\big) W_2
$$


```python
def swiglu(x, w1, w3, w2):
    # w1, w3: [D, D_ff], w2: [D_ff, D]
    return (F.silu(x @ w1) * (x @ w3)) @ w2
```

In both designs the MLP does not look at other positions. Mixing across the sequence already happened in attention. The MLP's job is a position-wise feature transform, and it holds a large share of the layer's parameters (the $D \times 4D$ style matrices). That is why, during decode, most of the bytes you load from memory are these weights, not the attention scores.

### A minimal causal attention step

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

### Track the shapes

Standard multi-head attention, with `B = 1`, `T = 4`, `D = 8`, `H = 2`, so `dh = 4`:

| Tensor | Shape | Why it has that shape |
|---|---|---|
| Input `X` | `[1, 4, 8]` | One hidden vector per token |
| `Q`, `K`, `V` after the head split | `[1, 2, 4, 4]` | Two heads, each of width 4 |
| Scores $QK^\top$ | `[1, 2, 4, 4]` | Every query against every key |
| Attention result | `[1, 2, 4, 4]` | Weighted values |
| Concatenated heads | `[1, 4, 8]` | Heads merged back to `D` |
| MLP hidden | `[1, 4, 32]` | Expansion, here `4D`, still one row per token |
| Block output | `[1, 4, 8]` | The residual restores width `D` |

The score tensor is the one that grows with $T^2$. For `T = 4` that is nothing. For a 128k context it is the tensor you do not want to materialize in GPU memory. Later chapters on kernels come back to that. Here, notice only that `Q`, `K`, and `V` themselves are linear in $T$, while the scores are quadratic.

Grouped-query attention and latent attention change the key and value shapes. The query path in this table is the multi-head case. [Attention variants](./transformer-forward-pass) redraws `K` and `V`.

### Why the blocks are stacked

One block lets each token mix with earlier tokens, then transform its own features. The next block receives that result, not the original token IDs. Depth is repeated refinement of the same `[B, T, D]` tensor.

A useful split:

- *Across tokens:* self-attention, limited by the causal mask.
- *Within one token:* the MLP, shared weights, no sequence axis.
- *Residual path:* the block's update added back onto the incoming hidden state.

The attention output is not a sentence. It is another stack of hidden vectors. Only the vocabulary projection in the following discussion turns the last vector into token scores.

### The language model head

After the last block and a final norm, every position still holds a vector of width $D$:


$$
H \in \mathbb{R}^{B \times T \times D}
$$


A linear map, the *LM head*, scores the whole vocabulary at every position:


$$
Z = H W_{\mathrm{vocab}}, \qquad W_{\mathrm{vocab}} \in \mathbb{R}^{D \times V}
$$


$V$ is the tokenizer vocabulary size, so $Z$ has shape `[B, T, V]`. For GPT-2, $V = 50257$. For a small Qwen it is on the order of 150,000. The head does not emit a word. It emits one score per vocabulary row.

Ordinary next-token generation reads only the last position, `Z[:, -1, :]`, a vector of length `V`. Earlier positions have logits too. During training those positions each predict the following token. During generation you already know the prompt, so you sample from the last row and append.

A *logit* is an unnormalized score. It can be negative. It does not sum to one. Softmax turns the row into a distribution:


$$
p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

### A four-token numerical example

Suppose a toy vocabulary has four tokens and the last position produced these logits. Softmax at temperature 1 gives:

| Token | Logit | Probability |
|---|---:|---:|
| `cooling` | 3.0 | 0.644 |
| `rising` | 2.0 | 0.237 |
| `normal` | 1.0 | 0.087 |
| `.` | 0.0 | 0.032 |

The probabilities sum to 1. They are the exact softmax of those four numbers, not a sample from a trained model.

Two different decisions sit on top of the same row:

- *Greedy decoding* picks the largest logit. Here that is `cooling`, every time.
- *Sampling* draws from $p$. `cooling` is the most likely draw. It is not a guaranteed draw.

Greedy is the usual meaning of temperature 0. Do not implement it as `logits / 0`.

### Temperature, top-k, and top-p

Temperature $\tau$ scales the logits *before* softmax:


$$
p_i = \mathrm{softmax}(z_i / \tau)
$$


A smaller positive $\tau$ sharpens the distribution. A larger $\tau$ flattens it. The ranking of the tokens does not change. Only the gaps do. These controls do not modify the trained weights. They modify the draw.

Top-k keeps the $k$ largest logits and sets the rest to $-\infty$ before softmax. Top-p (*nucleus* sampling) keeps the smallest set of tokens whose probabilities sum to at least $p$, walking from the largest probability downward. Both shrink the candidate set. They answer different questions: top-k caps the count, top-p caps the probability mass, so the count can change from step to step.

```python
import torch

torch.manual_seed(7)
logits = torch.tensor([3.0, 2.0, 1.0, 0.0])
tokens = ["cooling", "rising", "normal", "."]

def sample_from(logits, temperature=1.0, top_k=0, top_p=1.0):
    z = logits / temperature
    if top_k > 0:
        cutoff = torch.topk(z, k=top_k).values[-1]
        z = z.masked_fill(z < cutoff, float("-inf"))
    probs = torch.softmax(z, dim=-1)
    if top_p < 1.0:
        ordered, index = torch.sort(probs, descending=True)
        keep = ordered.cumsum(dim=-1) - ordered <= top_p
        keep[0] = True  # always retain the mode
        mask = torch.zeros_like(keep)
        mask.scatter_(0, index, keep)
        probs = probs * mask
        probs = probs / probs.sum()
    choice = torch.multinomial(probs, num_samples=1).item()
    return probs, tokens[choice]

for temperature in (0.5, 1.0, 2.0):
    probs, choice = sample_from(logits, temperature=temperature)
    shown = {tok: round(p, 3) for tok, p in zip(tokens, probs.tolist())}
    print(f"τ={temperature}", shown, "→", choice)

probs, choice = sample_from(logits, top_k=2)
print("top-2", {tok: round(p, 3) for tok, p in zip(tokens, probs.tolist())}, "→", choice)
```

At $\tau = 0.5$ almost all of the mass sits on `cooling`. At $\tau = 2$ the four probabilities move closer together, so `normal` and `.` show up more often if you draw many times. With `top_k=2`, `normal` and `.` have probability 0. Their logits were discarded before softmax, and the remaining two probabilities were renormalized.

Run the temperature loop a few dozen times if you want to see the draw change. A single `multinomial` call is one sample, not the distribution itself.

### The whole vocabulary is scored

Even when the next word feels obvious, the LM head writes a score for every row: whole words, fragments, spaces, digits, and special tokens. If the tokenizer splits `powertrain` into two pieces, the model cannot emit the whole string in one step. It samples an id, appends it, and samples again. A different tokenizer might do that in one id. The head only chooses among the rows it was trained with.

Softmax depends on differences, not on the absolute level of the logits. Subtracting the same constant from every entry leaves $p$ unchanged. Implementations subtract the max logit first so the exponentials do not overflow:

```python
def stable_softmax(z):
    shifted = z - z.max()
    ex = shifted.exp()
    return ex / ex.sum()
```

For `logits = [2.0, 1.0]`, `stable_softmax(logits)` and `stable_softmax(logits - 100)` match. The attention variants below `argmax` path never builds this distribution. Sampling does.

### The token generation loop

Decoder-only models are *autoregressive*: the distribution for the next id is conditioned on every id already in the sequence. The loop is:

1. Receive a prompt as token ids.
2. Run one forward pass over that whole prefix. This is the *prefill*. It writes keys and values for every prompt token.
3. Sample one id from the last position's logits. Append it.
4. Run the next forward pass for the new token, attending to the prefix. Repeat until an end-of-sequence id or a length limit.

```text
Prompt: "The battery"
Step 1: "The battery"                 → " is"
Step 2: "The battery is"              → " heating"
Step 3: "The battery is heating"      → " up"
Step 4: "The battery is heating up"   → "."
```

Those pieces are invented so the dependency is obvious. They are not a trace from a real model. One token is not necessarily one word. The stop condition is an id the tokenizer calls end-of-sequence, a stop string, or a maximum number of new tokens. It is not "the model decided the paragraph is finished" unless that decision was encoded as one of those rules.

Step 2 can run the prompt tokens together. Steps 3 and 4 cannot be skipped or reordered: the distribution at step 4 depends on which id step 3 actually drew. If you sampled `cooling` instead of `heating`, the following distribution would be a different row of the LM head.

### A loop you can run

This version sets `use_cache=False`. Every iteration feeds the entire id list and recomputes attention for every position. That is wasteful, and it is the right first implementation because the growing tensor is visible. Run it where the model can be downloaded, or point `model_id` at a local snapshot.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).eval()

messages = [{"role": "user", "content": "Explain why a battery needs cooling in one sentence."}]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
ids = tokenizer(prompt, return_tensors="pt").input_ids

with torch.inference_mode():
    for step in range(12):
        outputs = model(input_ids=ids, use_cache=False)
        next_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        piece = tokenizer.decode(next_id[0])
        ids = torch.cat([ids, next_id], dim=1)
        print(f"Step {step + 1}: {piece!r}  ids.shape={tuple(ids.shape)}")
        if tokenizer.eos_token_id is not None and next_id.item() == tokenizer.eos_token_id:
            break

print(tokenizer.decode(ids[0], skip_special_tokens=True))
```

`apply_chat_template` matters for instruct checkpoints. The model was trained to see a user turn and a generation marker. A raw sentence without that wrapper is a different input, even when the English looks the same.

`argmax` is greedy selection from the earlier forward-pass discussion. Swap it for the sampler you wrote there if you want temperature. Keep the loop structure.

This is not how a serving stack should decode. It rereads the whole prefix on every step. The following discussion keeps the keys and values so the new step only attends, instead of recomputing the prefix.

### The sequence grows by one id

With `B = 1` and a prompt of length `T = 5`:

| Step | `input_ids` shape | Logits used for the sample |
|---|---|---|
| Prefill | `[1, 5]` | `[1, V]` from position 5 |
| After one append | `[1, 6]` | `[1, V]` from position 6 |
| After another | `[1, 7]` | `[1, V]` from position 7 |

The forward pass returns logits at every position. Generation uses the last row only. The earlier rows are predictions for tokens you already have.

The second dimension of `ids` is the only shape that changes in this loop. `V` is fixed by the tokenizer. The hidden width is fixed by the checkpoint. What grows is the context the next distribution conditions on, and, once caching is on, the KV cache that stores it.

### Two choices that do not change the dependency

The example makes two independent choices:

1. *Decoding strategy.* `argmax` is greedy. Sampling would draw from the probabilities in the earlier forward-pass discussion. Either way you append exactly one id before the next forward pass.
2. *Computation strategy.* `use_cache=False` recomputes every position. A correct cache reuses keys and values for the prefix and runs the new token only. The following discussion builds that distinction. A wrong cache (feeding the full sequence *and* a past cache, or forgetting the position id) silently attends to the wrong places.

Changing either choice does not remove the dependency. Token $t+1$ is a function of the prefix through token $t$.

## 3.2 The KV cache: what it is and why it matters

### What the KV cache stores

Without a cache, step $n$ recomputes keys and values for all $n$ positions, in every layer. The uncached loop in the earlier forward-pass discussion does exactly that. The *KV cache* keeps those keys and values so the next step only projects the new token.

The cache is not the weights. Weights are the learned matrices $W_Q, W_K, W_V$, shared by every request and fixed at inference. The cache is activation state for *this* sequence. Drop the request and you may free the cache. You do not free the model.

For one sequence, one layer, standard attention stores:


$$
\text{bytes} = 2 \times n_{\text{kv}} \times T \times d_h \times \text{bytes per element}
$$


The leading 2 is key and value. $n_{\text{kv}}$ is the number of key/value heads, $T$ is tokens stored so far, and $d_h$ is the head width. Multiply by the layer count $L$ and the batch $B$.

```python
def kv_bytes(layers, kv_heads, head_dim, tokens, bytes_per_elem=2, batch=1):
    """K and V, all layers, one byte count. bytes_per_elem=2 is FP16 or BF16."""
    return 2 * batch * layers * kv_heads * tokens * head_dim * bytes_per_elem

# Llama-2-7B is multi-head: 32 layers, 32 KV heads, head dim 128.
per_token = kv_bytes(layers=32, kv_heads=32, head_dim=128, tokens=1)
print("bytes per token:", per_token)  # 524_288 = 512 KiB

# 512 prompt tokens + 1000 generated tokens.
print("MB at 1512 tokens:", kv_bytes(32, 32, 128, 1512) / 1e6)  # about 793
```

A decode step appends one token's K and V. The read is the whole cache. After a long conversation the cache, not the new token, dominates attention's memory traffic. The projections $W_Q, W_K, W_V, W_O$ and the MLP still have to be loaded too. Those weights do not grow with $T$. The cache does.

This educational step shows the append. `q`, `k`, and `v` would come from linear layers in a real block. Here they are random so you can watch the shapes.

```python
import math
import torch
import torch.nn.functional as F

def attend(q, k, v):
    """q is [B, H, Tq, Dh], k and v are [B, H, Tk, Dh]."""
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    if q.shape[-2] == k.shape[-2]:
        t = q.shape[-2]
        mask = torch.triu(torch.ones(t, t, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    return F.softmax(scores, dim=-1) @ v

torch.manual_seed(7)
B, T, H, Dh = 1, 4, 2, 4
q, k, v = torch.randn(B, H, T, Dh), torch.randn(B, H, T, Dh), torch.randn(B, H, T, Dh)
prefill = attend(q, k, v)          # [1, 2, 4, 4]
cache_k, cache_v = k, v

q_new = torch.randn(B, H, 1, Dh)
k_new, v_new = torch.randn(B, H, 1, Dh), torch.randn(B, H, 1, Dh)
cache_k = torch.cat([cache_k, k_new], dim=2)
cache_v = torch.cat([cache_v, v_new], dim=2)
decoded = attend(q_new, cache_k, cache_v)   # [1, 2, 1, 4]
print(tuple(prefill.shape), tuple(cache_k.shape), tuple(decoded.shape))
```

On the decode call, `Tq` is 1 and `Tk` is 5. There is no future token to mask: the single query is at the last position, so it may read every cached key. The attention variants below variants change `H` for `k` and `v` only. The append along the sequence axis stays the same.

### A worked memory estimate

Use KV heads, not query heads. Llama-2-70B has 80 layers, 64 query heads, and **8 KV heads**, with $d_h = 128$. At 4096 tokens, FP16, batch 1:

```text
2 × 80 × 8 × 4096 × 128 × 2 bytes ≈ 1.34 GB
```

If you plug in 64 heads by mistake, the same formula prints about 10.7 GB. That number is the multi-head counterfactual, not this checkpoint. The bug is common because model cards list both head counts, and the cache follows the smaller one. [Attention variants](./transformer-forward-pass) explains why the smaller count exists.

These are closed-form byte counts under the formula above. They omit allocator padding, paged-cache waste, and the weights. They are not a measured memory profile.

## 3.3 Prefill versus decode: two different workloads

### Two phases of one request

A prompt of 100 tokens and a reply of 3 new tokens is still one request. The work splits into *prefill* and *decode*.

*Prefill* runs the prompt in one forward pass. Attention can use every prompt token together, under the causal mask. The pass writes a key and a value for every prompt position, in every layer. The logits at the last prompt position select the first new token.

*Decode* then runs once per new token. Each step reads the keys and values already stored, appends the key and value of the new token, and samples the next id.

<figure>

![Prompt tokens processed together, then one new token at a time](./images/prefill-decode-flow.svg)

<figcaption><span class="figure-label">Figure 1.</span> Prefill handles the prompt in one pass; decode emits one token per step</figcaption>
</figure>

```text
Input prompt: [ P1 P2 P3 ... P100 ]

Prefill:    process P1..P100 together → select G1
            store K and V for every prompt position

Decode 1:   G1 attends to stored K/V → select G2
            append K/V for G1
Decode 2:   G2 attends to the longer cache → select G3
```

In a busy server a scheduler may interleave prefill work and decode work from different requests. That scheduling does not change the split: one phase consumes a known string in parallel, the other extends the string one id at a time.

### Why the two phases stress the GPU differently

Prefill multiplies a token matrix by a weight matrix: a GEMM whose token axis is the prompt length. Decode, at batch 1, multiplies a single vector by the same weight matrix: a GEMV.

A useful ratio is *arithmetic intensity*, FLOPs per byte moved. For $y = Wx$ with $W$ shaped `[rows, cols]` in FP16, you do about `2 × rows × cols` FLOPs and you must load the matrix:

```python
def gemv_intensity(rows, cols, bytes_per_elem=2):
    flops = 2 * rows * cols
    moved = (rows * cols + cols + rows) * bytes_per_elem
    return flops / moved

# One square projection at Llama-2-7B width.
print(round(gemv_intensity(4096, 4096), 3))  # about 1.0 FLOP/byte
```

The *ridge* of a chip is peak FLOP/s divided by peak memory bandwidth. An A100 80 GB SXM is about 312 TFLOP/s of FP16 tensor-core math (the dense rate, not the sparse 2×) and about 2.0 TB/s of HBM. The ridge is roughly $312 / 2.0 \approx 156$ FLOP/byte. A GEMV at 1 FLOP/byte sits far below that ridge, so the step is limited by how fast weights move from HBM, not by peak FLOP/s.

A 7-billion-parameter model in 16-bit weights is about $7 \times 10^9 \times 2$ bytes $= 14$ GB. Reading 14 GB at 2 TB/s takes $14 / 2000$ seconds $= 7$ ms. That is a lower bound for one decode step if the device streamed every weight byte once at the brochure bandwidth and did nothing else. Real decode also reads the KV cache, launches many kernels, and does not hold peak bandwidth on each of them. The bound is still the right picture: at batch 1 you pay a weight-load on the order of milliseconds to emit one token.

Raise the batch, or the prompt length, and the same loaded weight is reused across more rows. Intensity scales with that row count. A long prefill can approach the ridge. A single-token decode cannot. Batching many independent decodes amortizes the weight load. It does not shrink the latency of the slowest request for free, which is the tradeoff in the next section.

A useful benchmark measures a specific GPU, a specific kernel, and a specific model. The numbers in this section are arithmetic with stated assumptions, not a benchmark table.

### Latency, throughput, and cost

Three goals pull against each other once this loop is serving real traffic.

| Goal | What you watch | What usually helps | What gives |
|---|---|---|---|
| Latency | Time to first token (TTFT), time per output token (TPOT or inter-token latency) | Shorter prompts, a warm prefix cache, enough bandwidth for the weight load | A GPU that is not packed with other users |
| Throughput | Output tokens per second across concurrent requests | Larger batches, so one weight load serves many decodes | A slower reply for each user |
| Cost | Dollars per million tokens | Higher utilization, fewer bytes per weight (quantization, later) | Latency, or quality, depending on the method |

TTFT includes the prefill of *this* prompt plus queueing. A longer prompt increases TTFT even when you only ask for two output tokens, because prefill work scales with the prompt. A longer answer increases total time mostly through decode steps. It does not redo the prefill.

TPOT is the gap between streamed tokens. If each decode step is near that 7 ms weight-load bound, you are around 140 tokens per second for one stream before overhead. A chat that needs a smooth stream cares about this gap. A batch job cares about throughput. Packing the batch to raise throughput lengthens the step for everyone in it, because the GEMM gets larger and the slowest request waits for the step to finish.

Cost follows from how many tokens you get per second of GPU time you paid for. An idle decode at batch 1 burns the same weight-load as a full one and produces one token. That is the expensive regime.

### Two easy mistakes

Prefill does not write the answer. It writes the prompt's keys and values, and the logits that select the first new token. The rest of the answer is the decode loop.

The KV cache does not remove attention. The new query still scores against every cached key. Caching removes the *recompute* of old keys and values. The stored state grows by one token per step, per layer.



---

### Compare cached and uncached generation in the original lab

The [original notebook](https://github.com/rangarajp/rangarajp.github.io/blob/main/notebooks/llm-inference/basic_vLLM_v1.ipynb) used Qwen2.5-0.5B-Instruct with short prompts. Its uncached steps were about 31–35 ms and warmed cached steps about 30–31 ms. Those short traces do not establish a universal speedup or constant decode cost.

<figure>

![Original short-context run comparing per-step latency with and without a KV cache](../../llm-inference/images/kv-cache-comparison.png)

<figcaption><span class="figure-label">Figure 2.</span> Historical cached and uncached traces from the small-model notebook</figcaption>
</figure>

Without cache reuse, full causal attention recomputation over a prefix has quadratic pairwise work in its length, plus projections and MLP work over all prefix positions. With caching, only the new position is projected, but it still scores against the retained keys. Attention work and cache reads therefore grow with context. Weight streaming and launch overhead can conceal that growth in a short run.

<figure>

![Original lab separates prompt prefill from repeated single-token decode steps](../../llm-inference/images/prefill-vs-decode.png)

<figcaption><span class="figure-label">Figure 3.</span> The original lab separates the one-time prompt pass from later decode steps</figcaption>
</figure>

Use warmup and CUDA synchronization when comparing GPU timings. Hold the input, requested output length, model, and precision fixed, and record each step's prefix length. Report prefill separately from later decode steps; the first output token is selected by prefill.

### Compare the cache switch fairly

This complete example uses a small Hugging Face model on CUDA when available, otherwise CPU. It requires `torch` and `transformers`, a model download or local snapshot, and enough RAM for model weights and state. It measures whole generation calls, not TTFT or per-step decode latency.

```python
import statistics
import time
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = 'Qwen/Qwen2.5-0.5B-Instruct'  # or a local snapshot
print('torch:', torch.__version__, 'transformers:', transformers.__version__)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to(device).eval()
text = tokenizer.apply_chat_template(
    [{'role': 'user', 'content': 'Explain why a battery needs cooling in one sentence.'}],
    tokenize=False, add_generation_prompt=True,
)
inputs = tokenizer(text, return_tensors='pt').to(device)

def synchronize():
    if device == 'cuda':
        torch.cuda.synchronize()

def run(use_cache):
    synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        ids = model.generate(**inputs, max_new_tokens=32, do_sample=False,
                             use_cache=use_cache, pad_token_id=tokenizer.eos_token_id)
    synchronize()
    elapsed = time.perf_counter() - start
    new_ids = ids[0, inputs.input_ids.shape[1]:].cpu()
    return new_ids, elapsed

# Warm both execution paths before collecting timed samples.
run(False)
run(True)
results = {False: [], True: []}
outputs = {}
for _ in range(3):
    for cached in (False, True):
        tokens, seconds = run(cached)
        results[cached].append(seconds)
        outputs[cached] = tokens
for cached, samples in results.items():
    median = statistics.median(samples)
    print('cache:', cached, 'tokens:', len(outputs[cached]),
          'median seconds:', round(median, 4))
print('identical token IDs:', torch.equal(outputs[False], outputs[True]))
```

Early EOS may produce fewer than 32 tokens; report actual output lengths. Floating-point differences can change a near-tie in greedy decoding, so inspect output differences rather than assuming a timing comparison proves equivalence. Do not claim a fixed speedup from these few runs.

### Separate generation from serving

The timer starts after tokenization and ends after GPU completion. It excludes model loading, HTTP, queueing, and detokenization. Those boundaries make this a model-execution baseline. A production request timer must include the omitted service work.

To investigate context scaling, repeat with controlled longer prompts and record the actual tokenizer length. To investigate batching, use multiple prompts with left padding and report total emitted tokens, per-request latency, and padding costs. A single model call does not implement continuous admission of new HTTP requests.

### Move from baseline to engine

Use [prefill and decode](./transformer-forward-pass) to interpret state growth, [kernel timing](./flashattention) to isolate device work, and [vLLM](./continuous-batching) to study scheduling and paged allocation. Keep the model, revision, precision, prompt distribution, and stopping rules fixed when comparing them.

API details: [Hugging Face generation documentation](https://huggingface.co/docs/transformers/main_classes/text_generation).

## 3.4 Attention variants: MHA to MLA

### Multi-head attention

In *multi-head attention* (MHA) each query head has its own key head and value head. With 8 query heads:

```text
Queries:  Q0 Q1 Q2 Q3 Q4 Q5 Q6 Q7
Keys:     K0 K1 K2 K3 K4 K5 K6 K7
Values:   V0 V1 V2 V3 V4 V5 V6 V7
```

Heads can learn different comparisons. More heads do not by themselves raise quality. They do raise the cache: stored K and V scale with the head count. Llama-2-7B is this pattern (32 query heads and 32 KV heads).

### Multi-query attention

*Multi-query attention* (MQA) keeps many query heads and shares one key head and one value head:

```text
Queries:  Q0 Q1 Q2 Q3 Q4 Q5 Q6 Q7
Keys:                 K0
Values:               V0
```

The cache shrinks by about the query-head count, because $n_{\text{kv}} = 1$. Every query head attends to that single K/V pair. The sharing is a capacity limit: one key space has to serve every query head. Models that use it are choosing a smaller cache, and they have to make up for that sharing in training. It is not a free speedup you can patch onto an MHA checkpoint without further work.

### Grouped-query attention

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

### Multi-head latent attention

*Multi-head latent attention* (MLA), used by DeepSeek-V2 and later DeepSeek models, does not pick a smaller head count and stop. It compresses the key/value content of a token into a shorter latent vector of width $c$, with $c$ well below the usual $n_{\text{kv}} \times d_h$, and stores that latent in the cache. Queries are still multi-head. On the way into the score, the latent is mapped back up so the heads can attend. Position handling is part of that design (some dimensions are kept uncompressed so rotary position can be applied), which is why MLA is not "GQA with one more reduction."

The motive matches the earlier chapters: the cache grows with every token and every layer, and decode reads it on every step. A shorter stored vector is a shorter read. Published DeepSeek configurations cut that cached state several times versus MHA, on the order of 5–13× depending on the model, with the quality coming from training the compression jointly. Do not quote a single ratio as if it were a property of the name MLA.

A picture that is sometimes repeated, MHA : GQA : MLA ≈ 8 : 2 : 1, is one imagined stack (8 KV heads, then 2, then a latent near the size of 1 head). It is a mnemonic for that stack, not a measurement. Use the byte formula with the head counts or latent width you actually have.

### Compare the formats

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


These head counts determine the retained cache size. Use them when checking the PyTorch baseline and the recorded V100 cache measurements below.

**Papers worth the original wording:** Vaswani et al., *Attention Is All You Need* (the MHA formula). Shazeer, *Fast Transformer Decoding* (MQA). Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*. DeepSeek-AI, *DeepSeek-V2* (MLA). Su et al., *RoFormer* (the rotation introduced in [text to tokens](./transformer-forward-pass)).

## 3.5 Positional encodings at inference time

### Positional encodings at inference time

The embedding table has no notion of order. Swap two token IDs and, without a position signal, attention would see the same bag of vectors. Decoder models add order in one of two places.

*Absolute position embeddings* (GPT-2) are a second table, `nn.Embedding(max_length, D)`, added to the token embeddings once, before layer 1. The table has a fixed maximum length. Position `4096` does not exist if the table was built for `4096` rows (`0 … 4095`).

*Rotary position embeddings* (RoPE) leave the token embeddings alone. Inside each attention layer they rotate the query and key vectors by an angle that depends on the position index. The rotation is applied in pairs of dimensions:


$$
\begin{aligned}
q^{(2i)}_m &= q_{2i}\cos(m\theta_i) - q_{2i+1}\sin(m\theta_i) \\
q^{(2i+1)}_m &= q_{2i}\sin(m\theta_i) + q_{2i+1}\cos(m\theta_i)
\end{aligned}
$$


with $\theta_i = 10000^{-2i/d}$. The same rotation is applied to keys. For fixed, unrotated query and key vectors, the dot product $\langle q_m, k_n \rangle$ depends on the *difference* $m - n$, not on the absolute indices. Values are not rotated.

```python
import torch

def apply_rope(x, positions, base=10000.0):
    """Rotate pairs of features. x is [T, D], positions is [T]."""
    t, d = x.shape
    half = d // 2
    pair = torch.arange(half, device=x.device)
    theta = positions.float()[:, None] / (base ** ((2 * pair) / d))
    cos, sin = theta.cos(), theta.sin()
    x1, x2 = x[:, 0::2], x[:, 1::2]
    out = torch.empty_like(x)
    out[:, 0::2] = x1 * cos - x2 * sin
    out[:, 1::2] = x1 * sin + x2 * cos
    return out

torch.manual_seed(0)
q = torch.randn(1, 8)
k = torch.randn(1, 8)
# Same relative offset (3), different absolute positions.
dot_5_8 = (apply_rope(q, torch.tensor([5])) * apply_rope(k, torch.tensor([8]))).sum()
dot_0_3 = (apply_rope(q, torch.tensor([0])) * apply_rope(k, torch.tensor([3]))).sum()
print(float(dot_5_8), float(dot_0_3))  # equal, up to rounding
```

That relative-offset property is why a cached key can stay in the KV cache with the position it was given at prefill. On the next decode step you rotate only the new query and the new key. You do not re-embed the whole prompt.

Early dimension pairs rotate faster than later pairs. RoPE makes position differences available to attention, but it does not guarantee useful extrapolation beyond the training context. Longer distances change the distribution of relative-position patterns; repeated rotations and training exposure both matter. NTK-aware scaling and YaRN modify the frequency schedule to support longer contexts. Use the checkpoint's supported configuration and validate retrieval quality at the intended length rather than changing a maximum-length flag alone.

At inference time the practical rule is: the position id of a new token is the length of the prefix, not zero, and it must match the position used when that token's key was written into the cache.

### Try it

```bash
pip install tiktoken torch
```

```python
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

text = "I am doing excellent"
ids = tokenizer.encode(text)

print("Token IDs:", ids)
print("Pieces:", [tokenizer.decode([i]) for i in ids])
print("Recovered text:", tokenizer.decode(ids))
print("Vocab size:", tokenizer.n_vocab)
```

You should see `[40, 716, 1804, 6275]` and four pieces. Change `excellent` to `excaellent` and print the pieces again.

*Experiment:* encode `"powertrain"`, `"power train"`, and `"PowerTrain"` with the same tokenizer. Record the token count for each. Case and spaces change the split, and therefore the prefill length.

*Optional comparison:* load `Qwen/Qwen2.5-0.5B-Instruct` with Hugging Face `AutoTokenizer` and encode the same three strings. Two models can assign different IDs and different boundaries to the same sentence. The generation examples use Qwen; those IDs will not match the GPT-2 table above.


#### Quick shape practice

If `B=2`, `T=12`, and `D=768`, the token-ID tensor is `[2, 12]` and the token-embedding output is `[2, 12, 768]`. Batch size does not change embedding width. Sequence length does not change `V`.



**Notebooks:** [`notebooks/LLM_Scratch/transformers_101.ipynb`](https://github.com/rangarajp/rangarajp.github.io/blob/main/notebooks/LLM_Scratch/transformers_101.ipynb) · [`transformers_202.ipynb`](https://github.com/rangarajp/rangarajp.github.io/blob/main/notebooks/LLM_Scratch/transformers_202.ipynb) · companion walkthrough: [GPT from scratch](/build-from-scratch/gpt)

## Recorded notebook observations

These observations come from the saved notebook runs linked below. Their model, timing boundary, and hardware assumptions are part of the result.

### V100 lab: extending the cache comparison to a 7B model

The [KV cache notebook](https://github.com/rangarajp/rangarajp.github.io/blob/main/notebooks/llm-inference/observability/experiments/3.1_kv_cache.ipynb) records Qwen2.5-7B on a Tesla V100-SXM2-32GB. It times a complete prefix forward without caching separately from a one-token forward with a prepared cache. Each context uses two warmup iterations and five timed repetitions with CUDA synchronization.

| Retained context tokens | Full-prefix forward (ms) | One-token cached forward (ms) | Per-step ratio |
|---:|---:|---:|---:|
| 128 | 58.6 | 30.8 | 1.90× |
| 256 | 67.5 | 30.9 | 2.18× |
| 512 | 108.1 | 30.3 | 3.56× |
| 1024 | 210.9 | 32.0 | 6.59× |
| 1536 | 313.0 | 33.9 | 9.22× |

The table shows the work avoided by reusing a prefix. It compares different amounts of work and excludes cache-building time from the cached column, so 9.22× is not a whole-request generation speedup. The notebook also reconstructs an approximate 128-token generation cost by interpolating this table; that reconstruction is an estimate, not an independently timed generation run.

These 7B results are separate from the earlier 0.5B short-prompt traces. Both model size and context length changed. The cached step is comparatively flat over this range, but still reads a growing cache.

### V100 lab: verify the checkpoint's head counts

The [Qwen2.5-7B cache inspection](https://github.com/rangarajp/rangarajp.github.io/blob/main/notebooks/llm-inference/observability/experiments/3.1_kv_cache.ipynb) prints `heads=28`, `kv_heads=4`, and `head_dim=128`. That is seven query heads per KV head. Its observed key shape `[1, 4, 10, 128]` stores four KV heads, not 28. Substituting query heads into the cache formula would overestimate this configuration's logical KV storage by seven times.

This is an observed configuration and tensor-shape check. It is not an experiment comparing independently trained MHA and GQA models at equal quality.

### V100 lab: establish what each comparison timed

The [basic notebook](https://github.com/rangarajp/rangarajp.github.io/blob/main/notebooks/llm-inference/basic_vLLM_v1.ipynb) already exposes uncached generation and a short cached trace. The [7B cache notebook](https://github.com/rangarajp/rangarajp.github.io/blob/main/notebooks/llm-inference/observability/experiments/3.1_kv_cache.ipynb) then isolates full-prefix forwards and cached single-token forwards across context lengths. These are useful separate baselines: one shows a generation loop, the other isolates a mechanism.

The complete script above measures whole generation calls and can extend that work. Its results have not been recorded in this revision. In particular, the 7B notebook's interpolated generation-cost estimate is not a replacement for running this end-to-end comparison.

## Hands-on extensions

Use these experiments to connect the mechanisms above to measurements. They are proposed exercises unless a recorded result is explicitly identified.

### Hands-on: inspect tokenization and position

Extend the existing tokenizer script with normal text, misspellings, code, and a repeated system prompt. Publish token counts and embedding shapes; use the RoPE script to verify equal dot products at equal relative offsets.

### Hands-on: check attention shapes and masking

Run the toy attention block with three sequence lengths. Publish the Q/K/V and score shapes, a causal-weight heatmap, and the numerical check that future-token weights are zero. CPU is sufficient.

### Hands-on: plot a controlled sampler

Use the same four logits with several temperatures, top-k values, and top-p thresholds. Draw repeated samples with a fixed seed and publish probability bars plus empirical frequencies; this isolates sampling from model quality.

### Hands-on: expose the growing prefix

Use the existing uncached loop to print input length and selected token at every step. Publish a short trace with actual token IDs, and compare it with the later cached baseline under the same stopping rule.

### Hands-on: publish the context-length sweep

Turn the existing 7B cache notebook results into two plots: full-prefix forward time versus context length and one-token cached-decode time versus context length. Show prefill separately and include repeat counts; avoid calling their per-step ratio an end-to-end speedup.

### Hands-on: measure KV storage rather than expanded tensors

Instantiate toy MHA, GQA, and MQA caches with matched batch, context, layers, and head width. Publish tensor-storage bytes and the predicted ratio. If benchmarking repeated heads, identify any materialized copies so the implementation does not erase the intended saving.

### Hands-on: save a reproducible baseline report

Run cached and uncached paths with the same prompt set and actual output lengths. Save model revision, device, library versions, dtype, warmup, medians, peak memory, and output equality. Keep CPU and GPU timing results in separate reports.
