---
title: "From Text to Tokens"
description: "How a prompt becomes token IDs, embedding rows, and position-aware vectors before the first transformer block."
section: learn
pubDate: 2026-10-29
part: 1
chapter: 1
order: 1
tags: ["llm-inference", "fundamentals", "pytorch"]
---

# From text to tokens

**Difficulty:** Beginner  
**Big question:** How does a model see a sentence?  
**You will build:** A GPT-2 BPE inspection script, an embedding lookup, and a small rotary-position check.

The worked token IDs below come from the GPT-2 tokenizer in [`0_transformers_101.ipynb`](/build-from-scratch/gpt) and [`1_transformers_202.ipynb`](/build-from-scratch/gpt). A survey of tokenizer families is in [Tokenization](/concepts/transformers/tokenization).

## 1. The model does not read words

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

## 2. Byte-pair encoding handles unknown spellings

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

## 3. From IDs to vectors

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

## 4. Positional encodings at inference time

The embedding table has no notion of order. Swap two token IDs and, without a position signal, attention would see the same bag of vectors. Decoder models add order in one of two places.

*Absolute position embeddings* (GPT-2) are a second table, `nn.Embedding(max_length, D)`, added to the token embeddings once, before layer 1. The table has a fixed maximum length. Position `4096` does not exist if the table was built for `4096` rows (`0 … 4095`).

*Rotary position embeddings* (RoPE) leave the token embeddings alone. Inside each attention layer they rotate the query and key vectors by an angle that depends on the position index. The rotation is applied in pairs of dimensions:

\[
\begin{aligned}
q^{(2i)}_m &= q_{2i}\cos(m\theta_i) - q_{2i+1}\sin(m\theta_i) \\
q^{(2i+1)}_m &= q_{2i}\sin(m\theta_i) + q_{2i+1}\cos(m\theta_i)
\end{aligned}
\]

with \(\theta_i = 10000^{-2i/d}\). The same rotation is applied to keys. For fixed, unrotated query and key vectors, the dot product \(\langle q_m, k_n \rangle\) depends on the *difference* \(m - n\), not on the absolute indices. Values are not rotated.

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

## 5. Try it

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

*Optional comparison:* load `Qwen/Qwen2.5-0.5B-Instruct` with Hugging Face `AutoTokenizer` and encode the same three strings. Two models can assign different IDs and different boundaries to the same sentence. Later chapters use Qwen for generation; those IDs will not match the GPT-2 table above.


### Quick shape practice

If `B=2`, `T=12`, and `D=768`, the token-ID tensor is `[2, 12]` and the token-embedding output is `[2, 12, 768]`. Batch size does not change embedding width. Sequence length does not change `V`.


---

**Continue learning:** [Next chapter →](./transformer-forward-pass)

**Notebooks:** [`notebooks/LLM_Scratch/0_transformers_101.ipynb`](https://github.com/rangarajp/rangarajp.github.io/blob/main/notebooks/LLM_Scratch/0_transformers_101.ipynb) · [`1_transformers_202.ipynb`](https://github.com/rangarajp/rangarajp.github.io/blob/main/notebooks/LLM_Scratch/1_transformers_202.ipynb) · companion walkthrough: [GPT from scratch](/build-from-scratch/gpt)
