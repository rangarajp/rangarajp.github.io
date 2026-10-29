---
title: "From Hidden States to the Next Token"
description: "The LM head, softmax, temperature, top-k, and top-p: how a hidden vector becomes one token id."
section: learn
pubDate: 2026-10-29
part: 1
chapter: 3
order: 3
tags: ["llm-inference", "fundamentals", "pytorch"]
---

**Difficulty:** Beginner  
**Big question:** How does a vector become a token?  
**You will build:** A logits-to-token experiment with temperature, top-k, and top-p.

## 1. The language model head

After the last block and a final norm, every position still holds a vector of width \(D\):

\[
H \in \mathbb{R}^{B \times T \times D}
\]

A linear map, the *LM head*, scores the whole vocabulary at every position:

\[
Z = H W_{\mathrm{vocab}}, \qquad W_{\mathrm{vocab}} \in \mathbb{R}^{D \times V}
\]

\(V\) is the tokenizer vocabulary size, so \(Z\) has shape `[B, T, V]`. For GPT-2, \(V = 50257\). For a small Qwen it is on the order of 150,000. The head does not emit a word. It emits one score per vocabulary row.

Ordinary next-token generation reads only the last position, `Z[:, -1, :]`, a vector of length `V`. Earlier positions have logits too. During training those positions each predict the following token. During generation you already know the prompt, so you sample from the last row and append.

A *logit* is an unnormalized score. It can be negative. It does not sum to one. Softmax turns the row into a distribution:

\[
p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
\]

## 2. A four-token numerical example

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
- *Sampling* draws from \(p\). `cooling` is the most likely draw. It is not a guaranteed draw.

Greedy is the usual meaning of temperature 0. Do not implement it as `logits / 0`.

## 3. Temperature, top-k, and top-p

Temperature \(\tau\) scales the logits *before* softmax:

\[
p_i = \mathrm{softmax}(z_i / \tau)
\]

A smaller positive \(\tau\) sharpens the distribution. A larger \(\tau\) flattens it. The ranking of the tokens does not change. Only the gaps do. These controls do not modify the trained weights. They modify the draw.

Top-k keeps the \(k\) largest logits and sets the rest to \(-\infty\) before softmax. Top-p (*nucleus* sampling) keeps the smallest set of tokens whose probabilities sum to at least \(p\), walking from the largest probability downward. Both shrink the candidate set. They answer different questions: top-k caps the count, top-p caps the probability mass, so the count can change from step to step.

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

At \(\tau = 0.5\) almost all of the mass sits on `cooling`. At \(\tau = 2\) the four probabilities move closer together, so `normal` and `.` show up more often if you draw many times. With `top_k=2`, `normal` and `.` have probability 0. Their logits were discarded before softmax, and the remaining two probabilities were renormalized.

Run the temperature loop a few dozen times if you want to see the draw change. A single `multinomial` call is one sample, not the distribution itself.

## 4. The whole vocabulary is scored

Even when the next word feels obvious, the LM head writes a score for every row: whole words, fragments, spaces, digits, and special tokens. If the tokenizer splits `powertrain` into two pieces, the model cannot emit the whole string in one step. It samples an id, appends it, and samples again. A different tokenizer might do that in one id. The head only chooses among the rows it was trained with.

Softmax depends on differences, not on the absolute level of the logits. Subtracting the same constant from every entry leaves \(p\) unchanged. Implementations subtract the max logit first so the exponentials do not overflow:

```python
def stable_softmax(z):
    shifted = z - z.max()
    ex = shifted.exp()
    return ex / ex.sum()
```

For `logits = [2.0, 1.0]`, `stable_softmax(logits)` and `stable_softmax(logits - 100)` match. The next chapter's `argmax` path never builds this distribution. Sampling does.



---

**Continue learning:** [← Previous chapter](./transformer-forward-pass) · [Next chapter →](./autoregressive-generation)
