---
title: 'Query, Key, and Value'
description: 'How Transformers turn the same starting word embedding into a context-aware meaning using learned Query, Key, and Value projections.'
pubDate: 'Sep 10 2026'
order: 5
heroImage: '../../../assets/blog-placeholder-3.jpg'
---

In the previous article, we simplified attention by assuming:

$$Q = K = V = X$$

That was useful for understanding the basic idea: a word looks at the other words, decides which ones matter, and uses them to build a context-aware representation.

Now we will remove that simplification and see why real Transformers learn three different representations: Query, Key, and Value.

We will use one word throughout this article: **bank**

Consider these two sentences:

- I am sitting by the river **bank**.
- I went to the **bank** to deposit money.

The word `bank` begins with the same learned token embedding in both sentences. But after attention, we want its representation to become different:

```text
bank  +  river context          →  bank beside a river
bank  +  deposit/money context  →  financial institution
```

This is the main goal: understand how Query, Key, and Value make that change possible.

## 1. Start with the embedding of `bank`

Every token is represented by a vector of numbers. Let the starting representation of `bank` be $x_{\text{bank}}$ — a point in a large embedding space containing many learned properties of the word.

For intuition only, imagine a tiny embedding:

```text
x_bank = [0.5, 0.5, 0.3, ...]
```

The real vector may contain hundreds or thousands of numbers.

This starting vector represents the general word `bank`. It has not yet been adapted to this particular sentence. So we need the surrounding words to help.

```text
river  ───────────────► bank
                          ↓
                    river-side meaning

deposit ──┐
          ├───────────► bank
money ────┘               ↓
                    financial meaning
```

## 2. Why not simply compare the original embeddings?

In our simplified attention article, we compared `bank` directly with `river`:

$$x_{\text{bank}} \cdot x_{\text{river}}$$

This asks roughly: how well do these two vectors align in the original embedding space?

But attention needs something more flexible. `bank` and `river` are different concepts. What matters here is not whether the words are globally similar. What matters is: is `river` useful for understanding what `bank` means in this sentence?

Likewise, `deposit` and `money` are useful clues for the financial meaning of `bank`. So instead of forcing the original embedding space to do everything, the Transformer learns new spaces specifically for attention — that is what the Query and Key projections provide.

## 3. One embedding becomes Query, Key, and Value

For every token representation $x_i$, a Transformer attention head creates three new vectors:

$$q_i = x_i W_Q \qquad k_i = x_i W_K \qquad v_i = x_i W_V$$

where $W_Q$, $W_K$, and $W_V$ are learned matrices.

A useful mental model:

| Vector | Simple question | For our `bank` example |
|---|---|---|
| Query $q_i$ | What am I looking for? | `bank`: Which words help clarify my meaning? |
| Key $k_i$ | What can I be matched on? | `river`: I may be a useful geographical clue. |
| Value $v_i$ | What information can I contribute? | `river`: Information that shifts the representation toward the river-side meaning. |

These descriptions are only intuition. The model stores numbers, and training learns useful patterns in those numbers.

## 4. What do the matrices $W_Q$, $W_K$, $W_V$ do?

Suppose a token embedding has $d_{\text{model}}$ numbers. For example:

$$x_{\text{bank}} \in \mathbb{R}^{1 \times 768}$$

For one attention head with 64-dimensional Query and Key:

$$W_Q \in \mathbb{R}^{768 \times 64} \qquad \Rightarrow \qquad q_{\text{bank}} = x_{\text{bank}}W_Q \in \mathbb{R}^{1 \times 64}$$

Similarly for Key and Value:

$$W_K \in \mathbb{R}^{768 \times 64} \qquad W_V \in \mathbb{R}^{768 \times 64}$$

So you can think of the three matrices as three different learned lenses:

```text
                         W_Q
                      ┌───────► Query
                      │
Token representation ─┼─ W_K ─► Key
                      │
                      └─ W_V ─► Value
```

The same starting token is viewed differently depending on the job we need it to perform.

## 5. Query and Key decide what is relevant

In the sentence "I am sitting by the river bank", `bank` produces a Query and every token produces a Key. The Query of `bank` is compared with each Key.

Suppose the scores from one attention head look like this:

| Token | Query-Key score |
|---|---:|
| `I` | 0.1 |
| `am` | 0.1 |
| `sitting` | 0.3 |
| `by` | 0.2 |
| `the` | 0.1 |
| **`river`** | **2.8** |
| `bank` | 0.5 |

`river` gets the largest score. Query and Key determine who should pay attention to whom.

```text
Query(bank)
    │
    ├──── Key(I)       → low match
    ├──── Key(sitting) → low match
    ├──── Key(the)     → low match
    └──── Key(river)   → HIGH MATCH
```

## 6. Why this is better than comparing embeddings directly

Previously, relevance depended on the original embedding geometry:

$$x_{\text{bank}} \cdot x_{\text{river}}$$

Now we have:

$$(x_{\text{bank}}W_Q) \cdot (x_{\text{river}}W_K)$$

Since $W_Q$ and $W_K$ are trainable, the model can learn a space where useful relationships receive high scores:

```text
Before: "Are bank and river similar in the original embedding space?"
After:  "Is river useful for what bank is looking for right now?"
```

That is the main reason Query and Key exist.

## 7. Turn the scores into attention weights

First, attention scales the score:

$$s_{ij} = \frac{q_i \cdot k_j}{\sqrt{d_k}}$$

where $d_k$ is the number of dimensions in the Query and Key vectors. Then softmax converts all the scores for one Query into weights that add up to 1:

$$\alpha_{ij} = \operatorname{softmax}_j(s_{ij})$$

For our `bank` Query, imagine the result is:

| Token | Attention weight |
|---|---:|
| `I` | 0.075 |
| `am` | 0.075 |
| `sitting` | 0.087 |
| `by` | 0.081 |
| `the` | 0.075 |
| **`river`** | **0.507** |
| `bank` | 0.100 |

At this point we know where `bank` should get information from. But we still have not decided what information should actually flow from `river` to `bank` — that is the job of Value.

## 8. Value decides what information flows

Every token also has a Value vector:

$$v_i = x_i W_V \qquad v_{\text{river}} = x_{\text{river}} W_V$$

The Key and Value have different jobs:

```text
Key(river)   → "Should bank select me?"
Value(river) → "If selected, what information should I contribute?"
```

A representation useful for finding `river` does not have to be the same representation useful for updating `bank`.

## 9. Build the context vector for `bank`

Multiply each Value by its attention weight and sum:

$$c_{\text{bank}} = \sum_j \alpha_{\text{bank},j} \, v_j$$

For our example:

$$c_{\text{bank}} = 0.507 \, v_{\text{river}} + 0.100 \, v_{\text{bank}} + 0.087 \, v_{\text{sitting}} + \cdots$$

Because `river` has the largest attention weight, its Value contributes strongly to the context vector.

The key upgrade from simplified attention:

```text
Simplified:   context = Σ attention weight × original embedding
Learned QKV:  context = Σ attention weight × Value vector
```

Or mathematically: $c_i = \sum_j \alpha_{ij}(x_j W_V)$

## 10. How `bank` becomes a river bank

Starting from the general representation $x_{\text{bank}}$, attention discovers that `river` is highly relevant:

```text
bank Query → river Key gets a high match
           → high attention weight
           → river Value contributes strongly
           → context vector for bank
```

Conceptually:

```text
General BANK
    +
context dominated by RIVER
    ↓
BANK understood as "the side of a river"
```

The original token embedding is not a special `river-bank` embedding. Attention uses the sentence to create a context-dependent representation.

## 11. The same `bank` can become a financial bank

In "I went to the bank to deposit money", the `bank` Query may match strongly with the Keys of `deposit` and `money`:

```text
bank Query ──── deposit Key → high match
           └─── money Key   → high match
```

Their Values now dominate the context vector:

$$c_{\text{bank}} = \alpha_{\text{bank,deposit}} \, v_{\text{deposit}} + \alpha_{\text{bank,money}} \, v_{\text{money}} + \cdots$$

So the starting word is the same while its contextual representation becomes different:

```text
                        + river context      → river-side BANK
General BANK ──────────
                        + deposit/money      → financial BANK
```

The embedding gives the token a starting meaning. Attention changes that representation according to the surrounding context.

## 12. One important note about GPT-style models

The example above is easiest to understand using bidirectional attention, where a token can look at words on both sides.

Decoder-only models such as GPT use causal attention — a token can only attend to itself and earlier tokens. So in "I went to the **bank** to deposit money", `bank` cannot look forward to `deposit` or `money`.

For a GPT-style example, write the sentence as: "After depositing the money, I went to the bank." Now when the model processes `bank`, `depositing` and `money` are already in the past context and can influence its representation.

This does not change how Q, K, and V work. It only changes which Keys a Query is allowed to see.

## 13. The whole operation in matrix form

A Transformer performs these calculations for all tokens together. If the sequence contains $n$ tokens:

$$X \in \mathbb{R}^{n \times d_{\text{model}}}$$

For one attention head:

$$Q = XW_Q \qquad K = XW_K \qquad V = XW_V$$

with $Q, K \in \mathbb{R}^{n \times d_k}$ and $V \in \mathbb{R}^{n \times d_v}$.

The attention calculation is:

$$\operatorname{Attention}(Q,K,V) = \operatorname{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Reading it as a sequence of steps:

```text
QKᵀ → which tokens are relevant to each other?
  ↓
scale + softmax → how much attention should each token receive?
  ↓
× V → what information should actually be combined?
  ↓
context vectors
```

## 14. Example with 7 tokens

Our river sentence has 7 tokens: `I | am | sitting | by | the | river | bank`

With $d_{\text{model}} = 768$ and $d_k = d_v = 64$ for one head:

| Matrix | Shape |
|---|---|
| $X$ | $7 \times 768$ |
| $W_Q, W_K, W_V$ | $768 \times 64$ |
| $Q, K, V$ | $7 \times 64$ |
| $QK^T$ | $7 \times 7$ |

The $7 \times 7$ matrix means every one of the 7 Queries is being compared with every one of the 7 Keys:

```text
                 Keys
          I  am sitting by the river bank
Queries I  •   •    •    •  •    •    •
       am  •   •    •    •  •    •    •
  sitting  •   •    •    •  •    •    •
       by  •   •    •    •  •    •    •
      the  •   •    •    •  •    •    •
    river  •   •    •    •  •    •    •
     bank  •   •    •    •  •   ★★★   •
```

The highlighted cell is $q_{\text{bank}} \cdot k_{\text{river}}$.

## 15. Where are $W_Q$, $W_K$, and $W_V$ learned?

The matrices are model parameters — not manually written rules. During training:

1. The model makes predictions.
2. A loss function measures the error.
3. Backpropagation calculates how the parameters contributed to that error.
4. Gradient descent updates $W_Q$, $W_K$, and $W_V$.
5. Across large amounts of text, useful attention patterns emerge.

A head may become useful for relationships such as ambiguous word → contextual clue, pronoun → referent, verb → subject, noun → adjective. These are learned patterns, not hard-coded rules.

## 16. The mental model to keep

```text
                    W_Q
Token ───────────────► Query
representation          │
                        │ asks "Who matters?"
                        ▼
Token ─── W_K ───────► Keys
representations         │
                        ▼
                  attention weights
                        │
                        │ choose how much
                        ▼
Token ─── W_V ───────► Values
representations         │
                        ▼
                   context vector
```

Query asks what I need. Key helps decide whether a token is relevant. Value contains the information that token contributes.

For our example: the general `bank` representation asks its context for useful clues. `river` pushes it toward the river-side meaning; `deposit` and `money` push it toward the financial meaning.

## What comes next?

So far we have described one attention head. But one way of looking at a sentence is not enough — one head might focus on word meaning while another focuses on grammar, references, or position.

Transformers therefore run several attention heads in parallel. That leads us to Multi-Head Attention.
