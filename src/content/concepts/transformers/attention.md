---
title: 'Attention'
description: 'How tokens weigh the relevance of other tokens in context.'
pubDate: 'Aug 3 2026'
order: 4
heroImage: '../../../assets/blog-placeholder-3.jpg'
---

Attention computes contextual relevance scores between tokens. Given a sequence of embeddings (with positional information already mixed in), each token looks at every other token and decides how much to weigh each one. The result is a new representation that depends on context — the same token can mean different things depending on what surrounds it.

## Background

Long sequences are hard to handle well. Take English-to-German translation: you cannot translate word by word. Some words depend on words that appeared earlier or later. Encoder-decoder architectures with RNN-style encoding and decoding were an early answer to that problem.

![Why Attention](./images/attention-background-translation.png)

A big limitation of RNN encoder-decoder models is that they cannot directly access earlier hidden states. We assume the current hidden state captures everything that matters, which leads to loss of context on long sequences. RNN-style models worked for short translations, but longer texts still could not reach back to earlier words directly.

**Bahdanau attention** was an early fix: it let the decoder selectively access different parts of the input sequence at each decoding step.

![Bahdanau Attention](./images/attention-bahdanau.png)

**Attention** (as in the picture above) lets the decoder decide which parts of the encoder input to focus on while producing the output. **Self-attention** goes further — every token within a sequence can interact directly with every other token. Weights are computed by relating different positions inside a single sequence.

## Self-attention

### Why embeddings alone are not enough

Consider these two sentences:

- **Sentence A:** "I am sitting by the river **bank**"
- **Sentence B:** "I am going to the **bank** to deposit money"

Both contain the word "bank", but the meaning differs:

- Sentence A: **bank** = riverbank (geographic, nature-related)
- Sentence B: **bank** = financial institution (business, money-related)

A traditional word embedding treats "bank" the same way in both contexts:

```
embedding("bank") = [0.2, -0.5, 0.8, 0.1, -0.3]  ← always the same
```

That single vector does not know what other words are in the sentence, what the sentence is about, or how important those words are for understanding this "bank".

Self-attention fixes this by looking at all words, scoring how relevant each is to "bank", and building a **context vector** tailored to this sentence. Same word, different sentences → different context vectors.

### Attention scores

Walk through Sentence A with five embedding dimensions that capture meaning:

| Dimension | Meaning |
| --------- | ------- |
| Geographic | How much does the word relate to places or locations? |
| Financial | How much does the word relate to money or transactions? |
| Nature | How much does the word relate to natural elements? |
| Action | How much does the word relate to verbs or movement? |
| Person | How much does the word relate to pronouns or people? |

| Word | Geographic | Financial | Nature | Action | Person |
| ---- | ---------- | --------- | ------ | ------ | ------ |
| I | 0.1 | 0.0 | 0.0 | 0.0 | **0.95** |
| am | 0.0 | 0.0 | 0.0 | 0.5 | 0.1 |
| sitting | 0.1 | 0.0 | 0.1 | **0.9** | 0.0 |
| by | 0.2 | 0.0 | 0.0 | 0.2 | 0.0 |
| the | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| river | **0.8** | 0.0 | **0.9** | 0.0 | 0.0 |
| **bank** | **0.7** | 0.1 | 0.5 | 0.0 | 0.0 |

**Key observation:** in this sentence, "bank" has high Geographic (0.7) and Nature (0.5) values — not Financial.

For each word we have three roles:

- **Query (Q):** "What am I looking for?"
- **Key (K):** "What do I offer?"
- **Value (V):** "What information can I provide?"

For simplicity, assume `Q = K = V = embedding`. In real models these are learned linear transformations.

Focus on **"bank"** and compute how much attention it should pay to each other word. The score is the dot product `Q_bank · K_word`:

| Word | Key vector | Dot product with Q_bank | Score |
| ---- | ---------- | ----------------------- | ----- |
| I | [0.1, 0.0, 0.0, 0.0, 0.95] | `(0.7×0.1) + (0.1×0.0) + (0.5×0.0) + (0.0×0.0) + (0.0×0.95)` | **0.07** |
| am | [0.0, 0.0, 0.0, 0.5, 0.1] | `(0.7×0.0) + (0.1×0.0) + (0.5×0.0) + (0.0×0.5) + (0.0×0.1)` | **0.00** |
| sitting | [0.1, 0.0, 0.1, 0.9, 0.0] | `(0.7×0.1) + (0.1×0.0) + (0.5×0.1) + (0.0×0.9) + (0.0×0.0)` | **0.12** |
| by | [0.2, 0.0, 0.0, 0.2, 0.0] | `(0.7×0.2) + (0.1×0.0) + (0.5×0.0) + (0.0×0.2) + (0.0×0.0)` | **0.14** |
| the | [0.0, 0.0, 0.0, 0.0, 0.0] | `(0.7×0.0) + (0.1×0.0) + (0.5×0.0) + (0.0×0.0) + (0.0×0.0)` | **0.00** |
| river | [0.8, 0.0, 0.9, 0.0, 0.0] | `(0.7×0.8) + (0.1×0.0) + (0.5×0.9) + (0.0×0.0) + (0.0×0.0)` | **1.01** |
| bank | [0.7, 0.1, 0.5, 0.0, 0.0] | `(0.7×0.7) + (0.1×0.1) + (0.5×0.5) + (0.0×0.0) + (0.0×0.0)` | **0.75** |

**Raw attention scores:** `[0.07, 0.00, 0.12, 0.14, 0.00, 1.01, 0.75]`

"river" gets the highest score (1.01) — it is the most semantically similar to "bank" in this geographic/nature context.

Softmax converts scores into probabilities that sum to 1:

```
attention_weight = e^score / Σ e^score_i
```

| Word | Score | e^Score |
| ---- | ----- | ------- |
| I | 0.07 | 1.07 |
| am | 0.00 | 1.00 |
| sitting | 0.12 | 1.13 |
| by | 0.14 | 1.15 |
| the | 0.00 | 1.00 |
| river | 1.01 | 2.75 |
| bank | 0.75 | 2.12 |

**Sum:** `1.07 + 1.00 + 1.13 + 1.15 + 1.00 + 2.75 + 2.12 = 10.22`

| Word | Attention weight |
| ---- | ---------------- |
| I | 1.07 / 10.22 = **0.105** |
| am | 1.00 / 10.22 = **0.098** |
| sitting | 1.13 / 10.22 = **0.111** |
| by | 1.15 / 10.22 = **0.113** |
| the | 1.00 / 10.22 = **0.098** |
| river | 2.75 / 10.22 = **0.269** |
| bank | 2.12 / 10.22 = **0.207** |

Interpretation:

- **26.9%** of "bank"'s attention goes to "river"
- **20.7%** goes to itself
- The remaining ~52% is distributed among the other words

### Context vector

The context vector is each value vector weighted by its attention weight, then summed:

```
context_vector = Σ attention_weight_i × V_i
```

| Word | Attention wt | Value vector | Weighted vector |
| ---- | ------------ | ------------ | --------------- |
| I | 0.105 | [0.1, 0.0, 0.0, 0.0, 0.95] | [0.010, 0.0, 0.0, 0.0, 0.100] |
| am | 0.098 | [0.0, 0.0, 0.0, 0.5, 0.1] | [0.0, 0.0, 0.0, 0.049, 0.010] |
| sitting | 0.111 | [0.1, 0.0, 0.1, 0.9, 0.0] | [0.011, 0.0, 0.011, 0.100, 0.0] |
| by | 0.113 | [0.2, 0.0, 0.0, 0.2, 0.0] | [0.023, 0.0, 0.0, 0.023, 0.0] |
| the | 0.098 | [0.0, 0.0, 0.0, 0.0, 0.0] | [0.0, 0.0, 0.0, 0.0, 0.0] |
| river | 0.269 | [0.8, 0.0, 0.9, 0.0, 0.0] | [0.215, 0.0, 0.242, 0.0, 0.0] |
| bank | 0.207 | [0.7, 0.1, 0.5, 0.0, 0.0] | [0.145, 0.021, 0.104, 0.0, 0.0] |

```
Context Vector = [0.010 + 0.0 + 0.011 + 0.023 + 0.0 + 0.215 + 0.145,
                  0.0 + 0.0 + 0.0 + 0.0 + 0.0 + 0.0 + 0.021,
                  0.0 + 0.0 + 0.011 + 0.0 + 0.0 + 0.242 + 0.104,
                  0.0 + 0.049 + 0.100 + 0.023 + 0.0 + 0.0 + 0.0,
                  0.100 + 0.010 + 0.0 + 0.0 + 0.0 + 0.0 + 0.0]

Context Vector = [0.404, 0.021, 0.357, 0.172, 0.110]
                  Geographic, Financial, Nature, Action, Person
```

Compare with the original embedding for "bank" `[0.7, 0.1, 0.5, 0.0, 0.0]`:

| Dimension | Original | Context | Change | Interpretation |
| --------- | -------- | ------- | ------ | -------------- |
| Geographic | 0.7 | 0.404 | Diluted | Other words are less geographic |
| Financial | 0.1 | 0.021 | Slightly reduced | Minimal financial context |
| Nature | 0.5 | 0.357 | Diluted | Mixed with surrounding words |
| Action | 0.0 | 0.172 | Boosted | "sitting" and "by" contributed action |
| Person | 0.0 | 0.110 | Boosted | "I" contributed person perspective |

The context vector absorbed influence from "river" (geographic, nature) and "sitting" (action), while the direct "bank" signal stayed strong. That is a contextualized representation.

Apply the same process to **"I am going to the bank to deposit money"**:

| Word | Geographic | Financial | Nature | Action | Person |
| ---- | ---------- | --------- | ------ | ------ | ------ |
| bank | **0.2** | **0.8** | 0.1 | 0.1 | 0.0 |

- Attention would weight "money" (Financial: 0.9), "deposit" (Action: 0.95), and nearby function words more heavily
- The final context vector for "bank" would have **high Financial**, low Nature
- Completely different contextualization from Sentence A

### Putting it together

```
1. INPUT: word embedding (static, context-agnostic)
   "bank" = [0.7, 0.1, 0.5, 0.0, 0.0]
                │
                ▼
2. QUERY-KEY SIMILARITY: compute attention scores
   Scores: [0.07, 0.00, 0.12, 0.14, 0.00, 1.01, 0.75]
                │
                ▼
3. SOFTMAX: convert scores to attention weights
   Weights: [0.105, 0.098, 0.111, 0.113, 0.098, 0.269, 0.207]
                │
                ▼
4. WEIGHTED SUM: combine value vectors by weight
   context = Σ(weight_i × value_i)
   [0.404, 0.021, 0.357, 0.172, 0.110]
```

In short:

1. Static embeddings ignore context
2. Attention scores measure relevance between words
3. Softmax turns scores into weights (0–1, sum to 1)
4. Context vectors are weighted combinations of all words, tailored to the query
5. Same word, different context → different representation — the foundation of transformers

For a query word `q`:

```
score_i  = q · k_i
weight_i = e^score_i / Σ_j e^score_j
context  = Σ_i weight_i × v_i
```

where `k_i` and `v_i` are the key and value vectors of word `i`.
