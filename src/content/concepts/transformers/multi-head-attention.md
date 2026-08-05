---
title: 'Multi-Head Attention'
description: 'How multiple attention heads learn different relationship patterns in parallel.'
pubDate: 'Aug 5 2026'
order: 5
heroImage: '../../../assets/blog-placeholder-3.jpg'
---

Single-head attention produces one weighted view of a token's neighbors. Multi-head attention runs several attention computations in parallel, each with its own weight matrices, then concatenates and projects the results. That lets the model capture different relationship types at once — geography and action, syntax and semantics — instead of forcing one head to compromise.

## Background

From [attention](/concepts/transformers/attention), a single head can contextualize an ambiguous word like "bank" by looking at its neighbors. The same embedding `[0.5, 0.5, 0.3, 0.0, 0.0]` produces different context vectors in different sentences:

- **Sentence A (river context):** `[0.323, 0.100, 0.267, 0.189, 0.122]` — nature-oriented
- **Sentence B (deposit context):** `[0.164, 0.679, 0.088, 0.271, 0.000]` — finance-oriented

The limitation is that one head produces **one** weighted combination. What if "bank" needs to attend simultaneously to:

- Geographic / nature aspects (neighbor: "river")
- Action / activity aspects (neighbor: "sitting")
- Person perspective (neighbor: "I")

A single head must prioritize one pattern. It cannot strongly split attention across fundamentally different relationship types.

**Multi-head attention** fixes this: run N independent attention heads in parallel, each learning different relationship patterns.

## The core idea

Instead of computing attention once:

```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) V
```

Compute it N times with different weight matrices:

```
head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
```

Then concatenate and project:

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_N) W^O
```

Each head learns different weights (`W_i^Q`, `W_i^K`, `W_i^V`), so each weights neighbors differently. Single-head gave one perspective; multi-head gives N in parallel.

## Worked example: 2-head attention on "bank"

Continue with Sentence A: **"I am sitting by the river bank"**

Use the same embeddings as the attention walkthrough, but compute with two different heads.

### Word embeddings

| Word | Geographic | Financial | Nature | Action | Person |
| ---- | ---------- | --------- | ------ | ------ | ------ |
| I | 0.1 | 0.0 | 0.0 | 0.0 | 0.95 |
| am | 0.0 | 0.0 | 0.0 | 0.5 | 0.1 |
| sitting | 0.1 | 0.0 | 0.1 | 0.9 | 0.0 |
| by | 0.2 | 0.0 | 0.0 | 0.2 | 0.0 |
| the | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| river | 0.8 | 0.0 | 0.9 | 0.0 | 0.0 |
| **bank** | **0.5** | **0.5** | **0.3** | 0.0 | 0.0 |

**Recall:** single-head attention on "bank" produced:

- Attention weights: river (0.217), bank (0.200), by (0.123), sitting (0.120), I (0.117), am (0.111), the (0.111)
- Context vector: `[0.323, 0.100, 0.267, 0.189, 0.122]`

### Multi-head decomposition

What if that single context could be split into multiple learned perspectives?

- **Head 1** learns `W_1^Q`, `W_1^K`, `W_1^V` that specialize in **geographic / spatial** relationships
- **Head 2** learns `W_2^Q`, `W_2^K`, `W_2^V` that specialize in **action / verb** relationships

Weights start random. During training, Head 1 tends to learn "pay attention to geographic neighbors" while Head 2 learns "pay attention to action neighbors".

## Head 1: geographic focus

### Apply W₁^Q to "bank"

Head 1 transforms the embedding to search for geographic relevance:

```
bank embedding: [0.5, 0.5, 0.3, 0.0, 0.0]

W_1^Q (learned to emphasize geographic search):
[0.8  0.1  0.1  0.0  0.0]
[0.1  0.7  0.1  0.0  0.0]
[0.2  0.1  0.8  0.0  0.1]
[0.0  0.0  0.0  0.3  0.1]
[0.0  0.0  0.0  0.1  0.2]

Q_1 = [0.5, 0.5, 0.3, 0.0, 0.0] × W_1^Q
    = [0.5×0.8 + 0.5×0.1 + 0.3×0.2, ...]
    = [0.56, 0.40, 0.49, 0.01, 0.02]
```

This query emphasizes Geographic and Nature.

### Attention scores (Head 1)

Query for "bank" (Head 1): `[0.56, 0.40, 0.49, 0.01, 0.02]`

Dot product with key vectors (using `W_1^K`, also learned for geographic focus):

| Word | Key vector | Score | Why |
| ---- | ---------- | ----- | --- |
| I | [0.08, 0.00, 0.00, 0.00, 0.88] | 0.08 | High person, not geographic |
| am | [0.00, 0.00, 0.00, 0.40, 0.08] | 0.00 | Action-focused, not geographic |
| sitting | [0.09, 0.00, 0.09, 0.80, 0.00] | 0.12 | Some geography, but action-heavy |
| by | [0.18, 0.00, 0.00, 0.16, 0.00] | **0.16** | Geographic preposition |
| the | [0.00, 0.00, 0.00, 0.00, 0.00] | 0.00 | Stop word |
| river | [0.72, 0.00, 0.81, 0.00, 0.00] | **0.79** | Highest — geographic + nature |
| bank | [0.45, 0.45, 0.27, 0.00, 0.00] | **0.52** | Self-reference with geographic aspect |

**Raw scores (Head 1):** `[0.08, 0.00, 0.12, 0.16, 0.00, 0.79, 0.52]`

Head 1 gives river a higher score (0.79) than single-head (0.67) because it is specialized for geographic focus.

### Softmax (Head 1)

| Word | Score | e^Score | Attention weight |
| ---- | ----- | ------- | ---------------- |
| I | 0.08 | 1.08 | 0.11 |
| am | 0.00 | 1.00 | 0.10 |
| sitting | 0.12 | 1.13 | 0.11 |
| by | 0.16 | 1.17 | **0.12** |
| the | 0.00 | 1.00 | 0.10 |
| river | 0.79 | 2.20 | **0.22** |
| bank | 0.52 | 1.68 | **0.17** |

**Sum:** `1.08 + 1.00 + 1.13 + 1.17 + 1.00 + 2.20 + 1.68 = 9.26`

**Attention weights (Head 1):**

- river: 22% (geographic-focused)
- bank: 17% (self)
- by: 12% (location preposition)
- others: ~49% distributed

This head emphasizes geographic relationships more than single-head.

### Context vector (Head 1)

Apply `W_1^V` to value embeddings and weight by attention:

```
context_1 = Σ weight_i × (V_i × W_1^V)
```

| Word | Wt | Value | Weighted (Head 1) |
| ---- | -- | ----- | ----------------- |
| I | 0.11 | [0.1, 0.0, 0.0, 0.0, 0.95] | [0.011, 0.000, 0.000, 0.000, 0.105] |
| am | 0.10 | [0.0, 0.0, 0.0, 0.5, 0.1] | [0.000, 0.000, 0.000, 0.050, 0.010] |
| sitting | 0.11 | [0.1, 0.0, 0.1, 0.9, 0.0] | [0.011, 0.000, 0.011, 0.099, 0.000] |
| by | 0.12 | [0.2, 0.0, 0.0, 0.2, 0.0] | [0.024, 0.000, 0.000, 0.024, 0.000] |
| the | 0.10 | [0.0, 0.0, 0.0, 0.0, 0.0] | [0.000, 0.000, 0.000, 0.000, 0.000] |
| river | 0.22 | [0.8, 0.0, 0.9, 0.0, 0.0] | [0.176, 0.000, 0.198, 0.000, 0.000] |
| bank | 0.17 | [0.5, 0.5, 0.3, 0.0, 0.0] | [0.085, 0.085, 0.051, 0.000, 0.000] |

```
Geographic: 0.011 + 0.000 + 0.011 + 0.024 + 0.000 + 0.176 + 0.085 = 0.307
Financial:  0.000 + 0.000 + 0.000 + 0.000 + 0.000 + 0.000 + 0.085 = 0.085
Nature:     0.000 + 0.000 + 0.011 + 0.000 + 0.000 + 0.198 + 0.051 = 0.260
Action:     0.000 + 0.050 + 0.099 + 0.024 + 0.000 + 0.000 + 0.000 = 0.173
Person:     0.105 + 0.010 + 0.000 + 0.000 + 0.000 + 0.000 + 0.000 = 0.115
```

**Head 1 context:** `[0.307, 0.085, 0.260, 0.173, 0.115]`

Geographic (0.307) and Nature (0.260) stay strong; Action is suppressed; Financial is modest. This is the geographic perspective on "bank".

## Head 2: action / activity focus

### Apply W₂^Q to "bank"

Head 2 uses a different matrix, searching for action / verb relevance:

```
bank embedding: [0.5, 0.5, 0.3, 0.0, 0.0]

W_2^Q (learned to emphasize action search):
[0.3  0.2  0.1  0.2  0.1]
[0.2  0.3  0.1  0.2  0.2]
[0.1  0.1  0.2  0.2  0.1]
[0.2  0.1  0.1  0.6  0.3]
[0.1  0.2  0.1  0.3  0.5]

Q_2 = [0.5, 0.5, 0.3, 0.0, 0.0] × W_2^Q
    = [0.5×0.3 + 0.5×0.2 + 0.3×0.1, ...]
    = [0.25, 0.25, 0.12, 0.16, 0.12]
```

This query emphasizes Action more than Head 1.

### Attention scores (Head 2)

Query for "bank" (Head 2): `[0.25, 0.25, 0.12, 0.16, 0.12]`

Dot product with keys using `W_2^K` (learned for action focus):

| Word | Key vector | Score | Why |
| ---- | ---------- | ----- | --- |
| I | [0.08, 0.00, 0.00, 0.00, 0.88] | 0.11 | Person element, not action-central |
| am | [0.00, 0.00, 0.00, 0.65, 0.10] | **0.10** | Auxiliary verb (action-relevant) |
| sitting | [0.10, 0.00, 0.12, 0.92, 0.00] | **0.28** | Core action verb |
| by | [0.18, 0.00, 0.00, 0.22, 0.00] | 0.08 | Preposition, modest action relevance |
| the | [0.00, 0.00, 0.00, 0.00, 0.00] | 0.00 | Stop word |
| river | [0.72, 0.00, 0.81, 0.02, 0.00] | 0.24 | Nature, not action-focused |
| bank | [0.45, 0.45, 0.27, 0.00, 0.00] | **0.25** | Self, some action relevance |

**Raw scores (Head 2):** `[0.11, 0.10, 0.28, 0.08, 0.00, 0.24, 0.25]`

Head 2 gives sitting much higher relative weight than Head 1 (0.28 vs 0.12).

### Softmax (Head 2)

| Word | Score | e^Score | Attention weight |
| ---- | ----- | ------- | ---------------- |
| I | 0.11 | 1.12 | 0.11 |
| am | 0.10 | 1.11 | 0.11 |
| sitting | 0.28 | 1.32 | **0.13** |
| by | 0.08 | 1.08 | 0.11 |
| the | 0.00 | 1.00 | 0.10 |
| river | 0.24 | 1.27 | 0.12 |
| bank | 0.25 | 1.28 | **0.13** |

**Sum:** `1.12 + 1.11 + 1.32 + 1.08 + 1.00 + 1.27 + 1.28 = 9.18`

**Attention weights (Head 2):**

- sitting: 13% (action-focused)
- bank: 13% (self)
- river: 12% (some action from movement)
- others: ~62% distributed

Sitting is elevated compared to single-head, though the gap is less dramatic than Head 1's geographic focus.

### Context vector (Head 2)

Weighted sum of value vectors (using `W_2^V`):

| Word | Wt | Value | Weighted (Head 2) |
| ---- | -- | ----- | ----------------- |
| I | 0.11 | [0.1, 0.0, 0.0, 0.0, 0.95] | [0.011, 0.000, 0.000, 0.000, 0.105] |
| am | 0.11 | [0.0, 0.0, 0.0, 0.5, 0.1] | [0.000, 0.000, 0.000, 0.055, 0.011] |
| sitting | 0.13 | [0.1, 0.0, 0.1, 0.9, 0.0] | [0.013, 0.000, 0.013, 0.117, 0.000] |
| by | 0.11 | [0.2, 0.0, 0.0, 0.2, 0.0] | [0.022, 0.000, 0.000, 0.022, 0.000] |
| the | 0.10 | [0.0, 0.0, 0.0, 0.0, 0.0] | [0.000, 0.000, 0.000, 0.000, 0.000] |
| river | 0.12 | [0.8, 0.0, 0.9, 0.0, 0.0] | [0.096, 0.000, 0.108, 0.000, 0.000] |
| bank | 0.13 | [0.5, 0.5, 0.3, 0.0, 0.0] | [0.065, 0.065, 0.039, 0.000, 0.000] |

```
Geographic: 0.011 + 0.000 + 0.013 + 0.022 + 0.000 + 0.096 + 0.065 = 0.207
Financial:  0.000 + 0.000 + 0.000 + 0.000 + 0.000 + 0.000 + 0.065 = 0.065
Nature:     0.000 + 0.000 + 0.013 + 0.000 + 0.000 + 0.108 + 0.039 = 0.160
Action:     0.000 + 0.055 + 0.117 + 0.022 + 0.000 + 0.000 + 0.000 = 0.194
Person:     0.105 + 0.011 + 0.000 + 0.000 + 0.000 + 0.000 + 0.000 = 0.116
```

**Head 2 context:** `[0.207, 0.065, 0.160, 0.194, 0.116]`

Action (0.194) is elevated vs Head 1 (0.173). Person is preserved. Geographic (0.207) is lower than Head 1 (0.307). This is the action perspective on "bank".

## Concatenate head outputs

Two different context vectors for "bank":

```
Head 1: [0.307, 0.085, 0.260, 0.173, 0.115]  ← geographic-focused
Head 2: [0.207, 0.065, 0.160, 0.194, 0.116]  ← action-focused
        ──────────────────────────────────────────────
Concat: [0.307, 0.085, 0.260, 0.173, 0.115, 0.207, 0.065, 0.160, 0.194, 0.116]
        └────── 5 dims ────┘ └────── 5 dims ────┘
                = 10 dimensions total
```

The 10-dimensional vector keeps specialized information from both heads — more than either perspective alone.

## Project back with W^O

The concatenated vector has 10 dimensions; the original embedding had 5. Apply a learned output projection `W^O` (10 × 5):

```
W^O (learned to combine multi-head perspectives):
[0.6  0.1  0.2  0.0  0.1]
[0.1  0.5  0.1  0.1  0.2]
[0.2  0.1  0.7  0.0  0.0]
[0.1  0.0  0.1  0.5  0.2]
[0.0  0.2  0.0  0.2  0.4]
[0.3  0.1  0.1  0.2  0.1]
[0.1  0.3  0.0  0.2  0.0]
[0.2  0.0  0.1  0.1  0.2]
[0.1  0.1  0.0  0.2  0.3]
[0.2  0.1  0.1  0.1  0.2]

Final = [0.307, 0.085, 0.260, 0.173, 0.115, 0.207, 0.065, 0.160, 0.194, 0.116] × W^O
      = [0.293, 0.134, 0.256, 0.211, 0.125]
```

**Multi-head output for "bank":** `[0.293, 0.134, 0.256, 0.211, 0.125]`

## Comparing original, single-head, and multi-head

| Dimension | Original | Single-head | Head 1 (geo) | Head 2 (action) | Multi-head final |
| --------- | -------- | ----------- | ------------ | --------------- | ---------------- |
| Geographic | 0.5 | 0.323 | 0.307 | 0.207 | **0.293** |
| Financial | 0.5 | 0.100 | 0.085 | 0.065 | **0.134** |
| Nature | 0.3 | 0.267 | 0.260 | 0.160 | **0.256** |
| Action | 0.0 | 0.189 | 0.173 | 0.194 | **0.211** |
| Person | 0.0 | 0.122 | 0.115 | 0.116 | **0.125** |

1. **Original** — ambiguous: equal Geographic and Financial (0.5 each)
2. **Single-head** — one pattern: "river" pulls Geographic to 0.323; Financial drops to 0.100
3. **Multi-head** — both patterns:
   - Head 1 (geographic) sees river clearly → 0.307 geographic
   - Head 2 (action) sees sitting clearly → 0.194 action
   - Projection blends both → 0.293 geographic + 0.211 action
4. **Key difference** — multi-head final Action (0.211) is higher than single-head (0.189). The action signal was not lost; the action head preserved it alongside the geographic signal.

## What each head learned

**Head 1 (geographic focus):**

- "river" (22%) is most geographically relevant
- "bank" (17%) self-attention confirms geographic aspects
- "by" (12%) location preposition
- Result: pulls "bank" toward river / nature meanings

**Head 2 (action focus):**

- "sitting" (13%) action-central verb
- "bank" (13%) self-attention for action aspect
- "river" (12%) still relevant (movement in space)
- Result: preserves action / activity meanings

Neither head was told to specialize. Gradient descent optimizes `W^Q`, `W^K`, `W^V` for each head independently. Head 1 naturally learns geographic patterns; Head 2 learns action patterns. That emergent specialization is why multi-head attention is powerful.

## Why multi-head matters

### Single-head produces one view

On "I am sitting by the river bank", single-head produces:

- Context: `[0.323, 0.100, 0.267, 0.189, 0.122]`
- Dominated by "river"'s geographic pull (score 0.67)
- "sitting"'s action signal is present (0.189) but muted

Why muted? One set of `W_Q`, `W_K`, `W_V` must compromise:

- Attending to "river" → high geographic, low action
- Attending to "sitting" → high action, low geographic
- Cannot do both strongly at once

### Multi-head: parallel patterns

```
Head 1 asks: "How is 'bank' used geographically?" → focuses on river (0.22)
Head 2 asks: "How is 'bank' used in an action?"   → focuses on sitting (0.13)

Both patterns computed in parallel. W^O combines them.
```

**Result:** multi-head final `[0.293, 0.134, 0.256, 0.211, 0.125]`

- Geographic: 0.293 (vs single 0.323) — slight trade-off
- Action: **0.211** (vs single 0.189) — action signal survives better
- Financial: 0.134 (vs single 0.100) — still represents ambiguity

"bank"'s multi-head representation now captures:

- It is a location (river context)
- It is part of an activity (sitting context)
- It is not purely financial (ambiguity preserved)

## Real-world transformers

In production models (BERT, GPT, and others):

```
BERT-base:
- 12 layers
- 12 attention heads per layer
- 768-dim embeddings
- Each head operates on 64-dim subspaces (768 / 12)
- Total: 144 separate attention mechanisms

Example patterns heads may learn:
- Pronoun–antecedent relationships
- Subject–verb agreement
- Adjective–noun modification
- Clause boundaries
- ...and other domain-specific patterns
```

All 12 heads run in parallel for every token — highly expressive.

## The mathematics (compact form)

For N heads:

```
head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)

Attention(Q, K, V) = softmax(QKᵀ / √d_k) V

MultiHead = Concat(head_1, ..., head_N) W^O
```

**Parameters to learn:**

- `W_1^Q`, `W_1^K`, `W_1^V`
- `W_2^Q`, `W_2^K`, `W_2^V`
- …
- `W_N^Q`, `W_N^K`, `W_N^V`
- `W^O` (output projection)

All learned via backpropagation during training.

## Key takeaways

1. **Single-head is a bottleneck** — one pattern per sentence
2. **Multi-head allows parallel patterns** — each head specializes
3. **Heads learn different aspects** — geographic, action, syntax, semantic similarity, whatever the task needs
4. **Concatenation preserves information** from all heads
5. **Output projection** combines heads into the final representation
6. **No explicit instruction** — heads learn what matters via gradient descent
7. **Expressiveness** — many heads × many layers compounds the patterns the model can learn

## Quick intuition check

**Why not one large head instead of multiple small heads?**

Multiple heads force specialization; a single large head tends to average and miss detail. Multi-head is closer to an ensemble of learned experts, and is more robust across input types.

**Do all heads learn completely different things?**

Mostly yes, though some redundancy can appear. The network learns to use heads efficiently via the `W^O` projection.

**How many heads is optimal?**

Empirically, often 8–16 for many tasks. More heads add capacity and parameters — a trade-off between expressiveness and compute cost.
