---
title: "Speculative decoding"
description: "Draft-and-verify sampling, token trees, Medusa, EAGLE, and a cost model."
section: learn
pubDate: 2026-10-29
part: 3
chapter: 18
order: 18
tags: ["llm-inference", "systems", "hands-on"]
---

Quantization changes representation; speculation changes how much progress one target-model round can verify. Both need quality checks, but exact speculative sampling has a specific distribution-preservation requirement.

Autoregressive decoding repeatedly reads a large model to emit one token. Speculation asks a cheaper draft mechanism to propose several tokens, then verifies them with the target in a parallel forward pass.

## 1. Draft and verify

The target can score the positions of a proposed continuation together even though producing those positions from scratch would require repeated next-token decisions. A speculative round uses that difference:

1. A cheaper drafter proposes a short continuation from the current accepted prefix.
2. The target evaluates those proposed positions in a causally correct verification pass.
3. Acceptance decisions proceed in order, extending the accepted prefix until a rejection or the end of the draft.
4. After rejection, emit a correctly sampled correction and discard later unaccepted proposals. If the whole draft is accepted, the target can provide a bonus token.

Verification shares target-model work across proposed positions. It still pays for drafting, masks, cache management, and verification, and its benefit depends on how much proposed text survives.

A greedy demonstration accepts draft tokens while they equal the target's greedy choices, then emits the target correction at the first mismatch. Exact *sampling* requires more: if draft probability is $q(x)$ and target probability is $p(x)$, accept a proposed token with probability $\min(1,p(x)/q(x))$. After rejection, sample from the normalized positive residual $\max(p-q,0)$ at that prefix. Accept/reject decisions proceed along the accepted prefix. If all drafts pass, the target can supply an additional token.

Skipping the probability correction changes the target sampling distribution. Matching output text on one prompt does not prove exactness.

### A two-token probability check

The following experiment samples a draft and applies the exact rejection correction. Across many independent trials, the output frequencies approach the target distribution, even though the draft favors the other token. This demonstrates one verification position; a multi-token engine must repeat it with the correct prefix and manage both caches.

```python
import random
rng = random.Random(7)
p, q = [0.8, 0.2], [0.4, 0.6]
residual = [max(a-b, 0.0) for a,b in zip(p,q)]
counts = [0, 0]
for _ in range(100_000):
    proposed = rng.choices(range(2), weights=q)[0]
    if rng.random() < min(1.0, p[proposed] / q[proposed]):
        emitted = proposed
    else:
        emitted = rng.choices(range(2), weights=residual)[0]
    counts[emitted] += 1
print([round(n / sum(counts), 3) for n in counts])  # approximately [0.8, 0.2]
```

The residual is only used after rejection. When draft and target distributions are identical, acceptance is certain and no residual draw is needed.

## 2. Estimate useful tokens per round

```python
# Analytical illustration: independent, constant conditional acceptance alpha.
k = 4
for alpha in (0.3, 0.6, 0.9):
    emitted = sum(alpha ** i for i in range(k + 1))
    target_step_ms, draft_step_ms, verify_ms = 10, 1, 14
    speedup = emitted * target_step_ms / (k * draft_step_ms + verify_ms)
    print(alpha, round(emitted, 3), round(speedup, 3))
```

The geometric sum includes the correction or bonus token. Drafting four tokens is not a fourfold speedup: draft cost, verification cost, rejected work, and cache updates all consume time. Real acceptance probabilities vary by position and task.

## 3. Token tree verification

A linear draft commits to one proposed continuation at each position. A draft tree offers alternatives: two possible next tokens may each have their own proposed successors. The target can evaluate those nodes together, but the attention mask must make each node see only its prompt and ancestors, never unrelated branches.

Verification then selects a valid accepted path using the decoding procedure's rules. Unselected branches are discarded. Their temporary state cannot simply be appended to the active request as though every branch had happened. Correct positions, masks, acceptance, and cache ownership are all part of implementing tree verification.

A wider tree increases the chance of proposing a useful continuation, but also creates more target computation and temporary state. Choose tree size by accepted progress per unit of total work, not by candidate count alone.

## 4. Medusa: multiple prediction heads

Medusa adds trained heads that use a shared hidden representation to propose tokens at future offsets. Several inexpensive heads can therefore supply candidate continuations without running a separate full draft model for every proposal. The target still verifies candidates; a head's prediction is not evidence that every earlier predicted token will be accepted.

The shared representation is an efficiency advantage and a modeling constraint. Additional heads must learn useful future-offset predictions, and the engine needs a compatible tree/verification path. Extra heads, their training, and verification all have costs, so this is not a free configuration flag for an arbitrary existing checkpoint.

## 5. EAGLE: feature-based drafting

EAGLE trains a lightweight drafter around model features and token information. Conditioning on token information helps address uncertainty about which continuation led to a future feature. Later variants also adapt the draft tree, changing where proposal work is spent.

The operational question is the same as for a separate draft model or Medusa: how much accepted target-model progress is gained after paying for drafting, verification, state updates, and any extra memory? The mechanisms differ, so their acceptance rates and overhead must be measured on the actual task.

## 6. When speculation helps

Low-concurrency, bandwidth-heavy decode is a promising regime because verification can reuse target weights across proposed positions. At high concurrency the target may already exploit that reuse through batching. Poor draft agreement, expensive drafting, or tight memory can erase the benefit. Measure accepted tokens per round and end-to-end latency, including draft-model memory.

Primary reading: [exact speculative sampling](https://arxiv.org/abs/2211.17192), [Medusa](https://arxiv.org/abs/2401.10774), and [EAGLE](https://arxiv.org/abs/2401.15077).

## Hands-on: measure acceptance and its overhead

First publish the existing CPU acceptance experiment and its empirical distribution check. Then, with a compatible engine, sweep draft length and report accepted tokens per round, draft time, verification time, memory, and end-to-end latency. Treat the GPU stage as a proposed experiment.
