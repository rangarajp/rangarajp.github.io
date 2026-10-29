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

A tree offers multiple possible continuations rather than one chain. A tree-aware mask allows each node to see only its ancestors. Verification can cover more alternatives, but extra nodes consume compute and temporary memory. A larger tree is useful only when extra accepted progress pays for that work.

## 4. Medusa and EAGLE

Medusa adds prediction heads to propose future tokens from shared model features. EAGLE drafts through feature-level prediction and token information; its variants change how candidates are constructed and expanded. These require compatible trained components and engine support, rather than a generic setting usable by every checkpoint.

## 5. When speculation helps

Low-concurrency, bandwidth-heavy decode is a promising regime because verification can reuse target weights across proposed positions. At high concurrency the target may already exploit that reuse through batching. Poor draft agreement, expensive drafting, or tight memory can erase the benefit. Measure accepted tokens per round and end-to-end latency, including draft-model memory.

Primary reading: [exact speculative sampling](https://arxiv.org/abs/2211.17192), [Medusa](https://arxiv.org/abs/2401.10774), and [EAGLE](https://arxiv.org/abs/2401.15077).

## Hands-on: measure acceptance and its overhead

First publish the existing CPU acceptance experiment and its empirical distribution check. Then, with a compatible engine, sweep draft length and report accepted tokens per round, draft time, verification time, memory, and end-to-end latency. Treat the GPU stage as a proposed experiment.
