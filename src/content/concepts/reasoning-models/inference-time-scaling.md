---
title: 'Inference-Time Scaling'
description: 'Improving reasoning at decode time — CoT, parallel strategies, and sampling — without retraining.'
pubDate: 'Sep 5 2026'
order: 2
heroImage: '../../../assets/blog-placeholder-3.jpg'
---

You can improve LLM reasoning in two broad ways: spend more compute **while answering** (inference time), or change the **weights** so the model reasons better by default (training time). This page focuses on the first — inference-time scaling — and the main families of methods you will see in practice.

## 1. Two ways to improve reasoning

1. **Inference-time** — keep the same trained model; spend more tokens, samples, or search at decode time (prompting, sampling, parallel candidates)
2. **Training-time** — change the model itself with supervised fine-tuning, distillation, or reinforcement learning so better reasoning is baked into the weights

Inference-time methods are useful first: they need no retraining, are easy to A/B test, and make the cost–quality trade-off explicit — more compute usually means higher accuracy, until returns diminish.

### 1.1 Same model, more compute

**Problem:** Ram has 3 chocolates. Sam has 5 chocolates. They put them in one basket and each eats 1. How many chocolates are left?

**Low inference compute** (direct answer):

```
Answer: 7
```

Fast, but wrong — no intermediate checks.

**Higher inference compute** (same model, more tokens / better strategy):

```
Step 1: Ram has 3 chocolates.
Step 2: Sam has 5 chocolates.
Step 3: Together: 3 + 5 = 8.
Step 4: Each eats 1 → 2 eaten.
Step 5: Left: 8 − 2 = 6.
Answer: 6
```

Same weights; extra decode-time work produced a verifiable path and the correct answer. That is the core idea of inference-time scaling.

## 2. Types of inference-time reasoning

Inference-time methods fall into a few recurring families (often combined):

1. **Chain of thought (CoT)** — ask the model to write intermediate steps before the final answer (one longer trace)
2. **Parallel strategies** — explore several reasoning paths at once (Best-of-N, self-consistency / majority vote, tree-style search)
3. **Sampling** — draw multiple stochastic completions (temperature, top-p) and select or aggregate among them

CoT spends compute **sequentially** on one chain. Parallel strategies and sampling spend compute **across** many candidates — then pick a winner (vote, score, or verifier).

### 2.1 Example: one answer vs many samples

Same chocolate problem. Suppose the model samples three short answers (temperature > 0):

```
Sample 1: Answer: 6
Sample 2: Answer: 7
Sample 3: Answer: 6
```

**Without aggregation:** you might keep Sample 2 and ship `7`.

**With sampling + majority vote** (a simple parallel strategy):

```
Votes: 6, 7, 6 → majority = 6
Answer: 6
```

Benefit: wrong one-off samples get outvoted. Cost: ~3× decode work vs a single shot. CoT alone often fixes arithmetic by forcing steps; sampling + vote helps when the model is noisy but usually correct.

| Approach | Extra compute | Typical benefit |
| -------- | ------------- | --------------- |
| Direct answer | None | Fast; error-prone on multistep tasks |
| Chain of thought | Longer single trace | Checkable steps; fewer arithmetic slips |
| Sampling + vote / Best-of-N | Multiple traces | Reduces variance; needs a way to pick |

## 3. Summary

1. Reasoning improves via **inference-time** compute or **training-time** weight updates
2. Inference-time families include **CoT**, **parallel strategies**, and **sampling** (often mixed)
3. More decode-time compute can raise accuracy on the same model — at the cost of latency and tokens
4. Next steps in this series dig into each family and into training-time methods that complement them
