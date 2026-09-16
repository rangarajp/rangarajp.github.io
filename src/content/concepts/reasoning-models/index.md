---
title: 'Reasoning Models'
description: 'How LLMs allocate test-time compute to think step by step before answering.'
pubDate: 'Aug 24 2026'
seriesOrder: 3
heroImage: '../../../assets/blog-placeholder-3.jpg'
---

Reasoning models are LLMs trained to spend extra compute at inference time — generating intermediate chain-of-thought before a final answer. This book covers how that works: from prompting tricks to RL-trained long CoT, search, verifiers, and the cost–quality trade-offs.

Read the chapters in order. Each one builds on the last.

## Contents

1. [What reasoning means](./basics) — conventional training, what reasoning means, and how it is improved
2. [Inference-time scaling](./inference-time-scaling) — CoT, sampling, self-consistency, Best-of-N, and self-refinement without retraining
3. [Training for reasoning](./training-for-reasoning) — pretraining → SFT + preference RL → reasoning RL with verifiers
4. [GRPO](./grpo) — group relative policy optimization: rollouts, rewards, advantages, and KL penalty

## Coming later

- *Chain of thought in depth* — prompting and supervised CoT as the baseline
- *Search and verifiers* — Best-of-N, self-consistency, tree search, checkers
- *Trade-offs* — latency, cost, overthinking, and when reasoning helps or hurts

Together, these chapters explain how models move from “instant answers” to “think then answer.”
