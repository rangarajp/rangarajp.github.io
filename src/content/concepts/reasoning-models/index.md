---
title: 'Reasoning Models Overview'
description: 'How LLMs allocate test-time compute to think step by step before answering.'
pubDate: 'Aug 24 2026'
heroImage: '../../../assets/blog-placeholder-3.jpg'
---

Reasoning models are LLMs trained to spend extra compute at inference time — generating intermediate chain-of-thought before a final answer. This series covers how that works: from prompting tricks to RL-trained long CoT, search, verifiers, and the cost–quality trade-offs.

Read them in order for the full picture, or jump to any topic that interests you.

## 1. The pipeline

Topics to cover in this series:

1. **Basics** — conventional training, what reasoning means, and how it is improved
2. **Inference-time scaling** — CoT, sampling, self-consistency, Best-of-N, and self-refinement without retraining
3. **Chain of thought** — prompting and supervised CoT as the baseline
4. **Training for reasoning** — process rewards, outcome rewards, and RL on long traces
5. **Search and verifiers** — Best-of-N, self-consistency, tree search, checkers
6. **Trade-offs** — latency, cost, overthinking, and when reasoning helps or hurts

Together, these ideas explain how models move from “instant answers” to “think then answer.”
