---
title: "Inference-time compute scaling"
description: "Reasoning budgets, verifiers, process rewards, and bounded search."
section: learn
pubDate: 2026-10-29
part: 6
chapter: 38
order: 38
tags: ["llm-inference", "systems", "hands-on"]
---

Some tasks benefit from spending additional inference compute: generating alternatives, checking a solution, or revising after a tool result. The objective is better verified task success per unit of time and cost, not more generated text.

## 1. Longer reasoning and its cost

Longer generated sequences consume decode time and cache capacity. Multiple candidates add repeated generation and verification. Keep a total token, latency, and tool-call budget. More reasoning tokens can repeat mistakes or rationalize an incorrect answer.

## 2. Process rewards and search

A final-answer verifier scores outcomes; a process reward model scores intermediate steps. Search uses these signals to allocate further work. A learned reward is imperfect, so optimizing its score can amplify its errors. Executable checks provide stronger evidence where the task permits them.

```python
# CPU-only illustration of bounded candidate selection, not an LLM call.
candidates = [19, 21, 17, 25]
def valid(x):
    return x > 0 and x * x == 441
chosen, checked = None, 0
for candidate in candidates[:3]:
    checked += 1
    if valid(candidate):
        chosen = candidate
        break
print(chosen, checked)  # 21, 2
```

A real system would generate candidates and charge their generation and checking costs against the budget. Avoid majority voting for a task where all candidates can share the same systematic error.

## 3. Reasoning models

DeepSeek-R1 illustrates training methods that encourage reasoning behavior. Public descriptions of o1-style systems support a discussion of spending inference compute, not an assertion that their internal architectures are identical or fully disclosed. The serving consequence is variable output length and less predictable occupancy. Bound work and measure distributions, including long-tail requests.

Primary reading: [Scaling test-time compute](https://arxiv.org/abs/2408.03314) and [DeepSeek-R1](https://arxiv.org/abs/2501.12948).
