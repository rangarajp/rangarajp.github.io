---
title: "Inference-time compute scaling"
description: "Efficient execution makes each unit of model work cheaper."
section: learn
pubDate: 2026-10-29
chapter: 12
order: 12
---

Efficient execution makes each unit of model work cheaper. Inference-time scaling asks when spending more work improves the answer enough to justify the cost. The comparison must include the quality of the selection or verification process as well as generation.

## 12.1 The new scaling paradigm

Training compute determines the model we start with. Inference-time compute determines how much work that fixed model and its surrounding system spend on a particular problem. A service can generate additional steps, attempt alternative solutions, or invoke checks without changing the model's parameters during the request.

That creates an allocation problem: use a larger model once, or spend a comparable budget on more work with a smaller model? The answer depends on task difficulty, the generator's ability to improve with extra work, and the reliability of the selection or verification procedure. Easy tasks may gain little from the added cost.

## 12.2 Chain-of-thought and its inference cost

A short final answer may be preceded by substantial generated reasoning or by several attempted solutions. That extra work increases token processing, retained state, and tail latency. Its value depends on whether the model and checking procedure can turn the budget into better verified answers.

Two budgets can have equal token counts but different latency: one long sequential attempt has a long dependency chain; multiple shorter attempts may run concurrently but consume more simultaneous memory and capacity. Compare equal task-quality targets and account for verifier work, rather than assuming that longer reasoning always improves an answer.

## 12.3 Process reward models and search

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

#### Best-of-N, beam search, and tree search

Best-of-N generates several complete candidates and uses a verifier, reward model, or task-specific rule to choose among them. It is easy to parallelize but can spend the whole budget on independent copies of the same mistake. Majority voting is an aggregation rule, not a substitute for correctness checking.

Beam search keeps a limited set of partial paths and repeatedly expands/selects them. Tree search allocates work more selectively among partial alternatives using a value or reward signal. Both need a policy for when to stop, and both can exploit weaknesses in the scoring model rather than improve the actual task.

An outcome verifier evaluates a completed result. A process reward model evaluates intermediate steps and can guide expansion earlier. Training and validating that intermediate signal is a separate problem; useful process scores cannot be assumed merely because the generator writes plausible-looking reasoning.

## 12.4 DeepSeek-R1 and o1-style reasoning models

DeepSeek-R1 illustrates training methods that encourage reasoning behavior. Public descriptions of o1-style systems support a discussion of spending inference compute, not an assertion that their internal architectures are identical or fully disclosed. The serving consequence is variable output length and less predictable occupancy. Bound work and measure distributions, including long-tail requests.

Primary reading: [Scaling test-time compute](https://arxiv.org/abs/2408.03314) and [DeepSeek-R1](https://arxiv.org/abs/2501.12948).

## Hands-on extensions

Use these experiments to connect the mechanisms above to measurements. They are proposed exercises unless a recorded result is explicitly identified.

### Hands-on: spend a fixed candidate budget

Use an executable arithmetic or coding checker to compare one candidate against several bounded candidates. Publish verified success, total generated tokens, verifier cost, and latency. Include failures where extra candidates repeat the same error.
