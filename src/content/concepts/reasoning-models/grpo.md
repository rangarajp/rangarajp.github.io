---
title: 'GRPO — Group Relative Policy Optimization'
description: 'How GRPO trains reasoning by comparing a group of attempts on the same question — without a critic network.'
pubDate: 'Sep 5 2026'
order: 5
heroImage: '../../../assets/blog-placeholder-3.jpg'
---

[Training for reasoning](/concepts/reasoning-models/training-for-reasoning) introduced GRPO as the RL algorithm behind DeepSeek-R1. This page goes deeper: what GRPO actually does, worked through a concrete example with rollouts, rewards, advantages, and the KL penalty.

Notebook: [grpo-training.ipynb](https://github.com/rangarajp/rangarajp.github.io/blob/main/notebooks/reasoning-models/grpo-training.ipynb) — runs a mini-GRPO loop on math problems and compares the base model against the fine-tuned version.

## 1. The core idea — exam practice in a group

Imagine a class of students all attempting the **same problem** at the same time. After everyone submits:

1. The teacher grades each answer (reward)
2. Each student's score is compared **relative to the class average** — "you did better/worse than your peers on this question"
3. Students who scored above average get positive reinforcement; those below get a correction signal
4. Crucially: the class never sees a "model answer" written by a teacher — they only know their own answer and how it ranked in the group

That is GRPO. The "class" is a group of sampled rollouts from the same policy. There is no separate "teacher" (critic/value network) telling anyone what the ideal answer looks like — just relative performance within the group.

Compare with classic PPO:

| | PPO | GRPO |
|---|---|---|
| Advantage estimate | Learned value/critic network V(s) | Group mean and std of rewards |
| Extra network needed | Yes — critic trained alongside policy | No |
| Stability on long CoT | Critic hard to train for long sequences | More stable; no value learning |
| Introduced by | Schulman et al. 2017 | DeepSeek-Math / DeepSeek-R1, 2025 |

## 2. A concrete rollout example

**Question:** Ram has 3 chocolates. Sam has 5 chocolates. They put them in one basket and each eats 1. How many are left?  
**Correct answer:** 6

The policy generates **N = 5 rollouts** for this question:

| Rollout | Trace (abbreviated) | Format reward | Accuracy reward | Total reward |
|---------|---------------------|---------------|-----------------|--------------|
| 1 | `<think>…</think> Answer: 4` | 1.0 | 0.0 | 1.0 |
| 2 | `<think>3+5=8; 8-2=6</think> Answer: 6` | 1.0 | 1.0 | 2.0 |
| 3 | `Answer: 6` (no think tags) | 0.0 | 1.0 | 1.0 |
| 4 | `<think>…</think> Answer: 7` | 1.0 | 0.0 | 1.0 |
| 5 | `<think>…</think> Answer: 6` | 1.0 | 1.0 | 2.0 |

**Rewards used in DeepSeek-R1-Zero** (rule-based, no neural reward model):
- **Format reward** — does the response use `<think>…</think>` tags? (+1 if yes)
- **Accuracy reward** — does the boxed/final answer match the ground truth? (+1 if correct, checked by rule or SymPy)

> No process reward model (PRM). No human preference labelling. Just these two binary rules.

## 3. Group-relative advantage

Compute the mean and standard deviation of rewards **within this group**:

```
rewards  = [1.0, 2.0, 1.0, 1.0, 2.0]
mean_r   = 1.4
std_r    = 0.49
```

Normalise each rollout's reward to get its **advantage**:

```
A_i = (r_i − mean_r) / std_r
```

| Rollout | Reward | Advantage |
|---------|--------|-----------|
| 1 | 1.0 | −0.82 |
| 2 | 2.0 | +1.22 |
| 3 | 1.0 | −0.82 |
| 4 | 1.0 | −0.82 |
| 5 | 2.0 | +1.22 |

Rollouts 2 and 5 (correct answer + think tags) got **positive** advantage → the policy gradient will reinforce these token sequences. Rollouts 1, 3, 4 got **negative** advantage → those paths are discouraged.

```python
import torch

rewards = torch.tensor([1.0, 2.0, 1.0, 1.0, 2.0])
advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
# tensor([-0.82, +1.22, -0.82, -0.82, +1.22])
```

No critic network required — the group itself provides the baseline.

## 4. Policy gradient loss

For each rollout *i* of length *T* tokens, the GRPO policy loss is:

```
L_policy = − (1/T) Σ_t [ A_i · log π_θ(token_t | context_t) ]
```

Positive advantage → we want to **increase** the log-prob of those tokens (reinforce).  
Negative advantage → we want to **decrease** it (penalise).

```python
def grpo_policy_loss(log_probs: torch.Tensor, advantages: torch.Tensor) -> torch.Tensor:
    """
    log_probs: [N, T] — per-token log probs of generated tokens for each rollout
    advantages: [N]   — group-normalised advantage per rollout
    """
    # broadcast advantage over token dimension
    weighted = log_probs * advantages.unsqueeze(1)   # [N, T]
    return -weighted.mean()
```

## 5. KL divergence — don't drift too far

Without a constraint, RL can make the policy drift far from the original pretrained model — producing correct but incoherent or degenerate text, or exploiting reward loopholes.

The fix: add a **KL penalty** that measures how far the current policy *π_θ* has moved from a frozen **reference policy** *π_ref* (the model before RL started):

```
L_total = L_policy + β · KL(π_θ ‖ π_ref)
```

where β is a small coefficient (e.g. 0.01–0.1).

**Analogy:** the class can improve their scores, but they must not change their *thinking style* so drastically that it becomes unrecognisable. The reference model is the anchor.

The KL formula matters more than it looks. The naive estimate `mean(log π_θ − log π_ref)` can go **negative** if the policy assigns lower probability to the tokens it generated than the reference did. A negative KL × positive β means the optimiser is *rewarded* for making the policy drift further away — the model degrades into garbled output.

The correct approximation (used in TRL and DeepSeek-R1) is always ≥ 0:

```python
def kl_penalty(
    log_probs_policy: torch.Tensor,   # [T] current policy log-probs
    log_probs_ref: torch.Tensor,      # [T] reference (frozen) log-probs
) -> torch.Tensor:
    """
    Non-negative per-token KL approximation — same formula used in TRL / DeepSeek-R1.

    KL(π_ref ‖ π_θ) ≈ exp(log π_ref − log π_θ) − (log π_ref − log π_θ) − 1  ≥ 0

    This equals 0 iff π_θ = π_ref, and grows positively for any divergence,
    so the penalty always pulls the policy toward the reference, never away.
    """
    log_ratio = log_probs_ref - log_probs_policy   # log π_ref − log π_θ
    return (torch.exp(log_ratio) - log_ratio - 1).mean()
```

One more guard: when all rollouts in a group receive the **same reward** (e.g. all zeros), `std = 0` and all advantages collapse to 0. The policy gradient term vanishes, but the KL penalty still fires. The safe approach is to skip the update entirely for that step — no signal means no update.

```python
if rewards.std() < 1e-6:
    continue   # skip — no variance means no advantage signal
```

## 6. Full GRPO step

Putting it together for one training step:

```python
def grpo_step(
    policy,
    ref_policy,
    question: str,
    gold_answer: str,
    N: int = 8,
    beta: float = 0.04,
):
    # 1. Sample a group of rollouts
    rollouts = generate_with_logprobs(policy, question, num_sequences=N)

    # 2. Score each rollout
    rewards = torch.tensor([
        reward_fn(text, gold_answer) for text, _ in rollouts
    ])

    # 3. Group-relative advantages
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    # 4. Compute log-probs under current and reference policy
    log_probs_policy = torch.stack([lp for _, lp in rollouts])      # [N, T]
    log_probs_ref    = get_ref_logprobs(ref_policy, rollouts)       # [N, T]

    # 5. Combined loss
    loss = (
        grpo_policy_loss(log_probs_policy, advantages)
        + beta * kl_penalty(log_probs_policy, log_probs_ref)
    )

    loss.backward()
    return loss.item(), rewards.mean().item()
```

## 7. Training loop sketch

```python
optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-6)

for step, (question, gold) in enumerate(practice_set):
    optimizer.zero_grad()
    loss, mean_reward = grpo_step(policy, ref_policy, question, gold, N=8)
    optimizer.step()

    if step % 10 == 0:
        print(f"step {step:4d}  loss={loss:.4f}  mean_reward={mean_reward:.3f}")
```

Over many steps the policy learns to consistently produce `<think>` blocks and correct answers — without ever seeing a worked solution from a teacher.

## 8. Mini-training results (Qwen3-0.6B, 20 steps, CPU)

The notebook `grpo-training.ipynb` runs the loop above on 20 arithmetic problems and evaluates on 5 held-out questions. Here is what actually happened.

### Training curve

```
Mean reward per step (bar = proportion of max 2.0)
------------------------------------------------------------
step   1  0.500  ||||||||||
step   2  1.000  ||||||||||||||||||||
step   3  0.500  ||||||||||
step   5  1.000  ||||||||||||||||||||   ← SKIPPED (all same reward)
step   7  1.000  ||||||||||||||||||||   ← SKIPPED
step  10  0.750  |||||||||||||||
step  15  1.000  ||||||||||||||||||||   ← SKIPPED
step  20  1.000  ||||||||||||||||||||   ← SKIPPED
```

Steps marked **SKIPPED** had zero reward variance — all 4 rollouts got the same score, so advantages collapsed to 0 and the update was safely bypassed.

### KL divergence

```
step   1  kl=0.000   step  10  kl=0.135
step   3  kl=0.031   step  14  kl=0.293
step   6  kl=0.048   step  20  kl=0.000
```

KL stays small and non-negative throughout. The policy has learned without drifting.

### Before vs after

| Metric | Base model | GRPO (20 steps) |
|---|---|---|
| Accuracy | 0.467 | **0.800** |
| Format rate | 0.000 | 0.000 |
| Mean reward | 0.467 | **0.800** |
| Samples | 15 | 15 |

Accuracy improved by **+71%** in 20 steps on CPU. Format rate remains 0 because the prompt already opens with `<think>` so the model only needs to close it — a useful extension for further experiments.

### Qualitative examples

```
PROBLEM: What is 9 times 9?   GOLD: 81
BASE  → None   (garbled symbol output)
GRPO  → 81     "9 times 9 equals 81."

PROBLEM: A dozen eggs minus 5. How many are left?   GOLD: 7
BASE  → None   (garbled symbol output)
GRPO  → 7      "a dozen eggs is 12 … 12 - 5 = 7."

PROBLEM: What is the square root of 64?   GOLD: 8
BASE  → 8      (correct — base model already knew this one)
GRPO  → 8      "64 is the square of 8, so the square root is 8."
```

### What went wrong the first time

Running the same code with the MATH-500 training set (GCD of 7!, arithmetic sequences) and the naive KL formula produced the opposite result:

| Metric | MATH dataset + naive KL | Built-in problems + fixed KL |
|---|---|---|
| GRPO accuracy | **0.033** (worse than base) | **0.800** |
| KL at step 20 | −0.81 (drifting) | ≤ 0.29 (stable) |
| Model quality | Garbled Hebrew/symbol text | Clean, correct answers |

Two root causes:
1. **KL formula bug** — `mean(log π_θ − log π_ref)` can be negative; the optimiser then makes it more negative, actively degrading the model. Fix: use `exp(log π_ref − log π_θ) − (log π_ref − log π_θ) − 1 ≥ 0`.
2. **No reward signal** — MATH Level-1 problems like "Find the GCD of 7! and (5!)²" require multi-step factorisation that a 0.6B base model cannot do in 120 tokens; every rollout scores 0, every advantage is 0, every gradient is zero, but the KL penalty fires anyway.

## 9. What emerges

Because the only signal is "correct final answer + proper format," the model is free to develop **any** intermediate strategy that works. In DeepSeek-R1-Zero this led to spontaneous **"aha moments"**: the model learned to pause, re-examine a step that looked wrong, and self-correct — purely as a learned strategy for earning accuracy rewards.

This is qualitatively different from SFT: there, self-correction would only appear if a human had written it into the training traces. With GRPO it emerges from the reward.

## 10. GRPO vs alternatives

| Method | Signal | Critic? | When to use |
|--------|--------|---------|-------------|
| SFT | Teacher demonstrations | No | Warm-start; format learning |
| PPO + reward model | Learned reward model + critic | Yes (value network) | General RLHF |
| **GRPO** | Verifiable rule-based rewards | No | Math, code — checkable tasks |
| DPO | Preference pairs (chosen / rejected) | No | Preference alignment without RL loop |
| RLOO | Leave-one-out baseline within group | No | Similar to GRPO, slightly different baseline |

## 11. Summary

1. GRPO samples **N rollouts per question** and normalises rewards within the group to compute advantages — no critic needed
2. Rewards are **rule-based**: format check + accuracy check (no neural reward model in R1-Zero)
3. The **policy gradient loss** reinforces high-advantage token sequences, penalises low-advantage ones
4. The **KL penalty** keeps the policy anchored to the reference model — prevents drift and reward hacking
5. Together, these produce reasoning behaviour (self-checking, extended CoT) that SFT on fixed demos rarely matches
6. The **KL formula must be non-negative** — the approximation `exp(log π_ref − log π_θ) − (log π_ref − log π_θ) − 1` ensures the penalty always pulls toward the reference; the naive `log π_θ − log π_ref` can go negative and corrupt the model
7. See [grpo-training.ipynb](https://github.com/rangarajp/rangarajp.github.io/blob/main/notebooks/reasoning-models/grpo-training.ipynb) for a runnable mini-GRPO loop, corrected KL, and before/after comparison (+71% accuracy in 20 steps on CPU)
