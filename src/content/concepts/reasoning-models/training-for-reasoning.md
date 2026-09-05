---
title: 'Training for Reasoning'
description: 'From pretraining to SFT to RL with verifiers — how reasoning models learned to practice, not only read.'
pubDate: 'Sep 5 2026'
order: 4
heroImage: '../../../assets/blog-placeholder-3.jpg'
---

[Inference-time scaling](/concepts/reasoning-models/inference-time-scaling) keeps the weights fixed and spends more compute **while answering**. **Training-time reasoning** changes the weights so the model *practices* hard problems during training — and carries that habit into every later call.

A useful metaphor for the history:

| Era | Student analogy | What the model mainly saw |
| --- | --------------- | ------------------------- |
| **1. Pretraining** | Read the entire internet; little exam practice | Next-token prediction on raw text |
| **2. SFT + preference RL** | Tutor worked a fixed set of problems; graded on whether answers *felt* right | Demonstrated solutions + human preference rankings |
| **3. Reasoning RL** | Practice with a grader who only cares whether the *answer is actually correct* | Questions + **verifiable** answers (math checker, code tests, …) |

The key difference between Era 2 and Era 3 is not whether RL is used — both use it — but **what the reward measures**: human preference vs. ground-truth correctness.

## 1. Era 1 — Pretraining: knowledge without practice

**What happens:** train on a huge corpus for next-token prediction. The model absorbs language, facts, and *patterns that look like* reasoning because such text appears in books and forums.

**Analogy:** a student who read the whole internet but sat few timed exams. They can talk about math; they have not been graded for solving novel contest problems under pressure.

**What you get:**

- Broad knowledge and fluent text (the base for everything else)
- Some emergent multistep behavior when the prompt cues it
- Weak reliability on hard, unfamiliar problems — no training signal said "keep searching until the answer checks out"

**Typical artifacts:** GPT-3-class base models; Llama, Qwen, DeepSeek **base** checkpoints — strong priors, not yet "reasoning products."

Pretraining builds the **library**. It does not, by itself, install a reliable **exam strategy**.

## 2. Era 2 — SFT + preference RL: learning from a tutor, graded by feel

**What happens:** two steps on top of the pretrained base.

**Step A — Supervised fine-tuning (SFT):** show the model worked examples — instruction–response pairs, and for reasoning tasks, step-by-step solutions (supervised CoT, or distillation from a stronger teacher). This gives the model a starting policy in the right format.

**Step B — Preference RL (RLHF):** humans (or AI) compare pairs of responses and label which is better. A **reward model** is trained on these preference rankings and used as the RL signal, typically via PPO — this is the InstructGPT / ChatGPT pipeline (Ouyang et al., 2022). The result is a model that is helpful, safe, and stylistically appropriate.

**Analogy:** a tutor works through problems on the board (SFT). Then examiners grade answers on how clear and well-presented they are — not only whether the boxed answer is right (preference RL).

**What you get:**

- Instruction-following and chat behavior: **ChatGPT**, **GPT-4**, **GPT-4o**, Claude, Gemini chat models — all use SFT + preference-based RLHF
- Gains on in-distribution reasoning tasks, especially when prompted with CoT
- Cleaner output style and better safety behaviour

**Where it stalls:**

- The preference reward reflects what *looks* like a good answer to a human rater — not whether the arithmetic is provably right
- Unfamiliar problem structures still fail — there was no loop that rewarded *discovering* a correct answer for a novel problem
- Trace length and search depth stay close to what the tutor's demonstrations showed, not what the *problem* actually needs

**SFT teaches how solutions look. Preference RL teaches how answers feel. Neither directly teaches: keep searching until the answer verifiably checks out.**

### 2.1 Why preference RL is not enough for hard reasoning

Preference-based RL (RLHF) is still RL — it updates the policy with a signal. But the signal comes from a human comparing two responses, not from a math compiler checking the boxed result. On tasks where correctness can be verified (e.g. "is 6 the right answer?"), human preference is a noisy proxy — a confidently wrong answer that *reads* well can score higher than a hesitant right one.

## 3. Era 3 — Reasoning RL: practice with a checker

**What happens:** give the model **questions**, let it generate long traces, score the **final answer** with an automated verifier (exact match, SymPy symbolic check, code test runner, …) and update the policy with RL. The model is not copying a tutor's transcript or pleasing a human rater — it is reinforced whenever its trace ends in a provably correct answer.

**Analogy:** graded practice problems. Attempt → submit to compiler / answer key → grade updates strategy. Over many problems, the model learns patterns of search, self-check, and recovery that SFT never explicitly wrote down.

**Core ingredients:**

| Piece | Role |
| ----- | ---- |
| Policy model | Generates long CoT + final answer |
| Verifier ("compiler") | Outcome reward: correct / incorrect — rule-based, no neural network needed |
| Optional process reward (PRM) | Scores intermediate steps, not just the final answer |
| RL algorithm | PPO, or **GRPO** (Group Relative Policy Optimization) |

**What tends to emerge:** longer internal chains, spontaneous self-correction, re-examination — reasoning behavior that SFT on fixed demonstrations rarely achieves at the same level.

### 3.1 GRPO — avoiding a separate critic

Classic PPO requires training a **value/critic network** alongside the policy to estimate advantage (how much better a move was than expected). GRPO (DeepSeek-Math / DeepSeek-R1) sidesteps this by sampling **a group of rollouts** for the same question and computing relative advantages directly:

```
advantage_i = (reward_i − mean(group_rewards)) / std(group_rewards)
```

No separate critic network needed. This is cheaper and more stable for long reasoning traces, where a learned value function is hard to train accurately.

→ **Deep dive:** [GRPO — Group Relative Policy Optimization](/concepts/reasoning-models/grpo) — rollouts, advantages, the KL penalty, and a [mini-training experiment](/concepts/reasoning-models/grpo#8-mini-training-results-qwen3-06b-20-steps-cpu) on Qwen3-0.6B showing +71% accuracy in 20 steps on CPU. Notebook: [grpo-training.ipynb](https://github.com/rangarajp/rangarajp.github.io/blob/main/notebooks/reasoning-models/grpo-training.ipynb)

### 3.2 DeepSeek-R1 — the open recipe (DeepSeek, Jan 2025)

DeepSeek published the full pipeline for two models built on **DeepSeek-V3-Base**:

**DeepSeek-R1-Zero — pure RL, no SFT warm-up:**

1. Start from the base model
2. Apply GRPO with two simple reward signals:
   - **Accuracy reward** — is the final answer correct? (rule-based: SymPy match for math, test execution for code)
   - **Format reward** — does the response use the `<think>` … `</think>` structure? (no neural reward model)
3. No process reward model; no preference data at this stage

Result: the model learns to reason and spontaneously develops "aha moments" — backtracking, re-examining its own steps — purely from outcome feedback. Downside: outputs sometimes mix languages and lack readable formatting.

**DeepSeek-R1 — four-stage pipeline to fix readability and generality:**

| Stage | What happens |
| ----- | ------------ |
| **1. Cold-start SFT** | Fine-tune on thousands of carefully formatted long CoT examples (collected from R1-Zero outputs + human refinement) to seed readable format |
| **2. Reasoning-oriented RL** | Same GRPO training as R1-Zero on the SFT-warmed model |
| **3. Rejection-sampling SFT** | Generate candidates from the RL checkpoint, keep correct ones; merge with supervised data on writing / factual QA / general tasks; retrain base from scratch on this combined set |
| **4. Second RL pass** | RL across *all* task types — reasoning and general chat — to align preferences and maintain quality |

DeepSeek-R1 matches **OpenAI o1** on many benchmarks. Smaller distilled versions (R1-Distill-Qwen-7B, etc.) transfer the reasoning behavior into much smaller models via SFT on R1's traces.

### 3.3 OpenAI o1 / o3 — the closed but public story

OpenAI describes o1 as trained with **large-scale reinforcement learning to think productively using its chain of thought** (OpenAI, 2024). Key verified points from their published material:

- Performance consistently improves with **more RL training** (train-time compute) and **more thinking** (test-time compute)
- The model learns to recognize mistakes, break down hard steps, try alternative strategies — *these behaviors emerge from RL*, not hand-coded search
- The raw chain of thought is **not shown to users** (competitive + safety reasons); users see a model-generated summary
- Exact reward design is undisclosed; the public description is consistent with verifiable outcome rewards on math / coding

**GPT-4o** is a different product: frontier general model with strong SFT and preference RLHF, excellent when combined with inference-time CoT. It is not primarily trained for extended internal deliberation.

### 3.4 Outcome vs process rewards

| Signal | Grades | Pros | Cons |
| ------ | ------ | ---- | ---- |
| **Outcome** | Final answer only | Simple; matches compilers and unit tests; no step labels needed | Sparse; a wrong reasoning path can accidentally reach the right answer |
| **Process (PRM)** | Intermediate steps | Denser feedback; can catch early mistakes before the final answer | Requires step-level labels or a separately trained PRM; expensive |

DeepSeek-R1-Zero uses **outcome rewards only** (accuracy + format), no PRM. OpenAI's "Let's Verify Step by Step" (Lightman et al., 2023) is an example of process-reward research applied to math.

## 4. The arc in one picture

```text
Pretrain on text      SFT + preference RL      Reasoning RL (verifiable rewards)
(read everything)  →  (imitate; grade by feel)  →  (practice; grade by correctness)
      |                        |                              |
  knowledge               good style                 search until check passes
  fluency                 instruction-following      self-correction
                          preference-aligned         emerges from RL, not demos
```

| Stage | Primary signal | What it optimises | Failure mode if you stop here |
| ----- | -------------- | ----------------- | ----------------------------- |
| Pretrain | Next-token loss on raw text | Language, knowledge | Knows a lot; unreliable on hard exams |
| SFT + preference RL | Demonstrations + human preference | Style, helpfulness, safety | Copies tutors; preferences ≠ correctness |
| Reasoning RL (GRPO / PPO) | Verifiable outcome (+ optional PRM) | Correctness on hard tasks | Reward hacking if verifier is weak; high train cost |

Inference-time methods (CoT prompts, self-consistency, self-refinement) still help **any** of these checkpoints. Training-time RL changes what the model does **by default** before you add those tricks.

## 5. Minimal reasoning-era training sketch

Conceptual loop (DeepSeek-R1-Zero-style outcome RL with GRPO):

```python
for question, gold in practice_set:            # programmatically checkable tasks
    group = sample_group(policy, question, N)  # N rollouts for the same question
    rewards = [verifier(trace, gold) for trace in group]  # accuracy + format

    # GRPO: normalise within group — no separate critic needed
    mean_r, std_r = mean(rewards), std(rewards)
    advantages = [(r - mean_r) / (std_r + 1e-8) for r in rewards]

    policy = rl_update(policy, group, advantages)  # e.g. PPO-style step
```

Replace `verifier` with a SymPy checker (math), a test executor (code), or a mix.  
The key: **rule-based correctness**, not a human preference model.

## 6. Trade-offs

1. **Train cost** — RL on long traces is far more expensive than another SFT epoch; GRPO helps but the compute is still large
2. **Verifier quality matters critically** — a weak or gameable verifier leads to reward hacking (fluent nonsense that passes a bad test)
3. **Overthinking** — models trained this way may generate very long traces even for simple problems; this is an active research area
4. **SFT cold start stabilises RL** — as R1-Zero showed, pure RL-from-base produces correct but unreadable / language-mixed outputs; a small SFT cold start helps
5. **Preference RL is still needed** — Era 3 models also go through a second RL stage on human preferences to remain general-purpose assistants; reasoning RL alone does not handle tone, safety, or off-topic queries
6. **Verifiable tasks only** — this approach works cleanly for math and code (clear right/wrong); open-ended tasks (writing, analysis) need hybrid reward designs

## 7. Summary

1. **Pretraining** — read the internet; knowledge without graded practice
2. **SFT + preference RLHF** (InstructGPT / GPT-4 / GPT-4o era) — imitate good solutions; optimise for how answers *feel* to humans; better style and helpfulness, but preference ≠ correctness
3. **Reasoning RL** (o1 / DeepSeek-R1 era) — questions + rule-based verifiers + GRPO/PPO; practice until the answer **checks out**; self-correction and extended deliberation emerge from the reward signal
4. **GRPO** avoids a critic by normalising rewards within a group of rollouts for the same question — cheaper and more stable for long CoT
5. **DeepSeek-R1** (Jan 2025, open weights) uses a four-stage pipeline: cold-start SFT → reasoning RL → rejection-sampling SFT → second RL; R1-Zero skips the cold start and shows reasoning emerges from outcome rewards alone
6. Pair with [inference-time scaling](/concepts/reasoning-models/inference-time-scaling): train the habit, then spend more decode compute when the problem is worth it

Next in this series: deeper **Chain of thought**, then **Search and verifiers**.
