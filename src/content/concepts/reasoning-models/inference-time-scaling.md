---
title: 'Inference-Time Scaling'
description: 'Improving reasoning at decode time — CoT, sampling, self-consistency, Best-of-N, and self-refinement — without retraining.'
pubDate: 'Sep 5 2026'
order: 2
heroImage: '../../../assets/blog-placeholder-3.jpg'
---

Reasoning ability and answer accuracy can improve without retraining by spending more compute at decode time. That idea is inference-time scaling: same weights, more tokens, samples, search, or revise loops while answering.

The central trade-off is simple — higher accuracy in exchange for more compute (latency, tokens, wall-clock time).

Toy traces below are from `Qwen/Qwen3-0.6B-Base` in `notebooks/reasoning-models/inference-time-scaling.ipynb`. Each experiment highlights what changed (prompt vs decode knobs vs aggregation vs refine). Broader benchmark notes (e.g. MATH-500) sit beside those runs where they matter.

## 1. Two ways to improve reasoning

1. Inference-time — keep the same trained model; spend more tokens, samples, or search at decode time
2. Training-time — change the model itself (SFT, distillation, RL) so better reasoning is baked into the weights

Inference-time methods come first in practice: no retraining, easy to A/B test, and an explicit cost–quality curve.

## 2. What happens when you sample

At each step the model outputs logits over the vocabulary. Next-token probabilities come from softmax over those logits. Sampling strategies only change *how* you draw from that distribution — not the weights.

| Knob | Role |
| ---- | ---- |
| Temperature | Scales logits before softmax — higher → flatter / more diverse text; lower → peakier / more deterministic |
| Top-k | Keep only the *k* highest-probability tokens, then sample |
| Top-p (nucleus) | Keep the smallest set of tokens whose probabilities sum to ≥ *p*; filters low-mass tail tokens that often produce nonsense |

A small helper lets you plug these in without rewriting callers (notebook: `generate` / `generate_scored`):

```python
def generate(prompt, *, do_sample=False, temperature=0.8, top_k=None, top_p=None, num_return_sequences=1):
    # logits → (optional temp / top-k / top-p) → sample or greedy → tokens
    ...

def generate_scored(...):
    # same, but also return mean log-prob of generated tokens
    ...
```

## 3. Parallel experiments — what we change

Shared setup: one model, one chocolate problem. Correct answer 6.

```python
PROBLEM = (
    "Ram has 3 chocolates. Sam has 5 chocolates. "
    "They put them in one basket and each eats 1. "
    "How many chocolates are left?"
)
```

### 3.1 Experiment A — baseline (direct ask)

Short prompt, greedy decode.

```python
prompt_direct = f"Question: {PROBLEM}\nAnswer:"
out = generate(prompt_direct, max_new_tokens=40, do_sample=False)[0]
# → parsed answer 4 (wrong)
```

Fast, but wrong (4). Everything below keeps the same weights and spends more decode compute.

### 3.2 Experiment B — Chain of thought (prompt change)

The prompt only — still greedy. "Explain step by step" forces intermediate tokens (more runtime, often better accuracy).

```python
prompt_cot = (
    f"Question: {PROBLEM}\n"
    "Explain step by step and think, then write Answer: <number>.\n"
    "Step 1:"
)
out = generate(prompt_cot, max_new_tokens=120, do_sample=False)[0]
# → parsed answer 6 (correct)
```

### 3.3 Experiment C — top-k + top-p sampling (decode change)

Same CoT prompt; turn on sampling. Goal is diversity for later vote / select / refine.

```python
outs = generate(
    prompt_cot,
    max_new_tokens=120,
    do_sample=True,
    temperature=0.8,
    top_k=40,
    top_p=0.9,
    num_return_sequences=4,
)
```

### 3.4 Experiment D — Self-consistency (aggregate answers)

N samples + majority vote on extracted finals (Wang et al.).

```python
samples = generate(prompt_cot, do_sample=True, top_k=40, top_p=0.9, num_return_sequences=N)
answers = [extract_answer(t) for t in samples]
majority = Counter(a for a in answers if a).most_common(1)[0][0]
# example: Votes {6: 2, 4: 1, 7: 1, 3: 1} → 6
```

### 3.5 Experiment E — Best-of-N (select one trace)

Same samples; score and pick one path instead of voting. Scorer quality matters (length alone often fails).

```python
best = max(samples, key=score_length)  # toy — often wrong
```

### 3.6 Experiment F — MATH-500 (same recipe, larger set)

Evaluation scale — CoT + self-consistency over MATH-500. Typical finding: accuracy rises vs a no-sampling baseline; runtime grows with N and trace length.

## 4. Self-refinement — correct, then revise

Parallel methods draw many answers at once. Self-refinement is sequential:

1. Draft an answer
2. Score / critique it (or rank candidates)
3. Revise if needed — optionally loop

You can plug different correctors / scorers into the same loop:

| Strategy | What it does | Strength | Weakness |
| -------- | ------------ | -------- | -------- |
| Rule-based | Cheap checks (parseable answer, arithmetic patterns in the trace) | Fast, deterministic | Brittle; domain-specific rules |
| Length | Prefer longer completions | Trivial to implement | Length ≠ correctness |
| Avg log-prob | Prefer sequences with higher mean token log-prob | Uses the model's own uncertainty | Confident ≠ correct; length bias |
| LLM-as-judge | Same (or stronger) model picks / critiques candidates | Flexible natural-language criteria | Extra calls; small models are noisy judges |
| Refine loop | Critique → rewrite with feedback | Can fix a bad draft without N samples | Error can compound; more latency |

Best-of-N is "score once, pick once." Self-refinement is "score/critique, then generate again with that signal."

### 4.1 Experiment G — Compare scorers on the same candidates

After sampling, rank with rule / length / avg log-prob and compare picks.

```python
scored = generate_scored(prompt_cot, do_sample=True, top_k=40, top_p=0.9, num_return_sequences=5)
texts = [t for t, _ in scored]

def score_rule_based(text: str) -> float:
    # no ground truth — reward structure: 3+5, total 8, 8-2, explicit Answer:
    ...

pick_len  = max(texts, key=score_length)
pick_rule = max(texts, key=score_rule_based)
pick_lp   = max(scored, key=lambda p: p[1])[0]  # highest avg log-prob
```

Same candidates, different correctors → often different selected answers. That is why the scorer is part of the method, not an afterthought.

### 4.2 Experiment H — LLM-as-judge

Ask the model to output `Judge: <n>` over short candidate summaries.

```python
judge_prompt = (
    f"Question: {PROBLEM}\n"
    "Pick the best candidate. Reply with only: Judge: <number>\n\n"
    + format_candidates(texts)
    + "\n\nJudge:"
)
judge_out = generate(judge_prompt, max_new_tokens=16, do_sample=False)[0]
```

Tiny base models are unreliable judges; production systems often use a stronger judge or a trained reward model. The pattern is the same: extra decode compute to select.

### 4.3 Experiment I — Critique → revise (one round)

Start from a weak draft, generate a critique, then regenerate with that feedback.

```python
draft = generate(f"Question: {PROBLEM}\nAnswer:", do_sample=False)[0]

critique = generate(
    f"Question: {PROBLEM}\nDraft:\n{draft}\n\n"
    "Critique the arithmetic. End with Verdict: OK or Verdict: REVISE\nCritique:",
    do_sample=False,
)[0]

if "REVISE" in critique.upper() or score_rule_based(draft) < 3:
    refined = generate(
        f"Question: {PROBLEM}\nPrevious draft:\n{draft}\nFeedback:\n{critique}\n\n"
        "Corrected step-by-step solution. Answer: <number>\nStep 1:",
        max_new_tokens=120,
        do_sample=False,
    )[0]
```

Compared to Best-of-N: the compute goes into a second generation conditioned on feedback, not only ranking fixed samples.

### 4.4 Experiment J — Multi-round refine

Repeat critique → revise for a few rounds; keep the best state by a scorer (here: rule-based).

```python
def self_refine(problem: str, rounds: int = 2):
    state = generate(f"Question: {problem}\nAnswer:", do_sample=False)[0]
    best, best_score = state, score_rule_based(state)
    for _ in range(rounds):
        critique = generate(critique_prompt(problem, state), do_sample=False)[0]
        state = generate(revise_prompt(problem, state, critique), do_sample=False)[0]
        if score_rule_based(state) >= best_score:
            best, best_score = state, score_rule_based(state)
    return best
```

Stop early when `Verdict: OK` or when the rule score stops improving — another accuracy↔compute dial.

## 5. At a glance

| Exp | Experiment | Family | What we change | Code focus |
| --- | ---------- | ------ | -------------- | ---------- |
| A | Baseline (direct ask) | Baseline | — | greedy short prompt |
| B | Chain of thought | CoT | Prompt | "Explain step by step" |
| C | Top-k + top-p sampling | Sampling | Decode | `do_sample`, `top_k`, `top_p` |
| D | Self-consistency | Parallel | Aggregate | majority vote |
| E | Best-of-N | Parallel | Select | `max(..., key=score)` |
| F | MATH-500 | Scale | Dataset | MATH-500 + CoT + vote |
| G | Compare scorers | Refine / select | Scorer | rule vs length vs avg log-prob |
| H | LLM-as-judge | Refine / select | Judge | LLM-as-judge prompt |
| I | Critique → revise | Refine | Feedback | critique → revise |
| J | Multi-round refine | Refine | Loop | multi-round `self_refine` |

## 6. Summary

1. Inference-time scaling improves reasoning without retraining by spending more decode compute
2. Softmax(logits) + temperature / top-k / top-p reshape sampling behind one flexible generate helper
3. Parallel path: sample many → vote (self-consistency) or score (Best-of-N)
4. Sequential path: draft → rule / length / avg log-prob / LLM-as-judge → revise (self-refinement)
5. Scorers and judges are part of the method — a bad scorer (e.g. length-only) can undo good samples
6. Toy runs and MATH-500-style evals share the trade-off: accuracy up, compute up
7. Next in this series: deeper chain of thought, then training-time methods, search, and verifiers
