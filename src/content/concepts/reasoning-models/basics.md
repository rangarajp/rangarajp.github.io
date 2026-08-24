---
title: 'Basics'
description: 'Conventional LLM training, what reasoning means, and how it is improved.'
pubDate: 'Aug 24 2026'
order: 1
heroImage: '../../../assets/blog-placeholder-3.jpg'
---

Standard LLMs are strong at statistical pattern recognition. Reasoning methods sit on top of that stack — they push the model to generate intermediate steps before a final answer, which often helps on complex tasks such as coding and math.

## 1. Conventional LLM training

Conventional LLM training occurs in several stages:

1. **Pretraining** — the model learns language patterns from vast amounts of text (books, websites, articles, and more) via next-token prediction
2. **Instruction fine-tuning** — improves how the model responds to user prompts
3. **Preference tuning** — aligns outputs with human preferences (style, helpfulness, safety)

Reasoning methods are applied **on top of** a conventional LLM — after these stages have already produced a capable language model.

### 1.1 Same user message, two training stages

**User:**

```
How do I boil an egg?
```

**Pretrained LLM** (next-token completion — not yet a chat assistant):

```
How do I boil an egg? How do I scramble an egg? How do I poach an egg?
For more recipes, see chapter 4. Soft-boiled eggs are popular in…
```

It continues the text like a webpage or book — more questions, lists, or related prose — because it was trained to predict the next token, not to help a user.

**Post-trained LLM** (instruction + preference tuning):

```
User: How do I boil an egg?

Assistant: Place eggs in a pot, cover with water, bring to a boil,
then simmer 6–7 minutes for soft or 9–12 for hard. Cool under cold
water, peel, and serve.
```

It treats the input as an instruction and replies as a helpful assistant — clear steps, stops when done, matches preferred style.

Pretraining builds language knowledge; post-training teaches the model to *use* that knowledge in a conversation.

## 2. What reasoning means in LLMs

Reasoning in LLMs means improving a model so that it **explicitly generates intermediate steps** (chain of thought) before producing a final answer. That often increases accuracy on multistep tasks.

Standard chat models often jump straight to an answer. Reasoning models are optimized to spend extra test-time compute on those intermediate tokens when the problem needs it.

### 2.1 Example: share and count

**Problem:** Ram has 3 chocolates. Sam has 5 chocolates. They put them in one basket and each eats 1. How many chocolates are left?

**Without reasoning** (instant answer):

```
Answer: 6
```

Lucky guess — or a wrong one. No checkable path.

**With reasoning** (step by step):

```
Step 1: Ram has 3 chocolates.
Step 2: Sam has 5 chocolates.
Step 3: Together in the basket: 3 + 5 = 8 chocolates.
Step 4: They each eat 1 → 2 chocolates eaten.
Step 5: Left in the basket: 8 − 2 = 6 chocolates.
Answer: 6
```

Same final number, but the model wrote intermediate tokens that make the answer verifiable. That chain of steps is what reasoning methods aim to produce.

## 3. Pattern matching, not rules

Reasoning in LLMs is different from rule-based reasoning, and it likely also works differently from human reasoning. The current consensus is that reasoning in LLMs still relies on **statistical pattern matching**.

Pattern matching here means statistical associations learned from data: fluent text generation without explicit logical inference engines or hard-coded rules. The “steps” look like reasoning; under the hood they are still next-token predictions shaped by training.

## 4. Ways to improve reasoning

Improving reasoning in LLMs can be done in a few ways:

1. **Inference-time compute scaling** — improve reasoning without retraining (for example, chain-of-thought prompting, more samples, or search at decode time)
2. **Reinforcement learning** — train models explicitly with reward signals for better traces or final answers
3. **Supervised fine-tuning and distillation** — train on examples from stronger reasoning models

Building reasoning models from scratch is useful in practice: it surfaces capabilities, limitations, and computational trade-offs more clearly than using a black-box API alone.

## 5. Summary

1. Conventional training = pretraining → instruction fine-tuning → preference tuning
2. Reasoning methods sit on top of that conventional LLM
3. Reasoning here means generating intermediate steps (chain of thought) before the final answer
4. It is not rule-based logic — it is still statistical pattern matching
5. You can improve it via inference-time compute, RL, or SFT / distillation from stronger models
