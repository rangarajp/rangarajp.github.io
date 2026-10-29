---
title: "The future of inference"
description: "Diffusion, multi-token prediction, alternative hardware, and inference-first design."
section: learn
pubDate: 2026-10-29
part: 6
chapter: 32
order: 32
tags: ["llm-inference", "systems", "hands-on"]
---

The earlier chapters provide a comparison framework for new methods. Evaluate alternative generation mechanisms and hardware with the same latency, memory, energy, and quality definitions.

Future inference methods should be evaluated against the same contract as today's engines: useful outputs, resource use, latency, and quality under a defined workload. A new generation mechanism changes the work rather than making it disappear.

## 1. Diffusion-based language models

Masked diffusion language models iteratively refine token sequences. They can update multiple positions in a step and naturally support infilling. Repeated full-sequence passes can offset that parallelism, so compare total work and latency at matched quality. Autoregressive KV reuse does not transfer unchanged to a sequence whose tokens keep changing.

## 2. Multi-token prediction

Training additional prediction heads can improve representations and provide draft candidates at inference. The heads do not guarantee that every future prediction is correct. Verification, acceptance, and the extra head cost determine whether they improve latency. Relate this to [speculative decoding](./speculative-decoding).

## 3. Compare total work

```python
# Illustrative token-position evaluations, not FLOPs or speed measurements.
length = 256
for denoising_steps in (4, 16, 64):
    print(denoising_steps, 'steps:', length * denoising_steps, 'position evaluations')
print('autoregressive output positions:', length)
```

A parallel method may evaluate more positions but finish sooner on suitable hardware. These counts alone cannot rank speed: attention cost, reuse, kernels, and model size differ.

## 4. Neuromorphic and photonic computing

Alternative hardware explores different energy and data-movement tradeoffs. Optical computation still needs data conversion, storage, control, and interconnects; spiking systems need compatible representations and algorithms. Neither makes weight delivery free. Evaluate full-system measurements on the target model rather than extrapolating an isolated operation's efficiency.

## 5. Inference-first design

GQA, MLA, recurrent-state layers, conditional experts, and adaptive depth make serving constraints part of architecture design. Each has quality, training, and kernel implications. Byte-level tokenization changes sequence length and vocabulary costs rather than eliminating computation. Choose designs by useful work per byte and joule for the intended workload.

Primary reading: [LLaDA](https://arxiv.org/abs/2502.09992), [multi-token prediction](https://arxiv.org/abs/2404.19737), and [Mixture-of-Depths](https://arxiv.org/abs/2404.02258).

## Hands-on: compare one alternative at matched quality

Choose one compatible multi-token or masked-generation model and define a small common task set. Publish latency, generated work, memory, and task success under equal budgets. Keep theoretical position-evaluation counts separate from a measured model comparison.
