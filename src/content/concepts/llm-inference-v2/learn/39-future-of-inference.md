---
title: "The future of inference"
description: "New generation methods and hardware should be evaluated against the same question: how much useful output do they deliver within a latency, memory, energy, and quality budget? This chapter considers changes to the generation algorithm and to the architecture itself."
section: learn
pubDate: 2026-10-29
chapter: 14
order: 14
---

New generation methods and hardware should be evaluated against the same question: how much useful output do they deliver within a latency, memory, energy, and quality budget? This chapter considers changes to the generation algorithm and to the architecture itself.

## 14.1 Diffusion-based language models

Masked diffusion language models iteratively refine token sequences. They can update multiple positions in a step and naturally support infilling. Repeated full-sequence passes can offset that parallelism, so compare total work and latency at matched quality. Autoregressive KV reuse does not transfer unchanged to a sequence whose tokens keep changing.

## 14.2 Multi-token prediction

### Multi-token prediction

Training additional prediction heads can improve representations and provide draft candidates at inference. The heads do not guarantee that every future prediction is correct. Verification, acceptance, and the extra head cost determine whether they improve latency. Relate this to [speculative decoding](./speculative-decoding).

### Compare total work

```python
# Illustrative token-position evaluations, not FLOPs or speed measurements.
length = 256
for denoising_steps in (4, 16, 64):
    print(denoising_steps, 'steps:', length * denoising_steps, 'position evaluations')
print('autoregressive output positions:', length)
```

A parallel method may evaluate more positions but finish sooner on suitable hardware. These counts alone cannot rank speed: attention cost, reuse, kernels, and model size differ.

## 14.3 Neuromorphic and photonic computing

Alternative hardware explores different energy and data-movement tradeoffs. Optical computation still needs data conversion, storage, control, and interconnects; spiking systems need compatible representations and algorithms. Neither makes weight delivery free. Evaluate full-system measurements on the target model rather than extrapolating an isolated operation's efficiency.

## 14.4 The inference-first design philosophy

GQA, MLA, recurrent-state layers, conditional experts, and adaptive depth make serving constraints part of architecture design. Each has quality, training, and kernel implications. Byte-level tokenization changes sequence length and vocabulary costs rather than eliminating computation. Choose designs by useful work per byte and joule for the intended workload.

Adaptive-depth approaches allocate different amounts of layer computation to different tokens or inputs. Early exit stops at a chosen intermediate representation; mixture-of-depths-style routing can skip some updates while continuing through later layers. Both need training, calibration, and a runtime that can exploit the changed workload without losing the quality target.

Activation sparsity is useful when the zero or inactive pattern can be identified cheaply and a kernel avoids corresponding work. Unstructured sparsity that still executes dense matrix operations may provide little latency benefit. Likewise, byte-level modeling changes vocabulary and sequence-length tradeoffs; it does not simply remove the cost of representing text.

These ideas share an inference-first question: which work and state are actually necessary for the intended task? The answer must include training changes, numerical behavior, data movement, and scheduler efficiency, not only a smaller theoretical operation count.

Primary reading: [LLaDA](https://arxiv.org/abs/2502.09992), [multi-token prediction](https://arxiv.org/abs/2404.19737), and [Mixture-of-Depths](https://arxiv.org/abs/2404.02258).

## Hands-on extensions

Use these experiments to connect the mechanisms above to measurements. They are proposed exercises unless a recorded result is explicitly identified.

### Hands-on: compare one alternative at matched quality

Choose one compatible multi-token or masked-generation model and define a small common task set. Publish latency, generated work, memory, and task success under equal budgets. Keep theoretical position-evaluation counts separate from a measured model comparison.
