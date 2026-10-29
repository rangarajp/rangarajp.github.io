---
title: "Quantization"
description: "Number formats, PTQ, QAT, activation outliers, and a runnable rounding experiment."
section: learn
pubDate: 2026-10-29
part: 3
chapter: 17
order: 17
tags: ["llm-inference", "systems", "hands-on"]
---

Quantization represents values with a smaller set of numbers. It reduces storage and sometimes data movement or arithmetic cost. It also introduces error; the useful result is the smallest representation that preserves the task quality you need.

## 1. Why quantization matters

Seven billion parameters require about 14 GB at two bytes each, 7 GB at one byte, or 3.5 GB at four bits before metadata. KV state, activations, workspace, and allocator overhead remain. Weight-only quantization does not automatically shrink the KV cache.

## 2. Number formats

FP16, BF16, and FP8 are floating-point families with different range and precision. INT8 and INT4 commonly use scales (and sometimes zero points) to map integers back to approximate real values. Grouped quantization gives different weight groups different scales. Smaller groups can reduce error but add metadata and kernel work.

For symmetric signed quantization, use $q=\operatorname{clip}(\operatorname{round}(x/s),-q_{max},q_{max})$ and reconstruct $\hat{x}=sq$. This toy uses the symmetric subset of the signed integer range.

```python
# Pure Python; no GPU, model download, or external packages.
weights = [-1.0, -0.31, 0.0, 0.12, 0.73, 0.95]
for bits in (8, 4):
    qmax = 2 ** (bits - 1) - 1
    scale = max(abs(w) for w in weights) / qmax or 1.0
    q = [max(-qmax, min(qmax, round(w / scale))) for w in weights]
    restored = [scale * n for n in q]
    mse = sum((a-b)**2 for a,b in zip(weights, restored)) / len(weights)
    print(bits, q, round(mse, 7))
```

The integer list demonstrates rounding; Python integers do not occupy four bits. Actual compression needs packed storage. Actual speed needs a kernel that consumes that packing efficiently.

## 3. Post-training quantization

*GPTQ* uses calibration activations and approximate second-order information to compensate quantization error across weights. *AWQ* uses activation information to choose channel scaling that protects important weights before quantization. Neither is simply rounding every tensor with one global scale.

*QuIP#* and *AQLM* investigate very low bit representations using transformations or codebooks. Compare their full runtime paths and quality, rather than treating a nominal bit width as a speed ranking. Calibration examples should resemble production prompts, languages, and code distributions.

## 4. Quantization-aware training

QAT simulates quantization effects during training so weights can adapt. It requires training resources and a deployment representation that matches the simulation. PTQ is often a cheaper initial experiment; QAT becomes relevant when its quality improvement justifies that extra work.

## 5. Activation outliers and validation

An unusually large channel can force a broad scale and waste integer levels on ordinary values. Per-channel scaling, smoothing transformations, clipping, or higher precision for sensitive operations address different versions of this problem. Clipping itself changes the computation.

Hold model, prompt set, concurrency, and output budget fixed. Compare memory, TTFT, decode latency, and task success against a higher-precision baseline. A checkpoint that fits but silently loses numerical accuracy is not a successful optimization.

Primary reading: [GPTQ](https://arxiv.org/abs/2210.17323) and [AWQ](https://arxiv.org/abs/2306.00978).
