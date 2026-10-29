---
title: "The inference problem"
description: "A reading map from token generation to production, with latency and bandwidth estimates."
section: learn
pubDate: 2026-10-29
part: 1
chapter: 0
order: 0
tags: ["llm-inference", "systems", "hands-on"]
---

# The inference problem

A model produces token scores. An inference system must turn those scores into useful answers while sharing finite memory, compute, and time across users. This guide follows the topic progression of Lamhot Siagian's *Efficient LLM Inference: Systems, Algorithms & Production Engineering*, second edition (2026), with original explanations and coding experiments adapted to this blog. Its interview material is omitted.

## 1. Training versus inference

Training computes a loss, backpropagates gradients, and updates parameters. Ordinary inference freezes those parameters. It still allocates temporary activations and sequence state, so fitting the weights is only the first capacity check.

Training often favors large batches and sustained throughput. Interactive inference also cares about queueing and the time each user waits. Offline inference can favor throughput instead. These are workload choices, rather than properties of the word inference.

## 2. The token generation loop

Read [text to tokens](./from-text-to-tokens), [the forward pass](./transformer-forward-pass), [sampling](./logits-and-sampling), and [autoregressive generation](./autoregressive-generation) as one worked sequence. Prefill computes the prompt and selects the first output token; subsequent decode passes consume previously selected tokens. [Prefill and KV caching](./prefill-decode-kv-cache) explains how to avoid projecting old tokens again.

## 3. Latency, throughput, and cost

Time to first token (TTFT) includes queueing, preprocessing, prefill, sampling, and delivery. Inter-token latency measures gaps in the stream. Throughput counts tokens across requests over wall time. Goodput counts work delivered within a specified service objective. Always state whether a throughput number counts input tokens, output tokens, or both.

```python
# A hypothetical request, in seconds; this is arithmetic, not a benchmark.
queue, preprocess, prefill, first_delivery = 0.020, 0.005, 0.080, 0.005
output_tokens, decode_step = 101, 0.010
ttft = queue + preprocess + prefill + first_delivery
end_to_end = ttft + (output_tokens - 1) * decode_step
print(round(ttft, 3), round(end_to_end, 3))  # 0.11, 1.11
```

There are 100 gaps after the first of 101 tokens. Doubling answer length mostly increases decode time; doubling prompt length changes prefill and cache requirements. Batching can improve efficiency, but increases waiting or step duration in some regimes. Measure both latency and throughput.

## 4. Memory bandwidth and the roofline

For a dense model with $P$ parameters, a useful first approximation is $2P$ forward FLOPs per processed token, excluding attention and other overhead. A single FP16 weight read costs $2P$ bytes. At batch one that is roughly one FLOP per weight byte: a bandwidth-heavy operation.

```python
parameters = 7e9
weight_bytes = parameters * 2
bandwidth = 2e12  # illustrative bytes/s, not measured bandwidth
print(weight_bytes / bandwidth * 1000)  # 7 ms, ideal weight-read bound
```

This is a lower bound under a full weight-streaming assumption, not a predicted token latency. The roofline bound is `min(peak_flops, intensity * bandwidth)`. Batch reuse raises intensity; long-context attention adds work and cache traffic. A bottleneck can move as concurrency and context change.

## 5. Read the PDF topics through this blog

| PDF topic | Blog chapters and experiments |
|---|---|
| The inference problem | This chapter; request timing and bandwidth bounds |
| Hardware foundations | [GPU fundamentals](./gpu-fundamentals); memory hierarchy |
| Transformer mechanics | Existing chapters 1–6; tensor shapes and generation |
| Quantization | [Quantization](./quantization); integer rounding and reconstruction |
| Speculative decoding | [Draft and verify](./speculative-decoding); acceptance and cost |
| KV cache optimization | [KV cache](./kv-cache) and [PagedAttention](./paged-attention) |
| Kernels and FlashAttention | [FlashAttention](./flashattention) and [kernel engineering](./kernel-fusion-cuda-graphs) |
| Serving architecture | [Continuous batching](./continuous-batching); prefill/decode scheduling |
| Parallelism and MoE | [Tensor parallelism](./tensor-parallelism), [pipeline parallelism](./pipeline-parallelism), [expert parallelism](./expert-parallelism) |
| Long context | [Context parallelism](./context-parallelism); compression and recurrent state |
| Edge inference | [On-device inference](./edge-inference); a memory budget |
| Inference-time scaling | [Compute budgets](./inference-time-scaling); verified candidates |
| Observability and production | [Metrics and SLOs](./observability-slos), [capacity and cost](./capacity-and-cost) |
| Future directions | [The future of inference](./future-of-inference); useful work per byte |

The examples distinguish executable CPU experiments from deployment concepts. GPU kernels and engine features require compatible hardware and runtime versions. No speedup in this guide should be read as universal.
