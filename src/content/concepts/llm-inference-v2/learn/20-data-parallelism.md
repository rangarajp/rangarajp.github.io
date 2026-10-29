---
title: "Parallelism for large model inference"
description: "A model can exceed one device\u2019s capacity, or a service can outgrow one worker\u2019s throughput."
section: learn
pubDate: 2026-10-29
chapter: 9
order: 9
---

A model can exceed one device’s capacity, or a service can outgrow one worker’s throughput. Replicating a model and splitting a model address different constraints. Choose the parallel strategy by the state and computation that must be distributed, then account for communication.

## 9.1 Why parallelism is necessary

A model may need multiple devices because its weights and request state exceed one GPU's memory, or because a particular workload benefits from extra compute. Parallelism adds communication, so capacity gain and latency gain are separate claims.

Data parallel inference runs independent request streams against complete model replicas. Each replica can itself be a group of tensor-parallel GPUs. A router decides which replica receives each request; a replica's local scheduler then batches its assigned sequences.

### Same model, different requests

Four replicas can serve requests A–D concurrently without sharing each request's KV cache. There are no gradient all-reduces during ordinary inference. This differs from data-parallel training, where replicas synchronize gradients to maintain a common parameter update.

A model that does not fit on one device still needs model sharding or lower precision within each replica. Adding independent copies cannot split a single replica's weight footprint. Total resident weights across the fleet scale with replica count.

### Round-robin versus work-aware routing

```python
# CPU-only queue simulation. Costs are arbitrary work units, not latency predictions.
requests = [20, 2, 2, 20, 2, 2]
round_robin, least_work = [0, 0], [0, 0]
for i, cost in enumerate(requests):
    round_robin[i % 2] += cost
    worker = min(range(2), key=lambda j: least_work[j])
    least_work[worker] += cost
print('round robin:', round_robin, 'estimated work:', least_work)
```

Both policies see the same number of requests, but request costs differ. Production routing can consider prompt length, active output budgets, queue depth, free KV blocks, and prefix locality. Exact future answer length is unknown, so estimates need headroom and feedback.

### Capacity and failure

Two workers do not guarantee twice the useful throughput: shared CPU, storage, network, and cold model loading can limit the fleet. Test one-replica loss at peak traffic. Decide whether to reject excess work, degrade output budgets, or preserve latency for a smaller admitted load.

Compare [tensor parallelism](./data-parallelism), [pipeline parallelism](./data-parallelism), and [multi-model routing](./observability-slos). The original working notes' distinction remains useful: replicas split requests; tensor parallelism splits operations inside a layer; pipeline parallelism splits layer depth.

## 9.2 Tensor parallelism

### Tensor parallelism

Split a layer's matrices across devices. Column-parallel projections produce output shards; row-parallel projections combine partial sums. Attention heads and MLP intermediate channels provide natural partitions. Exact placement depends on architecture and implementation.

```python
# CPU PyTorch: algebra behind two output shards, not distributed execution.
import torch
torch.manual_seed(7)
x, w = torch.randn(3, 8), torch.randn(8, 12)
left, right = w.chunk(2, dim=1)
y = torch.cat([x @ left, x @ right], dim=1)
assert torch.allclose(y, x @ w, atol=1e-6)
# Split the reduction dimension instead: partial outputs must be summed.
xa, xb = x.chunk(2, dim=1)
wa, wb = w.chunk(2, dim=0)
assert torch.allclose(xa @ wa + xb @ wb, x @ w, atol=1e-6)
```

The second case illustrates why a collective reduction is needed. Actual latency depends on message size, interconnect, synchronization, and work available between collectives.

### Choose the parallel axis

| Strategy | What is split | Main cost |
|---|---|---|
| Replicas | Independent requests | Extra copies of weights |
| Tensor parallelism | Work within layers | Frequent collectives |
| Pipeline parallelism | Layer depth | Stage transfers and bubbles |
| Expert parallelism | Expert placement | Token dispatch and load imbalance |
| Context parallelism | Sequence positions | Attention-state communication |

Replicas improve aggregate capacity but do not make an oversized model fit per replica. Tensor parallelism is most attractive with a fast interconnect. Continue with [pipeline](./data-parallelism), [MoE](./data-parallelism), and [long context](./context-parallelism).

### Multi-GPU communication

On-chip HBM is not the only roof. When a matmul is split across GPUs, *partial results* must cross NVLink / InfiniBand, and that link has its own bandwidth.

Split the $D$ dimension across two devices:

$$
Z_0 = X[:, :D/2]\, Y[:D/2, :], \qquad
Z_1 = X[:, D/2:]\, Y[D/2:, :]
$$

then exchange and add the partial $Z$ tiles. Assume the shards of $X$ and $Y$ already live on each GPU, so a simplified per-rank exchange payload is one BF16 partial output — $2BF$ bytes per rank — over a link of $45$ GB/s with per-GPU compute $1.97 \times 10^{14}$ FLOPs/s:

$$
T_{\text{math}} = \frac{BDF}{1.97 \times 10^{14}}, \qquad
T_{\text{comm}} = \frac{2BF}{4.5 \times 10^{10}}
$$

The compute term exceeds this ideal payload-transfer term when $T_{\text{math}} > T_{\text{comm}}$:

$$
\frac{D}{2} > \frac{1.97 \times 10^{14}}{4.5 \times 10^{10}}
\quad\Rightarrow\quad
D \gtrsim 8756
$$

Larger inner dimension → more local FLOPs per byte shipped. Actual collectives also incur startup latency, algorithm-dependent byte counts, and synchronization. Overlap is not guaranteed. Tensor-parallel width trades per-device memory for communication; benchmark the collective on the actual topology.

#### Decode floor for a sharded 70B

Llama-70B BF16 is $\approx 140$ GB of weights → $17.5$ GB per GPU on 8-way tensor parallel. Each decode step must touch those weights once:

$$
T_{\text{decode}} \ge \frac{17.5\ \text{GB}}{3.35\ \text{TB/s}} \approx 5.2\ \text{ms}
$$

That is an ideal lower bound under the assumption of streaming every weight byte once from HBM (before attention KV traffic or network). Pushing the same 17.5 GB gradient shard over NVLink at $900$ GB/s takes $\approx 19$ ms; over a $50$ GB/s link it jumps to $\approx 350$ ms — interconnect math for training-style collectives, same bandwidth thinking.

## 9.3 Pipeline parallelism

Pipeline parallelism assigns groups of layers to different devices. Activations move between stages while weights stay on their assigned device. It reduces per-device weight storage without requiring every layer to run across every GPU.

### Why bubbles appear

A stage cannot process a microbatch until the previous stage has produced its input. Under an ideal equal-stage schedule with $p$ stages and $m$ independent microbatches, utilization is approximately $m/(m+p-1)$ for a forward pipeline.

```python
stages = 4
for microbatches in (1, 4, 16):
    utilization = microbatches / (microbatches + stages - 1)
    print(microbatches, round(utilization, 3))
```

One microbatch uses only a quarter of the pipeline's stage-time capacity. More independent work fills it, but a single autoregressive stream still needs its next token before beginning its next step. Pipeline throughput and single-user decode latency therefore need separate measurements.

### Balance the stages

Equal layer counts do not guarantee equal work: embeddings, output projections, attention lengths, and MoE routing can differ. The slowest stage sets the steady-state pace. Account for activation transfer and runtime workspace as well as weights. The formula above ignores imbalance, communication, and scheduler interference.

## 9.4 Mixture of experts inference

A mixture-of-experts (MoE) layer routes each token to a subset of expert feed-forward networks. Attention and shared components still run. Fewer active parameters per token can reduce arithmetic relative to activating every expert, but all potentially selected experts must remain accessible somewhere.

### Total versus active parameters

A model with many total parameters and a small active subset is not equivalent in memory to a dense model of the active size. It also pays routing, dispatch, and combination costs. Expert parallelism distributes experts across devices, often sending token activations through an all-to-all exchange.

For selected experts $\mathcal{K}(x)$ and their routing weights $g_k(x)$, the expert contribution can be expressed as

$$
y(x)=\sum_{k\in\mathcal{K}(x)}g_k(x)\,E_k(x).
$$

The router selects computation, not free memory. A rarely selected expert still needs accessible weights, and moving a token to its device adds dispatch and combination traffic. Offloading experts to host memory or storage trades resident device capacity for loading latency and bandwidth. Report resident state and cold expert access separately from active arithmetic.

### Inspect routing balance

```python
from collections import Counter
# Toy top-2 routes for six tokens, four experts.
routes = [(0, 1), (0, 2), (0, 1), (0, 3), (1, 2), (0, 1)]
loads = Counter(e for selected in routes for e in selected)
mean = sum(loads.values()) / 4
print(dict(loads), 'max/mean:', max(loads.values()) / mean)
```

A busy expert can delay the iteration while another device waits. In production, monitor dispatch bytes, expert load, padding, and any overflow policy. Dropping tokens to enforce capacity can change outputs and is not a transparent optimization.

### Combine parallel strategies

Tensor parallelism can split work inside experts while expert parallelism distributes expert groups. The best layout depends on batch size, routing balance, and topology. Report total weight footprint and active computation separately when comparing MoE with dense models.

Primary reading: [DeepSeek-V2](https://arxiv.org/abs/2405.04434).

## Recorded notebook observations

These observations come from the saved notebook runs linked below. Their model, timing boundary, and hardware assumptions are part of the result.

### Existing sharding notebook: separate algebra from distributed execution

The [sharding notebook](https://github.com/rangarajp/rangarajp.github.io/blob/main/notebooks/llm-inference/gpu_fundamentals/sharding.ipynb) provides the companion path from matrix partitioning to real multi-GPU work. Its distributed section explicitly requires at least two CUDA devices and a `torchrun` launch. A one-V100 algebra demonstration can verify reconstruction, but it cannot measure cross-device collectives or tensor-parallel scaling.

## Hands-on extensions

Use these experiments to connect the mechanisms above to measurements. They are proposed exercises unless a recorded result is explicitly identified.

### Hands-on: compare routing policies before adding replicas

Extend the CPU queue simulation with measured service-time samples and uneven request lengths. Publish per-replica work and tail latency. A real two-replica GPU throughput/failure experiment is a later step that needs two devices or two independent worker allocations.

### Hands-on: verify sharding algebra, then communication

Use the existing sharding notebook to reconstruct a matrix result from row and column shards. Publish numerical error on one device first. With at least two GPUs, separately measure compute, collective communication, and complete step time; report the interconnect.

### Hands-on: draw a stage utilization timeline

Simulate two stages with unequal service times and several microbatch counts. Publish occupied and idle stage intervals and compare utilization with the ideal formula. Claim a measured multi-GPU benefit only after running an actual distributed pipeline.

### Hands-on: stress uneven routing

Extend the top-2 routing example with balanced and skewed token assignments. Publish expert loads, maximum-to-mean load, and estimated dispatch bytes. This CPU simulation demonstrates imbalance; distributed all-to-all timing requires multiple devices.
