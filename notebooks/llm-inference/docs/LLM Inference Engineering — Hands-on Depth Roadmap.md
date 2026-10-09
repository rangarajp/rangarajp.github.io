# LLM Inference Engineering — Hands-on Depth Roadmap

## Goal

I have completed the conceptual foundation of LLM inference and serving through:

- *Inference Engineering* — Phillip Keily
- *Hands-On LLM Serving* — Chi Wang
- My learning notes: `https://rangarajp.github.io/concepts/llm-inference/`

I already have a working multi-model serving architecture:

```text
Client
  │
  ▼
FastAPI Gateway :8000
  │
  ├──► Qwen2.5-7B / vLLM :8001
  │
  └──► TinyLlama-1.1B / vLLM :8002
                         │
                         ▼
                   NVIDIA V100 32GB
```

The objective now is **not to learn more definitions**.

The objective is to become capable of answering:

> Given this workload, model and GPU, why am I getting this latency, throughput and GPU utilization — and what should I change?

The learning methodology will be:

```text
Measure
   ↓
Explain
   ↓
Predict
   ↓
Modify
   ↓
Measure Again
```

---

# Phase 1 — Build Proper Inference Observability

## Objective

Upgrade the existing benchmark and serving architecture so that inference performance can be understood quantitatively.

I currently measure:

- End-to-end latency
- Prompt tokens
- Completion tokens
- Aggregate tokens/sec

Extend this to measure:

### Request Metrics

- End-to-end latency
- Time to First Token (TTFT)
- Inter Token Latency (ITL)
- Time Per Output Token (TPOT)
- Prompt tokens
- Completion tokens
- Queue/waiting time
- Prefill time
- Decode time

### Serving Metrics

- Requests running
- Requests waiting
- Request throughput
- Input token throughput
- Output token throughput
- Batch size / active sequences

### GPU Metrics

- GPU utilization
- GPU memory utilization
- GPU memory bandwidth where measurable
- KV-cache utilization

## Desired Benchmark Output

Produce results similar to:

| Concurrent Requests | TTFT | TPOT | Throughput | GPU Util | KV Cache | Queue Time |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | | | | | | |
| 2 | | | | | | |
| 4 | | | | | | |
| 8 | | | | | | |
| 16 | | | | | | |

The goal is not merely to collect numbers.

For every benchmark result, answer:

> Why did the metrics change?

---

# Phase 2 — Controlled Performance Experiments

## Experiment 1 — Concurrency Sweep

Run the same workload with:

```text
1
2
4
8
16
32
```

concurrent requests, stopping when the current hardware/configuration reaches saturation.

Measure:

- TTFT
- TPOT
- E2E latency
- Throughput
- GPU utilization
- KV-cache utilization
- Queue time

Before running the experiment, write a prediction.

Example:

> I expect throughput to initially increase because continuous batching improves GPU utilization. After saturation, throughput should flatten while queue time and TTFT increase rapidly.

Then compare prediction vs measurement.

---

## Experiment 2 — Prompt Length

Keep output length approximately constant.

Test different prompt lengths:

```text
10 tokens
50 tokens
100 tokens
250 tokens
500 tokens
```

Observe:

```text
Prompt Length
      ↓
Prefill Compute
      ↓
TTFT
      ↓
GPU Utilization
```

Determine which metrics are affected strongly and which remain relatively stable.

---

## Experiment 3 — Output Length

Keep prompt approximately constant.

Test:

```text
10 output tokens
50 output tokens
100 output tokens
250 output tokens
```

Study the relationship between:

```text
Decode Length
      ↓
Decode Iterations
      ↓
KV Cache Reads
      ↓
TPOT / ITL
      ↓
Total Latency
```

---

## Experiment 4 — Continuous Batching

Experiment with serving parameters such as:

```text
--max-num-seqs
```

Try different values appropriate for the available GPU memory.

Study:

```text
Batch Size
   ↓
GPU Utilization
   ↓
Throughput
   ↓
Latency
```

Find the point where increased batching stops helping.

---

## Experiment 5 — KV Cache Pressure

Increase:

- Prompt length
- Output length
- Number of concurrent sequences

Observe KV-cache utilization.

Understand experimentally:

```text
More Sequences
      +
Longer Context
      ↓
More KV Cache
      ↓
Memory Pressure
      ↓
Reduced Serving Capacity
```

Identify what happens when KV-cache capacity becomes a limiting resource.

---

## Experiment 6 — Prefix Reuse

Create two workloads.

### Workload A

Every request contains a common large prefix/system prompt.

### Workload B

Every request contains a different prefix.

Compare performance.

Understand:

```text
Repeated Prefix
      ↓
Reusable KV State
      ↓
Reduced Prefill Work
      ↓
Potential TTFT Improvement
```

---

## Experiment 7 — Precision / Quantization

Where supported by the V100 and current software stack, compare practical configurations.

Current baseline:

```text
FP16
```

Explore supported lower-precision/quantized alternatives only where technically compatible.

Compare:

- Model memory
- KV-cache capacity
- TTFT
- TPOT
- Throughput
- Output quality

Understand the distinction between:

```text
Weight Memory
KV Cache Memory
Compute Cost
Memory Bandwidth
Accuracy
```

---

## Experiment 8 — Mixed Workloads

Simulate realistic traffic.

For example:

```text
Request A → 500-token prompt → 20-token output
Request B → 20-token prompt  → 200-token output
Request C → 100-token prompt → 50-token output
Request D → 400-token prompt → 100-token output
```

Observe how the scheduler handles heterogeneous requests.

Study:

- Scheduling fairness
- TTFT
- ITL
- Queueing
- Continuous batching
- Prefill interference with decode

---

# Phase 3 — Understand the GPU

Move below the serving-engine abstraction.

The goal is to connect:

```text
LLM
 ↓
PyTorch
 ↓
CUDA Kernels
 ↓
GPU
 ↓
Serving Engine
 ↓
User-visible Latency
```

---

# Prefill vs Decode

Develop a strong mental model for why these phases behave differently.

## Prefill

Conceptually:

```text
Many Tokens
    ↓
Large Matrix Operations
    ↓
High Parallelism
    ↓
High Arithmetic Intensity
    ↓
Often More Compute-Oriented
```

## Decode

Conceptually:

```text
Generate One Token
      ↓
Read Model Weights
      ↓
Read KV Cache
      ↓
Small Incremental Computation
      ↓
Repeat
```

Decode can therefore become strongly constrained by memory movement/bandwidth.

Verify this experimentally rather than treating it as a rule.

---

# GPU Profiling

Use:

```text
NVIDIA Nsight Systems
```

first.

Understand:

- CPU/GPU timeline
- CUDA kernel launches
- GPU idle periods
- Synchronization
- Memory operations

Then selectively use:

```text
NVIDIA Nsight Compute
```

for deeper kernel analysis.

Investigate differences between:

```text
Prefill
vs
Decode
```

Try to identify which CUDA kernels dominate each phase.

---

# Phase 4 — Understand vLLM Internals

Do not attempt to understand the entire vLLM repository.

Focus on four major areas.

## 1. Scheduler

Understand:

- How requests enter scheduling
- How sequences are selected
- How batching decisions happen
- How prefill and decode are scheduled
- What happens when requests are waiting

---

## 2. KV Cache Manager

Understand:

- KV block allocation
- Block ownership
- Block reuse
- Free blocks
- Memory pressure
- PagedAttention relationship

Be able to explain why PagedAttention exists from first principles.

---

## 3. Model Runner

Understand how scheduled work reaches the model/GPU.

Trace approximately:

```text
Scheduler
   ↓
Model Runner
   ↓
Model Forward
   ↓
Attention
   ↓
CUDA Kernels
   ↓
GPU
```

---

## 4. Attention Implementation

Connect the theory already learned to actual serving behavior.

Understand:

```text
Q
K
V
↓
Attention
↓
KV Cache
↓
Paged KV Cache
↓
Attention Kernel
```

---

# Core Depth Exercise

Be able to explain **one decode iteration**.

Without referring to documentation, explain:

```text
Existing Request
      ↓
Scheduler selects sequence
      ↓
KV blocks identified
      ↓
Latest token enters model
      ↓
Q calculated
      ↓
Existing K/V retrieved
      ↓
Attention calculated
      ↓
Layer computation
      ↓
Logits
      ↓
Sampling
      ↓
Next token
      ↓
New K/V stored
      ↓
Scheduler repeats
```

If this can be explained confidently at both the conceptual and implementation level, the serving-engine understanding is becoming strong.

---

# Phase 5 — Production Failure Engineering

The existing architecture is:

```text
                  ┌── Qwen :8001
Client → Gateway ─┤
                  └── TinyLlama :8002
                         │
                         ▼
                       V100
```

Now intentionally break it.

---

# Failure Experiment 1 — Kill a Model

Generate traffic.

Then terminate:

```text
Qwen :8001
```

Observe:

- Gateway behavior
- Timeout behavior
- Error propagation
- User experience

Implement health-aware routing.

---

# Failure Experiment 2 — Fallback

Desired architecture:

```text
Request
   ↓
Router
   ↓
Qwen healthy?
   │
 YES ───► Qwen
   │
  NO
   ↓
TinyLlama
```

Record when fallback occurred.

---

# Failure Experiment 3 — Slow Model

Artificially create slow responses.

Study:

- Connect timeout
- Read timeout
- Overall request timeout
- Retry behavior

Avoid uncontrolled retries that amplify load.

---

# Failure Experiment 4 — Overload

Generate:

```text
10
20
50
100
```

simultaneous requests where practical.

Observe:

```text
Traffic
   ↓
Queue
   ↓
GPU Saturation
   ↓
TTFT Increase
   ↓
Timeouts
```

---

# Implement Production Protection Mechanisms

Progressively add:

```text
Health-aware routing
        ↓
Timeouts
        ↓
Retry policy
        ↓
Fallback
        ↓
Backpressure
        ↓
Admission control
        ↓
Rate limiting
        ↓
Circuit breaker
```

Understand why each mechanism exists rather than simply implementing it.

---

# Phase 6 — Load Balancing and Replicas

Once failure handling is understood, evolve toward:

```text
                     ┌── Qwen Replica 1
                     │
Gateway → Router ────┼── Qwen Replica 2
                     │
                     ├── TinyLlama Replica 1
                     │
                     └── TinyLlama Replica 2
```

Study routing strategies such as:

```text
Round Robin
Least Requests
Queue-Aware Routing
Latency-Aware Routing
KV/Prefix-Aware Routing
```

Compare them experimentally.

---

# Phase 7 — Distributed Inference

Only after the single-GPU serving architecture is understood well.

Learn the distinction between:

```text
Data Parallelism
Tensor Parallelism
Pipeline Parallelism
```

Then explore:

```text
Multi-GPU Serving
        ↓
Replica Routing
        ↓
KV Cache Transfer
        ↓
KV Offloading
        ↓
Prefill/Decode Disaggregation
        ↓
Multi-Node Serving
```

---

# Phase 8 — Containers and Kubernetes

Do not introduce Kubernetes merely to make the architecture look production-like.

First understand the serving system.

Then containerize:

```text
Gateway
Router
Model Servers
Metrics
```

Eventually reproduce:

```text
                 Public API
                     │
                     ▼
               Load Balancer
                     │
                     ▼
                API Gateway
                     │
                     ▼
              Request Router
                     │
             ┌───────┴───────┐
             ▼               ▼
        Model Pool A     Model Pool B
             │               │
        ┌────┴────┐      ┌───┴────┐
        ▼         ▼      ▼        ▼
      GPU       GPU     GPU      GPU
```

Then learn:

- Kubernetes deployments
- Services
- Replicas
- Health probes
- Resource limits
- GPU scheduling
- Autoscaling
- Failure recovery
- Rolling deployments
- Multi-node concepts

---

# Three-Week Execution Plan

## Week 1 — Inference Performance

### Build

- [ ] TTFT measurement
- [ ] TPOT/ITL measurement
- [ ] Queue-time measurement
- [ ] GPU metrics
- [ ] KV-cache metrics
- [ ] Better benchmark result reporting

### Experiments

- [ ] Concurrency sweep
- [ ] Prompt-length sweep
- [ ] Output-length sweep
- [ ] Continuous batching
- [ ] KV-cache pressure
- [ ] Prefix reuse
- [ ] Precision/quantization where supported
- [ ] Mixed workloads

### Outcome

I should be able to look at benchmark results and explain:

> Why did latency/throughput change?

---

# Week 2 — GPU + vLLM Internals

### GPU

- [ ] Profile inference
- [ ] Identify prefill
- [ ] Identify decode
- [ ] Understand compute vs memory behavior
- [ ] Inspect important CUDA kernels

### vLLM

- [ ] Scheduler
- [ ] KV Cache Manager
- [ ] Model Runner
- [ ] Attention implementation

### Outcome

Be able to explain:

```text
HTTP Request
     ↓
Gateway
     ↓
vLLM
     ↓
Scheduler
     ↓
KV Manager
     ↓
Model Runner
     ↓
Attention
     ↓
CUDA Kernel
     ↓
GPU
     ↓
Generated Token
```

---

# Week 3 — Production Serving

Implement:

- [ ] Health-aware routing
- [ ] Failure detection
- [ ] Fallback
- [ ] Timeouts
- [ ] Retry policy
- [ ] Backpressure
- [ ] Admission control
- [ ] Rate limiting
- [ ] Circuit breaker
- [ ] Load-balancer layer

Then intentionally break the architecture and verify recovery.

---

# Final Depth Test

Given:

```text
Model: 7B

Concurrency:
5 → 40 users

Observed:

Throughput: 1.4× increase
TTFT: 6× increase
TPOT: 15% increase
GPU utilization: 96%
KV cache utilization: 72%
```

Answer:

1. What is likely happening?
2. Why did TTFT increase much more than TPOT?
3. Why did throughput stop scaling?
4. Is KV-cache capacity currently the primary bottleneck?
5. What does 96% GPU utilization actually tell us?
6. Is the workload compute-bound or memory-bandwidth-bound?
7. What additional metrics are required to answer confidently?
8. What three experiments should be run next?
9. What serving-engine changes might improve the situation?
10. Which optimization would likely improve TTFT versus TPOT?

Do not simply propose configuration changes.

First establish a hypothesis, identify evidence needed, run an experiment, and then make the optimization.

---

# Final Project

Create a technical article:

# Anatomy of an LLM Request: From API Call to GPU Kernel

Use measurements from the actual serving system.

Cover:

```text
Request
  ↓
Gateway
  ↓
Router
  ↓
vLLM Scheduler
  ↓
Continuous Batching
  ↓
KV Cache Manager
  ↓
PagedAttention
  ↓
Model Runner
  ↓
CUDA Kernels
  ↓
GPU
  ↓
Sampling
  ↓
Next Token
```

Include actual measurements for:

- TTFT
- TPOT
- Throughput
- Queue time
- GPU utilization
- KV-cache utilization
- Concurrency

The final objective is to be able to connect:

> **User-visible latency → serving-engine behavior → model execution → GPU behavior.**

---

# Working Principle

For every new inference optimization:

```text
1. Understand the bottleneck
2. Predict the behavior
3. Measure the baseline
4. Change ONE variable
5. Measure again
6. Explain the result
7. Document what was learned
```

Do not optimize blindly.

Do not add distributed infrastructure before understanding single-GPU behavior.

Do not treat higher GPU utilization as automatically better.

Do not treat higher throughput as automatically better.

The target is to understand the trade-off between:

```text
Latency
Throughput
Memory
Cost
Reliability
Quality
```

That is the core of the next stage of inference engineering.