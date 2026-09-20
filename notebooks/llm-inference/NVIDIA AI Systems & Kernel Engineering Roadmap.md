# NVIDIA AI Systems & Kernel Engineering Journey

> **Goal:** Spend the next 6–12 months building demonstrable depth in GPU computing, kernel engineering, and LLM inference systems — from Python all the way down to GPU hardware.

This repository is not intended to be a collection of course notes.

The objective is to develop the ability to answer:

> **Why is this AI workload slow, what exactly is the GPU doing, and how can I make it faster?**

The learning philosophy is simple:

> **Explain → Implement → Measure → Diagnose → Optimize**

No concept is considered fully learned until I can explain it, implement it, benchmark it, profile it, and reason about its performance.

---

# 1. North Star

The long-term technical stack I want to understand is:

```text
                         AI / LLM
                            │
                    Transformer Internals
                            │
                       LLM Inference
                            │
                 vLLM / TensorRT-LLM
                            │
             Batching / Scheduling / KV Cache
                            │
                  PyTorch Custom Operators
                            │
                      Triton Kernels
                            │
                        CUDA C++
                            │
                       PTX / SASS
                            │
                      Tensor Cores
                            │
                GPU Architecture & Memory
                            │
                 Hopper / Blackwell GPUs
```

The goal is not simply to know what these layers are.

The goal is to understand **how they interact**.

For example:

```text
User sends prompt
      ↓
Inference server schedules request
      ↓
Model performs prefill
      ↓
Attention accesses KV cache
      ↓
GPU kernels are launched
      ↓
Warps execute instructions
      ↓
Data moves through HBM / L2 / Shared Memory / Registers
      ↓
Tensor Cores / CUDA Cores perform computation
```

Eventually I should be able to reason about performance across this entire path.

---

# 2. Learning Philosophy

Every major topic follows the same loop:

```text
Understand
    ↓
Implement
    ↓
Benchmark
    ↓
Profile
    ↓
Identify Bottleneck
    ↓
Form Hypothesis
    ↓
Optimize
    ↓
Benchmark Again
    ↓
Explain Why
```

For example:

```text
PyTorch RMSNorm
      ↓
Implement naive RMSNorm
      ↓
Benchmark
      ↓
Write Triton RMSNorm
      ↓
Benchmark
      ↓
Profile memory traffic
      ↓
Optimize
      ↓
Write CUDA implementation
      ↓
Compare
```

The final output should contain evidence:

```text
Implementation     Latency     Bandwidth     Notes
---------------------------------------------------
PyTorch            measured    measured      baseline
Triton             measured    measured      optimized
CUDA               measured    measured      optimized
```

Never invent benchmark numbers.

All performance results in this repository should come from actual experiments.

---

# 3. Phase I — GPU Foundations

**Target: Months 0–2**

Objective:

> Build a strong mental model of how GPUs execute programs.

Topics:

- CPU vs GPU architecture
- Streaming Multiprocessors
- CUDA cores
- Tensor Cores
- threads
- blocks
- grids
- warps
- warp scheduling
- SIMT execution
- registers
- shared memory
- L1 cache
- L2 cache
- HBM
- memory latency
- memory bandwidth
- coalesced memory access
- branch divergence
- synchronization
- occupancy
- arithmetic intensity
- compute-bound workloads
- memory-bound workloads
- kernel launch overhead
- streams
- asynchronous execution

---

# 4. Kernel Engineering Chapters

## Chapter 01 — What Actually Happens When You Run `torch.add()`?

Start with:

```python
a = torch.randn(1_000_000, device="cuda")
b = torch.randn(1_000_000, device="cuda")

c = a + b
```

Understand the complete execution path:

```text
Python
   ↓
PyTorch
   ↓
ATen
   ↓
CUDA Runtime
   ↓
Kernel Launch
   ↓
GPU
   ↓
Streaming Multiprocessor
   ↓
Warp
   ↓
Threads
   ↓
Load
   ↓
Add
   ↓
Store
```

Questions to answer:

- What is a GPU kernel?
- Who launches the kernel?
- Where does the kernel execute?
- What is an SM?
- What is a warp?
- How are one million elements processed?
- Where are `a`, `b`, and `c` stored?
- How much data must move?
- Is vector addition compute-bound or memory-bound?
- What overhead does PyTorch introduce?
- What does asynchronous GPU execution mean?

Experiment:

```text
torch.add
    ↓
profile
    ↓
identify CUDA kernel
    ↓
measure execution time
```

---

## Chapter 02 — My First CUDA Kernel

Implement vector addition.

CPU mental model:

```python
for i in range(N):
    c[i] = a[i] + b[i]
```

GPU mental model:

```text
Thread 0 → c[0] = a[0] + b[0]
Thread 1 → c[1] = a[1] + b[1]
Thread 2 → c[2] = a[2] + b[2]
...
```

CUDA:

```cpp
__global__
void vector_add(
    const float* a,
    const float* b,
    float* c,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
```

Understand:

- `threadIdx`
- `blockIdx`
- `blockDim`
- grid dimensions
- kernel launch syntax
- bounds checking

---

## Chapter 03 — How GPUs Execute Millions of Threads

Understand:

```text
Grid
 │
 ├── Block
 │    │
 │    ├── Warp
 │    │    ├── Thread
 │    │    ├── Thread
 │    │    └── ...
 │    │
 │    └── Warp
 │
 └── Block
```

Key question:

> If I launch one million threads, does the GPU have one million processors?

Topics:

- SM scheduling
- warp scheduling
- resident warps
- latency hiding
- occupancy

---

## Chapter 04 — GPU Memory Hierarchy

Understand:

```text
               Fast / Small
                    ↑

                Registers
                    │
              Shared Memory
                    │
                 L1 Cache
                    │
                 L2 Cache
                    │
                   HBM

                    ↓
               Large / Slow
```

Study:

- latency
- bandwidth
- capacity
- scope
- lifetime

Experiment with memory-access patterns.

---

## Chapter 05 — Memory Coalescing

Compare:

```text
Thread 0 → address 0
Thread 1 → address 1
Thread 2 → address 2
Thread 3 → address 3
```

against scattered accesses:

```text
Thread 0 → address 0
Thread 1 → address 100
Thread 2 → address 200
Thread 3 → address 300
```

Measure the performance difference.

Understand why memory layout matters.

---

## Chapter 06 — Shared Memory

Implement tiled operations.

Understand why:

```text
HBM
 ↓
Shared Memory
 ↓
Many computations
 ↓
HBM
```

can outperform repeatedly accessing HBM.

---

## Chapter 07 — Warp Divergence

Investigate:

```cpp
if (condition) {
    ...
} else {
    ...
}
```

Understand what happens when threads inside the same warp follow different execution paths.

---

## Chapter 08 — Profiling GPU Kernels

Tools:

- PyTorch Profiler
- NVIDIA Nsight Systems
- NVIDIA Nsight Compute
- NVTX

Learn to inspect:

- kernel execution time
- memory throughput
- SM utilization
- achieved occupancy
- warp stalls
- memory transactions
- instruction mix

Primary question:

> **Why is this kernel slow?**

---

# 5. Phase II — AI Kernel Engineering

**Target: Months 2–4**

Move quickly from educational kernels into kernels used by modern neural networks.

Implement:

```text
01 Vector Add
02 Matrix Add
03 Reduction
04 Softmax
05 LayerNorm
06 RMSNorm
07 SiLU
08 SwiGLU
09 RoPE
10 GEMM
11 Quantized GEMM
12 Attention
13 Fused Attention
14 FlashAttention-like Kernel
```

For important kernels implement:

```text
              Operation
                  │
        ┌─────────┼─────────┐
        ↓         ↓         ↓
     PyTorch    Triton     CUDA
```

Then compare them.

---

# 6. Triton

Learn Triton after establishing the CUDA mental model.

Topics:

- Triton programming model
- programs
- blocks
- masks
- pointer arithmetic
- memory loads
- memory stores
- reductions
- block sizes
- number of warps
- autotuning

Implement:

```text
Vector Add
    ↓
Softmax
    ↓
RMSNorm
    ↓
Matrix Multiplication
    ↓
RoPE
    ↓
Fused Operations
```

For each kernel:

```text
PyTorch baseline
       ↓
Triton implementation
       ↓
Benchmark
       ↓
Profile
       ↓
Optimize
```

---

# 7. Kernel Fusion

Start with:

```text
Input
 │
 ├── Operation A
 │
 ├── Operation B
 │
 └── Operation C
```

Understand the cost of:

```text
HBM → Kernel A → HBM
HBM → Kernel B → HBM
HBM → Kernel C → HBM
```

Then investigate:

```text
             HBM
              │
              ▼
      ┌────────────────┐
      │  Fused Kernel  │
      │                │
      │ Operation A    │
      │ Operation B    │
      │ Operation C    │
      └────────────────┘
              │
              ▼
             HBM
```

Measure when fusion helps and when it doesn't.

---

# 8. Roofline Model

Learn to classify kernels as:

```text
Memory Bound
      │
      │
      ├──────── Balanced
      │
      │
Compute Bound
```

Understand arithmetic intensity:

```text
Arithmetic Intensity =
    FLOPs
    -----
    Bytes moved
```

Use this to predict whether optimization should focus on:

- memory traffic
- computation
- occupancy
- instruction efficiency

---

# 9. LLM Kernels

Implement important Transformer operations from scratch.

## RMSNorm

Understand:

```text
Input
  ↓
Square
  ↓
Reduction
  ↓
Mean
  ↓
rsqrt
  ↓
Normalize
  ↓
Scale
```

Determine whether the kernel is memory or compute bound.

---

## RoPE

Understand how rotary embeddings transform query and key vectors.

Implement:

```text
PyTorch
   ↓
Triton
   ↓
CUDA
```

---

## SwiGLU

Study:

```text
x
│
├── Linear A → SiLU ─┐
│                    × → output
└── Linear B ────────┘
```

Investigate fusion opportunities.

---

# 10. Attention

Start with naive attention:

```text
Q
│
├── QKᵀ
│
▼
Scores
│
├── Scale
│
├── Softmax
│
▼
Probabilities
│
├── × V
│
▼
Output
```

Understand the memory problem created by materializing:

```text
N × N attention matrix
```

---

# 11. FlashAttention

Study attention as an IO problem.

Understand:

- tiling
- SRAM/shared memory
- HBM traffic
- online softmax
- recomputation
- block-wise attention

Build a simplified implementation.

Primary question:

> Why can FlashAttention be faster even though mathematically it computes the same attention?

---

# 12. Quantization Kernels

Study:

- FP32
- TF32
- FP16
- BF16
- FP8
- INT8
- INT4

Understand:

```text
Model precision
      ↓
Memory footprint
      ↓
Memory bandwidth
      ↓
Tensor Core utilization
      ↓
Inference throughput
```

Implement at least one quantized operation.

---

# 13. Phase III — LLM Inference Systems

**Target: Months 4–7**

Move above individual kernels into inference-engine architecture.

Study:

- prefill
- decode
- KV cache
- batching
- continuous batching
- request scheduling
- memory management
- PagedAttention
- chunked prefill
- prefix caching
- speculative decoding
- CUDA graphs
- quantization
- tensor parallelism

Study systems such as:

- vLLM
- TensorRT-LLM

---

# 14. Build a Mini LLM Inference Engine

This becomes a flagship project.

Architecture:

```text
                    Request
                       │
                       ▼
                ┌─────────────┐
                │  Tokenizer  │
                └──────┬──────┘
                       │
                       ▼
                 Request Queue
                       │
                       ▼
                   Scheduler
                  /         \
             Prefill       Decode
                │             │
                └──────┬──────┘
                       ▼
                   KV Cache
                       │
                       ▼
                 Transformer
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
       RMSNorm        GEMM       Attention
          │            │            │
          └────── GPU Kernels ──────┘
                       │
                       ▼
                     Token
```

Implement progressively.

### Version 0

Basic autoregressive inference.

### Version 1

KV caching.

### Version 2

Batching.

### Version 3

Continuous batching.

### Version 4

Paged KV cache.

### Version 5

Chunked prefill.

### Version 6

Prefix caching.

### Version 7

Custom Triton kernels.

### Version 8

CUDA graphs.

### Version 9

Quantization.

Every version should include benchmarks.

---

# 15. Performance Metrics

Understand and measure:

## TTFT

Time To First Token.

```text
Request
   │
   ├──────────── TTFT ────────────►
   │                              First Token
```

---

## ITL

Inter-Token Latency.

```text
Token 1
   │
   ├── ITL ── Token 2
                  │
                  ├── ITL ── Token 3
```

---

## Throughput

Measure:

```text
tokens / second
requests / second
```

Study the tradeoff between:

```text
Latency ↔ Throughput
```

---

# 16. Phase IV — Deep GPU Optimization

**Target: Months 7–12**

Move below CUDA source code.

Understand:

```text
CUDA C++
   ↓
PTX
   ↓
SASS
   ↓
GPU Instructions
```

Topics:

- PTX
- SASS
- compiler behavior
- warp-level primitives
- asynchronous copies
- Tensor Cores
- WMMA
- CUTLASS
- persistent kernels
- CUDA graphs
- memory pipelines
- advanced synchronization
- FP8
- Hopper architecture
- Blackwell architecture

---

# 17. Tensor Cores

Understand why:

```text
Normal CUDA Core

a × b + c
```

differs from matrix-oriented Tensor Core execution.

Study:

- MMA
- WMMA
- tile shapes
- mixed precision
- Tensor Core data types
- matrix multiplication pipelines

Eventually inspect generated instructions.

---

# 18. CUTLASS

Understand NVIDIA's approach to high-performance matrix multiplication.

Study the hierarchy:

```text
Device
   ↓
Thread Block
   ↓
Warp
   ↓
MMA
   ↓
Tensor Core
```

Build and modify small CUTLASS examples.

---

# 19. PTX and SASS

Take simple CUDA:

```cpp
c[i] = a[i] + b[i];
```

Follow it through:

```text
CUDA
  ↓
PTX
  ↓
SASS
```

Learn to answer:

- What instructions were generated?
- How many loads?
- How many stores?
- Which instructions execute on which units?
- What is the compiler optimizing?
- Where are registers used?

---

# 20. Distributed LLM Inference

Once single-GPU fundamentals are strong, study:

- data parallelism
- tensor parallelism
- pipeline parallelism
- expert parallelism
- sequence/context parallelism

Understand communication primitives:

- AllReduce
- AllGather
- ReduceScatter
- AllToAll

Study NCCL.

Understand when inference becomes:

```text
Compute Bound

vs

Memory Bound

vs

Communication Bound
```

---

# 21. Repository Structure

Suggested structure:

```text
gpu-learning/
│
├── README.md
│
├── docs/
│   │
│   ├── 01_gpu_foundations/
│   │   ├── 01_torch_add.md
│   │   ├── 02_first_cuda_kernel.md
│   │   ├── 03_gpu_execution.md
│   │   ├── 04_memory_hierarchy.md
│   │   ├── 05_memory_coalescing.md
│   │   ├── 06_shared_memory.md
│   │   ├── 07_warp_divergence.md
│   │   └── 08_profiling.md
│   │
│   ├── 02_triton/
│   │
│   ├── 03_ai_kernels/
│   │
│   ├── 04_attention/
│   │
│   ├── 05_quantization/
│   │
│   ├── 06_llm_inference/
│   │
│   └── 07_advanced_gpu/
│
├── kernels/
│   │
│   ├── cuda/
│   ├── triton/
│   └── pytorch/
│
├── benchmarks/
│
├── profiling/
│
├── mini_inference_engine/
│
└── experiments/
```

---

# 22. Every Chapter Must Contain

Every chapter should follow the same structure.

## 1. Problem

What are we trying to solve?

## 2. Intuition

Explain it without GPU terminology first.

## 3. GPU Mental Model

What actually happens on the hardware?

## 4. Implementation

Build the simplest implementation.

## 5. Prediction

Before profiling, predict:

> What do I think the bottleneck is?

## 6. Benchmark

Measure actual performance.

## 7. Profile

Inspect GPU behavior.

## 8. Diagnose

Explain the bottleneck using evidence.

## 9. Optimize

Change the implementation.

## 10. Benchmark Again

Determine whether the hypothesis was correct.

## 11. LLM Connection

Explain where the concept appears in modern LLM systems.

## 12. Questions

Answer conceptual questions without looking at notes.

---

# 23. Learning Allocation

Approximate personal technical-learning allocation:

```text
GPU / CUDA / Kernels / Inference Systems
███████████████████████████████████ 70%

Transformers / LLM Architecture
██████████ 20%

Broader AI Research
█████ 10%
```

The purpose is to build **compounding depth** instead of continuously expanding breadth.

---

# 24. Portfolio Projects

By the end of this journey, aim to have several substantial public artifacts.

## Project 1 — GPU Kernels From Scratch

```text
PyTorch
vs
Triton
vs
CUDA
```

for important neural-network operations.

---

## Project 2 — LLM Kernels From Scratch

Implement:

- RMSNorm
- RoPE
- SwiGLU
- Softmax
- Attention
- FlashAttention-like kernel
- Quantized operations

---

## Project 3 — Mini LLM Inference Engine

Implement:

- KV cache
- batching
- continuous batching
- paged memory
- chunked prefill
- prefix caching
- CUDA graphs
- custom kernels

---

## Project 4 — GPU Performance Case Studies

Take slow kernels and document:

```text
Problem
   ↓
Profiler Evidence
   ↓
Hypothesis
   ↓
Optimization
   ↓
Measured Improvement
   ↓
Technical Explanation
```

This demonstrates performance-engineering ability rather than just implementation ability.

---

# 25. Blog Strategy

The Learning Hub should tell one coherent story:

```text
Learning Hub
│
├── Transformers
│   ├── Attention
│   ├── QKV
│   └── KV Cache
│
├── GPU Computing
│   ├── GPU Architecture
│   ├── CUDA
│   ├── Memory Hierarchy
│   ├── Warps
│   └── Profiling
│
├── Kernel Engineering
│   ├── Vector Add
│   ├── Reduction
│   ├── Softmax
│   ├── RMSNorm
│   ├── GEMM
│   ├── RoPE
│   └── FlashAttention
│
└── LLM Systems
    ├── Inference
    ├── KV Cache
    ├── Continuous Batching
    ├── PagedAttention
    ├── Quantization
    ├── CUDA Graphs
    └── Mini Inference Engine
```

The story should be:

> **Rebuilding modern AI inference from first principles.**

---

# 26. Six-Month Milestone

At approximately six months, I should be comfortable explaining and implementing:

```text
GPU architecture
      ↓
CUDA kernels
      ↓
GPU memory optimization
      ↓
Triton
      ↓
LLM kernels
      ↓
Attention
      ↓
FlashAttention
      ↓
KV cache
      ↓
PagedAttention
      ↓
Continuous batching
      ↓
LLM inference engine
```

I should also be able to profile unfamiliar GPU workloads and systematically investigate their bottlenecks.

---

# 27. Twelve-Month Milestone

At approximately twelve months, I should be capable of reasoning across:

```text
LLM Architecture
      ↓
Inference Engine
      ↓
Runtime
      ↓
CUDA Kernel
      ↓
PTX / SASS
      ↓
GPU Architecture
      ↓
Distributed GPUs
```

The objective is not memorization.

The objective is to develop the ability to look at an AI workload and ask:

> **Where is the bottleneck?**

Then investigate whether it comes from:

```text
Algorithm
Memory
Kernel
Scheduling
Runtime
Communication
Precision
Hardware Utilization
```

and produce evidence supporting the conclusion.

---

# 28. The Rule

For every topic:

> **Don't just read it. Build it.**

For every implementation:

> **Don't just run it. Measure it.**

For every benchmark:

> **Don't just report it. Explain it.**

For every optimization:

> **Don't just make it faster. Understand why it became faster.**

---

# 29. Starting Point

The journey begins with one deceptively simple line:

```python
c = a + b
```

The first question is:

> **What actually happens inside the GPU when PyTorch executes this line?**

From there:

```text
torch.add()
    ↓
CUDA kernel
    ↓
Grid
    ↓
Blocks
    ↓
Warps
    ↓
Threads
    ↓
Memory
    ↓
Instructions
    ↓
Performance
```

That is **Chapter 01**.

---

# Mission

> **Build enough depth that my work itself demonstrates what I can do.**

The next 6–12 months are about accumulating evidence:

**code, kernels, benchmarks, profiler traces, technical explanations, experiments, systems, and open-source work.**

One chapter at a time.

One kernel at a time.

One layer deeper.