# LLM Inference Engineering --- Concept Summary

This note summarizes the main concepts we clarified during our
inference-engineering discussions, with emphasis on **intuition,
bottlenecks, and how the optimizations connect together**.

------------------------------------------------------------------------

## 1. The Big Picture: What Are We Optimizing?

LLM inference is fundamentally a balancing act between:

-   **Compute** --- how many FLOPs the GPU can perform.
-   **Memory bandwidth** --- how quickly weights, activations, and
    KV-cache data can move between HBM and compute units.
-   **GPU memory capacity** --- how much model state, KV cache, and
    active workload can fit.
-   **Latency** --- how quickly an individual user receives a response.
-   **Throughput** --- how much total work the system completes per
    second.
-   **Concurrency** --- how many requests can be served simultaneously.

A recurring theme is:

> **Maximize useful GPU work while minimizing unnecessary memory
> movement, idle time, and wasted memory.**

------------------------------------------------------------------------

# 2. Arithmetic Intensity and the Roofline Mental Model

Arithmetic intensity (AI) is:

$$ AI = `\frac{\text{FLOPs}}{\text{Bytes moved from memory}}`{=tex} $$

It tells us how much computation we perform for every byte moved from
memory.

### Low arithmetic intensity

The GPU spends much of its time waiting for memory.

This is a **memory-bound** workload.

### High arithmetic intensity

There is enough computation per byte moved that GPU compute becomes the
limiting factor.

This is a **compute-bound** workload.

### Ridge point

The approximate transition occurs around:

$$ AI_{`\text{ridge}`{=tex}} `\approx`{=tex}
`\frac{\text{Peak FLOPs/s}}`{=tex}
{`\text{Memory Bandwidth Bytes/s}`{=tex}} $$

Below the ridge point → memory bound.

Above the ridge point → compute bound.

Importantly, reaching the ridge point does **not** mean that larger
batches make throughput fall. Instead:

-   throughput grows strongly initially,
-   then approaches the hardware ceiling,
-   additional batching gives diminishing throughput improvement,
-   latency and queueing may continue increasing.

------------------------------------------------------------------------

# 3. Why Batching Helps

Suppose a linear operation is:

$$ X$$B,D$$ `\times `{=tex}W$$D,F$$ `\rightarrow `{=tex}Y$$B,F$$ $$

Approximate FLOPs:

$$ 2BDF $$

With a very small batch, especially (B=1), the large weight matrix may
need to be read while relatively little computation is performed.

Increasing (B) lets many inputs reuse the same weights.

Therefore:

> **Batching increases arithmetic intensity through weight reuse.**

This is particularly important during autoregressive decode, which is
often memory-bandwidth bound at small batch sizes.

------------------------------------------------------------------------

# 4. Static, Dynamic, and Continuous Batching

## Static batching

Collect a fixed batch and process it together.

Problem:

-   requests have different output lengths,
-   short requests finish early,
-   their slots may remain unused,
-   the longest request becomes a straggler.

Static batching is not useless, but it is often inefficient for online
LLM serving.

## Dynamic batching

The server waits briefly and forms a batch according to conditions such
as:

-   maximum batch size,
-   timeout,
-   number of waiting requests.

This improves batch formation.

However, once that batch starts, it is still essentially a fixed group.

## Continuous batching

Continuous batching moves scheduling into the inference engine's
decoding loop.

At each decode iteration:

1.  process the currently active sequences,
2.  detect sequences that have finished,
3.  return their results,
4.  remove them,
5.  insert waiting requests into the freed capacity,
6.  execute the next decoding iteration.

The important insight is:

> We are **not interrupting a GPU kernel halfway through**.\
> The serving engine changes the active roster **between decoding
> iterations**.

### Why this helps

Without continuous batching:

batch size gradually shrinks → weight reuse decreases → arithmetic
intensity decreases → GPU utilization falls.

With continuous batching:

> **The engine tries to keep the active workload populated.**

So a useful mental model is:

> **Continuous batching keeps the assembly line full.**

------------------------------------------------------------------------

# 5. Chunked Prefill

Prefill and decode have very different characteristics.

### Prefill

For a long prompt, many tokens are processed together.

This tends to involve large matrix multiplications and can be highly
compute intensive.

### Decode

Usually one new token per active sequence is processed at each
iteration.

At small batches this is commonly memory-bandwidth bound.

## The problem

Imagine active users are decoding and suddenly a request containing a
huge prompt arrives.

If its entire prefill runs as one large operation, it can occupy the GPU
for a relatively long period.

Existing decoding requests can suffer latency spikes.

## Chunked prefill

Instead of processing the huge prompt's prefill in one shot:

$$ `\text{Large Prefill}`{=tex} `\rightarrow`{=tex} C_1 + C_2 + C_3 +
`\dots`{=tex} $$

The scheduler can interleave those chunks with decode work.

Important clarification:

> A partially prefetched request does **not** begin normal
> autoregressive decoding before its required prefill is complete.

The chunks allow the long prefill to make incremental progress without
monopolizing scheduling opportunities.

### Mental model

> **Continuous batching keeps the assembly line full.**

> **Chunked prefill prevents one giant job from blocking the assembly
> line.**

------------------------------------------------------------------------

# 6. Speculative Decoding

Traditional autoregressive decoding with a large model does:

$$ t_1 `\rightarrow `{=tex}t_2 `\rightarrow `{=tex}t_3
`\rightarrow `{=tex}t_4 $$

Each token normally requires another sequential target-model decode
iteration.

## Speculative decoding

Introduce a cheaper **draft model**.

It predicts several candidate tokens:

$$ $$d_1,d_2,d_3,d_4$$ $$

The large target model then verifies multiple draft positions in a more
parallel operation.

If the proposed tokens are accepted, several tokens can advance while
avoiding the same number of sequential large-model decode iterations.

If a mismatch occurs, the accepted prefix is retained and generation
continues according to the speculative-decoding algorithm.

## Why can this be faster even though two models run?

Because the draft model is cheap.

The expensive target model gets to verify several positions together
rather than performing the same number of strictly sequential one-token
decode iterations.

This can improve hardware utilization and amortize expensive
target-model work.

## Arithmetic-intensity connection

Traditional low-batch decode:

-   little computation per model-weight load,
-   often memory bound.

Speculative verification:

-   more token positions processed together,
-   greater reuse of weights,
-   potentially higher arithmetic intensity.

## Important limitation

Speculative decoding is workload dependent.

At high continuous-batching levels, the GPU may already be highly
utilized.

In that case, speculation may provide much less benefit because:

-   the target workload already has substantial parallelism,
-   the draft model adds additional compute and memory overhead.

So:

> **Speculative decoding is particularly attractive when decode is
> latency-sensitive and underutilizing the GPU, but it should be
> benchmarked rather than assumed to help every workload.**

Standard speculative decoding usually uses one draft continuation. More
advanced approaches can use tree or multi-branch speculation.

------------------------------------------------------------------------

# 7. Quantization

Quantization reduces the precision used to represent model values.

Examples include:

-   FP32
-   BF16
-   FP16
-   FP8
-   INT8
-   INT4
-   even lower-bit representations in specialized methods

The fundamental trade-off is:

$$ `\text{Lower precision}`{=tex} `\rightarrow`{=tex}
`\text{Less memory + less bandwidth}`{=tex} `\rightarrow`{=tex}
`\text{Potential numerical/quality loss}`{=tex} $$

------------------------------------------------------------------------

## 7.1 Quantization as a staircase

A high-precision value can take many possible numerical values.

After quantization, many nearby values map onto the same representable
level.

Conceptually:

``` text
High precision
      /
     /
    /
   /

Quantized approximation

      __
    _|
  _|
_|
```

Nearby original values can collapse to the same quantized value.

The difference between the original and reconstructed value is
**quantization error**.

------------------------------------------------------------------------

## 7.2 Dynamic range

Dynamic range describes the span of magnitudes that a number format or
quantization scheme can represent.

Outliers matter enormously.

Suppose most values are around:

``` text
-2 ... +2
```

but one value is:

``` text
+100
```

If a low-bit quantizer uses one scale covering the entire range:

``` text
-2 ------------------------------ +100
```

many quantization levels are effectively spent covering the outlier.

The dense region around -2 to +2 then receives relatively coarse
resolution.

So:

> **Large outliers can force a large range, increasing the quantization
> step size for ordinary values.**

Possible remedies include:

-   per-channel quantization,
-   group-wise quantization,
-   dynamic scaling,
-   outlier-aware methods,
-   mixed precision.

------------------------------------------------------------------------

## 7.3 BF16 intuition

BF16 is useful because it retains an **8-bit exponent**, like FP32.

Therefore it preserves roughly the same broad exponent range while using
fewer mantissa bits.

So compared with FP32:

-   similar dynamic range,
-   lower precision,
-   half the storage.

This is why BF16 is useful for numerically stable lower-precision
computation.

It is better thought of as a lower-precision floating-point format
rather than integer quantization.

------------------------------------------------------------------------

# 8. What Can Be Quantized in an LLM?

For inference, the major targets are:

1.  **Weights**
2.  **Activations**
3.  **KV cache**
4.  Selected intermediate/sensitive operations

Training additionally introduces:

-   gradients,
-   optimizer states.

------------------------------------------------------------------------

## 8.1 Weight quantization

Weights are comparatively convenient because they are static after
training.

Their distributions can be inspected ahead of inference.

This enables approaches such as weight-only quantization.

Example:

``` text
Weights: INT4
Activations: BF16
```

The major benefit is reducing:

-   model memory footprint,
-   weight-memory bandwidth.

------------------------------------------------------------------------

## 8.2 Activation quantization

Activations are harder.

Why?

They depend on the actual input.

Different prompts can produce very different activation distributions
and outliers.

Therefore the quantization range may need to adapt dynamically or use
finer-grained scaling.

This is why activation quantization often requires more care than
weight-only quantization.

------------------------------------------------------------------------

## 8.3 KV-cache quantization

During autoregressive inference, previous keys and values are stored and
repeatedly read.

Quantizing the KV cache can significantly reduce memory consumption and
memory bandwidth.

But KV precision can be quality-sensitive because cached values
participate repeatedly in future attention computations.

Important refinement:

> KV quantization errors do not simply "compound" mechanically every
> step, but inaccurate cached K/V values are repeatedly reused and can
> influence future token predictions.

Therefore specialized scaling and mixed-precision approaches are often
used.

------------------------------------------------------------------------

# 9. Attention --- Core Intuition

Consider:

> **"A blue fluffy creature is walking on the road."**

The embedding of **creature** gives an initial token representation.

But the desired representation should incorporate context such as:

-   blue,
-   fluffy,
-   walking,
-   road.

Attention creates this contextualization.

------------------------------------------------------------------------

# 10. Query, Key, and Value

For each token representation (x_i):

$$ Q_i = x_iW_Q $$

$$ K_i = x_iW_K $$

$$ V_i = x_iW_V $$

where (W_Q), (W_K), and (W_V) are learned matrices.

A useful intuition:

### Query

> **What information am I looking for?**

For "creature":

> Which tokens help explain what kind of creature this is?

### Key

> **What kind of information do I contain / when am I relevant?**

"blue" may be highly relevant to a query seeking descriptive
information.

"fluffy" may also be highly relevant.

### Value

> **If you attend to me, what information should I contribute?**

This was an important clarification:

The value vector is **not the original embedding itself**.

For "blue":

$$ V_{`\text{blue}`{=tex}} = x_{`\text{blue}`{=tex}}W_V $$

It is a learned transformation of the current representation.

------------------------------------------------------------------------

# 11. Attention Scores

For the query corresponding to "creature":

$$ score_j = `\frac{
Q_{\text{creature}}K_j^T
}{
\sqrt{d_k}
}`{=tex} $$

Softmax converts these scores into attention weights:

$$ `\alpha`{=tex}_j = `\text{softmax}`{=tex}(score_j) $$

Conceptually:

``` text
blue       -> 0.35
fluffy     -> 0.40
creature   -> 0.15
walking    -> 0.07
road       -> 0.03
```

These are illustrative numbers only.

------------------------------------------------------------------------

# 12. The Key Clarification: How the Context Vector Is Built

A tempting but incorrect mental model is:

> Start with (V_{`\text{creature}`{=tex}}) and pull it toward blue and
> fluffy.

That is **not** the direct attention operation.

Instead:

$$ Context_{`\text{creature}`{=tex}} = `\sum`{=tex}_j
`\alpha`{=tex}_jV_j $$

For example:

$$ Context_{`\text{creature}`{=tex}} = 0.35V_{`\text{blue}`{=tex}} +
0.40V_{`\text{fluffy}`{=tex}} + 0.15V_{`\text{creature}`{=tex}} +
0.07V_{`\text{walking}`{=tex}} + 0.03V_{`\text{road}`{=tex}} $$

So:

> **The query determines the weights. The weights mix the value
> vectors.**

Residual connections later combine the attention output with the token's
prior representation.

------------------------------------------------------------------------

# 13. Matrix View of Attention

Suppose there are 5 tokens.

After projections:

$$ Q `\in `{=tex}`\mathbb{R}`{=tex}\^{5 `\times `{=tex}d} $$

$$ K `\in `{=tex}`\mathbb{R}`{=tex}\^{5 `\times `{=tex}d} $$

$$ V `\in `{=tex}`\mathbb{R}`{=tex}\^{5 `\times `{=tex}d_v} $$

Attention scores:

$$ S = QK\^T $$

Therefore:

$$ S `\in `{=tex}`\mathbb{R}`{=tex}\^{5 `\times 5`{=tex}} $$

After scaling and softmax:

$$ A = `\text{softmax}`{=tex} `\left`{=tex}(
`\frac{QK^T}{\sqrt{d_k}}`{=tex} `\right`{=tex}) $$

Then:

$$ C = AV $$

where (C) contains one contextualized attention output per token.

For "creature", take the **creature row** from (A) and multiply it
against the entire value matrix (V).

That produces one context vector for "creature".

------------------------------------------------------------------------

# 14. KV Cache Dimensions

An important distinction:

For sequence length (T=5) and head dimension (d_h=128):

$$ K: $$5,128$$ $$

$$ V: $$5,128$$ $$

The KV cache is **not** (5`\times5`{=tex}).

The (5`\times5`{=tex}) object is the attention-score matrix:

$$ QK\^T $$

This distinction becomes important when understanding FlashAttention and
PagedAttention.

------------------------------------------------------------------------

# 15. FlashAttention

FlashAttention primarily optimizes **attention computation and memory
I/O**.

Naive attention conceptually computes:

$$ S = QK\^T $$

then:

$$ P = softmax(S) $$

then:

$$ O = PV $$

A naive implementation can involve substantial reads/writes of large
intermediate tensors to GPU HBM.

For sequence length (T), the attention score matrix is:

$$ T `\times `{=tex}T $$

For long sequences this becomes enormous.

------------------------------------------------------------------------

## 15.1 The core FlashAttention idea

Do not materialize the entire (T`\times `{=tex}T) attention matrix in
HBM.

Instead:

1.  divide Q/K/V processing into tiles,
2.  move manageable blocks into fast on-chip memory,
3.  calculate partial (QK\^T),
4.  maintain the necessary running softmax statistics,
5.  combine with V,
6.  discard temporary score blocks,
7.  continue with the next tile,
8.  write the final attention output to HBM.

So:

> **The mathematical attention remains global, but the physical
> computation is tiled.**

This is not local/chunked attention.

Token 4096 can still attend to all allowed previous tokens.

The algorithm simply computes that full attention in memory-efficient
pieces.

------------------------------------------------------------------------

## 15.2 Why FlashAttention helps prefill especially

During prefill with a long sequence:

$$ T `\times `{=tex}T $$

attention work can be substantial.

FlashAttention avoids repeatedly materializing huge intermediate
score/probability tensors in HBM.

During decode, however, there is usually only one new query position per
sequence attending over the existing KV cache.

So the giant (T`\times `{=tex}T) intermediate no longer appears in the
same way.

Flash-style kernels can still help decode, but the dramatic (O(T\^2))
intermediate-I/O saving is primarily a prefill story.

### Mental model

> **FlashAttention = reduce attention-related HBM traffic by doing tiled
> attention work on-chip.**

------------------------------------------------------------------------

# 16. PagedAttention

PagedAttention addresses a different problem:

> **Efficient management of the KV cache.**

During autoregressive generation, each active sequence has a growing KV
cache.

Different requests have different:

-   prompt lengths,
-   generated lengths,
-   lifetimes.

Managing many variable-length, dynamically growing KV caches with large
contiguous allocations can waste memory and make growth difficult.

------------------------------------------------------------------------

## 16.1 What does contiguous mean?

Contiguous means:

> One unbroken range of physical memory addresses.

Imagine:

``` text
[1][2][3][4][5][6][7][8][9][10]
```

Suppose request A occupies:

``` text
[1][2][3]
```

If A must remain contiguous and needs another block, ideally it wants:

``` text
[4]
```

But if 4 is occupied, growing A becomes awkward.

You either:

-   reserve more memory in advance, causing waste,
-   relocate/reallocate memory,
-   or use another memory-management strategy.

------------------------------------------------------------------------

# 17. Paging the KV Cache

PagedAttention borrows the idea of virtual-memory paging.

A logical sequence's KV cache is split into fixed-size blocks/pages.

Physically they might live like:

``` text
Request A logical blocks:

A1 -> physical block 2
A2 -> physical block 9
A3 -> physical block 5
A4 -> physical block 14
```

A block table remembers the mapping.

The blocks do not need to be physically adjacent.

When attention needs the KV cache, the system follows the mapping and
accesses the appropriate blocks.

Therefore:

> **Logical continuity does not require physical continuity.**

------------------------------------------------------------------------

# 18. Why PagedAttention Helps

Without paging, systems can face a bad trade-off:

### Reserve large contiguous space

Pros:

-   easy growth.

Cons:

-   potentially large unused capacity.

### Allocate only current needs contiguously

Pros:

-   less initial waste.

Cons:

-   future growth may require expensive allocation/reorganization.

Paging allows incremental allocation:

``` text
Need more KV?
       ↓
Allocate another available block
       ↓
Update block table
```

This improves memory utilization.

Better KV-cache utilization means:

$$ `\text{More free GPU memory}`{=tex} `\rightarrow`{=tex}
`\text{More active sequences}`{=tex} `\rightarrow`{=tex}
`\text{Higher concurrency}`{=tex} $$

### Mental model

> **FlashAttention optimizes attention computation/I/O.**

> **PagedAttention optimizes KV-cache memory management.**

They solve different problems and complement each other.

------------------------------------------------------------------------

# 19. Core Serving Metrics

## TTFT --- Time to First Token

Time between request arrival and the first generated token becoming
available.

It includes effects such as:

-   queueing,
-   scheduling,
-   prefill,
-   first decode work.

As batch/concurrency increases, TTFT commonly increases somewhat.

Once the system becomes saturated, queueing can make TTFT increase
sharply.

------------------------------------------------------------------------

## TPOT --- Time Per Output Token

After the first token appears, TPOT describes the average time between
subsequent output tokens.

Example:

``` text
First token: 500 ms
Later tokens: one every ~20 ms
```

Then approximately:

``` text
TTFT = 500 ms
TPOT = 20 ms/token
```

As batch size increases:

-   system throughput can improve,
-   each decode iteration handles more sequences,
-   per-request TPOT can worsen.

So:

> **System throughput and individual streaming speed are different
> metrics.**

------------------------------------------------------------------------

## End-to-end latency

Approximate total response completion time:

$$ Latency `\approx`{=tex} TTFT +
(N_{`\text{output}`{=tex}}-1)`\times `{=tex}TPOT $$

The exact definition used by a benchmark should always be checked.

------------------------------------------------------------------------

## Throughput

Throughput measures total work completed per unit time.

For output throughput:

$$ `\text{Output Throughput}`{=tex} =
`\frac{\text{Total generated tokens}}`{=tex} {`\text{Time}`{=tex}} $$

Input and output throughput can also be reported separately.

Typical batching behavior:

``` text
Throughput
   ^
   |                 _________
   |             ___/
   |          __/
   |       __/
   |_____/
   +--------------------------> Batch / concurrency
```

Initially batching produces large gains.

As the GPU approaches saturation, throughput approaches a ceiling.

------------------------------------------------------------------------

# 20. Tail-Latency Percentiles

Suppose:

``` text
P99 latency = 200 ms
```

That means approximately:

> **99% of measured requests completed in 200 ms or less.**

About 1% took longer.

It does **not** mean that 200 ms is the worst-case latency.

The maximum might be much larger.

Usually:

$$ P50 `\le `{=tex}P95 `\le `{=tex}P99 $$

These metrics expose the slow tail that an average can hide.

For production serving, P95 and P99 can be particularly important.

------------------------------------------------------------------------

# 21. The Batching Trade-off in One Picture

As batch/concurrency increases:

  Metric                      Typical behavior
  --------------------------- ------------------
  Arithmetic intensity        ↑ initially
  GPU utilization             ↑
  System throughput           ↑ then plateaus
  TTFT                        generally ↑
  TPOT per request            generally ↑
  End-to-end latency          generally ↑
  Queueing after saturation   ↑ sharply

So the optimization problem is not:

> "Find the largest possible batch."

It is:

> **Find the operating point that gives enough throughput while meeting
> latency SLOs.**

------------------------------------------------------------------------

# 22. Parallelism

Parallelism determines how work and model state are distributed across
GPUs.

The main types we discussed were:

1.  Data Parallelism
2.  Tensor Parallelism
3.  Pipeline Parallelism

Additional important types include:

4.  Sequence / Context Parallelism
5.  Expert Parallelism

------------------------------------------------------------------------

# 23. Data Parallelism --- DP

### Mental model

> **Same model, different data.**

Suppose there are 4 GPUs and a training batch of 64:

``` text
GPU 0 -> 16 samples
GPU 1 -> 16 samples
GPU 2 -> 16 samples
GPU 3 -> 16 samples
```

Each GPU holds a complete model replica.

During inference:

``` text
GPU 0 -> requests A, B
GPU 1 -> requests C, D
GPU 2 -> requests E, F
GPU 3 -> requests G, H
```

### Primary purpose

Increase:

-   throughput,
-   concurrency.

### Important limitation

The complete model must fit on every GPU.

Therefore DP does **not** solve the problem:

> "My model is too large for one GPU."

### Training

Gradients must be synchronized between replicas, commonly through
collective communication such as AllReduce.

### Inference

There is no gradient synchronization.

The main systems problems become:

-   load balancing,
-   request routing,
-   batching,
-   keeping replicas busy.

------------------------------------------------------------------------

# 24. Tensor Parallelism --- TP

### Mental model

> **Split tensors within a layer across GPUs.**

Suppose:

$$ Y = XW $$

and (W) is huge.

Instead of putting all of (W) on one GPU, shard it:

``` text
GPU 0 -> W shard 0
GPU 1 -> W shard 1
GPU 2 -> W shard 2
GPU 3 -> W shard 3
```

The GPUs cooperate to execute the layer.

This is different from pipeline parallelism.

TP does **not** primarily mean:

``` text
GPU 0 -> layer 1
GPU 1 -> layer 2
```

Instead, multiple GPUs cooperate **inside the same layer's large tensor
operations**.

### Why use TP?

-   model/layer tensors may not fit on one GPU,
-   large matrix operations can be distributed.

### Main cost

Frequent inter-GPU communication.

Therefore high-speed interconnects matter:

-   NVLink / NVSwitch,
-   InfiniBand or other fast networking when communication crosses
    nodes.

### Mental model

> **TP trades communication for distributed memory capacity and
> compute.**

It is useful in both training and inference.

------------------------------------------------------------------------

# 25. Pipeline Parallelism --- PP

### Mental model

> **Different layers live on different GPUs.**

Example with a 32-layer transformer:

``` text
GPU 0 -> Layers  1-8
GPU 1 -> Layers  9-16
GPU 2 -> Layers 17-24
GPU 3 -> Layers 25-32
```

Activations flow through the GPUs:

``` text
Input
  ↓
GPU 0
  ↓
GPU 1
  ↓
GPU 2
  ↓
GPU 3
  ↓
Output
```

This helps distribute a large model across devices.

### The problem: pipeline bubbles

If only one input is moving through the pipeline:

``` text
Time →

GPU0: [WORK][idle][idle][idle]
GPU1: [idle][WORK][idle][idle]
GPU2: [idle][idle][WORK][idle]
GPU3: [idle][idle][idle][WORK]
```

Lots of hardware sits idle.

Using multiple micro-batches creates an assembly line:

``` text
GPU0: [M1][M2][M3][M4]
GPU1:     [M1][M2][M3][M4]
GPU2:         [M1][M2][M3][M4]
GPU3:             [M1][M2][M3][M4]
```

The goal is to reduce idle pipeline bubbles.

------------------------------------------------------------------------

# 26. DP vs TP vs PP

  -----------------------------------------------------------------------
  Parallelism             What is split?          Primary reason
  ----------------------- ----------------------- -----------------------
  Data Parallel           Requests / training     Throughput
                          samples                 

  Tensor Parallel         Tensor operations       Model size +
                          inside layers           distributed compute

  Pipeline Parallel       Layers / model stages   Model size

  Context Parallel        Sequence/context        Very long context
                          dimension               

  Expert Parallel         MoE experts             Large sparse MoE models
  -----------------------------------------------------------------------

A useful memory trick:

``` text
DP = split DATA

TP = split TENSOR inside a layer

PP = split PIPELINE of layers
```

------------------------------------------------------------------------

# 27. Context / Sequence Parallelism

For extremely long contexts, the sequence itself can become expensive to
process or store.

Context/sequence parallel techniques distribute work along the sequence
dimension across devices.

Conceptually:

``` text
Long sequence

[token 1 ........ token 100000]

GPU0 -> earlier segment
GPU1 -> next segment
GPU2 -> next segment
GPU3 -> later segment
```

The exact algorithm and communication pattern depend on the
implementation.

The important intuition is:

> **Instead of splitting requests, weights, or layers, split work
> associated with the long sequence dimension.**

------------------------------------------------------------------------

# 28. Expert Parallelism

Used especially for Mixture-of-Experts (MoE) models.

Different experts can be placed on different GPUs.

For example:

``` text
GPU0 -> Experts 0,1
GPU1 -> Experts 2,3
GPU2 -> Experts 4,5
GPU3 -> Experts 6,7
```

A router chooses which experts process each token.

This introduces significant communication because tokens may need to
move to the GPU holding their selected expert.

Hence **all-to-all communication** is an important concern.

------------------------------------------------------------------------

# 29. How These Optimizations Fit Together

A useful inference stack is:

``` text
                    LLM INFERENCE
                         |
          +--------------+--------------+
          |                             |
       COMPUTE                         MEMORY
          |                             |
          |                       +-----+------+
          |                       |            |
       Batching                Weights       KV Cache
          |                       |            |
   +------+-------+          Quantization  PagedAttention
   |              |
Continuous     Speculative
Batching       Decoding
   |
Chunked Prefill

Attention compute
      |
FlashAttention

Multi-GPU execution
      |
+-----+------+------+
|            |      |
DP           TP     PP
```

These techniques are not mutually exclusive.

A production serving system may combine several of them.

------------------------------------------------------------------------

# 30. Optimization → Bottleneck Mapping

  Technique              Main bottleneck addressed
  ---------------------- -------------------------------------------------
  Batching               Poor GPU utilization / low arithmetic intensity
  Continuous batching    Slots becoming idle as requests finish
  Chunked prefill        Long prefills blocking latency-sensitive work
  Speculative decoding   Sequential autoregressive decode
  Quantization           Weight/KV memory capacity and bandwidth
  FlashAttention         Attention intermediate-memory I/O
  PagedAttention         KV-cache allocation and fragmentation/waste
  Data parallelism       Throughput / replica scaling
  Tensor parallelism     Model/tensor too large for one GPU
  Pipeline parallelism   Model layers too large for one GPU
  Context parallelism    Very long sequence/context
  Expert parallelism     Distribution of MoE experts

------------------------------------------------------------------------

# 31. A Better Way to Think About Inference Optimization

Instead of memorizing optimization names, ask:

### 1. What is the bottleneck?

``` text
Compute?
Memory bandwidth?
Memory capacity?
Communication?
Scheduling?
Queueing?
```

### 2. Which phase?

``` text
Prefill?
Decode?
Both?
```

### 3. What resource is underutilized?

``` text
Tensor cores?
HBM bandwidth?
GPU memory?
Multiple GPUs?
Network?
```

### 4. What optimization attacks that bottleneck?

For example:

``` text
Decode is memory bound
        ↓
Increase batching
        ↓
Increase weight reuse
        ↓
Increase arithmetic intensity
```

Or:

``` text
KV cache consumes too much VRAM
        ↓
Paged KV management
        +
KV quantization
        ↓
More concurrent sequences
```

Or:

``` text
Long prefill causes attention I/O overhead
        ↓
FlashAttention
        ↓
Less HBM traffic
```

------------------------------------------------------------------------

# 32. Recommended Hands-On Learning Path

The next step should be to reproduce these ideas in **small PyTorch
experiments before relying entirely on vLLM**.

The objective is not to rebuild vLLM.

It is to understand enough of the internals that when vLLM behaves a
certain way, you can reason about **why**.

## Experiment 1 --- Baseline decode

Implement a tiny decoder-only transformer.

Measure:

-   TTFT,
-   TPOT,
-   tokens/sec,
-   GPU memory.

------------------------------------------------------------------------

## Experiment 2 --- KV cache

Compare:

``` text
Decode without KV cache
vs
Decode with KV cache
```

Observe the reduction in repeated computation.

------------------------------------------------------------------------

## Experiment 3 --- Batching

Run:

``` text
Batch = 1
Batch = 2
Batch = 4
Batch = 8
...
```

Measure:

-   throughput,
-   TTFT,
-   TPOT,
-   latency,
-   GPU utilization.

Connect the measurements back to arithmetic intensity.

------------------------------------------------------------------------

## Experiment 4 --- Continuous batching

Build a tiny scheduler simulation.

Requests should have different output lengths.

Compare:

``` text
Fixed batch
vs
Continuous replacement of finished sequences
```

Observe utilization.

------------------------------------------------------------------------

## Experiment 5 --- Quantization

Compare:

``` text
FP32
FP16/BF16
INT8 or weight-only lower precision
```

Measure:

-   model memory,
-   latency,
-   throughput,
-   output quality.

Inspect the numerical distributions and outliers.

------------------------------------------------------------------------

## Experiment 6 --- Attention memory

Implement normal attention.

Inspect the shapes and memory associated with:

``` text
Q
K
V
QK^T
softmax(QK^T)
output
```

Then compare with PyTorch's optimized scaled-dot-product attention /
FlashAttention-compatible path where supported.

------------------------------------------------------------------------

## Experiment 7 --- Paged KV-cache simulation

You do not initially need a production CUDA implementation.

Create a toy block allocator:

``` text
logical KV block
      ↓
block table
      ↓
physical block
```

Generate variable-length sequences and compare:

``` text
contiguous reservation
vs
paged allocation
```

Measure memory waste.

------------------------------------------------------------------------

## Experiment 8 --- Data Parallelism

Two GPUs:

``` text
GPU0 -> model copy -> requests A
GPU1 -> model copy -> requests B
```

Measure aggregate throughput.

------------------------------------------------------------------------

## Experiment 9 --- Tensor Parallelism

Start with one simple matrix multiplication:

$$ Y=XW $$

Manually split (W) across two GPUs.

Compute each shard and reconstruct the correct output.

Then measure:

-   compute time,
-   communication time,
-   total time.

This makes TP communication cost tangible.

------------------------------------------------------------------------

## Experiment 10 --- Pipeline Parallelism

Split a tiny transformer:

``` text
GPU0 -> first layers
GPU1 -> later layers
```

First run one input.

Observe idle time.

Then introduce micro-batches and visualize the pipeline.

------------------------------------------------------------------------

# 33. PyTorch First, vLLM Second

Recommended progression:

``` text
Mathematics
    ↓
Tiny PyTorch implementation
    ↓
Measure/profile
    ↓
Understand bottleneck
    ↓
Apply optimization
    ↓
Measure improvement
    ↓
Study how vLLM implements the production version
```

vLLM should not be thought of as merely a wrapper.

It is a sophisticated inference engine containing substantial systems
engineering around:

-   scheduling,
-   batching,
-   KV-cache management,
-   memory allocation,
-   optimized kernels,
-   distributed execution,
-   request lifecycle management.

The goal of the PyTorch experiments is therefore:

> **Build enough first-principles intuition to understand and debug the
> behavior of production inference engines.**

------------------------------------------------------------------------

# 34. Final Mental Map

If only a few ideas are remembered, keep these:

### Batching

**Reuse weights across more work.**

### Continuous batching

**Keep the active workload populated.**

### Chunked prefill

**Do not let one giant prompt monopolize scheduling.**

### Speculative decoding

**Use cheap drafting to reduce expensive sequential target-model decode
iterations.**

### Quantization

**Move/store fewer bits while preserving enough numerical fidelity.**

### FlashAttention

**Do the same attention math with dramatically less intermediate HBM
traffic.**

### PagedAttention

**Let KV-cache blocks live wherever memory is available and map them
logically.**

### Data parallelism

**Same model, different requests/data.**

### Tensor parallelism

**Same layer, split tensor computation across GPUs.**

### Pipeline parallelism

**Different layer groups on different GPUs.**

### TTFT

**How long until the response starts.**

### TPOT

**How quickly tokens arrive after it starts.**

### Throughput

**How much total work the system completes per second.**

### P95 / P99

**How bad latency is for the slower tail of requests.**

And the overall inference-engineering question is always:

> **Where is the bottleneck, and what resource is currently being
> wasted?**
