---
title: "The inference problem"
description: "A user sends a prompt and waits for an answer."
section: learn
pubDate: 2026-10-29
chapter: 1
order: 1
---

A user sends a prompt and waits for an answer. The model has already been trained, yet producing that answer still requires repeated computation, access to billions of stored weights, and memory for the conversation. When many users arrive together, the server must share those resources without making every answer slow.

This is the inference problem: turn a trained model into a service that delivers useful results within a latency, memory, and cost budget. Before studying individual optimizations, we need to understand the work an answer creates and why different parts of that work stress the machine differently.

## 1.1 Training versus inference: fundamentally different goals

*Training* changes a model. It processes examples, calculates a loss, propagates gradients backward through the network, and updates learned parameters. The system needs memory for parameters, gradients, optimizer state, and activations retained for backpropagation. A common systems goal is to process a large training workload efficiently enough to finish within the available time and budget.

*Inference* uses a trained model to compute predictions without updating its parameters. For a decoder language model, those predictions are distributions over the next token. Weights remain fixed, but the input and the sequence state change as an answer grows.

That distinction changes both the memory budget and the performance objective. Removing backpropagation eliminates gradients and optimizer state from ordinary inference. It does not make inference stateless: every active conversation still needs its own context, temporary computation, and usually a growing key/value cache.

| Concern | Training | Language-model inference |
|---|---|---|
| Parameters | Updated through optimization | Fixed during ordinary generation |
| Repeated work | Forward, backward, and optimizer steps | Prompt processing followed by generation steps |
| Additional state | Gradients, optimizer state, saved activations | Temporary activations and per-sequence KV state |
| Typical batch objective | Efficient progress over a large dataset | Useful service across requests with different lengths |
| User-visible constraint | Training completion time | First-token delay and the pace of the answer |

Interactive chat emphasizes responsiveness. An offline document-processing job may instead favor throughput. Neither objective removes quality and cost: a fast incorrect response or a cheap service that misses every deadline is not a successful deployment.

The practical consequence is that fitting the weights is only the first check. We also need enough memory for the requests we admit, and enough execution capacity to serve them at the desired pace.

## 1.2 The token generation loop explained

Consider a prompt that has already been converted to token IDs. The model reads that known prefix and predicts what can come next. After a token is selected, it becomes part of the prefix for the following prediction. This dependency is what makes ordinary autoregressive generation a loop.

The request has two phases:

1. *Prefill:* process the known prompt positions together under a causal attention mask. Build the keys and values needed for later attention. The logits at the final prompt position select the first output token.
2. *Decode:* feed the previously selected token through the model, add its keys and values to the retained state, attend over the available prefix, and select the next output token.
3. Repeat decode until an end condition is reached, such as an end-of-sequence token, a configured stop condition, or an output-token limit.

For a prompt with 512 tokens and an answer with three generated tokens, prefill processes the 512 known positions and selects the first output token. The first decode pass consumes that token and selects the second. A second decode pass consumes the second and selects the third. If the request stops there, the third selected token need not be fed through another forward pass.

This detail explains why the first token behaves differently from the rest: a nonempty answer of $C$ tokens normally has one prefill and $C-1$ subsequent generation steps.

Prefill exposes many known positions to the GPU at once. Their projection and feed-forward computations can reuse weights across a matrix of token vectors. Decode at low concurrency exposes much less work per step, even though it still needs the model weights. Batching independent decode requests can restore some of that reuse; it cannot remove the within-request dependency between successive selected tokens.

The *KV cache* avoids recomputing old keys and values. It is not a cache of model weights or finished answers. Each new query still attends to retained context, so longer context increases cache storage and attention work. We will derive the tensor shapes later; for now, distinguish the shared weights from state that belongs to a particular request.

## 1.3 Latency, throughput, and cost: the iron triangle

A serving system has several clocks and one shared resource budget. The right configuration depends on which clock matters to the application.

| Objective | What it measures | A useful improvement | The tradeoff to measure |
|---|---|---|---|
| Latency | How long a request waits and how quickly it streams | Less queueing, less prompt work, efficient decode | Reserved capacity can sit idle |
| Throughput | Work delivered across requests per wall-clock second | Reuse weights across a well-filled batch | More waiting or longer execution steps can hurt individual requests |
| Cost | Resources paid for per useful completed task or token | Better utilization, fewer bytes or less unnecessary work | Quality and latency must still meet the target |

*Time to first token* (TTFT) is measured from request arrival to first delivery. It includes queueing, preprocessing, prompt execution, sampling, and delivery work. *Inter-token latency* (ITL) is the gap between delivered tokens. *Time per output token* (TPOT), as used here, is the mean post-first-token gap. It is undefined for a one-token answer.

For $C>1$ and consistently defined delivery timestamps:

$$
T_{\mathrm{E2E}} = T_{\mathrm{TTFT}} + (C-1)\,\overline{T_{\mathrm{ITL}}}
$$

Queueing is already included in TTFT; adding it again would double-count waiting. A separate timer around only the model's prompt forward pass is a prefill measurement, not client TTFT.

```python
# Illustrative request timings in seconds, not a measured model run.
queue, preprocessing, prefill, first_delivery = 0.020, 0.005, 0.080, 0.005
output_tokens, mean_gap = 101, 0.010
ttft = queue + preprocessing + prefill + first_delivery
e2e = ttft + (output_tokens - 1) * mean_gap
print('TTFT:', round(ttft, 3), 's')  # 0.11
print('Complete answer:', round(e2e, 3), 's')  # 1.11
```

Two users can see different responsiveness from the same model. A long prompt adds prompt-processing work; a long answer adds more sequential generation steps; a busy queue adds waiting before either starts. An optimization that improves one of these does not necessarily improve the others.

Larger batches can amortize weight movement and improve total output throughput. They can also increase batch-formation delay or execution time per step. The relationship is not a rule that latency always worsens: under some loads, improved utilization reduces queueing. Measure the actual operating point rather than maximizing batch size in isolation.

A useful production objective is *goodput*: requests or tokens delivered within the stated latency and quality objectives per wall-clock second. High raw throughput with many missed deadlines can be less valuable than a lower throughput that consistently meets them.

## 1.4 Memory bandwidth: the decode bottleneck

A GPU needs both arithmetic units and a way to supply their operands. Its device memory stores far more data than the on-chip caches can retain. A large dense model therefore repeatedly moves weight data from device memory toward the compute units during generation.

For seven billion parameters stored at two bytes each:

$$
M_{\mathrm{weights}} = 7\times10^9\times2 = 14\times10^9\ \mathrm{bytes}
$$

If a step streams those 14 GB once through a memory system sustaining 2 TB/s, the ideal weight-read time is:

$$
T_{\mathrm{weight\ read}} = \frac{14\times10^9}{2\times10^{12}} = 0.007\ \mathrm{s} = 7\ \mathrm{ms}
$$

At batch one, that read supports one next-token prediction. With multiple independent token rows, a matrix operation can reuse loaded weights across them, producing more useful work for the traffic paid. This is the reason batching can improve throughput even when the model parameters do not change.

The 7 ms calculation is an ideal resource bound under the stated streaming and bandwidth assumptions. It is not a benchmark for the V100, and arithmetic time should not simply be added to it as though computation and movement never overlap. Actual execution also includes KV reads, activation traffic, launch overhead, synchronization, and imperfect use of the memory system.

Caching and bandwidth solve different problems. KV reuse removes repeated computation over old prefix positions. Quantization can reduce the weight bytes moved. Grouped-query attention can reduce retained KV bytes. Batching increases reuse across independent work. None makes memory traffic disappear.

## 1.5 The roofline model

The roofline model connects those observations using *arithmetic intensity*: operations performed per byte transferred across a specified memory boundary.

$$
I = \frac{\mathrm{FLOPs}}{\mathrm{bytes\ moved}}, \qquad
P_{\mathrm{attainable}} \leq \min\!\left(P_{\mathrm{peak}},\ I\,B_{\mathrm{memory}}\right)
$$

Here $P_{\mathrm{peak}}$ is the relevant peak operation rate and $B_{\mathrm{memory}}$ is bandwidth in bytes per second. Their ratio is the *ridge intensity*. Below that ridge, the bandwidth term is the tighter ideal bound. Above it, peak arithmetic throughput is tighter. Real kernels can fall below either roof.

A matrix-vector projection at batch one performs roughly two operations per weight. If each weight occupies two bytes and is read once, its weight-dominated intensity is about one FLOP per byte. More token rows reuse those weights, raising intensity toward the matrix-matrix regime.

Counting only weight traffic gives a useful first approximation:

$$
I_{\mathrm{weights}}
\approx \frac{2M\,d_{\mathrm{in}}d_{\mathrm{out}}}
{2\,d_{\mathrm{in}}d_{\mathrm{out}}}
= M\ \mathrm{FLOPs/byte}
$$

The numerator counts a multiply and an add per weight per row; the denominator assumes two-byte weights read once. This is a weight-only estimate, not a prediction that intensity grows without limit. Activation traffic, output writes, and repeated tile loads also matter. The example below includes input and output traffic in its idealized count.

```python
# One FP16 projection; count input reads, weight reads, and output writes.
inner = output = 4096
for token_rows in (1, 64, 512):
    flops = 2 * token_rows * inner * output
    moved = 2 * (token_rows * inner + inner * output + token_rows * output)
    print(token_rows, round(flops / moved, 2), 'FLOP/byte')
# Approximately 1.00, 62.06, 409.60 FLOP/byte.
```

This is why prefill and decode often respond to different optimizations. Long prompt matrices can provide enough reuse to make arithmetic throughput important. Single-stream decode often remains limited by moving weights and state. Long-context attention, higher concurrency, quantization, and different kernels can move the bottleneck, so these are workload tendencies rather than permanent labels attached to a model.

#### Loading the model and reading its weights are different transfers

“The model is already on the GPU” means its weights are resident in *high-bandwidth memory* (HBM). The arithmetic units still need those weights brought into the much smaller on-chip storage used during computation.

| Transfer | When it happens |
| --- | --- |
| Disk/CPU RAM → GPU HBM | Usually once when loading the model |
| GPU HBM → on-chip storage used by compute units | Repeated during model execution |

The first row describes a typical serving setup where the model remains resident; CPU offloading or swapping can introduce further host-to-device transfers. The second row is the traffic relevant to the decode bandwidth limit discussed here. On-chip storage includes caches, shared memory, and registers; the exact data path depends on the kernel and GPU architecture.

During a forward pass, each layer reads the weights it needs. A kernel can reuse a loaded *tile* of weights while processing that operation, but a large model cannot retain all its weights on chip across the entire pass. As subsequent layers run, earlier data is generally displaced. When the next generated token reaches the first layer again, that layer generally reads its weights from HBM again. Keeping the model in HBM avoids reloading it from the host; it does not eliminate these repeated device-memory reads.

The KV cache stores previous tokens' keys and values, avoiding their recomputation. It does not hold the model's weights close to the arithmetic units, and attention must also read the relevant cached K/V values during decoding.

#### Training also reads weights: it gets more work from each read

Training moves weights from HBM into on-chip storage too. The difference is how much computation a kernel can perform with each loaded tile. For a linear layer:

$$
Y = XW,\qquad
X\in\mathbb{R}^{M\times d_{\mathrm{in}}},\qquad
W\in\mathbb{R}^{d_{\mathrm{in}}\times d_{\mathrm{out}}}
$$

Here $M$ is the number of token rows processed together. During training, a projection commonly combines batch and sequence positions into many rows. Although autoregressive attention uses a causal mask, the known training sequence lets the layer process those positions in parallel. During ordinary batch-one decoding, the same projection receives just one new token row. Batched decoding supplies one row per active sequence.

| Property | Training with many token rows | Batch-one decoding |
| --- | --- | --- |
| Weight movement from HBM | Repeated during execution | Repeated during execution |
| Reuse of a loaded weight tile | Across many token rows | For one token row |
| Work per byte read | Usually higher | Usually lower |
| Common limit for large linear layers | Compute throughput | Memory bandwidth |

Training therefore *amortizes* weight movement across more arithmetic. Its backward pass adds further matrix multiplications, along with gradient and activation traffic; optimizer updates add memory traffic too. This does not make every training operation compute-bound. Small matrices and operations with little reuse can still be memory-bound. Long-prompt prefill benefits from processing many rows for the same reason that training's forward projections do.

NVIDIA's [matrix multiplication performance guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html) explains the relationship between matrix shape, reuse, and arithmetic intensity. The arithmetic-intensity estimate above puts numbers on this distinction.

The following discussion examines the hardware behind these limits: compute units, memory hierarchy, and precision. We then follow tokens through the transformer before studying the optimizations that reduce its work and memory traffic.

## Hands-on extensions

Use these experiments to connect the mechanisms above to measurements. They are proposed exercises unless a recorded result is explicitly identified.

### Hands-on: trace one request

Send one streamed request to the teaching server and record arrival, first delivery, final delivery, input length, and output length. Publish an annotated timing table that separates prompt execution from client-visible waiting and delivery. Keep the calculations above as estimates and the request trace as a separate measurement.

The conceptual sequence follows the opening chapter of the supplied *Efficient LLM Inference: Systems, Algorithms & Production Engineering*, second edition (Lamhot Siagian, 2026), adapted here with original examples and consistent metric definitions.
