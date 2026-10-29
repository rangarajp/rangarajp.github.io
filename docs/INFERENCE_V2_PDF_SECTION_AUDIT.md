# PDF section adaptation audit

Source: user-supplied *Efficient LLM Inference: Systems, Algorithms & Production Engineering*, second edition, Lamhot Siagian (2026). The conceptual sections were re-read directly from the attached PDF, excluding its interview Q&A. The blog uses original adapted prose and retains the existing single chapter list, working examples, and sourced V100 observations.

The first chapter is rewritten as a self-contained explanation using the PDF's five-section progression. Its token-generation section now explains prefill, first-token selection, subsequent decode, state, and stopping without requiring readers to follow links first.

| PDF section group | Blog destination | Coverage |
|---|---|---|
| 1.1–1.5 Inference problem | `00-inference-problem.md` | Training/inference, state and memory, complete generation loop, latency/throughput/cost tradeoff, 14 GB/7 ms worked bound, roofline and matrix-intensity code |
| 2.1–2.5 Hardware foundations | `10-gpu-fundamentals.md`, `07-memory-footprint.md` | GPU organization, tensor cores, formats, hierarchy, CPU/edge constraints, accelerator tradeoffs; V100 and H100 assumptions remain separate |
| 3.1–3.5 Transformer mechanics | `01` through `06` | Tokens/positions, forward pass, Q/K/V, cache shapes, prefill/decode, MHA/MQA/GQA/MLA, RoPE; existing detailed code retained |
| 4.1–4.5 Quantization | `17-quantization.md` | Memory need, numeric formats/ranges, GPTQ, AWQ, QuIP#/AQLM, QAT and surrogate gradients, activation outliers and transformed scaling, validation |
| 5.1–5.5 Speculation | `18-speculative-decoding.md` | Ordered draft/verify algorithm, exact correction, tree masks/state, separate Medusa and EAGLE explanations, acceptance-versus-overhead cost model |
| 6.1–6.5 KV optimization | `13-kv-cache.md`, `15-paged-attention.md` | Cache pressure, block allocation, prefix reuse, attention sinks/windowing, independent KV precision choice; tail waste and error reuse explained |
| 7.1–7.4 Kernels/FlashAttention | `14-flashattention.md`, `19-kernel-fusion-cuda-graphs.md` | Intermediate IO, fusion, tiling, online normalization/value recurrence, numerical CPU example, kernel tooling and backend constraints |
| 8.1–8.4 Serving architecture | `16-continuous-batching.md`, `27`, `28` | Production components, static/dynamic/continuous batching, chunked prefill, KV migration, priorities, preemption, fairness, routing and admission |
| 9.1–9.4 Parallelism/MoE | `20` through `23`, `08-inference-mathematics.md` | Capacity and replicas, row/column sharding, collectives, pipeline bubbles, routed expert equation and residency/dispatch costs |
| 10.1–10.5 Long context | `24-context-parallelism.md` | Quadratic scores, tiled memory, ring attention, prompt/learned compression, LLMLingua, state-space recurrence and retrieval tradeoffs |
| 11.1–11.4 Edge | `37-edge-inference.md` | Device budgets, low precision, runtime/operator compatibility, measured thermal behavior, hardware-aware architecture choices, mmap/unified-memory limits |
| 12.1–12.4 Inference-time scaling | `38-inference-time-scaling.md` | Training versus request compute, reasoning-token cost, sequential/parallel attempts, best-of-N/beam/tree search, process/outcome verification, reasoning-model context |
| 13.1–13.4 Production | `33-observability-slos.md`, `36-capacity-and-cost.md` | Traces, metrics, alerts, rollout/cost attribution, TTFT/TPOT, observed joint compliance versus goodput rate, MFU limits, failures and economics |
| 14.1–14.4 Future inference | `39-future-of-inference.md` | Diffusion, multi-token prediction, alternative hardware, adaptive depth/early exit, sparsity and byte-level tradeoffs |

Technical adaptations preserve accuracy rather than copying simplified claims: Llama-2-70B KV uses 8 KV heads; roofline is a resource bound; prefill/decode regimes depend on workload; paged allocation still has tail waste; retained quantization error can affect later tokens; speculation is not universally lossless without the exact sampling procedure; separate SLO pass rates do not establish joint compliance; no unverified hardware speedup, price, model-quality guarantee, or unpublished system detail is asserted.

The book's training-oriented backward-pass discussion is not presented as inference work. Hardware-specific kernel examples are explained conceptually or linked to the relevant chapter rather than claiming V100 support. No interview questions were added.

## Editorial order and use of evidence

The reading order follows the PDF topic progression: inference problem → hardware → transformer mechanics → quantization → speculation → KV optimization → kernels → serving → parallelism → long context → edge → inference-time scaling → production → future directions. Detailed blog chapters and notebook examples remain within their relevant topic, using a single chapter list. Memory sizing and mathematics extend hardware foundations; the PyTorch baseline closes transformer mechanics.

Within the opening chapter, keep the PDF’s five main sections. Introduce bandwidth and roofline first, then explain model loading versus repeated HBM reads and training’s weight reuse as subsections of roofline. Use theory → mechanism → worked example → recorded notebook evidence or proposed hands-on where applicable. Preserve links and distinguish measured V100 outputs from illustrative estimates and unexecuted exercises. Exclude interview questions. Future additions should extend the relevant explanation rather than interrupting the prerequisite sequence.
