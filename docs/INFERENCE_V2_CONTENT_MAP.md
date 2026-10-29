# Inference Engineering V2 — content map

Inventory of existing inference writing and labs, mapped onto the six-part V2 curriculum. Canonical V2 pages are **new**; old `/concepts/llm-inference/` articles stay published and are linked as related reading, not replaced.

Status key for V2:

- *have source* — existing article or notebook can inform a rewrite
- *gap* — no canonical page yet
- *do not invent* — labs exist but published numbers must be re-measured before citing

---

## Existing public book (`/concepts/llm-inference/`)

Leave these files and URLs unchanged.

| Order | Title | Path | Notes |
|------:|-------|------|-------|
| hub | LLM Inference | `/concepts/llm-inference/` | `seriesOrder: 2` |
| 1 | Inference Engineering Basics | `.../basics/` | Runtime / infra / tooling; TTFT, ITL, TPS, arithmetic intensity |
| 1.5 | Inference Engineering Mathematics | `.../inference-engineering-mathematics/` | Formulas + `Inference_Math_Basics.xlsx` |
| 2 | GPU Basics — Inside the Chip | `.../gpu-basics/` | SMs, warps, caches, H100 example |
| 3 | GPU Architecture for LLM Inference | `.../gpu-architecture/` | Memory, FLOPS, bandwidth, interconnect, card choice |
| 4 | vLLM Basics and Why KV Cache Matters | `.../vllm-basics-kv-cache/` | Prefill/decode, KV, vLLM wiring |
| 5 | LLM Serving Engine Internals | `.../serving-engine-internals/` | HTTP → GPU loop; `single_model_llm_serving` lab |
| 6 | Serving Multiple Models | `.../serving-multi-models/` | FastAPI + LRU; Qwen / TinyLlama |
| 7 | Batching | `.../inference-optimizations-batching/` | Static / dynamic / continuous / chunked prefill |
| 8 | Quantization | `.../inference-optimizations-quantization/` | Bits, absmax, figures |

Hub “coming later”: speculative decoding, serving trade-offs. Those remain gaps in both books until written.

**Not a routed article:** `src/content/concepts/llm-inference/llm_inference_engineering_concepts_summary_fixed.md` (no frontmatter). Useful private summary; do not glob it into v2.

### Related books (do not merge into v2 routes)

| Series | Useful overlap |
|--------|----------------|
| Transformers | Embeddings, attention, MHA, transformer block — Part 1 prerequisites |
| LLM system design | Latency timeline, reliability, routing — Part 6 product view |
| Reasoning models | Inference-time scaling — out of v2 V1 scope |

---

## Labs and notebooks (not public routes)

Under `notebooks/llm-inference/`:

| Asset | Maps to | Cite in handbook? |
|-------|---------|-------------------|
| `basic_vLLM_v1.ipynb` | Part 3 KV / Part 5 engines | Link after version check |
| `single_model_llm_serving/` | Part 5–6 | Link; do not invent metrics |
| `multi_model_serving/` | Part 6 | Link |
| `gateway.py`, `start_servers.sh` | Part 6 | Phase E |
| `observability/experiments/1.1_single_model_performance.*` | Part 2 benchmarking | Re-run before quoting |
| `observability/experiments/3.1_kv_cache.*` | Part 3 KV | Re-run before quoting |
| `gpu_fundamentals/sharding.ipynb` | Part 4 | Later |
| Roadmaps in `notebooks/llm-inference/docs/` | Planning only | Do not publish as chapters |

---

## Six-part curriculum → V2 slugs

Each topic gets **one** Learn page. Sidebar lists honest `draft` / `in-progress` / `complete` / `planned`. V1 ships the shell plus **one** complete sample (`prefill-vs-decode`). Phase C ships the three flagship chapters.

### Part 1 — Inside LLM inference

| V2 slug | Title | Existing source | V1 status |
|---------|-------|-----------------|-----------|
| `learn/transformer-inference/` | Transformer inference walkthrough | Transformers book + old basics | planned |
| `learn/autoregressive-generation/` | Autoregressive generation and sampling | Old KV chapter §1 | planned |
| `learn/prefill-vs-decode/` | Prefill versus decode | Old `vllm-basics-kv-cache`, GPU arch | **sample chapter (rewrite)** |
| `learn/decoder-tensor-shapes/` | Tensor shapes and data movement | Summary note; transformer figures | planned |
| `learn/mha-mqa-gqa-mla/` | MHA, MQA, GQA, MLA | Transformers MHA; MLA is a **gap** | planned |

### Part 2 — Performance fundamentals

| V2 slug | Title | Existing source | V1 status |
|---------|-------|-----------------|-----------|
| `learn/memory-footprint/` | Parameters, activations, KV memory | Math chapter, GPU arch | planned |
| `learn/flops-and-arithmetic-intensity/` | FLOPs, traffic, intensity | Math chapter, basics | planned (flagship in Phase C as *inference mathematics*) |
| `learn/roofline/` | Roofline, compute vs memory bound | Math + `roofline.png` | planned |
| `learn/gpu-fundamentals/` | SMs, warps, HBM, tensor cores | `gpu-basics`, `gpu-architecture` | planned |
| `learn/latency-throughput-goodput/` | TTFT, ITL/TPOT, concurrency, goodput | Basics; system-design latency | planned |
| `learn/benchmarking/` | Methodology and profiling | Observability notebooks | planned; no fake numbers |

Flagship **Inference mathematics** (`learn/inference-mathematics/`): Phase C rewrite of the 7B-class worked example (FLOPs, parameter memory, KV, intensity, labeled roofline assumptions). Old math article remains at its URL.

### Part 3 — Making inference faster

| V2 slug | Title | Existing source | V1 status |
|---------|-------|-----------------|-----------|
| `learn/kv-cache/` | KV caching and prefix caching | Old KV chapter; `3.1_kv_cache` | Phase C flagship |
| `learn/flashattention/` | FlashAttention / IO-aware kernels | **gap** | planned |
| `learn/paged-attention/` | PagedAttention | Partial in old KV/vLLM | planned |
| `learn/continuous-batching/` | Continuous batching, scheduling, chunked prefill | Batching chapter | planned |
| `learn/quantization/` | FP16/BF16, FP8, INT8, INT4 | Quantization chapter | planned |
| `learn/speculative-decoding/` | Speculative decoding | Hub “coming later” | planned |
| `learn/kernel-fusion-cuda-graphs/` | Fusion, compile, CUDA graphs | **gap** | planned |

### Part 4 — Scaling beyond one GPU

| V2 slug | Title | Existing source | V1 status |
|---------|-------|-----------------|-----------|
| `learn/data-parallelism/` | Replicas | GPU arch interconnect | planned |
| `learn/tensor-parallelism/` | TP | Thin coverage | planned |
| `learn/pipeline-parallelism/` | PP | **gap** | planned |
| `learn/expert-parallelism/` | MoE EP | **gap** | planned |
| `learn/context-parallelism/` | Sequence/context parallel | **gap** | planned |
| `learn/nccl-collectives/` | NCCL and communication cost | GPU multi-node figure | planned |

`sharding.ipynb` is a starting lab, not a chapter.

### Part 5 — Building inference engines

| V2 slug | Title | Existing source | V1 status |
|---------|-------|-----------------|-----------|
| `learn/pytorch-baseline/` | Minimal native PyTorch inference | **gap** (must write, not claim) | planned |
| `learn/caching-batching-scheduling/` | Incremental engine | Serving internals lab | planned |
| `learn/vllm-internals/` | vLLM vs baseline | Old KV + engine internals | planned |
| `learn/sglang-trtllm/` | SGLang / TensorRT-LLM overviews | **gap** | planned |
| `learn/engine-benchmarks/` | Engine benchmark methodology | Observability docs | planned |

### Part 6 — Production LLM serving

| V2 slug | Title | Existing source | V1 status |
|---------|-------|-----------------|-----------|
| `learn/gateway-registry-routing/` | Gateway, registry, routing, health | `gateway.py`, multi-model | Phase E |
| `learn/queues-admission-lb/` | Queues, admission, fallback | System-design reliability | Phase E |
| `learn/observability-slos/` | Metrics, tracing, logging, SLOs | Observability PLAN | Phase E |
| `learn/containers-k8s/` | Deploy; K8s when justified | **gap** | Phase E |
| `learn/autoscaling-fault-tolerance/` | Replicas, FT | **gap** | Phase E |
| `learn/capacity-and-cost/` | Utilization, cost per token | Math chapter cost section | Phase E |

---

## Reference and journal (not Learn silos)

| URL | V1 | Later |
|-----|-----|--------|
| `/concepts/llm-inference-v2/reference/` | Glossary + formula list skeleton | Calculators, GPU/model tables |
| `/concepts/llm-inference-v2/journal/` | Empty state | Dated experiments, failures, decisions |

No top-level Labs section. Long labs link from the chapter (`notebooks/llm-inference/...` or later `experiments/inference-v2/`).

---

## Learning tracks (metadata only in V1)

Tracks reuse canonical chapters; do not duplicate pages.

| Track | First chapters (when they exist) |
|-------|----------------------------------|
| Fundamentals | transformer-inference → prefill-vs-decode → gpu-fundamentals → inference-mathematics |
| Optimization | kv-cache → continuous-batching → quantization → speculative-decoding |
| Production serving | pytorch-baseline → vllm-internals → gateway-registry-routing → observability-slos |

---

## Duplicate / rewrite policy

- Old URLs stay canonical for the **current** book.
- V2 chapters are rewritten to the ten-section template (problem, intuition, mechanics, math, build, optimize, measure, trade-offs, exercises, sources).
- Do not copy-paste old markdown into v2.
- Cross-link: “Narrative version in the original notes: …”
- Never publish benchmark tables without hardware, software versions, model, precision, lengths, batch/concurrency, warmup, reps, metric definitions, and limitations.

---

## V1 content actually written in Phase B

1. Landing page with six-part map and tracks (tracks may point at planned chapters).
2. Learn index with statuses.
3. One substantive sample: **Prefill vs decode**.
4. Reference and journal empty states.
5. Sidebar entries for the rest of the curriculum marked *planned*.

Phase C (after B): deepen prefill vs decode if needed; add **Inference mathematics** and **KV cache**.
