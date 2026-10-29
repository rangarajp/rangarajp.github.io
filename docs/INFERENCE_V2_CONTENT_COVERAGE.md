# Inference v2 content coverage

The original inference guide is retained. Its published content is adapted into the single v2 chapter list, with shared figure assets and preserved companion-lab links.

| Original source | V2 destination | Material carried forward |
|---|---|---|
| basics.md | latency-throughput-goodput | Runtime/infrastructure, streaming metrics, percentiles, GQA traffic calculation |
| gpu-basics.md | gpu-fundamentals | SMs, execution units, warps, registers, memory hierarchy, divergence |
| gpu-architecture.md | memory-footprint; inference-mathematics | Weight/KV/workspace budgets, concurrency examples, capacity versus bandwidth, device selection |
| inference-engineering-mathematics.md | inference-mathematics | Dot products, GEMMs, roofline, precision, collectives, 70B memory floor, metrics and cost |
| vllm-basics-kv-cache.md | prefill-decode-kv-cache; vllm-internals | Historical cache traces, figures, engine setup, sampling, batch observations, streaming |
| inference-optimizations-batching.md | continuous-batching | Ferry analogy, static/dynamic/continuous batching, chunked prefill, timeline figures |
| inference-optimizations-quantization.md | quantization | Bits, float layouts, 0.73 worked example, staircases, histograms, error and absmax figures |
| serving-engine-internals.md | caching-batching-scheduling | Request lifecycle, IDs, batching, worker queues, full recorded FLOW traces, streaming, replicas |
| serving-multi-models.md | gateway-registry-routing | Registry, LRU workers, APIs, recorded scenarios, eviction, cluster placement |
| llm_inference_engineering_concepts_summary_fixed.md (unpublished working notes) | Existing attention, cache, FlashAttention, paging, speculation and parallelism chapters | Replica serving and measured PyTorch baseline added; consolidated explanations already represented; repeated roadmaps and study questions are not added |

Corrections made while adapting: queueing is counted once in TTFT/E2E; TPOT excludes first-token delay; serialized compute plus memory is not a runtime upper bound; cached attention still grows with context; dense versus sparse FLOP rates are distinguished; quantization speedups are conditional; zero-point mapping and numerical precision definitions are corrected. Hardware price snapshots and speculative roadmap assertions are omitted. Recorded lab outputs are historical and have not been remeasured.

Math uses dollar delimiters outside code. The math audit parses Markdown with the same math syntax and validates every formula with KaTeX. Production build checks catch renderer failures; search checks exercise both guides.
