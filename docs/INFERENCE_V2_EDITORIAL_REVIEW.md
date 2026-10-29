# Inference v2 editorial review

All 32 written chapters were reviewed for scope, transitions, terminology, source context, and hands-on opportunities. A single internal reading order powers chapter numbers, sidebar, and previous/next navigation. GPU fundamentals now precedes memory sizing, the measured PyTorch baseline precedes advanced optimization, and vLLM precedes the single-model server walkthrough. No extra user-facing tracks or roadmap were added.

Each chapter now has a purpose paragraph and a specific proposed hands-on extension with an artifact to publish. Duplicate difficulty/Q&A badges and embedded Previous/Next links were removed; layout navigation remains authoritative. Paragraph justification is scoped to article prose; headings, code, equations, tables, and navigation keep their existing layout.

## Evidence boundaries

- `basic_vLLM_v1.ipynb`: recorded Tesla V100-SXM2-32GB, Torch 2.4.0+cu118, Qwen2.5-0.5B-Instruct. Four-prompt batch/sequential times are 0.379/1.145 s. They are workload-specific wall times. The introductory Markdown currently names 7B while saved setup output names 0.5B; the blog discloses this inconsistency and attributes the historical numbers to the saved output rather than an assumed rerun.
- `single_model_llm_serving/main.ipynb`: records V100, Torch 2.4.0+cu118, vLLM 0.6.1.post1 and eager execution. No graph-replay comparison is claimed.
- `observability/experiments/3.1_kv_cache.ipynb`: records V100, Qwen2.5-7B, 28 layers/28 query heads/4 KV heads/128 head dimension. Full-prefix versus cached-forward sweep, actual cache tensor bytes, and explicit Hugging Face prefix-reuse timings are carried into the relevant chapters. Its interpolated total-generation estimate is identified as an estimate.
- `observability/experiments/1.1_single_model_performance.md`: supplies the server concurrency summary. It is separate from the 0.5B offline batch result. The associated notebook warns that chunked-prefill timelines can be reconstructions; no reconstructed gain is labeled a live measurement.
- `gpu_fundamentals/sharding.ipynb`: algebraic sharding and a distributed section requiring two devices are distinguished. No single-V100 multi-GPU scaling claim is made.

Volta compatibility notes are based on NVIDIA's Volta tuning documentation and the FlashAttention implementation requirements. H100 reference calculations remain explicitly separate from measured V100 results. GPU experiments were not rerun during this editorial revision.

## Per-chapter hands-on proposals

- `00-inference-problem.md`: trace one request.
- `01-from-text-to-tokens.md`: inspect tokenization and position.
- `02-transformer-forward-pass.md`: check attention shapes and masking.
- `03-logits-and-sampling.md`: plot a controlled sampler.
- `04-autoregressive-generation.md`: expose the growing prefix.
- `05-prefill-decode-kv-cache.md`: publish the context-length sweep.
- `06-attention-variants.md`: measure KV storage rather than expanded tensors.
- `07-memory-footprint.md`: compare predicted and allocated cache bytes.
- `08-inference-mathematics.md`: overlay estimates and measurements.
- `10-gpu-fundamentals.md`: profile matrix shapes on the V100.
- `11-latency-throughput-goodput.md`: show the throughput knee.
- `13-kv-cache.md`: measure prefix reuse with cold and warm cases.
- `14-flashattention.md`: validate IO savings on supported backends.
- `15-paged-attention.md`: simulate block allocation and reclamation.
- `16-continuous-batching.md`: separate live stalls from reconstructed timelines.
- `17-quantization.md`: connect weight error to task behavior.
- `18-speculative-decoding.md`: measure acceptance and its overhead.
- `19-kernel-fusion-cuda-graphs.md`: isolate dispatch overhead.
- `20-data-parallelism.md`: compare routing policies before adding replicas.
- `21-tensor-parallelism.md`: verify sharding algebra, then communication.
- `22-pipeline-parallelism.md`: draw a stage utilization timeline.
- `23-expert-parallelism.md`: stress uneven routing.
- `24-context-parallelism.md`: compare memory scaling and retained information.
- `26-pytorch-baseline.md`: save a reproducible baseline report.
- `27-caching-batching-scheduling.md`: trace cancellation and cleanup.
- `28-vllm-internals.md`: compare batch and sequential engine calls.
- `31-gateway-registry-routing.md`: expose cold starts and safe eviction.
- `33-observability-slos.md`: build a small load-to-SLO report.
- `36-capacity-and-cost.md`: turn measured goodput into a sensitivity table.
- `37-edge-inference.md`: measure a real edge device.
- `38-inference-time-scaling.md`: spend a fixed candidate budget.
- `39-future-of-inference.md`: compare one alternative at matched quality.

The PDF topic sequence in `INFERENCE_V2_PDF_SECTION_AUDIT.md` now governs chapter order; earlier numbering in this review describes the previous arrangement. Content and recorded-lab evidence remain applicable.
