# PDF structure and content mapping

The attached second edition supplies the 14-chapter structure and numbered conceptual sections. Existing explanations, equations, code, and notebook evidence are integrated below those sections. Interview Q&A is excluded. Hardware and performance claims retain the corrections documented in the original content audit.

| PDF chapter | Blog file | Main sections |
|---|---|---|
| 1. The inference problem | `00-inference-problem.md` | 1.1 Training versus inference: fundamentally different goals; 1.2 The token generation loop explained; 1.3 Latency, throughput, and cost: the iron triangle; 1.4 Memory bandwidth: the decode bottleneck; 1.5 The roofline model |
| 2. Hardware foundations for inference | `10-gpu-fundamentals.md` | 2.1 GPU architecture for the inference engineer; 2.2 Tensor cores and mixed precision; 2.3 Memory hierarchy and bandwidth; 2.4 CPU and edge hardware; 2.5 Emerging hardware trends |
| 3. Transformer inference mechanics | `02-transformer-forward-pass.md` | 3.1 Anatomy of a transformer forward pass; 3.2 The KV cache: what it is and why it matters; 3.3 Prefill versus decode: two different workloads; 3.4 Attention variants: MHA to MLA; 3.5 Positional encodings at inference time |
| 4. Quantization | `17-quantization.md` | 4.1 Why quantization matters; 4.2 Number formats; 4.3 Post-training quantization methods; 4.4 Quantization-aware training; 4.5 Activation quantization challenges |
| 5. Speculative decoding | `18-speculative-decoding.md` | 5.1 The core insight: draft and verify; 5.2 Token tree verification; 5.3 Medusa: multi-head speculative decoding; 5.4 EAGLE: feature-based drafting; 5.5 When speculative decoding helps |
| 6. KV cache optimization | `13-kv-cache.md` | 6.1 The KV cache memory crisis; 6.2 PagedAttention: virtual memory for KV caches; 6.3 Prefix caching: reusing shared prompts; 6.4 StreamingLLM and attention sinks; 6.5 KV cache quantization |
| 7. Kernel engineering and FlashAttention | `14-flashattention.md` | 7.1 Why custom CUDA kernels matter; 7.2 FlashAttention: IO-aware exact attention; 7.3 The online softmax trick; 7.4 Triton: Pythonic kernel programming |
| 8. Serving systems architecture | `16-continuous-batching.md` | 8.1 Anatomy of a production inference server; 8.2 Batching strategies; 8.3 Disaggregated prefill and decode; 8.4 SLO-aware scheduling |
| 9. Parallelism for large model inference | `20-data-parallelism.md` | 9.1 Why parallelism is necessary; 9.2 Tensor parallelism; 9.3 Pipeline parallelism; 9.4 Mixture of experts inference |
| 10. Long-context and memory management | `24-context-parallelism.md` | 10.1 The quadratic attention problem; 10.2 FlashAttention for long context; 10.3 Ring attention; 10.4 Context compression; 10.5 State space models and recurrent memory |
| 11. Edge and on-device inference | `37-edge-inference.md` | 11.1 The edge inference landscape; 11.2 Memory constraints and quantization for edge; 11.3 Runtime frameworks; 11.4 Neural architecture search for efficiency |
| 12. Inference-time compute scaling | `38-inference-time-scaling.md` | 12.1 The new scaling paradigm; 12.2 Chain-of-thought and its inference cost; 12.3 Process reward models and search; 12.4 DeepSeek-R1 and o1-style reasoning models |
| 13. Observability and production engineering | `33-observability-slos.md` | 13.1 The production inference stack; 13.2 Key metrics in depth; 13.3 Failure modes and debugging; 13.4 Cost modeling |
| 14. The future of inference | `39-future-of-inference.md` | 14.1 Diffusion-based language models; 14.2 Multi-token prediction; 14.3 Neuromorphic and photonic computing; 14.4 The inference-first design philosophy |

Recorded notebook observations and proposed hands-on extensions follow each chapter’s conceptual sections. The HBM transfer table and training-reuse explanation remain after the roofline introduction in 1.5. Old chapter URLs redirect to the consolidated chapter; original inference v1 remains available.
