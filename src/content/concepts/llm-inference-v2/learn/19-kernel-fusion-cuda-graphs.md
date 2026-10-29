---
title: "Kernel fusion, compilation, and CUDA graphs"
description: "Launch overhead, fusion, graph capture, and honest GPU timing."
section: learn
pubDate: 2026-10-29
part: 3
chapter: 19
order: 19
tags: ["llm-inference", "systems", "hands-on"]
---

# Kernel fusion, compilation, and CUDA graphs

Decode consists of many operations on relatively small activation tensors. When kernels are short, launch and dispatch overhead can become a material part of latency.

## 1. Fusion

Consider `y = silu(x + residual)`. Separate kernels can write and reread the addition result. A fused kernel can compute the addition and activation while the intermediate stays in registers. Fewer memory transfers and launches help; excessive fusion can raise register pressure and reduce occupancy.

## 2. Compilation and CUDA graphs

A compiler can specialize operations and fuse compatible regions. CUDA graphs record a compatible GPU work sequence for replay with lower dispatch overhead. They do not remove the underlying matrix multiplications. Dynamic shapes, changing addresses, and control flow complicate capture; engines often maintain graph variants or padded execution sizes.

## 3. Time GPU work correctly

```python
# Requires a CUDA-capable PyTorch installation and GPU.
import torch
if torch.cuda.is_available():
    x = torch.randn(1024, 1024, device='cuda')
    with torch.inference_mode():
        for _ in range(10):
            y = x @ x
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(50):
            y = x @ x
        end.record()
        end.synchronize()
        print(start.elapsed_time(end) / 50, 'ms per matmul')
else:
    print('CUDA required for this measurement')
```

Warmup avoids counting first-use initialization. Events measure device work between the events; request wall time includes queueing and CPU work too. Record model, GPU, dtype, shapes, software versions, and concurrency alongside results. The example measures repeated matrix multiplication, not LLM serving throughput.
