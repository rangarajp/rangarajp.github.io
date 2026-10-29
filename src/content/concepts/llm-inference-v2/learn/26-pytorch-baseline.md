---
title: "A measured PyTorch inference baseline"
description: "Compare cached and uncached generation with matched prompts, warmup, output checks, and explicit timing boundaries."
section: learn
pubDate: 2026-10-29
part: 5
chapter: 12
order: 12
---

The model mechanics, memory budget, and metrics are in place. Establish a controlled reference here before introducing paging, batching, lower precision, or a serving engine.

Before tuning a serving engine, establish a small baseline that you can inspect. The original guide and working notes recommend building intuition in PyTorch first: measure, identify the resource limit, change one mechanism, and measure again.

## 1. Compare the cache switch fairly

This complete example uses a small Hugging Face model on CUDA when available, otherwise CPU. It requires `torch` and `transformers`, a model download or local snapshot, and enough RAM for model weights and state. It measures whole generation calls, not TTFT or per-step decode latency.

```python
import statistics
import time
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = 'Qwen/Qwen2.5-0.5B-Instruct'  # or a local snapshot
print('torch:', torch.__version__, 'transformers:', transformers.__version__)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to(device).eval()
text = tokenizer.apply_chat_template(
    [{'role': 'user', 'content': 'Explain why a battery needs cooling in one sentence.'}],
    tokenize=False, add_generation_prompt=True,
)
inputs = tokenizer(text, return_tensors='pt').to(device)

def synchronize():
    if device == 'cuda':
        torch.cuda.synchronize()

def run(use_cache):
    synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        ids = model.generate(**inputs, max_new_tokens=32, do_sample=False,
                             use_cache=use_cache, pad_token_id=tokenizer.eos_token_id)
    synchronize()
    elapsed = time.perf_counter() - start
    new_ids = ids[0, inputs.input_ids.shape[1]:].cpu()
    return new_ids, elapsed

# Warm both execution paths before collecting timed samples.
run(False)
run(True)
results = {False: [], True: []}
outputs = {}
for _ in range(3):
    for cached in (False, True):
        tokens, seconds = run(cached)
        results[cached].append(seconds)
        outputs[cached] = tokens
for cached, samples in results.items():
    median = statistics.median(samples)
    print('cache:', cached, 'tokens:', len(outputs[cached]),
          'median seconds:', round(median, 4))
print('identical token IDs:', torch.equal(outputs[False], outputs[True]))
```

Early EOS may produce fewer than 32 tokens; report actual output lengths. Floating-point differences can change a near-tie in greedy decoding, so inspect output differences rather than assuming a timing comparison proves equivalence. Do not claim a fixed speedup from these few runs.

## 2. Separate generation from serving

The timer starts after tokenization and ends after GPU completion. It excludes model loading, HTTP, queueing, and detokenization. Those boundaries make this a model-execution baseline. A production request timer must include the omitted service work.

To investigate context scaling, repeat with controlled longer prompts and record the actual tokenizer length. To investigate batching, use multiple prompts with left padding and report total emitted tokens, per-request latency, and padding costs. A single model call does not implement continuous admission of new HTTP requests.

## 3. Move from baseline to engine

Use [prefill and decode](./prefill-decode-kv-cache) to interpret state growth, [kernel timing](./kernel-fusion-cuda-graphs) to isolate device work, and [vLLM](./vllm-internals) to study scheduling and paged allocation. Keep the model, revision, precision, prompt distribution, and stopping rules fixed when comparing them.

API details: [Hugging Face generation documentation](https://huggingface.co/docs/transformers/main_classes/text_generation).

## V100 lab: establish what each comparison timed

The [basic notebook](https://github.com/rangarajp/rangarajp.github.io/blob/main/notebooks/llm-inference/basic_vLLM_v1.ipynb) already exposes uncached generation and a short cached trace. The [7B cache notebook](https://github.com/rangarajp/rangarajp.github.io/blob/main/notebooks/llm-inference/observability/experiments/3.1_kv_cache.ipynb) then isolates full-prefix forwards and cached single-token forwards across context lengths. These are useful separate baselines: one shows a generation loop, the other isolates a mechanism.

The complete script above measures whole generation calls and can extend that work. Its results have not been recorded in this revision. In particular, the 7B notebook's interpolated generation-cost estimate is not a replacement for running this end-to-end comparison.

## Hands-on: save a reproducible baseline report

Run cached and uncached paths with the same prompt set and actual output lengths. Save model revision, device, library versions, dtype, warmup, medians, peak memory, and output equality. Keep CPU and GPU timing results in separate reports.
