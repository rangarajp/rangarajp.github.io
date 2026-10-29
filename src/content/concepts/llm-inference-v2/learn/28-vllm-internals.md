---
title: "vLLM from a generate call to streaming"
description: "Load once, batch prompts, control sampling, and stream through an OpenAI-compatible server."
section: learn
pubDate: 2026-10-29
part: 5
chapter: 25
order: 25
tags: ["llm-inference", "systems", "hands-on"]
---

The individual mechanisms are now familiar. Use vLLM as the integrated engine example: load a model, submit prompts, and stream results and connect its behavior to the request lifecycle in the teaching server.

Start with a supported GPU installation of vLLM. This chapter adapts the original [vLLM notebook](https://github.com/rangarajp/rangarajp.github.io/blob/main/notebooks/llm-inference/basic_vLLM_v1.ipynb). Its recorded environment used vLLM 0.6.1.post1; the basic `LLM`/`SamplingParams` interface remains the entry point, while internal async APIs differ by release.

## 1. vLLM: all of this managed for you

Manually managing K/V tensors, handling multiple concurrent users, and keeping the GPU busy is a significant engineering job. *vLLM* is an inference engine that handles it — you write prompts and sampling parameters, it handles the rest.

### Load once, reuse everywhere

```python
from vllm import LLM, SamplingParams
MODEL_PATH = "Qwen/Qwen2.5-0.5B-Instruct"  # or an existing local snapshot directory

llm = LLM(
    model=MODEL_PATH,
    dtype="float16",
    gpu_memory_utilization=0.30,
    max_model_len=2048,
    max_num_seqs=8,
)
```

Weights load once — ≈ 0.93 GB for this 0.5B model. vLLM then pre-allocates a *paged KV block pool* so it can serve many concurrent requests without memory fragmentation.

| Knob | Role |
| ---- | ---- |
| `dtype` | Weight precision |
| `gpu_memory_utilization` | Fraction of GPU for weights + KV blocks |
| `max_model_len` | Max prompt + generation length |
| `max_num_seqs` | Max concurrent sequences in one batch |

### First completion

```python
params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=128)
prompt = "Explain why a battery needs cooling in one sentence."
outputs = llm.generate([prompt], params)
text = outputs[0].outputs[0].text
```

*Run output:*
```
prompt tokens ~ 29
new tokens    = 128

est. speed input: 48.53 toks/s, output: 214.20 toks/s
```

The call blocks until all 128 tokens are ready. Fine for batch jobs.

### Sampling knobs

`SamplingParams` controls *how* you draw from the next-token distribution — the model weights stay fixed.

| Param | Effect |
| ----- | ------ |
| `temperature` | Higher → more random; `0` = greedy (always pick highest probability) |
| `top_p` | Keep smallest token set covering p% of probability mass |
| `top_k` | Keep only the top-k most probable tokens |
| `max_tokens` | Stop after this many new tokens |
| `stop` | Stop on specific strings (e.g. `["\n"]`) |

*Same prompt, two settings — from the run:*

```
[greedy (T=0)]    The capital of France is Paris. It is the largest city in
                  Europe and the third largest city in the world...

[creative (T=1.0)] The capital of France is:
                   A) Paris
                   B) London
                   C) Berlin
                   D) Moscow
                   To determine the capital of France...
```

Same model. Same weights. Entirely different output character.

### Batching: one of the biggest wins

vLLM can run multiple prompts through shared scheduled model iterations. The scheduler can combine their work and reuse loaded weights. Occupancy and speedup depend on shapes and hardware.

*Measured:*

```
4 prompts in one batch : 0.379s   (~688 output tokens/s)
4 prompts one-by-one   : 1.145s   (~226 output tokens/s)
speedup                : 3.02×
```

These numbers belong to the original small-model notebook run, not a guarantee for other workloads. The high-level API accepts a list of prompts; compare that batch with sequential calls using matched output lengths and warmup.



## 2. Serve and stream through the HTTP API

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct --dtype float16 --max-model-len 2048
```

With the server running, this client consumes OpenAI-compatible server-sent events and prints content deltas. It does not assume that one HTTP chunk equals one token.

```python
import json
import httpx

with httpx.Client(timeout=120.0) as client:
    with client.stream("POST", "http://127.0.0.1:8000/v1/chat/completions", json={
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "messages": [{"role": "user", "content": "Explain KV caching briefly."}],
        "max_tokens": 128, "temperature": 0, "stream": True,
    }) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            event = json.loads(payload)
            for choice in event.get("choices", []):
                print(choice.get("delta", {}).get("content") or "", end="", flush=True)
print()
```

Model download, GPU memory, dtype, and attention kernels must be compatible with the environment. Some GPUs do not support BF16 or FP8. Match the engine version to the model rather than changing flags until it happens to load.

## 3. What the engine manages

The scheduler selects request work; the allocator assigns KV blocks; the executor runs model operations; sampling updates sequence state; delivery streams results to the caller. Prefix caching reuses identical prompt state, while paging manages its placement. A request ending early must release its reservation.

Engine configuration names and defaults vary by version. Record model revision, runtime version, token budgets, KV dtype, and warmup in every benchmark. See [the official quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart/) and [OpenAI-compatible serving](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/).

## V100 lab: label the model, engine version, and timing boundary

The original [generation notebook](https://github.com/rangarajp/rangarajp.github.io/blob/main/notebooks/llm-inference/basic_vLLM_v1.ipynb) uses Qwen2.5-0.5B-Instruct on Tesla V100-SXM2-32GB. The saved [server environment](https://github.com/rangarajp/rangarajp.github.io/blob/main/notebooks/llm-inference/single_model_llm_serving/main.ipynb) records vLLM `0.6.1.post1` with PyTorch `2.4.0+cu118`. The batch timings reproduced above belong to that historical setup; the command-line/API example describes the engine interface and is not evidence that the newest release supports Volta.

The basic notebook’s introductory Markdown names a 7B checkpoint, but its saved setup output prints `Qwen2.5-0.5B-Instruct`. The batch figures are therefore presented as historical outputs associated with the printed setup, not as a fresh 7B benchmark. Re-running the notebook requires recording the resolved model path/configuration and actual completion lengths.

For the separate Qwen2.5-7B experiment, the [cache notebook](https://github.com/rangarajp/rangarajp.github.io/blob/main/notebooks/llm-inference/observability/experiments/3.1_kv_cache.ipynb) records a vLLM sample with prefix caching off: TTFT 42.7 ms, end-to-end 1583.5 ms, and 64 output tokens. Its displayed 40.4 tok/s is $64/1.5835$, an end-to-end output rate including startup delay. It is not the reciprocal of steady inter-token latency.

## Hands-on: compare batch and sequential engine calls

Reproduce the original four-prompt comparison using a compatible pinned engine. Report actual generated tokens, wall time, model revision, batch settings, and warmup. Add a streamed-client trace and compare server processing time with client delivery time.
