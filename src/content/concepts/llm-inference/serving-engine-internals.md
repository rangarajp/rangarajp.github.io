---
title: 'LLM Serving Engine Internals'
description: 'A restaurant walkthrough of a single-model serving stack — then real HTTP examples that show which layers run and which are skipped.'
pubDate: 'Sep 13 2026'
order: 3
heroImage: '../../../assets/blog-placeholder-3.jpg'
---

Once you can generate tokens (see [vLLM Basics and Why KV Cache Matters](./vllm-basics-kv-cache)), the next question is: **how does a request travel from HTTP to GPU and back?**

This article walks a teaching stack in `notebooks/llm-inference/single_model_llm_serving/`. It is intentionally explicit — separate processes, queues, and a workload manager — so you can see roles that production engines often hide inside one binary.

Lab setup: run the server from `main.ipynb` (or `main.py`), then hit it from `test.ipynb`. Server-side FLOW prints show every station that cooked — and every station that was **skipped**.

## 1. Restaurant analogy

Think of inference as a busy kitchen.

| Restaurant role | Component | Job |
| --------------- | --------- | --- |
| Customer | Client (`test.ipynb` / curl) | Places an order (prompt) |
| Waiter | FastAPI | Takes the order, brings the plate back |
| Head chef | `LLMEngine` | Owns the ticket end-to-end; picks the cooking path |
| Host | `WorkloadManager` | Tracks open tickets; seats them into batches (tables of up to 4) |
| Kitchen manager | `ModelExecutor` | Talks to the line cooks; owns the pass window |
| Pass window | `task_queue` / `result_queue` | Orders out, plates back (cross-process) |
| Line cook | `ModelWorker` | Runs the model on GPU |
| Pantry | `ModelManager` | Loads weights + tokenizer **once** at open |

![Single-model serving architecture](./images/serving-engine-architecture.png)

**Main process** (front of house + head chef + manager) stays responsive. **Worker process** (cooks) does the heavy GPU work.

```text
Client
  → FastAPI endpoint
  → LLMEngine
  → WorkloadManager          (sometimes skipped)
  → ModelExecutor
  → task_queue  |  result_queue
  → ModelWorker
  → ModelManager / model
  → GPU
```

Not every menu item walks every station. That is the whole point of the examples below.

---

## 2. Example A — Basic generate (one prompt)

**Customer order** (`test.ipynb`):

```python
import httpx

with httpx.Client(proxy=None, trust_env=False) as client:
    response = client.post(
        "http://127.0.0.1:8000/basic_generate",
        json={"prompt": "What is the capital of the United States?"},
        timeout=60.0,
    )

print(response.status_code)
print(response.text)
```

**Client output:**

```text
200
{"generated_text":"What is the capital of the United States? The capital of the United States is Washington, D.C. ..."}
```

**What the kitchen did** (FLOW on the server):

```text
============================================================
 REQUEST  POST /basic_generate
============================================================
>>> [FastAPI] /basic_generate — prompt='What is the capital of the United States?'
>>> [FastAPI] → LLMEngine.basic_generate()
--- SKIP [WorkloadManager] basic_generate bypasses queue/batching; goes straight to ModelExecutor
>>> [LLMEngine] basic_generate(...)
--- SKIP [WorkloadManager] basic_generate builds Sequence locally; no add_request / get_next_batch
>>> [LLMEngine] → ModelExecutor.execute_batch([sequence])
>>> [ModelExecutor] execute_batch — sending 1 item(s) to worker (is_streaming=False)
>>> [ModelWorker] task received — is_streaming=False, batch_size=1
>>> [ModelWorker] generate() — batch size=1
--- SKIP [ModelManager] model already loaded at worker startup; not reloaded per request
>>> [GPU] model.generate() on cuda  input_shape=(1, 9)
...
 DONE     POST /basic_generate
```

**How to read this**

| Layer | Called? | Why |
| ----- | ------- | --- |
| FastAPI | Yes | Waiter takes the order |
| LLMEngine | Yes | Head chef builds one local `Sequence` |
| WorkloadManager (Host) | **Skip** | No queue / no batch — one plate goes straight to the kitchen |
| ModelExecutor → Worker → GPU | Yes | Full `model.generate` for that one prompt |
| ModelManager | **Skip** (per request) | Pantry already stocked at worker startup |

Restaurant version: customer asks for one dish; waiter skips the host’s seating list and sends the ticket straight to the kitchen manager.

---

## 3. Example B — Batch generate (four prompts, one request)

**Customer order:**

```python
payload = {
    "prompts": [
        "The capital of France is",
        "The capital of India is",
        "The capital of Japan is",
        "The largest planet is",
    ]
}

with httpx.Client(trust_env=False) as client:
    response = client.post(
        "http://127.0.0.1:8000/generate",
        json=payload,
        timeout=60.0,
    )

print(response.status_code)
print(response.json())
```

**Client output (shape):**

```text
200
{
  "generated_texts": [
    "The capital of France is Paris. ...",
    "The capital of India is ...",
    "The capital of Japan is ...",
    "The largest planet is ..."
  ]
}
```

**What the kitchen did:**

```text
============================================================
 REQUEST  POST /generate
============================================================
>>> [FastAPI] /generate — 4 prompts
>>> [FastAPI] → LLMEngine.generate()  [uses WorkloadManager batching]
>>> [LLMEngine] generate(4 prompts) — uses WorkloadManager
>>> [WorkloadManager] add_request → incoming_queue  id=4e010141… prompt='The capital of France is'
>>> [WorkloadManager] add_request → incoming_queue  id=e7720f6e… prompt='The capital of India is'
>>> [WorkloadManager] add_request → incoming_queue  id=2a54b380… prompt='The capital of Japan is'
>>> [WorkloadManager] add_request → incoming_queue  id=b586257b… prompt='The largest planet is'
>>> [WorkloadManager] get_next_batch(batch) — pulled 4, active=4
>>> [LLMEngine] batch loop — 4 seq(s) → ModelExecutor.execute_batch()
>>> [ModelExecutor] execute_batch — sending 4 item(s) to worker (is_streaming=False)
>>> [ModelWorker] task received — is_streaming=False, batch_size=4
>>> [ModelWorker] generate() — batch size=4
--- SKIP [ModelManager] model already loaded at worker startup; not reloaded per request
>>> [GPU] model.generate() on cuda  input_shape=(4, 5)
>>> [GPU] ← model.generate() finished
>>> [WorkloadManager] update_sequence_output ... finished=True
>>> [LLMEngine] ← returning 4 generated texts
 DONE     POST /generate
```

**How to read this**

| Layer | Called? | Why |
| ----- | ------- | --- |
| WorkloadManager (Host) | **Yes** | Four tickets queued, then seated as one table (`batch_size=4`) |
| ModelWorker / GPU | **Yes** | One batched generate — note `input_shape=(4, 5)` (4 sequences) |
| ModelManager | **Skip** per request | Same pantry rule |

Restaurant version: host seats four guests at one table; kitchen cooks the table together instead of four separate fires.

Contrast with basic generate: **Host is on the path** here; it was **skipped** for `/basic_generate`.

---

## 4. Example C — Concurrent clients (many basic_generate at once)

**Customer order** — four async clients, each calling `/basic_generate`:

```python
prompts = [
    "What is KV cache?",
    "What is attention?",
    "What is speculative decoding?",
    "What is continuous batching?",
]

async def call_api(prompt):
    async with httpx.AsyncClient(trust_env=False) as client:
        start = time.perf_counter()
        r = await client.post(
            "http://127.0.0.1:8000/basic_generate",
            json={"prompt": prompt},
            timeout=60.0,
        )
        return {"prompt": prompt, "time": round(time.perf_counter() - start, 2), "response": r.json()}

results = await asyncio.gather(*[call_api(p) for p in prompts])
```

**Client output (from the lab):**

```text
{'prompt': 'What is KV cache?',             'time': 1.61, ...}
{'prompt': 'What is attention?',            'time': 3.04, ...}
{'prompt': 'What is speculative decoding?', 'time': 4.45, ...}
{'prompt': 'What is continuous batching?',  'time': 5.86, ...}
```

**What the kitchen did:** four separate `POST /basic_generate` FLOW banners — each still **skips WorkloadManager** and sends `batch_size=1` to the worker.

So concurrency at the HTTP layer ≠ batching inside the custom stack. These four orders arrive as four tickets; each walks the “basic generate” path (Host skipped). Wall-clock times climb because the worker still finishes one full generate before the next (or interleaves poorly compared to a true batch).

That is why Example B’s `/generate` matters: **one HTTP request, Host forms a table of 4, one GPU call with `input_shape=(4, …)`**.

---

## 5. Example D — Streaming

**Customer order:**

```python
with httpx.Client(trust_env=False) as client:
    with client.stream(
        "POST",
        "http://127.0.0.1:8000/generate_stream",
        json={"prompt": "What is the capital of the United States?"},
        timeout=60.0,
    ) as response:
        for line in response.iter_lines():
            if line:
                print(line)
```

**Expected SSE shape:**

```text
data: {"token": " Washington", "sequence_id": "…"}
data: {"token": ",", "sequence_id": "…"}
...
```

**What the kitchen does (FLOW pattern):**

```text
 REQUEST  POST /generate_stream
>>> [FastAPI] → LLMEngine.event_generator()  [uses WorkloadManager + streaming loop]
>>> [WorkloadManager] add_streaming_request → streaming_queue
>>> [LLMEngine] streaming loop — N active seq(s) → ModelExecutor.execute_forward_batch()
>>> [ModelExecutor] execute_forward_batch — is_streaming=True
>>> [ModelWorker] generate_forward_batch() — one token each
>>> [GPU] model forward() ...
>>> [LLMEngine] streaming loop — token=… → client queue
```

| Layer | Called? | Why |
| ----- | ------- | --- |
| WorkloadManager (Host) | **Yes** | Streaming queue + active streaming sequences |
| ModelExecutor | **Yes** | `execute_forward_batch` (one token / step), not full `generate` |
| ModelWorker | **Yes** | Forward + sample, repeatedly until done |

Restaurant version: waiter keeps the plate at the table and brings **bite by bite**; host keeps the ticket on the streaming board; cook fires one step at a time through the pass window.

---

## 6. Example E — vLLM shortcut (`/generate_vllm`)

Same dining room (FastAPI), different kitchen:

```text
 REQUEST  POST /generate_vllm
>>> [FastAPI] → LLMEngine.generate_vllm()
--- SKIP [WorkloadManager] vLLM path does not use WorkloadManager
--- SKIP [ModelExecutor] vLLM path does not use ModelExecutor
--- SKIP [ModelWorker] vLLM path does not use ModelWorker
--- SKIP [ModelManager] vLLM loads/runs its own engine
>>> [LLMEngine] → vLLM.generate()  [GPU via vLLM engine]
```

| Layer | Called? |
| ----- | ------- |
| Custom Host / Executor / Worker / Pantry | **All skipped** |
| vLLM engine | **Yes** |

Restaurant version: head chef uses the industrial oven and ignores the teaching line. Useful to contrast **orchestration you can step through** vs **production engine internals**.

---

## 7. Cheat sheet — who cooks for which menu

| Endpoint | Host (`WorkloadManager`) | Executor → Worker → GPU | Notes |
| -------- | ------------------------ | ----------------------- | ----- |
| `/basic_generate` | **Skip** | Yes (batch size 1) | One plate, no seating list |
| `/generate` | **Yes** (batch up to 4) | Yes (`input_shape=(N, …)`) | Table seating |
| `/generate_stream` | **Yes** (stream queue) | Yes (one token / step) | Bites at the table |
| `/generate_vllm` | **Skip** | **Skip** custom kitchen | Industrial oven |

`>>> [Layer]` = that station cooked.  
`--- SKIP [Layer]` = that ticket never visited it.

---

## 8. Takeaways

1. Serving is **orchestration** around the same prefill/decode loop from Article 1.  
2. **Waiter / head chef / host / kitchen manager / cook** map to API → engine → workload → executor → worker.  
3. Different endpoints **intentionally skip** stations — basic generate vs batch is the clearest demo.  
4. Concurrent HTTP clients ≠ automatic batching; `/generate` is where the Host packs a table.  
5. Streaming keeps Host + forward steps; `/generate_vllm` skips the whole teaching kitchen.

**Previous:** [vLLM Basics and Why KV Cache Matters](./vllm-basics-kv-cache)  
**Next ideas:** continuous batching, paged attention, and multi-model routing (`multi_model_serving`).
