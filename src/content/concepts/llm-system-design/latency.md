---
title: 'Designing for Latency'
description: 'How to measure and reduce LLM latency using streaming, batching, caching, routing, parallelism, and inference-aware architecture.'
pubDate: 'Sep 10 2026'
order: 3
heroImage: '../../../assets/blog-placeholder-3.jpg'
---

Latency is the time between a user starting an action and the system delivering a useful result. In an LLM application, that delay is not one number. The request may wait in a queue, retrieve context, process a long prompt, generate hundreds of tokens, call tools, and pass through output validation before the user sees the final answer.

A system can therefore feel fast while still taking several seconds to finish, or feel slow even when its total runtime is reasonable. Designing for latency starts by measuring the right milestones.

---

## 1. The latency timeline

For a single LLM request:

```text
User sends request
      │
      ├─ network + gateway
      ├─ queue wait
      ├─ retrieval / context assembly
      ├─ prompt processing (prefill)
      │
      ├──────── first token visible
      │
      ├─ token generation (decode)
      │
      ├──────── last token generated
      │
      └─ validation / rendering
                         User receives complete result
```

The main metrics are:

| Metric | Meaning | Matters most for |
|---|---|---|
| **Time to first token (TTFT)** | Time from request submission until the first generated token arrives | Interactive chat and perceived responsiveness |
| **Inter-token latency (ITL)** | Delay between consecutive streamed tokens | How smooth the response feels |
| **Time per output token (TPOT)** | Average decode time for each generated token after the first | Generation speed and capacity planning |
| **End-to-end latency (E2E)** | Time until the complete, usable response is available | APIs, structured output, agents, and batch jobs |

A useful approximation is:

```text
E2E latency =
  network
  + queue
  + context preparation
  + prompt prefill
  + output token count × time per output token
  + post-processing
```

Prompt processing and output generation behave differently:

- **Prefill** processes prompt tokens largely in parallel. TTFT grows with prompt length, queue time, and model load.
- **Decode** generates output autoregressively, one token after another. Total generation time grows roughly with output length.

This is why trimming 5,000 input tokens primarily improves TTFT, while trimming 500 output tokens primarily improves completion time.

### Measure percentiles, not only averages

An average hides the requests users remember: the slow ones. Track at least p50, p95, and p99 for TTFT and end-to-end latency, segmented by:

- Model and provider
- Input and output token buckets
- Streaming vs non-streaming
- Cache hit vs miss
- Request type and route
- Region and deployment
- Number of tool or retrieval calls

Also track queue time separately. If model execution remains stable but queue time rises, the problem is capacity or admission control—not the prompt.

---

## 2. Streaming vs non-streaming responses

### Non-streaming

In a non-streaming request, the server buffers the full answer and returns it only after generation finishes:

```text
Request ───────── model generates full answer ─────────▶ Complete response
```

This is the right choice when:

- A machine must parse the complete JSON or schema
- The output must be validated before anything is exposed
- The response is short
- The job runs asynchronously and no user is waiting

### Streaming

In a streaming request, the server forwards tokens or semantic chunks as they become available:

```text
Request ──▶ first token ─▶ more tokens ─▶ more tokens ─▶ done
```

Streaming usually does **not** reduce model execution time. It reduces *perceived latency* by letting the user read while generation continues.

Use streaming for:

- Chat and copilots
- Long explanations or summaries
- Progress and intermediate status events
- Any experience where the user can benefit from a partial answer

Streaming introduces design constraints:

- The client must handle partial chunks, reconnects, and cancellation
- Markdown, code blocks, and citations may be incomplete mid-stream
- A safety or schema validator may not approve content until completion
- A failed stream cannot always be retried invisibly after partial output was shown
- Backpressure is required when the client consumes data slower than the model produces it

Prefer **semantic streaming** when raw token streaming creates unstable UI. Buffer a sentence, JSON event, or tool status and emit complete units:

```text
event: status   data: "Searching documents"
event: source   data: {"title": "Design specification", "id": "D-42"}
event: answer   data: "The primary constraint is..."
event: done     data: {"finish_reason": "stop"}
```

### Streaming and batching are orthogonal

Streaming determines **how results are delivered**. Batching determines **how inference work is scheduled**. A serving engine can continuously batch several active requests while streaming each response to a different user.

---

## 3. Online inference vs offline batch processing

The word *batch* is used for two related ideas.

### Offline batch jobs

Many inputs are submitted for completion without an interactive user waiting:

- Generate summaries for a document collection overnight
- Compute embeddings for a new corpus
- Classify historical support tickets
- Pre-compute answers for known queries

Offline jobs optimize for throughput and cost, not TTFT. They can use larger batches, lower-priority capacity, and longer deadlines.

### Inference batching

The serving engine groups token-processing work from multiple requests so the accelerator is used efficiently.

**Static batching** waits for a fixed group and processes it together. It is simple but suffers from head-of-line blocking: short requests wait for the longest request in the batch.

**Continuous batching** adds and removes requests between decoding steps. It gives much better utilization for variable-length LLM workloads and is the normal choice for high-throughput serving.

The trade-off is fundamental:

```text
Larger batching window
  → better accelerator utilization and throughput
  → more queue delay and worse TTFT
```

For interactive traffic, use a very small batching window and explicit latency objectives. Put offline traffic in a separate queue so it cannot consume capacity needed by users.

---

## 4. Caching

Caching avoids repeated work, but there are several distinct caches in an LLM system. They have different keys, risks, and invalidation rules.

### 4.1 Exact-match response cache

Return a stored response when the normalized request is identical:

```text
key = hash(
  tenant
  + model_version
  + system_prompt_version
  + normalized_user_input
  + relevant_context_version
  + generation_parameters
)
```

Exact matching is fast and predictable. It works well for deterministic or low-temperature tasks, FAQs, classification, and repeated API calls.

Do not key only on the user prompt. Two requests with the same text may differ by tenant, permissions, conversation history, retrieved evidence, model version, tools, locale, or temperature.

### 4.2 Semantic response cache

A semantic cache embeds a new request and searches for a sufficiently similar cached request:

```text
query
  → embedding
  → nearest cached queries
  → similarity + policy checks
  → return cached answer or call model
```

It can match paraphrases such as:

```text
"How do I reset my password?"
"I forgot my password—how can I change it?"
```

Semantic caching has higher hit potential but is not safe for similarity alone. These queries are lexically close but require different answers:

```text
"Can employee A approve this release?"
"Can employee B approve this release?"
```

Add hard filters before accepting a semantic hit:

- Same tenant, user scope, language, and task type
- Same policy and knowledge-base version
- No user-specific or rapidly changing data
- Similarity above a threshold calibrated on an evaluation set
- Optional deterministic or model-based verification for high-risk cases

False hits are usually more damaging than misses. Tune for precision first.

### 4.3 Prompt or prefix caching

Many providers and inference engines can reuse computation for a repeated prompt prefix, such as:

- A stable system prompt
- Tool definitions
- Few-shot examples
- A long shared document

This reduces repeated prefill work and therefore TTFT. Structure prompts with stable content first and request-specific content last:

```text
[stable system instructions]
[stable tool schemas]
[stable few-shot examples]
[variable retrieved context]
[variable conversation and user query]
```

Changing even a small item near the beginning can invalidate reuse for everything after it. Version stable prefixes deliberately and avoid dynamic timestamps, random IDs, or request-specific text in the reusable section.

### 4.4 KV cache

During inference, the model stores attention keys and values for tokens it has already processed. This **KV cache** prevents recomputing the whole sequence for every generated token.

It is primarily a model-serving optimization, not an application response cache. Its main trade-off is memory: long contexts and many concurrent requests consume significant accelerator memory. Serving systems manage this with techniques such as paged allocation, prefix sharing, eviction, and context limits.

### 4.5 Retrieval and tool-result caches

Sometimes the best optimization is to cache work around the model:

- Query embeddings
- Retrieval results for stable corpora
- Reranker results
- Database or API tool responses
- Document parsing and chunking
- Safety classifications

Cache each deterministic stage independently. Reusing a fresh database result can preserve answer quality while avoiding a full tool round trip.

### 4.6 Request coalescing

Request coalescing—also called request collapsing or single-flight—is useful when many identical requests miss the cache at the same time.

```text
Request A ─┐
Request B ─┼─ same key ─▶ one model call ─▶ populate cache ─▶ fan out result
Request C ─┘
```

Without coalescing, a cache expiry can trigger a stampede of expensive duplicate model calls. Keep one in-flight operation per cache key and let followers await its result.

```python
import asyncio

cache: dict[str, str] = {}
in_flight: dict[str, asyncio.Task[str]] = {}
lock = asyncio.Lock()

async def cached_generate(key: str, generate) -> str:
    if key in cache:
        return cache[key]

    async with lock:
        # Recheck after waiting: another request may have filled the cache.
        if key in cache:
            return cache[key]
        task = in_flight.get(key)
        if task is None:
            task = asyncio.create_task(generate())
            in_flight[key] = task

    try:
        result = await task
        cache[key] = result
        return result
    finally:
        async with lock:
            if in_flight.get(key) is task:
                in_flight.pop(key, None)
```

In production, add timeouts, bounded cache size, exception handling, cancellation policy, and a distributed lock or coordination layer when instances do not share memory.

### Cache strategy and invalidation

Choose freshness rules from the data, not from convenience:

| Strategy | Use when | Main risk |
|---|---|---|
| Time to live (TTL) | Data changes at a known approximate rate | Stale results within the TTL |
| Versioned keys | Prompts, models, policies, or corpora have explicit versions | Old entries consume storage |
| Event-based invalidation | Source changes emit reliable events | Missed events leave stale data |
| Stale-while-revalidate | Slightly stale answers are acceptable | Users may briefly see old content |
| Negative caching | Repeated misses or failures are expensive | Temporary failure may be cached too long |

Never share cached responses across security boundaries. Encrypt sensitive cache data, set retention limits, and avoid caching prompts or answers containing personal or confidential information unless the use case explicitly permits it.

---

## 5. Reduce the work on the critical path

The fastest model call is the one the system does not make. Before tuning inference, remove unnecessary sequential work.

### Run independent operations in parallel

Sequential:

```text
retrieve A (300 ms) → retrieve B (400 ms) → policy check (100 ms)
total before model: 800 ms
```

Parallel:

```text
max(retrieve A, retrieve B, policy check) ≈ 400 ms
```

Only parallelize operations that are genuinely independent. Speculatively executing expensive branches can lower latency while sharply increasing cost.

### Minimize serial LLM calls

An agent that makes five 2-second model calls in sequence has a minimum model latency near 10 seconds. Ask whether steps can be:

- Combined into one structured call
- Replaced with deterministic code
- Executed concurrently
- Moved to the background
- Skipped when confidence is already sufficient

### Bound tool latency

Every external tool joins the critical path. Give tools explicit deadlines, cache stable results, and return partial results when optional tools time out. For multi-step agents, enforce both per-step and total request budgets.

```text
Total user deadline: 8 s
  retrieval budget: 800 ms
  model TTFT budget: 1.5 s
  tools budget: 2 s
  final generation budget: remaining time
```

---

## 6. Reduce model inference latency

### Route to the smallest model that meets the quality target

Smaller models usually process and generate tokens faster. Route simple extraction, classification, and formatting requests to a fast model; reserve larger reasoning models for tasks that need them.

Do not route on latency alone. Validate each route against a task-specific quality evaluation set.

### Control input length

- Retrieve only relevant chunks instead of inserting full documents
- Remove duplicated instructions and repeated tool outputs
- Summarize old conversation turns
- Keep tool schemas concise
- Put a hard token budget on context assembly

### Control output length

Generation is sequential. Ask for the shortest useful answer, use structured outputs where appropriate, set a realistic output limit, and stop generation when the task is complete.

A high `max_tokens` value does not necessarily slow a response if the model stops early, but it weakens the upper latency bound. Explicit output contracts make latency more predictable.

### Use inference-engine optimizations

When self-hosting, important serving choices include:

- Continuous batching
- Efficient KV-cache memory management
- Quantization, after measuring quality impact
- Tensor or pipeline parallelism for models that do not fit efficiently on one device
- Speculative decoding with a smaller draft model
- Optimized attention kernels

These are workload-dependent. Benchmark with representative prompt lengths, output lengths, and concurrency—not a single short prompt.

### Keep capacity warm and local

Cold starts, model loading, and cross-region network calls can dominate short requests. Keep interactive capacity warm, place inference near application services and users, reuse connections, and autoscale from queue depth or in-flight token load rather than CPU usage alone.

---

## 7. Design different latency paths

Not every request needs the same service-level objective (SLO).

| Traffic class | Example | Optimize for | Typical design |
|---|---|---|---|
| Interactive | Chat response | TTFT and smooth streaming | Priority queue, streaming, warm capacity |
| Synchronous API | JSON extraction | End-to-end latency | Short output, schema enforcement, no token streaming |
| Agent workflow | Research or tool use | Progress and bounded completion | Async execution, status events, step deadlines |
| Offline batch | Corpus summarization | Throughput and cost | Large batches, low-priority queue |

Separate these workloads. If an overnight embedding job and an interactive chatbot share one unconstrained queue, high-throughput work will damage user latency.

For long operations, return a job ID quickly and move execution out of the request:

```text
POST /reports  →  202 Accepted + job_id
worker         →  retrieval + model + validation
client         →  status stream, webhook, or polling
```

Async execution does not make the work faster. It prevents a long task from blocking a synchronous user interaction and makes progress visible.

---

## 8. A practical latency architecture

```text
                            ┌── exact / semantic response cache ── hit ─▶ return
                            │
User ─▶ regional gateway ─▶ request coalescer
                            │ miss
                            ▼
                    route by task + SLO
                     ┌──────┴────────┐
                     ▼               ▼
              interactive queue   batch queue
                     │
          retrieval + tools in parallel
                     │
             prefix-cache-friendly prompt
                     │
        continuously batched model inference
                     │
             semantic stream to client
                     │
             validate + populate cache
```

The order matters:

1. Check whether safe reusable work already exists
2. Collapse duplicate in-flight requests
3. Route by task complexity and latency objective
4. Parallelize independent context operations
5. Minimize prompt and expected output
6. Stream useful units to interactive clients
7. Validate before storing a reusable result

---

## 9. Common mistakes

**Optimizing only average latency**  
A good average can coexist with an unusable p99. Design and alert on percentile SLOs.

**Calling streaming a reduction in total latency**  
Streaming usually improves perceived responsiveness, not completion time.

**Treating semantic similarity as answer equivalence**  
Similar wording does not guarantee the same authorization, context, time, or intent.

**Using retries to fix overload**  
Retries add traffic to an already overloaded service. Use bounded retries with jitter, admission control, backpressure, and circuit breakers.

**Increasing batch size without a queue budget**  
Throughput improves while interactive TTFT becomes worse.

**Caching without model, prompt, or data versions**  
Old answers silently survive behavior and knowledge changes.

**Optimizing model inference while serial tools dominate**  
Trace the full request before deciding where to optimize.

---

## 10. Measurement-driven optimization

Use a trace for every request:

```text
request_id
  gateway_ms
  queue_ms
  retrieval_ms
  tool_ms
  cache_lookup_ms + cache_type + hit/miss
  model + input_tokens + output_tokens
  ttft_ms + tpot_ms + generation_ms
  validation_ms
  end_to_end_ms
```

Then optimize the largest contributor:

- High queue time → add capacity, isolate traffic, or apply admission control
- High TTFT with long prompts → reduce context or improve prefix reuse
- High generation time → shorten output or use a faster model
- Repeated identical work → exact cache and request coalescing
- Repeated paraphrases → carefully evaluated semantic cache
- Long serial traces → parallelize or remove steps
- Slow but unavoidable jobs → async execution with progress

Every latency optimization can affect another system property. Caching can reduce freshness, smaller models can reduce quality, batching can increase TTFT, parallel branches can increase cost, and aggressive timeouts can reduce successful completion. Evaluate latency together with quality, reliability, and cost.

---

## Summary

1. LLM latency is a timeline: queueing, retrieval, prompt prefill, token decoding, tools, and post-processing.
2. Track TTFT for interactive responsiveness and end-to-end latency for complete machine-consumable results.
3. Streaming improves perceived latency; batching improves inference utilization. They can be used together.
4. Use separate cache layers for exact responses, semantic matches, repeated prompt prefixes, model KV state, and deterministic retrieval or tool work.
5. Request coalescing prevents duplicate calls and cache stampedes.
6. Remove serial work, parallelize independent operations, route to the smallest adequate model, and bound input and output tokens.
7. Separate interactive, synchronous API, agent, and offline batch traffic because they optimize for different goals.
8. Trace the full critical path and optimize the measured bottleneck—not the most visible component.
