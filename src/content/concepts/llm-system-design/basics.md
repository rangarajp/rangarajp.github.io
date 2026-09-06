---
title: 'How LLM Systems Differ'
description: 'The new design primitives — context window, probabilistic output, latent reasoning — and why they break traditional software assumptions.'
pubDate: 'Sep 6 2026'
order: 1
heroImage: '../../../assets/blog-placeholder-3.jpg'
---

In traditional software, a function is a contract: same input, same output, every time. You test it once, trust it, and move on. An LLM is not that. It is a probabilistic reasoning engine that operates over text, and designing systems around it requires rethinking the assumptions that software engineering has held for decades.

This article covers the foundational shift — the new primitives, the new failure modes, and the new design questions that arise the moment an LLM becomes a component in your system.

## 1. The old model vs the new

In a traditional system, components are functions or services:

```
Input → deterministic function → Output
```

You validate the input schema, unit-test the function, and trust the composition. The system is only as unpredictable as its external dependencies (databases, APIs, clocks).

An LLM application looks structurally similar but behaves very differently:

```
Input + Context + Prompt → LLM (probabilistic) → Text → Parser → Output
```

Every arrow in this pipeline can fail in ways that do not raise exceptions. The model can produce a plausible-sounding wrong answer. The parser can fail silently on unexpected formatting. The context can be stale, truncated, or irrelevant. None of these are stack traces — they are quality failures that only show up in evaluation.

## 2. The new primitives

### Context window

The context window is the model's working memory. Everything the model can reason about in a single call must fit in it — the system prompt, conversation history, retrieved documents, tool outputs, and the user's question. Typical sizes range from 8K to 1M tokens depending on the model.

This creates a hard constraint with no equivalent in traditional systems: you cannot simply pass "all relevant information" to the model. You must decide what to include, in what order, and what to leave out. That decision is an architectural one, not a plumbing one.

Key implications:
- Long conversations must be summarised or truncated — you cannot keep the full history forever
- Retrieved documents must be ranked and pruned before insertion
- Tool outputs can quickly consume the budget if not managed
- Prompt text itself has a cost — every token in the system prompt is paid on every call

### Probabilistic output

The model samples from a probability distribution at each token. Given the same input, two calls can produce different outputs. This is not a bug — it is how the model explores the space of plausible continuations — but it means:

- You cannot unit-test an LLM call with `assertEqual`
- Retry logic must account for the possibility that retrying produces a different (possibly worse) answer
- Caching must be done carefully — a cached response may be wrong for a slightly different context
- Output parsing must be robust to variation in phrasing, structure, and format

### Latent reasoning

The model's "reasoning" is implicit in its weights, not inspectable. When it produces a wrong answer, you often cannot tell *why* — whether it misunderstood the question, lacked relevant knowledge, was distracted by earlier context, or was simply unlucky in sampling. This makes debugging fundamentally harder than tracing a stack.

Chain-of-thought prompting and reasoning models (like DeepSeek-R1) externalise some of this reasoning into tokens, which helps — but the underlying computation is still opaque.

### Latency profile

LLM calls are slow and variable. A single generation can take anywhere from 200ms to 30+ seconds depending on output length, model size, and load. Unlike a database query where you can add an index to reduce latency, you have limited levers:

- Reduce max output tokens
- Use a smaller / faster model
- Cache frequent responses
- Stream the output so the user sees tokens as they arrive
- Pre-compute responses offline where the query is predictable

This changes how you architect flows. Synchronous user-facing calls need to be short. Long chains of LLM calls should run asynchronously with progress signals.

## 3. What breaks when you use an LLM as a component

| Traditional assumption | What happens with LLMs |
|---|---|
| Same input → same output | Same input → distribution of outputs; non-deterministic |
| Failures raise exceptions | Quality failures are silent — wrong answers look like correct ones |
| Test once, trust everywhere | Behaviour shifts with prompt changes, model updates, context changes |
| Latency is bounded | Output length is unbounded; latency scales with generation |
| Memory is explicit state | Context window is implicit, lossy, and bounded |
| Composition is predictable | Chaining LLM calls multiplies uncertainty; errors accumulate |

### The silent failure problem

This is the most important difference. A microservice that crashes returns a 500. An LLM that produces a confident wrong answer returns a 200 with a plausible-looking body. Detecting the failure requires evaluation, not monitoring in the traditional sense.

In engineering applications this matters enormously: a hallucinated parameter value, a wrong unit, or a misremembered constraint looks identical to a correct answer in the response. The system cannot tell the difference. Only domain validation, downstream checks, or human review can catch it.

## 4. The new design questions

Every LLM application forces you to make choices that do not exist in traditional systems:

**What goes in the context?**  
Not everything relevant can fit. You need a retrieval strategy (RAG), a ranking strategy, and a truncation policy. The wrong choice here is the most common source of quality failures.

**How do you handle state?**  
Conversations have history. Agents have memory. Neither fits neatly in a stateless request/response model. You need to decide: full history (expensive), sliding window (loses early context), summary (lossy), external memory store (latency + retrieval quality), or a combination.

**How do you parse and validate output?**  
Asking the model to respond in JSON is not a guarantee. Structured output modes (function calling, JSON mode) help but do not eliminate failures. You need a fallback: retry with a stricter prompt, extract with regex, or reject and surface an error.

**How do you evaluate?**  
You cannot rely on unit tests. You need eval sets, quality metrics (accuracy, faithfulness, groundedness, format compliance), and a way to detect regressions when the prompt or model changes. LLM-as-judge is useful but adds cost and latency.

**When do you call the model vs use deterministic code?**  
This is the most underrated question. Many tasks that reach for an LLM can be done more reliably and cheaply with a lookup, a regex, a small classifier, or a rules engine. Use the LLM where language understanding, generation, or flexible reasoning is genuinely required.

## 5. The component model

A useful way to think about an LLM application is as a pipeline of components, each with a clear contract:

```
┌─────────────┐     ┌──────────────┐     ┌───────────┐     ┌──────────────┐
│   Input     │────▶│   Context    │────▶│   Model   │────▶│   Output     │
│  (query,    │     │  (prompt +   │     │  (LLM)    │     │  (parser +   │
│   history)  │     │   retrieval) │     │           │     │  validator)  │
└─────────────┘     └──────────────┘     └───────────┘     └──────────────┘
                            ▲                                      │
                            │                                      ▼
                     ┌──────────────┐                     ┌──────────────┐
                     │   Memory     │◀────────────────────│   Actions    │
                     │  (history,   │                     │  (tools,     │
                     │   summaries) │                     │   writes)    │
                     └──────────────┘                     └──────────────┘
```

Each component in this diagram is a design decision:
- Context assembly: what retrieval strategy, what ranking, what truncation?
- Model: which model, what temperature, what output format?
- Output: what parser, what validator, what fallback on failure?
- Memory: full history, window, summary, or vector store?
- Actions: which tools, what permissions, what rate limits?

The rest of this series covers each of these decisions in depth.

## 6. Production design principles

Once the prototype works, the real design work begins. Each of the following concerns has a well-established answer in traditional systems engineering; with LLMs, each needs a fresh treatment.

### Resilience and reliability

LLM APIs fail — rate limits, timeouts, model unavailability, and upstream outages are all routine. Unlike a flaky microservice, a failed LLM call cannot always be retried identically, because:

- The retry may return a different (potentially better or worse) answer
- Retrying an expensive call under load amplifies the problem
- A partial streamed response needs a different recovery path than a clean timeout

Design for this explicitly:

- Add a retry policy with exponential backoff, but cap retries at 2–3 — beyond that, fail fast and surface the error
- Define a fallback for every LLM call: a cached response, a simpler deterministic answer, a "try again later" message, or escalation to a human
- Use circuit breakers: if a model endpoint is failing repeatedly, stop sending requests for a window rather than hammering it
- For critical paths, run two models in parallel and take the first successful response (more expensive but dramatically more reliable)
- Distinguish between quality failures (model returned something wrong) and infrastructure failures (model did not return at all) — they need different recovery strategies

### Low latency

LLM calls are 10–100x slower than a typical database query. Users notice. The levers are:

| Technique | Latency reduction | Trade-off |
|---|---|---|
| Streaming | Time-to-first-token drops dramatically | Parser must handle partial output |
| Prompt caching | First-token latency cut by 50–90% on repeated prefixes | Cache invalidation complexity |
| Smaller / faster model | 2–10x speedup | Quality may drop; need eval to validate |
| Reduce max output tokens | Linear improvement | Model may truncate useful output |
| Async / background generation | Perceived latency near zero | Adds polling or push infrastructure |
| Pre-computation | Zero latency for known queries | Only works for predictable inputs |

Streaming is the highest-leverage change for user-facing applications: even if the total time is the same, showing tokens as they arrive makes the system feel fast. Prompt caching (available on most hosted APIs) is the highest-leverage change for repeated or templated prompts — the system prompt and few-shot examples are paid once, not on every call.

Do not conflate total latency with time-to-first-token. For a user reading a response, time-to-first-token governs perceived responsiveness. For a downstream system parsing the full response, total latency is what matters.

### Cost optimisation

Token cost compounds quickly in production. A system that costs $5 in a demo can cost $5,000 at scale, often because of choices made without cost in mind.

The biggest cost drivers, in order:

1. Model choice — GPT-4-class models cost 10–50x more per token than small or open models; use them only where the quality delta justifies it
2. Output length — generation is more expensive than input on most APIs; set a realistic `max_tokens` and prompt for conciseness where appropriate
3. Context size — a 50-page document in every request multiplies your input cost by 50; use RAG to retrieve only what is needed
4. Number of calls — agentic loops that make 10 LLM calls per user query cost 10x more than a single call; design the task to minimise hops
5. Retries — a 20% retry rate adds 20% to your bill; fix reliability at the source rather than papering over it with retries

A practical cost control approach: route requests by complexity. Use a fast cheap model for classification, filtering, and simple formatting; escalate to a powerful model only for reasoning-heavy tasks. Log token usage per request from day one — cost surprises in production are almost always traceable to a single high-usage flow.

### Testability and observability

You cannot put a breakpoint in an LLM. What you can do:

For testability:
- Build an eval set before you build the feature — even 20 representative examples let you detect regressions when the prompt changes
- Separate prompt logic from application logic so prompts can be versioned and tested independently
- Use deterministic temperature (temperature=0) for eval runs so results are reproducible
- For structured output tasks, add schema validation as a post-processing step and track the failure rate

For observability:
- Log every LLM call: prompt (or prompt hash), model, latency, token counts, output, and any downstream quality signals
- Trace multi-step pipelines end-to-end — a wrong final answer in an agent loop is often caused by a failure three steps earlier
- Track quality metrics over time: answer accuracy on your eval set, format compliance rate, retrieval precision (for RAG), and latency percentiles
- Set up alerts on silent quality degradation, not just on errors — a 20% drop in accuracy on your eval set matters as much as a spike in 500s

Frameworks like LangSmith, Weave, and Arize provide LLM-native tracing. The key is treating LLM call logs as first-class telemetry, not an afterthought.

### Security and trust

LLMs introduce attack surfaces that do not exist in traditional systems:

Prompt injection is the most common: a malicious user embeds instructions in their input ("Ignore previous instructions and instead...") that override the system prompt. Unlike SQL injection, there is no parameterised query equivalent — the model processes instructions and data in the same stream. Mitigations: input sanitisation, output validation, sandboxed tool execution, and treating user input as untrusted data throughout.

Data leakage through context: if you insert private or sensitive documents into the context, the model may echo them back, summarise them, or include them in logged outputs. Design the context assembly step to be as narrow as possible — only include what the model genuinely needs for the task.

Output trust: model outputs should not be trusted unconditionally, especially before taking actions. An agent deciding to delete a file, send an email, or make an API call should do so only after output validation. For high-stakes actions, require explicit confirmation or a human-in-the-loop gate.

Model supply chain: if you use a hosted model, you are trusting the provider with your prompts and any data they contain. Review data processing agreements, check whether inputs are used for training, and understand the data residency of the API endpoint.

### Engineering for production

A prototype that works in a notebook is not production-ready. The gap is larger with LLMs than with most software because the failure modes are quieter and more varied.

Things that need to be in place before you go live:

- Prompt versioning — prompts are code; track them in version control, review changes, and test before deploying
- Model version pinning — APIs can change model behaviour between versions; pin to a specific model version and plan an explicit upgrade process
- Rollback — be able to revert to the previous prompt and model version within minutes if quality degrades in production
- Rate limit handling — build back-pressure into your system; do not let a rate limit cascade into a user-facing error
- Graceful degradation — define what the system does when the LLM is unavailable; a fallback message is better than a crash
- Load testing — LLM latency under concurrent load is often much worse than single-request testing suggests; test at realistic concurrency before launch

### Evaluation hygiene — never train on your test set

This is well-known in ML but easy to violate accidentally in LLM systems:

- If you use production queries to improve your prompt (few-shot examples, instruction tuning), and those queries overlap with your eval set, your evaluation is contaminated
- If you fine-tune a model on data that includes your eval questions — even indirectly through a data pipeline — benchmark numbers are meaningless
- LLM judges that have seen the task at training time (e.g. GPT-4 evaluating tasks that were in its pretraining data) can give artificially high scores

Keep a held-out eval set that never touches the training or prompt-tuning pipeline. Refresh it periodically as the distribution of real queries shifts.

### User privacy

Users interacting with LLM systems often share personal information they would not expect to be retained or processed beyond the immediate conversation.

Key principles:

- Do not log prompt content that contains personal data unless you have a legal basis and user consent to do so
- PII (names, emails, health data, financial data) should be detected and masked before logs are stored — treat LLM call logs like any other sensitive telemetry
- If you use user conversations to improve the model or prompt, obtain explicit opt-in consent — not buried in terms of service
- Understand the data handling policies of your model provider; in regulated industries (healthcare, finance, automotive), enterprise API agreements with data processing addenda are typically required
- For on-device or on-premise deployments, document what data leaves the device/network boundary and what stays local

In automotive and engineering contexts, this extends to operational data: CAD files, diagnostic logs, and simulation results embedded in prompts may contain IP, safety-critical parameters, or regulatory artefacts — treat them with the same access controls as the source systems.

## 7. Summary

1. LLMs introduce probabilistic, latency-variable, context-bounded components into otherwise deterministic systems
2. The hardest failure mode is silent quality failure — a wrong answer that looks correct
3. Context window management is the central design constraint — you decide what the model can reason about
4. State, output parsing, and evaluation all need explicit strategies that have no direct equivalent in traditional software
5. Use deterministic code where you can; reach for an LLM only where language understanding or flexible reasoning is genuinely needed
6. Design for the failure modes specific to LLMs: prompt injection, context leakage, silent quality degradation, and latency at scale — see [Designing for reliability](/concepts/llm-system-design/reliability) for fallback chains, model routing, and gateway patterns
7. Treat prompts as versioned code, outputs as untrusted data, and LLM call logs as first-class telemetry
8. Evaluation hygiene and user privacy are not afterthoughts — contaminated evals produce false confidence, and unmanaged data handling creates legal and trust risk
9. Next: the anatomy of an LLM application — how the components above connect into a working system
