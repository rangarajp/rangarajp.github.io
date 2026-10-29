---
title: "Observability and production engineering"
description: "Request metrics, traces, failure diagnosis, and goodput."
section: learn
pubDate: 2026-10-29
part: 6
chapter: 28
order: 28
tags: ["llm-inference", "systems", "hands-on"]
---

The serving components are in place. Instrument their boundaries here to distinguish model execution, queueing, and delivery, then use SLO-qualified results for capacity planning.

An engine benchmark cannot tell you whether users receive timely answers through the full application. Instrument arrival, queue admission, prefill, first delivered token, subsequent tokens, completion, and cancellation. A correlation ID connects those events without logging private prompts by default.

## Production responsibilities beyond the engine

A production service needs more than a model endpoint. Logging and tracing establish which model version handled a request and where time was spent. Metrics and alerts expose latency tails, failures, queue pressure, and resource saturation. Controlled rollouts compare versions under bounded traffic exposure, while capacity planning and cost attribution connect resource use to the teams or features consuming it.

Keep request identifiers, model/configuration versions, token counts, timings, and error state available without indiscriminately logging private prompt content. The purpose of an observability field is to resolve a concrete operational question.

## 1. Key metrics

| Metric | Definition | Diagnostic use |
|---|---|---|
| TTFT | Arrival to first delivered token | Queueing, prefill, delivery |
| ITL | Gaps between delivered tokens | Streaming stalls |
| TPOT | Mean post-first-token time per token | Request decode pace |
| Throughput | Defined token count / wall time | Aggregate capacity |
| Goodput | Work meeting a stated SLO / wall time | Useful capacity |
| KV occupancy | Allocated blocks / usable blocks | Admission pressure |

```python
# Synthetic delivery times in seconds since request arrival.
times = [0.12, 0.14, 0.17, 0.19]
gaps = [b-a for a,b in zip(times,times[1:])]
tpot = (times[-1]-times[0]) / (len(times)-1)
print('TTFT:', times[0], 'TPOT:', round(tpot, 4), 'max gap:', round(max(gaps), 4))
```

TPOT is undefined for a one-token answer; do not divide by zero or substitute TTFT. Report request percentiles and output lengths together. Pooling every token gap weights long answers more heavily than short ones, so state your aggregation method.

### Goodput, compliance, and model utilization

Define goodput as SLO-qualified requests or tokens per unit time, and report the compliance fraction separately. TTFT and TPOT targets must be evaluated jointly on each request. If 95% meet each target independently, joint compliance would be 90.25%; real request timings are often correlated, so calculate the observed joint fraction instead of multiplying marginals.

Model FLOP utilization compares estimated useful model operations per second with a relevant device peak. It can help interpret compute-heavy prefill, but a low number during bandwidth-limited decode does not by itself mean the implementation is poor. Pair it with memory traffic, achieved bandwidth, kernel shapes, and launch overhead. Use dense operation rates and the actual arithmetic format consistently.

## 2. Failure modes and debugging

High TTFT with low prefill time suggests queueing or upstream work. Good TTFT with intermittent streaming stalls suggests scheduler interference, delivery buffering, or contention. Out-of-memory failures late in answers suggest insufficient growth reservations or memory accounting. Inspect traces before changing hardware.

Cancellation must release request state and stop unnecessary GPU work. Readiness should mean the model can accept work, not merely that an HTTP process exists. Retry policies need bounded attempts: restarting a streamed answer can duplicate output and multiply load.

## 3. Benchmark and operate

Use realistic prompt/output distributions and arrival rates, plus a concurrency sweep. Warm the engine, separate cold-cache from warm-prefix results, and include failures and rejected requests. Record software, model revision, quantization, hardware, and scheduler settings. Pair performance changes with quality checks.

Production layers include authentication, quotas, bounded queues, routing, healthy model replicas, and telemetry. Containers or orchestration do not remove these requirements. Continue with [capacity and cost](./capacity-and-cost).

## Related server experiment: observe the knee before choosing concurrency

The repository's [single-model performance summary](https://github.com/rangarajp/rangarajp.github.io/blob/main/notebooks/llm-inference/observability/experiments/1.1_single_model_performance.md) reports the following client-load sweep for its Qwen2.5-7B teaching-server setup. This is a separate recorded server report, not the 0.5B offline batch comparison.

| Concurrent clients | TTFT p50 (s) | TPOT p50 (s) | System output tok/s |
|---:|---:|---:|---:|
| 1 | 0.056 | 0.031 | 31 |
| 2 | 0.061 | 0.032 | 60 |
| 4 | 0.167 | 0.038 | 91 |
| 8 | 0.580 | 0.038 | 96 |

Increasing concurrency from four to eight added little throughput while median TTFT rose substantially. That suggests an operating knee for this workload, not a production capacity guarantee. The table does not provide p99, arrival-process details, or a quality-qualified goodput measurement, so those remain useful additions to the report.

## Hands-on: build a small load-to-SLO report

Reuse the serving experiment with a fixed prompt/output distribution and a request ID in every trace. Publish latency percentiles, rejected/cancelled counts, and goodput at increasing arrival rates. Keep client delivery timestamps distinct from engine execution timers.
