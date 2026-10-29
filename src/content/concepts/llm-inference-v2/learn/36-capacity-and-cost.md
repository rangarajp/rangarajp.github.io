---
title: "Capacity planning and cost per token"
description: "Goodput-based economics, headroom, and a worked cost calculator."
section: learn
pubDate: 2026-10-29
part: 6
chapter: 36
order: 36
tags: ["llm-inference", "systems", "hands-on"]
---

# Capacity planning and cost per token

Plan capacity from a measured workload that satisfies latency and quality objectives. Maximum unconstrained tokens per second can overstate the amount of useful service you can sell or deliver.

## 1. Cost modeling

```python
# Hypothetical all-in worker price and measured output goodput.
worker_cost_per_hour = 3.0
output_goodput_per_second = 100
cost_per_million = worker_cost_per_hour * 1e6 / (3600 * output_goodput_per_second)
print(round(cost_per_million, 2))  # 8.33 currency units per million output tokens
```

This denominator counts output tokens from requests meeting the SLO. Input work is included in worker time but not counted as output. State this convention when comparing services. Add CPU, networking, idle replicas, storage, and operational costs if they are not already included.

## 2. Estimate replicas

```python
import math
peak_requests_per_second = 12
measured_requests_per_worker_second = 4  # at the required length mix and SLO
operating_fraction = 0.75
workers = math.ceil(peak_requests_per_second /
                    (measured_requests_per_worker_second * operating_fraction))
print(workers)  # 4; redundancy may require additional capacity
```

This simple model reserves headroom, but assumes workers scale independently. Shared bottlenecks and prefix locality can violate that assumption. Test burst arrivals and worker loss. Include model-loading time when sizing warm capacity and autoscaling lead time.

## 3. Improve efficiency systematically

Measure queueing and cache occupancy first. Then test prompt reduction, prefix reuse, batching, quantization, and replica placement one change at a time. Recheck quality and tail latency after each change. Buying more hardware is useful when a validated bottleneck remains and its cost is justified.
