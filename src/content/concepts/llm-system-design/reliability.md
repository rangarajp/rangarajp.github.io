---
title: 'Designing for Reliability'
description: 'Fallback strategies, model routing by task, and gateway logic for LLM systems that need to stay up when models do not.'
pubDate: 'Sep 6 2026'
order: 2
heroImage: '../../../assets/blog-placeholder-3.jpg'
---

Notebook: [reliability-demo.ipynb](https://github.com/rangarajp/rangarajp.github.io/blob/main/notebooks/llm-system-design/reliability-demo.ipynb) — runs all patterns layer by layer against two local Qwen3-0.6B checkpoints (base + reasoning); uses real connection errors and real threading-based timeouts; shows before/after availability numbers.

A traditional API call either succeeds or fails cleanly. An LLM call can fail in four distinct ways: the endpoint is down, the rate limit is hit, the response times out, or the model returns something that passes the HTTP check but is wrong. The last failure is the hardest — it is invisible to infrastructure monitoring and only surfaces in downstream quality metrics.

Reliability in LLM systems therefore means two things: keeping the system responsive when the infrastructure fails, and keeping the output acceptable when the model fails. Both need explicit design.

## 1. The three layers of failure

```
┌─────────────────────────────────────────────────────┐
│  Infrastructure layer                               │
│  Endpoint down · Rate limit · Timeout · Network     │
│  → Raises an exception. Easy to detect.             │
├─────────────────────────────────────────────────────┤
│  Output layer                                       │
│  Wrong answer · Bad format · Truncated · Hallucina  │
│  → Returns 200. Silent. Only caught by evaluation.  │
├─────────────────────────────────────────────────────┤
│  Quality drift layer                                │
│  Model update · Prompt regression · Data shift      │
│  → No error. Gradual degradation over days/weeks.   │
└─────────────────────────────────────────────────────┘
```

Most reliability work focuses on the infrastructure layer because it is easy to observe. A resilient LLM system also has guards on the output and quality drift layers.

## 2. Fallback strategies

### The fallback chain

For any LLM call, define a chain of fallbacks before you write the call:

```
Primary model call
  → failed or slow?   →  Retry (1–2 times, exponential backoff)
  → still failing?    →  Secondary model (cheaper, different provider)
  → still failing?    →  Cached response (if available for this query)
  → no cache?         →  Degraded response (template, rule-based answer)
  → nothing works?    →  Graceful error ("unavailable, try later")
```

Flowchart:

```
         User request
              │
              ▼
     ┌─────────────────┐
     │  Primary model  │──── success ──────────────────▶ Return response
     └─────────────────┘
              │ fail / timeout
              ▼
     ┌─────────────────┐
     │  Retry (≤2x)    │──── success ──────────────────▶ Return response
     └─────────────────┘
              │ still failing
              ▼
     ┌─────────────────┐
     │ Secondary model │──── success ──────────────────▶ Return response
     │ (fallback LLM)  │                                 + log downgrade
     └─────────────────┘
              │ still failing
              ▼
     ┌─────────────────┐
     │  Cache lookup   │──── hit ──────────────────────▶ Return cached
     └─────────────────┘                                 + warn stale
              │ miss
              ▼
     ┌─────────────────┐
     │ Degraded / rule │──── always succeeds ──────────▶ Return fallback
     │  based answer   │
     └─────────────────┘
```

Each step in the chain should be logged with the reason for the downgrade. The caller should never know — from a latency perspective — which layer served the response, but the observability layer must.

### Code: fallback chain

```python
import time
import logging
from typing import Callable, Any

logger = logging.getLogger(__name__)


def call_with_fallback(
    prompt: str,
    primary: Callable,
    secondary: Callable,
    cache: Callable,
    degraded: Callable,
    max_retries: int = 2,
    timeout: float = 10.0,
) -> dict:
    """
    Try primary → retry → secondary → cache → degraded.
    Returns {"text": ..., "source": "primary|secondary|cache|degraded"}.
    """
    # 1. Primary + retries
    for attempt in range(max_retries + 1):
        try:
            text = primary(prompt, timeout=timeout)
            return {"text": text, "source": "primary"}
        except Exception as e:
            logger.warning(f"Primary attempt {attempt + 1} failed: {e}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)   # 1s, 2s

    # 2. Secondary model
    try:
        text = secondary(prompt, timeout=timeout)
        logger.info("Serving from secondary model")
        return {"text": text, "source": "secondary"}
    except Exception as e:
        logger.warning(f"Secondary failed: {e}")

    # 3. Cache
    cached = cache(prompt)
    if cached:
        logger.info("Serving stale cache")
        return {"text": cached, "source": "cache"}

    # 4. Degraded
    text = degraded(prompt)
    logger.error("Serving degraded response")
    return {"text": text, "source": "degraded"}
```

### Output-layer fallback

Infrastructure fallbacks handle cases where the model does not respond. Output fallbacks handle cases where it responds wrongly. The pattern is: validate → retry with a tighter prompt → give up gracefully.

A `Validator` base class keeps the retry loop format-agnostic. Every output type — number, JSON, table, bullets, or any LLM-judged criteria — is just a subclass:

```python
from abc import ABC, abstractmethod
import re, json

class Validator(ABC):
    @abstractmethod
    def check(self, text: str) -> tuple[bool, str]:
        """Returns (is_valid, detail). detail is shown to the model on retry."""

    def tighten(self, prompt: str, reason: str) -> str:
        return prompt + f"\n\n[Previous attempt was invalid: {reason}. Please fix.]"


class NumberValidator(Validator):
    def check(self, text):
        m = re.search(r"-?\d+(?:\.\d+)?", text)
        return (True, m.group()) if m else (False, "no number found — reply with digits only")

class JSONValidator(Validator):
    def check(self, text):
        m = re.search(r'(\{.*\}|\[.*\])', text, re.S)
        candidate = m.group(1) if m else text
        try:
            return True, json.dumps(json.loads(candidate), indent=2)
        except json.JSONDecodeError as e:
            return False, f"invalid JSON: {e}"

class MarkdownTableValidator(Validator):
    def check(self, text):
        lines = [l for l in text.splitlines() if '|' in l]
        return (True, text) if len(lines) >= 2 else (False, "no markdown table — use | separators")

class BulletListValidator(Validator):
    def __init__(self, min_bullets=3): self.min = min_bullets
    def check(self, text):
        bullets = re.findall(r'^\s*[-*•]\s+.+', text, re.MULTILINE)
        return (True, text) if len(bullets) >= self.min else (False, f"only {len(bullets)} bullets, need {self.min}")

class LLMJudgeValidator(Validator):
    """Any criteria you can describe in plain English."""
    def __init__(self, criteria: str, model_fn=None):
        self.criteria = criteria
        self.model_fn = model_fn or fast_model
    def check(self, text):
        verdict = self.model_fn(
            f"Does this response meet: {self.criteria}\n\nResponse: {text[:300]}\n\nReply YES or NO."
        ).strip().upper()
        return (True, text) if verdict.startswith("YES") else (False, f"judge said NO — {self.criteria}")


def validated_ask(prompt, validator, model_fn, max_attempts=3):
    current_prompt = prompt
    for attempt in range(1, max_attempts + 1):
        response = model_fn(current_prompt)
        ok, detail = validator.check(response)
        if ok:
            return {"valid": True, "value": detail, "attempts": attempt}
        current_prompt = validator.tighten(prompt, detail)
    return {"valid": False, "value": response, "attempts": max_attempts}
```

### Dynamic validator dispatch

Hardcoding the validator at the call site leaks format knowledge into every caller. A cleaner pattern: pass a format hint (or nothing) and let the system pick automatically.

```
ask(prompt)
  └─ keyword_infer(prompt)      ← zero tokens — scan for signal words
      └─ no match → llm_infer() ← ~6 tokens — fast model classifies format
          └─ VALIDATORS[fmt]    ← right validator, no hardcoding in caller
```

```python
VALIDATORS = {
    "number":  NumberValidator(),
    "json":    JSONValidator(),
    "table":   MarkdownTableValidator(),
    "bullets": BulletListValidator(min_bullets=3),
}

FORMAT_KEYWORDS = {
    "number":  ["how many", "what is", "calculate", "sum", "multiply", "plus", "minus"],
    "json":    ["json", "object", "dict", "key", "structured"],
    "table":   ["table", "compare", "comparison", "versus", "vs", "across"],
    "bullets": ["list", "bullet", "steps", "advantages", "pros", "cons", "reasons"],
}

def keyword_infer(prompt):
    p = prompt.lower()
    for fmt, kws in FORMAT_KEYWORDS.items():
        if any(kw in p for kw in kws):
            return fmt
    return None

def llm_infer(prompt):
    reply = fast_model(
        f"What output format does this prompt expect?\n"
        f"Reply with one word: number / json / table / bullets / text\n\nPrompt: {prompt}\nFormat:"
    ).strip().lower().split()[0]
    return reply if reply in VALIDATORS else "text"

def ask(prompt, format="auto", max_tokens=150):
    fmt = keyword_infer(prompt) if format == "auto" else format
    if fmt is None:
        fmt = llm_infer(prompt)   # LLM fallback only when keywords give no signal
    validator = VALIDATORS.get(fmt)
    if validator is None:
        return {"valid": True, "format": "text", "value": fast_model(prompt)}
    result = validated_ask(prompt, validator, fast_model, max_tokens=max_tokens)
    result["format"] = fmt
    return result
```

| Strategy | Extra cost | When to use |
|---|---|---|
| Keyword scan | 0 tokens | Structured task pipelines |
| LLM classify | ~6 tokens | Free-form user prompts |
| Explicit `format=` | 0 tokens | When caller knows the schema |

## 3. Model routing

Not every request needs the same model. A routing layer dispatches each request to the appropriate model based on what the task actually requires.

### Why route?

| Task | Needs | Best fit |
|---|---|---|
| Classify intent from a short message | Speed, low cost | Small model (1–7B) |
| Extract structured fields from a document | Accuracy, JSON output | Mid-size model |
| Multi-step reasoning over technical content | Deep reasoning | Large / reasoning model |
| Generate a one-sentence summary | Fluency | Small model |
| Write and debug code | Context + reasoning | Coding-specialist model |

Routing by task reduces cost (small models are 10–50x cheaper) and reduces latency (small models are 5–10x faster), at the expense of router complexity and the risk of misclassification.

### Router design

```
         Incoming request
                │
                ▼
        ┌──────────────┐
        │    Router    │  ← classification model, rules, or embeddings
        └──────┬───────┘
               │
       ┌───────┼───────────┐
       ▼       ▼           ▼
  ┌─────────┐ ┌─────────┐ ┌──────────────┐
  │  Small  │ │   Mid   │ │    Large /   │
  │  model  │ │  model  │ │  Reasoning   │
  │ (fast,  │ │         │ │    model     │
  │  cheap) │ │         │ │  (slow, $$$) │
  └─────────┘ └─────────┘ └──────────────┘
```

Three approaches to the router itself:

1. Rule-based — classify by input length, keyword presence, or task type detected from the request structure. Fast, predictable, no extra model call needed.
2. Classifier model — run a tiny classifier (fine-tuned or few-shot) that predicts task complexity from the input. Adds ~50ms but handles edge cases better.
3. LLM-based — ask a cheap model "how complex is this task?" and route on the answer. Slowest but most flexible.

### Code: rule-based router

The notebook routes across three tiers using keyword matching and prompt length. FAST and DEEP use genuinely different model checkpoints (base vs reasoning); DEEP also gets a chain-of-thought prefix.

```python
import time

REASONING_WORDS = (
    "compare", "explain", "why", "difference", "analyse", "analyze",
    "design", "trade-off", "when should", "how does", "evaluate",
    "pros and cons", "versus", "vs",
)

def classify_tier(prompt: str) -> str:
    lower   = prompt.lower()
    n_words = len(prompt.split())
    if any(kw in lower for kw in REASONING_WORDS) or n_words > 30:
        return "DEEP"
    if n_words > 10:
        return "MID"
    return "FAST"

def routed_call(prompt: str) -> dict:
    tier = classify_tier(prompt)
    t0   = time.time()

    if tier == "FAST":
        response = base_model(prompt, max_tokens=40)

    elif tier == "MID":
        response = base_model(prompt, max_tokens=120)

    else:   # DEEP — use the reasoning model + CoT prefix
        cot_prompt = "Think step by step, then give a clear, structured final answer.\n" + prompt
        response   = reasoning_model(cot_prompt, max_tokens=250)

    ms = round((time.time() - t0) * 1000)
    return {"tier": tier, "response": response, "ms": ms}
```

Example output from the notebook (Qwen3-0.6B base + reasoning, CPU):

| Tier | ms | Task type | Prompt |
|------|----|-----------|--------|
| FAST | ~1 200 | arithmetic | What is 9 times 9? |
| FAST | ~900 | fact | Name the capital of Japan. |
| MID | ~3 000 | moderate | List two advantages of caching. |
| DEEP | ~8 000 | reasoning | Compare retry logic and circuit breakers. |
| DEEP | ~9 000 | reasoning | Why does model routing reduce cost? |

### Routing + fallback combined

The real design combines both: route first to decide which model to try, then fall back if that model fails.

```
Request → Router → Tier selection
                        │
                   Primary model for tier
                        │ fail
                   Fallback within tier (same class, different provider)
                        │ fail
                   Downgrade to next tier down
                        │ fail
                   Cache / degraded
```

This means a request that needed a deep reasoning model can gracefully serve from the mid-tier model when the reasoning model is unavailable — still better than a cached response, while preserving availability.

## 4. Gateway logic

At scale, individual fallback and routing logic inside each service becomes hard to maintain and observe. A gateway centralises this into one layer that all LLM calls pass through.

### What a gateway does

```
                    ┌────────────────────────────────┐
                    │          LLM Gateway            │
                    │                                │
  App service ────▶ │  Auth & rate limit per caller  │
                    │  Model routing                 │
  App service ────▶ │  Retry & fallback logic        │ ────▶ Model A (OpenAI)
                    │  Response caching              │
  App service ────▶ │  Circuit breaker per provider  │ ────▶ Model B (Anthropic)
                    │  Cost tracking per caller      │
                    │  Request / response logging    │ ────▶ Model C (local)
                    │                                │
                    └────────────────────────────────┘
```

Each application service calls the gateway with a task type and a prompt. The gateway handles everything else — which model to use, how to retry, whether to serve from cache, and how to log the call. Application services never talk directly to model providers.

### Circuit breaker

A circuit breaker prevents repeated calls to a provider that is known to be failing. Without it, every user request during an outage fires a slow timeout before hitting the fallback — multiplying latency by the number of retries.

```
States:
  CLOSED   → normal operation; calls pass through
  OPEN     → provider known to be failing; calls short-circuit to fallback immediately
  HALF-OPEN → testing if provider has recovered; let one call through

Transitions:
  CLOSED  → OPEN     : N consecutive failures within a time window
  OPEN    → HALF-OPEN: after a cooldown period (e.g. 30 seconds)
  HALF-OPEN → CLOSED : probe call succeeds
  HALF-OPEN → OPEN   : probe call fails; reset cooldown
```

The circuit breaker wraps the primary provider. When it trips OPEN, calls skip to the fallback immediately — no timeout wasted.

```python
import time
from enum import Enum

class CBState(Enum):
    CLOSED    = "CLOSED"      # normal — calls pass through
    OPEN      = "OPEN"        # provider known down — skip instantly
    HALF_OPEN = "HALF-OPEN"   # cooldown done — send one probe

class CircuitBreaker:
    def __init__(self, failure_threshold=3, cooldown_s=5.0):
        self.state      = CBState.CLOSED
        self.failures   = 0
        self.threshold  = failure_threshold
        self.cooldown   = cooldown_s
        self._opened_at = None

    def call(self, primary_fn, fallback_fn, prompt):
        skip_primary = False

        if self.state == CBState.OPEN:
            waited = time.time() - self._opened_at
            if waited < self.cooldown:
                skip_primary = True   # ⚡ no timeout wasted
            else:
                self.state = CBState.HALF_OPEN   # 🔶 send probe

        if not skip_primary:
            try:
                result = primary_fn(prompt)
                self.failures = 0
                self.state    = CBState.CLOSED
                return {"text": result, "source": "primary"}
            except Exception:
                self.failures += 1
                if self.failures >= self.threshold or self.state == CBState.HALF_OPEN:
                    self.state      = CBState.OPEN
                    self._opened_at = time.time()

        result = fallback_fn(prompt)
        return {"text": result, "source": "secondary"}
```

Observed behaviour from the notebook (threshold=3, cooldown=5s, primary = non-existent HTTP endpoint):

```
Request 1  [CLOSED  failures=0]  PRIMARY ❌ (1/3)  → SECONDARY ✅
Request 2  [CLOSED  failures=1]  PRIMARY ❌ (2/3)  → SECONDARY ✅
Request 3  [CLOSED  failures=2]  PRIMARY ❌ → circuit OPEN  → SECONDARY ✅
Request 4  [OPEN    failures=3]  ⚡ skipped instantly → SECONDARY ✅
Request 5  [OPEN    failures=3]  ⚡ skipped instantly → SECONDARY ✅
--- 5s cooldown ---
Request 6  [HALF-OPEN]  🔶 probe sent → PRIMARY ✅ → circuit CLOSED again
Request 7  [CLOSED  failures=0]  PRIMARY ✅
```

The key gain: requests 4 and 5 cost near-zero latency instead of waiting for a 2–10s timeout on each retry.

### Cost and quota tracking

The gateway is the right place to enforce per-caller token budgets. Each service or user gets a quota; the gateway rejects or downgrades calls that would exceed it.

```python
from collections import defaultdict


class TokenBudget:
    def __init__(self, limit_per_minute: dict[str, int]):
        self.limits = limit_per_minute
        self.usage: dict[str, list] = defaultdict(list)   # caller → [(timestamp, tokens)]

    def check_and_record(self, caller: str, estimated_tokens: int) -> bool:
        now = time.time()
        window = [u for u in self.usage[caller] if now - u[0] < 60]
        used = sum(u[1] for u in window)
        limit = self.limits.get(caller, self.limits.get("default", 10_000))
        if used + estimated_tokens > limit:
            return False   # over budget
        window.append((now, estimated_tokens))
        self.usage[caller] = window
        return True
```

## 5. Putting it together — a minimal reliable gateway

```python
class LLMGateway:
    def __init__(self, router_cfg, budget, breakers: dict):
        self.router = router_cfg
        self.budget = budget
        self.breakers = breakers   # provider name → CircuitBreaker

    def call(self, caller: str, prompt: str) -> dict:
        # 1. Budget check
        estimated = len(prompt.split()) * 2   # rough token estimate
        if not self.budget.check_and_record(caller, estimated):
            return {"text": None, "error": "quota_exceeded", "source": None}

        # 2. Route to model tier
        model_fn, tier = route(prompt, self.router)
        breaker = self.breakers.get(tier.value)

        # 3. Call through circuit breaker + fallback
        try:
            if breaker:
                text = breaker.call(model_fn, prompt)
            else:
                text = model_fn(prompt)
            return {"text": text, "source": tier.value, "error": None}
        except Exception:
            # Tier failed — try next tier down or degraded
            pass

        return {"text": "Service temporarily unavailable.", "source": "degraded", "error": None}
```

## 6. Summary

| Pattern | Problem it solves | When to add it |
|---|---|---|
| Timeout + retry | Brief blips, transient API errors | Always — every call needs this |
| Fallback chain | Primary model unavailable | Day one — before anything else |
| Output validator + retry | Silent quality failures, wrong format | As soon as output structure matters |
| Dynamic validator dispatch | Format knowledge leaking into callers | When multiple output types exist |
| Model routing | Wrong model for the task; unnecessary cost | When traffic is mixed-complexity |
| Circuit breaker | Slow timeouts during sustained outages | When you have more than one provider |
| Centralised gateway | Scattered retry/routing; no unified observability | When multiple services call LLMs |
| Token budget enforcement | Runaway cost from a single caller | Early — before production |

Reliability is not a single feature; it is a layered property. Start with a fallback chain on every call, add output validation next, and introduce a gateway when you have more than two services making LLM calls.

## 7. Full reliability flowchart

Every request through a production LLM system should pass through these layers in order:

```
                        User request
                             │
                             ▼
              ┌──────────────────────────┐
              │     Budget / quota       │──── over limit ────▶ 429 Quota exceeded
              │     (token allowance)    │
              └──────────────┬───────────┘
                             │
                             ▼
              ┌──────────────────────────┐
              │       Router             │
              │  keyword / classifier /  │──▶ FAST tier (base model, short prompt)
              │  LLM judge               │──▶ MID  tier (base model, longer prompt)
              └──────────────┬───────────┘──▶ DEEP tier (reasoning model + CoT)
                             │ (selected tier)
                             ▼
              ┌──────────────────────────┐
              │    Circuit breaker       │
              │    per provider          │
              └──────────────┬───────────┘
                    OPEN ────┘  ⚡ skip primary instantly
                    │  CLOSED / HALF-OPEN
                    ▼
     ┌──────────────────────────────┐
     │  Primary model call          │
     │  + threading timeout         │──── success ──┐
     └──────────────┬───────────────┘               │
                    │ fail / timeout                 ▼
                    │                  ┌─────────────────────────┐
          retry ◀───┤ (≤2 retries)     │   Output validator       │
          (backoff) │ still failing    │   NumberValidator /       │
                    ▼                  │   JSONValidator /         │
     ┌──────────────────────────────┐  │   MarkdownTableValidator /│
     │  Secondary model call        │  │   BulletListValidator /   │
     │  (fallback tier / endpoint)  │──┤   LLMJudgeValidator       │
     └──────────────┬───────────────┘  └──────────┬────────────────┘
                    │ fail                  valid  │  invalid
                    ▼                         ▼   │   ▼
     ┌──────────────────────────────┐    Return ✅ │  tighter prompt
     │  Cache lookup                │              │  retry (≤3x)
     │  (exact-match or semantic)   │              │  still invalid
     └──────────────┬───────────────┘              ▼
              hit   │   miss              secondary model
               ▼    │                          │
          Return     ▼                         │ fail
          cached  ┌──────────────────────┐     ▼
          (stale  │  Degraded response   │  cache → degraded
          warning)│  "unavailable, retry"│
                  └──────────────────────┘
                             │
                             ▼
                        Return ✅
                   (always returns something)
```

### What each layer saves you from

```
 Layer           Failure it prevents                     Without it
 ─────────────────────────────────────────────────────────────────────
 Timeout+retry   Transient blip → user sees 500          Every brief hiccup = error
 Fallback chain  Primary down → user sees 500            Primary down = outage
 Output validator Wrong format silently reaches product  Bad output ships undetected
 Dynamic dispatch Format leak into every call site       Brittle, hard to extend
 Routing         Simple question uses slow $$ model      5-10x higher cost + latency
 Circuit breaker Outage → 10s timeout × N retries × M   Request storm, timeout storms
                 users → server meltdown
 Gateway         Retry logic duplicated in every service Inconsistent, unobservable
 Token budget    One noisy caller drains quota           Cost spike, starvation for others
```
