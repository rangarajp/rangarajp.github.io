---
title: "Autoregressive Generation: Why Inference Is a Loop"
description: "Why inference freezes the weights and still runs the model once per new token, and a generation loop you can read line by line."
section: learn
pubDate: 2026-10-29
part: 1
chapter: 5
order: 5
tags: ["llm-inference", "fundamentals", "pytorch"]
---

Sampling chooses one token. This chapter places that choice inside a complete loop, deliberately recomputing the prefix first so the later benefit of KV reuse is visible.

## 1. Training and inference want different things

Training and inference use the same forward pass. They keep different state, and they score success differently.

| | Training | Inference |
|---|---|---|
| Parameters | Updated from gradients | Frozen |
| Extra tensors | Activations for the backward pass, gradients, optimizer moments | The KV cache for tokens already seen |
| What you repeat | Full sequences, in large batches, for days | One new token, for a person who is waiting |
| The number you care about | Tokens per second across the job, and the loss | Time to the first token, time between later tokens, and cost per request |

During training, a single slow step is fine if the GPU stays full and the loss moves. During inference, the weight values stay fixed, while their data still moves from HBM toward the compute units. The work is a forward pass from a prompt to the next id, then another forward pass, until the reply is done. Optimizer state is gone. The KV cache, introduced in the next chapter, is the new piece of memory that grows with the conversation.

A 200 ms reply feels immediate. A multi-second wait before the first token feels broken, even when the model is the same checkpoint you trained. That is why inference work is organized around the loop below, not around a throughput number for a week-long job.

## 2. The token generation loop

Decoder-only models are *autoregressive*: the distribution for the next id is conditioned on every id already in the sequence. The loop is:

1. Receive a prompt as token ids.
2. Run one forward pass over that whole prefix. This is the *prefill*. It writes keys and values for every prompt token.
3. Sample one id from the last position's logits. Append it.
4. Run the next forward pass for the new token, attending to the prefix. Repeat until an end-of-sequence id or a length limit.

```text
Prompt: "The battery"
Step 1: "The battery"                 → " is"
Step 2: "The battery is"              → " heating"
Step 3: "The battery is heating"      → " up"
Step 4: "The battery is heating up"   → "."
```

Those pieces are invented so the dependency is obvious. They are not a trace from a real model. One token is not necessarily one word. The stop condition is an id the tokenizer calls end-of-sequence, a stop string, or a maximum number of new tokens. It is not "the model decided the paragraph is finished" unless that decision was encoded as one of those rules.

Step 2 can run the prompt tokens together. Steps 3 and 4 cannot be skipped or reordered: the distribution at step 4 depends on which id step 3 actually drew. If you sampled `cooling` instead of `heating`, the following distribution would be a different row of the LM head.

## 3. A loop you can run

This version sets `use_cache=False`. Every iteration feeds the entire id list and recomputes attention for every position. That is wasteful, and it is the right first implementation because the growing tensor is visible. Run it where the model can be downloaded, or point `model_id` at a local snapshot.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).eval()

messages = [{"role": "user", "content": "Explain why a battery needs cooling in one sentence."}]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
ids = tokenizer(prompt, return_tensors="pt").input_ids

with torch.inference_mode():
    for step in range(12):
        outputs = model(input_ids=ids, use_cache=False)
        next_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        piece = tokenizer.decode(next_id[0])
        ids = torch.cat([ids, next_id], dim=1)
        print(f"Step {step + 1}: {piece!r}  ids.shape={tuple(ids.shape)}")
        if tokenizer.eos_token_id is not None and next_id.item() == tokenizer.eos_token_id:
            break

print(tokenizer.decode(ids[0], skip_special_tokens=True))
```

`apply_chat_template` matters for instruct checkpoints. The model was trained to see a user turn and a generation marker. A raw sentence without that wrapper is a different input, even when the English looks the same.

`argmax` is greedy selection from the previous chapter. Swap it for the sampler you wrote there if you want temperature. Keep the loop structure.

This is not how a serving stack should decode. It rereads the whole prefix on every step. The next chapter keeps the keys and values so the new step only attends, instead of recomputing the prefix.

## 4. The sequence grows by one id

With `B = 1` and a prompt of length `T = 5`:

| Step | `input_ids` shape | Logits used for the sample |
|---|---|---|
| Prefill | `[1, 5]` | `[1, V]` from position 5 |
| After one append | `[1, 6]` | `[1, V]` from position 6 |
| After another | `[1, 7]` | `[1, V]` from position 7 |

The forward pass returns logits at every position. Generation uses the last row only. The earlier rows are predictions for tokens you already have.

The second dimension of `ids` is the only shape that changes in this loop. `V` is fixed by the tokenizer. The hidden width is fixed by the checkpoint. What grows is the context the next distribution conditions on, and, once caching is on, the KV cache that stores it.


## 5. Two choices that do not change the dependency

The example makes two independent choices:

1. *Decoding strategy.* `argmax` is greedy. Sampling would draw from the probabilities in the previous chapter. Either way you append exactly one id before the next forward pass.
2. *Computation strategy.* `use_cache=False` recomputes every position. A correct cache reuses keys and values for the prefix and runs the new token only. The next chapter builds that distinction. A wrong cache (feeding the full sequence *and* a past cache, or forgetting the position id) silently attends to the wrong places.

Changing either choice does not remove the dependency. Token $t+1$ is a function of the prefix through token $t$.

## Hands-on: expose the growing prefix

Use the existing uncached loop to print input length and selected token at every step. Publish a short trace with actual token IDs, and compare it with the later cached baseline under the same stopping rule.
