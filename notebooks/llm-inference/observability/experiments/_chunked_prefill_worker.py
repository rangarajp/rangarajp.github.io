"""Stress demo: chat streaming + one or more long_doc prefills.

Usage:
  python _chunked_prefill_worker.py \\
      <model> <doc.txt> <chat.txt> <chunked:0|1> <max_batched> <out.json> \\
      [max_model_len=4096] [n_docs=3] [delay_s=0.35]
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import traceback
import uuid
from pathlib import Path


def main() -> None:
    model = sys.argv[1]
    doc_prompt = Path(sys.argv[2]).read_text(encoding="utf-8")
    chat_prompt = Path(sys.argv[3]).read_text(encoding="utf-8")
    chunked = sys.argv[4] == "1"
    max_batched = int(sys.argv[5])
    out_path = Path(sys.argv[6])
    max_model_len = int(sys.argv[7]) if len(sys.argv) > 7 else 4096
    n_docs = int(sys.argv[8]) if len(sys.argv) > 8 else 3
    delay_s = float(sys.argv[9]) if len(sys.argv) > 9 else 0.35

    log_path = out_path.with_suffix(".log")

    def log(msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    if log_path.exists():
        log_path.unlink()

    log(
        f"start chunked={chunked} max_batched={max_batched} "
        f"max_model_len={max_model_len} n_docs={n_docs} delay={delay_s}"
    )
    log(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine

    kwargs = dict(
        model=model,
        dtype="float16",
        gpu_memory_utilization=0.85,
        max_model_len=max_model_len,
        max_num_seqs=max(4, 1 + n_docs),
        disable_log_requests=True,
        enable_chunked_prefill=chunked,
        enforce_eager=True,  # more stable on V100; chunked may still abort on some builds
    )
    if chunked:
        kwargs["max_num_batched_tokens"] = max_batched
        kwargs["enforce_eager"] = False

    log(f"engine kwargs={ {k: kwargs[k] for k in ('enable_chunked_prefill','enforce_eager','max_model_len','max_num_seqs','max_num_batched_tokens') if k in kwargs} }")
    try:
        engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**kwargs))
    except Exception:
        log("engine init failed:\n" + traceback.format_exc())
        raise
    log("engine ready")

    async def generate_traced(prompt: str, max_tokens: int) -> dict:
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        rid = uuid.uuid4().hex
        t_send = time.perf_counter()
        times: list[float] = []
        text = ""
        async for out in engine.generate(prompt, sp, rid):
            n = len(out.outputs[0].token_ids)
            now = time.perf_counter()
            while len(times) < n:
                times.append(now)
            text = out.outputs[0].text
        return {
            "ok": bool(times),
            "n_tok": len(times),
            "ttft_s": (times[0] - t_send) if times else None,
            "decode_s": (times[-1] - times[0]) if len(times) >= 2 else 0.0,
            "e2e_s": (times[-1] - t_send) if times else None,
            "t_send": t_send,
            "t_first": times[0] if times else None,
            "t_last": times[-1] if times else None,
            "token_times": times,
            "text": (text or "")[:200],
        }

    async def run():
        log("warmup")
        await generate_traced("warmup", 4)
        t0 = time.perf_counter()

        async def chat_job():
            log("chat start")
            m = await generate_traced(chat_prompt, 80)
            log(f"chat done tok={m['n_tok']}")
            return m

        async def doc_job(i: int):
            await asyncio.sleep(delay_s)
            prompt = f"{doc_prompt}\n\n[doc={i}]"
            log(f"long_doc[{i}] start")
            m = await generate_traced(prompt, 8)
            log(f"long_doc[{i}] done tok={m['n_tok']} ttft={m['ttft_s']:.3f}")
            return m

        if n_docs <= 0:
            chat_m = await chat_job()
            docs = []
        else:
            results = await asyncio.gather(chat_job(), *[doc_job(i) for i in range(n_docs)])
            chat_m = results[0]
            docs = list(results[1:])
        wall = time.perf_counter() - t0

        def pack(name, m, arrive):
            return {
                "name": name,
                "n_tok": m["n_tok"],
                "arrive": arrive,
                "start": m["t_send"] - t0,
                "first": m["t_first"] - t0,
                "done": m["t_last"] - t0,
                "release": m["t_last"] - t0,
                "ttft_s": m["ttft_s"],
                "decode_s": m["decode_s"],
                "e2e_s": m["e2e_s"],
            }

        rows = [pack("chat", chat_m, 0.0)]
        for i, d in enumerate(docs):
            rows.append(pack(f"long_doc_{i}", d, delay_s))

        if docs:
            doc_send0 = min(d["t_send"] for d in docs)
            doc_first1 = max(d["t_first"] for d in docs)
            freight = {
                "ttft_s": doc_first1 - doc_send0,
                "n_tok": sum(d["n_tok"] for d in docs),
                "t_send_rel": doc_send0 - t0,
                "t_first_rel": doc_first1 - t0,
            }
        else:
            freight = {
                "ttft_s": 0.0,
                "n_tok": 0,
                "t_send_rel": 0.0,
                "t_first_rel": 0.0,
            }

        return {
            "chunked": chunked,
            "simulated": False,
            "wall": wall,
            "n_docs": n_docs,
            "delay_s": delay_s,
            "rows": rows,
            "rider": {
                "ttft_s": chat_m["ttft_s"],
                "n_tok": chat_m["n_tok"],
                "token_times_rel": [t - chat_m["t_send"] for t in chat_m["token_times"]],
                "t_send_rel": chat_m["t_send"] - t0,
            },
            "freight": freight,
        }

    try:
        result = asyncio.run(run())
    except Exception:
        log("run failed:\n" + traceback.format_exc())
        raise

    out_path.write_text(json.dumps(result), encoding="utf-8")
    log(
        f"WROTE {out_path} wall={result['wall']:.3f} "
        f"doc_wave_prefill={result['freight']['ttft_s']:.3f}s"
    )


if __name__ == "__main__":
    main()
