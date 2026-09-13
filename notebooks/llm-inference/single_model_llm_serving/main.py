from fastapi import FastAPI, BackgroundTasks, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from llm import LLMEngine
from llm.flow import flow, flow_banner, flow_done, flow_skip
from typing import List
import asyncio
import multiprocessing
import atexit
import signal
import os

# Local Qwen checkpoint (override with env MODEL_PATH, or local_paths.json)
def _default_model_path() -> str:
    env = os.environ.get("MODEL_PATH")
    if env:
        return env
    try:
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from local_paths import model_path

        return str(model_path("QWEN_MODEL"))
    except Exception:
        return "Qwen2.5-0.5B-Instruct"


MODEL_PATH = _default_model_path()

# Create FastAPI app
app = FastAPI()

# Create LLM instance
_llm = None
_llm_lock = multiprocessing.Lock()

def cleanup():
    global _llm
    if _llm is not None:
        try:
            _llm._cleanup()
        except:
            pass
        _llm = None

def get_llm():
    global _llm
    with _llm_lock:
        if _llm is None:
            flow("FastAPI", f"get_llm() — creating LLMEngine(model_path={MODEL_PATH!r})")
            _llm = LLMEngine(
                model_path=MODEL_PATH,
                vllm_kwargs={
                    "dtype": "float16",
                    "gpu_memory_utilization": 0.15,
                    "max_model_len": 1024,
                    "max_num_seqs": 4,
                    "enforce_eager": True,
                },
            )
            # Register cleanup
            atexit.register(cleanup)
        return _llm

class GenerateRequest(BaseModel):
    prompt: str

class GenerateResponse(BaseModel):
    generated_text: str

class BatchGenerateRequest(BaseModel):
    prompts: List[str]

class BatchGenerateResponse(BaseModel):
    generated_texts: List[str]

@app.post("/generate_stream")
async def generate_stream(request: GenerateRequest, llm: LLMEngine = Depends(get_llm)):
    flow_banner("POST /generate_stream")
    flow("FastAPI", f"/generate_stream — prompt={request.prompt!r}")
    flow("FastAPI", "→ LLMEngine.event_generator()  [uses WorkloadManager + streaming loop]")

    async def event_generator():
        loop = asyncio.get_event_loop()
        async for token in llm.event_generator(loop, request.prompt):
            # token = 'data: {"token": " a", "sequence_id": "8310f5e1-6f6f-480e-b2f9-c8144a12cc17"}\n\n'
            yield token
        flow_done("POST /generate_stream")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )

# process 1 request with only one prompt at a time.
@app.post("/basic_generate", response_model=GenerateResponse)
async def basic_generate(request: GenerateRequest, llm: LLMEngine = Depends(get_llm)):
    flow_banner("POST /basic_generate")
    flow("FastAPI", f"/basic_generate — prompt={request.prompt!r}")
    flow("FastAPI", "→ LLMEngine.basic_generate()")
    flow_skip("WorkloadManager", "basic_generate bypasses queue/batching; goes straight to ModelExecutor")

    generated_text = llm.basic_generate(request.prompt)

    flow("FastAPI", "← returning GenerateResponse")
    flow_done("POST /basic_generate")
    return GenerateResponse(generated_text=generated_text)

# process multiple prompts in a request
@app.post("/generate", response_model=BatchGenerateResponse)
async def generate(request: BatchGenerateRequest, llm: LLMEngine = Depends(get_llm)):
    flow_banner("POST /generate")
    flow("FastAPI", f"/generate — {len(request.prompts)} prompts")
    flow("FastAPI", "→ LLMEngine.generate()  [uses WorkloadManager batching]")

    generated_texts = llm.generate(request.prompts)

    flow("FastAPI", "← returning BatchGenerateResponse")
    flow_done("POST /generate")
    return BatchGenerateResponse(generated_texts=generated_texts)

@app.post("/generate_vllm", response_model=BatchGenerateResponse)
async def generate_vllm(request: BatchGenerateRequest, llm: LLMEngine = Depends(get_llm)):
    """
    Generate text using vLLM for multiple prompts.
    This endpoint uses vLLM's efficient batched inference capabilities.
    """
    flow_banner("POST /generate_vllm")
    flow("FastAPI", f"/generate_vllm — {len(request.prompts)} prompts")
    flow("FastAPI", "→ LLMEngine.generate_vllm()")
    flow_skip("WorkloadManager", "vLLM path does not use WorkloadManager")
    flow_skip("ModelExecutor", "vLLM path does not use ModelExecutor")
    flow_skip("ModelWorker", "vLLM path does not use ModelWorker")
    flow_skip("ModelManager", "vLLM loads/runs its own engine")

    generated_texts = llm.generate_vllm(request.prompts)

    flow("FastAPI", "← returning BatchGenerateResponse")
    flow_done("POST /generate_vllm")
    return BatchGenerateResponse(generated_texts=generated_texts)

def signal_handler(signum, frame):
    cleanup()
    exit(0)

if __name__ == "__main__":
    import uvicorn
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Initialize LLM before starting the server
    get_llm()
    uvicorn.run(app, host="0.0.0.0", port=8000)
