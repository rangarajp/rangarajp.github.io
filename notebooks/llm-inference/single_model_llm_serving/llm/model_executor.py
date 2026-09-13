import multiprocessing as mp
from queue import Empty
from typing import List, Dict, Any, Optional
from .model_worker import ModelWorker
from .flow import flow
import logging
import sys

# Set up logging with stream handler
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)

# spawn avoids CUDA-fork hangs when vLLM already initialized GPU in the parent
_CTX = mp.get_context("spawn")


class ModelExecutor:
    def __init__(self):
        self.task_queue = _CTX.Queue()
        self.result_queue = _CTX.Queue()
        self.worker_process = None
        flow("ModelExecutor", "__init__ — spawn queues created (safe with CUDA/vLLM in parent)")
    
    def setup_worker(self, model_name: str, ready_timeout: float = 600.0):
        flow("ModelExecutor", f"setup_worker({model_name!r}) — spawn ModelWorker process")
        self.worker_process = _CTX.Process(
            target=ModelWorker.run,
            args=(model_name, self.task_queue, self.result_queue),
            daemon=True,
        )
        self.worker_process.start()
        flow("ModelExecutor", f"worker process started pid={self.worker_process.pid} — waiting for ready…")

        # Block until worker finished loading model (or died)
        msg = self._get_result(timeout=ready_timeout, label="worker ready")
        if not isinstance(msg, tuple) or msg[0] != "ready":
            raise RuntimeError(f"ModelWorker failed to start, got: {msg!r}")
        flow("ModelExecutor", "← ModelWorker ready")
    
    def _ensure_worker_alive(self):
        if self.worker_process is not None and not self.worker_process.is_alive():
            raise RuntimeError(
                f"ModelWorker process is dead (exitcode={self.worker_process.exitcode}). "
                "Check worker stderr / OOM. Not caused by flow imports — worker crashed "
                "before putting a result on result_queue."
            )

    def _get_result(self, timeout: Optional[float] = None, label: str = "result"):
        """Wait for result_queue; fail fast if worker dies (avoids infinite hang)."""
        deadline_chunks = 1.0
        waited = 0.0
        while True:
            self._ensure_worker_alive()
            try:
                return self.result_queue.get(timeout=deadline_chunks)
            except Empty:
                waited += deadline_chunks
                if timeout is not None and waited >= timeout:
                    self._ensure_worker_alive()
                    raise TimeoutError(
                        f"Timed out after {waited:.0f}s waiting for {label} from ModelWorker"
                    )
                if int(waited) % 10 < 1:
                    flow("ModelExecutor", f"still waiting for {label}… ({waited:.0f}s)")
    
    def execute_batch(self, prompts: List[Any]) -> Any:
        if not prompts:
            flow("ModelExecutor", "execute_batch — empty batch, skip")
            return []

        # Serialize to plain dicts (safer across spawn than Sequence objects)
        payload = [
            {"prompt": p.prompt, "id": p.id} if hasattr(p, "prompt") else p
            for p in prompts
        ]

        flow("ModelExecutor", f"execute_batch — sending {len(payload)} item(s) to worker (is_streaming=False)")
        self._ensure_worker_alive()
        self.task_queue.put((payload, False))

        flow("ModelExecutor", "waiting on result_queue…")
        results = self._get_result(timeout=300.0, label="execute_batch")
        if isinstance(results, tuple) and results[0] == "error":
            raise RuntimeError(f"ModelWorker error: {results[1]}")
        flow("ModelExecutor", "← result received from ModelWorker")
        return results
    
    def execute_forward_batch(self, prompts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not prompts:
            flow("ModelExecutor", "execute_forward_batch — empty batch, skip")
            return []
        
        flow("ModelExecutor", f"execute_forward_batch — sending {len(prompts)} item(s) to worker (is_streaming=True)")
        self._ensure_worker_alive()
        self.task_queue.put((prompts, True))
        
        flow("ModelExecutor", "waiting on result_queue…")
        result = self._get_result(timeout=300.0, label="execute_forward_batch")
        if isinstance(result, tuple) and result[0] == "error":
            raise RuntimeError(f"ModelWorker error: {result[1]}")

        result_type, results = result
        flow("ModelExecutor", f"← streaming result received (type={result_type})")

        if result_type == 'stream':
            return results
        else:
            raise Exception(f"Unexpected result type from worker: {result_type}")
    
    def __del__(self):
        if self.worker_process and self.worker_process.is_alive():
            try:
                self.task_queue.put(None)
            except Exception:
                pass
            self.worker_process.terminate()
            self.worker_process.join(timeout=5)
