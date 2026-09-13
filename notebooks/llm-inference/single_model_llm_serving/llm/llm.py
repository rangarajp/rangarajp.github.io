from typing import List, Dict, Any
from .workload_manager import WorkloadManager, Sequence
from .model_executor import ModelExecutor
from .flow import flow, flow_skip
import asyncio
import json
import atexit
import threading
import time
import uuid
from vllm import LLM as VLLM
from vllm import SamplingParams

class LLMEngine:
    def __init__(self, model_path: str, vllm_model=None, vllm_kwargs: Dict[str, Any] | None = None):
        """
        Args:
            model_path: Local path (or HF id) used by ModelWorker / ModelManager.
            vllm_model: Optional already-constructed vLLM `LLM` instance to reuse
                        (e.g. from `LLM(model=MODEL_PATH, ...)` in the notebook).
            vllm_kwargs: Kwargs for `LLM(model=model_path, **vllm_kwargs)` when
                         `vllm_model` is not provided.
        """
        flow("LLMEngine", f"__init__ — model_path={model_path!r}")
        self.model_path = model_path
        self.model_executor = ModelExecutor()
        self.workload_manager = WorkloadManager()
        self.max_tokens = 20
        
        # Transformers worker (custom stack: /basic_generate, /generate, /generate_stream)
        flow("LLMEngine", f"setup_worker({model_path!r}) — starts ModelWorker process")
        self.model_executor.setup_worker(model_path)
        
        # vLLM engine (only /generate_vllm)
        if vllm_model is not None:
            flow("LLMEngine", "reusing provided vLLM LLM(model_path) instance")
            self.vllm_model = vllm_model
        else:
            flow("LLMEngine", f"loading vLLM from {model_path!r}")
            self.vllm_model = VLLM(model=model_path, **(vllm_kwargs or {}))
        
        # Start processing loop in a separate thread
        flow("LLMEngine", "starting requests_processing_loop thread (streaming only)")
        self.thread = threading.Thread(target=self.requests_processing_loop, daemon=True)
        self.thread.start()
        
        # Register cleanup
        atexit.register(self._cleanup)
    
    def requests_processing_loop(self):
        """Process streaming requests in a loop."""
        while True:
            try:
                active_sequences = self.workload_manager.get_next_batch(is_streaming=True)
                if not active_sequences:
                    time.sleep(0.1)
                    continue

                flow("LLMEngine", f"streaming loop — {len(active_sequences)} active seq(s) → ModelExecutor.execute_forward_batch()")
                    
                # Process batch through model, forward pass.
                prompts = [{'prompt': seq.prompt, 'request_id': seq.id} for seq in active_sequences]
                prompts_results = self.model_executor.execute_forward_batch(prompts)
                
                # Stream tokens back to respective clients
                for result in prompts_results:
                    seq = self.workload_manager.get_sequence(result['request_id'])
                    if result['is_finished'] or seq.token_count > self.max_tokens:
                        flow("LLMEngine", f"streaming loop — seq {result['request_id'][:8]}… finished → put None on client queue")
                        # Use run_coroutine_threadsafe to safely put None in the main loop's queue
                        asyncio.run_coroutine_threadsafe(
                            seq.client_stream.put(None),
                            seq.loop
                        )
                        seq.finished = True
                        self.workload_manager.remove_finished_sequence(result['request_id'])
                    else:
                        flow("LLMEngine", f"streaming loop — seq {result['request_id'][:8]}… token={result['token']!r} → client queue")
                        # Use run_coroutine_threadsafe to safely put data in the main loop's queue
                        asyncio.run_coroutine_threadsafe(
                            seq.client_stream.put(
                                json.dumps({"token": result['token'], "sequence_id": result['request_id']})
                            ),
                            seq.loop
                        )
                        self.workload_manager.update_sequence_output(result['request_id'], result['token'])
                
            except Exception as e:
                print(f"Error in processing loop: {e}", flush=True)
                time.sleep(0.1)
    
    def _cleanup(self):
        """Cleanup function to be called when the program exits."""
        # The thread will be automatically terminated since it's a daemon thread
        pass

    # process 1 request with only one prompt at a time.
    def basic_generate(self, prompt: str) -> str:
        flow("LLMEngine", f"basic_generate(prompt={prompt!r})")
        flow_skip("WorkloadManager", "basic_generate builds Sequence locally; no add_request / get_next_batch")

        sequence = Sequence(str(uuid.uuid4()), prompt, None, None)
        
        flow("LLMEngine", "→ ModelExecutor.execute_batch([sequence])")
        # Execute the batch
        results = self.model_executor.execute_batch([sequence])
        flow("LLMEngine", "← result received from ModelExecutor")

        return results[1][0]['generated_text']
    
    def _is_batch_finished(self, request_ids: List[str]) -> bool:
        for id in request_ids:
            if not self.workload_manager.is_sequence_finished(id):
                return False
        return True

    # process multiple prompts in a request
    def generate(self, prompts: List[str]) -> List[str]:
        flow("LLMEngine", f"generate({len(prompts)} prompts) — uses WorkloadManager")

        # Add all requests to workload manager
        request_ids = []
        for prompt in prompts:
            request_id = self.workload_manager.add_request(prompt)
            request_ids.append(request_id)
        
        # Process requests in batches (from LoadManager) until all prompts of the request are finished
        while not self._is_batch_finished(request_ids):
            # Get next batch of requests
            sequences = self.workload_manager.get_next_batch()
            if not sequences:
                time.sleep(0.1)
                continue

            flow("LLMEngine", f"batch loop — {len(sequences)} seq(s) → ModelExecutor.execute_batch()")
            # Execute the next batch in one go, it may not be the same prompts as the prompts in the request.
            results = self.model_executor.execute_batch(sequences)
        
            # Update results in workload manager
            for result in results[1]:
                self.workload_manager.remove_active_sequence(result['request_id'])
                self.workload_manager.update_sequence_output(result['request_id'], result['generated_text'], is_finished=True)

        # Remove finished sequences from workload manager
        generated_texts = []
        for request_id in request_ids:
            generated_texts.append(self.workload_manager.get_sequence(request_id).output[0])
            self.workload_manager.remove_finished_sequence(request_id)

        flow("LLMEngine", f"← returning {len(generated_texts)} generated texts")
        return generated_texts 
    
    async def event_generator(self, loop, prompt: str):
        flow("LLMEngine", f"event_generator(prompt={prompt!r})")
        asyncio.set_event_loop(loop)
        # Create a queue for this client's stream
        queue = asyncio.Queue()
        
        # Add streaming request to workload manager with the queue
        seq_id = self.workload_manager.add_streaming_request(prompt, queue, loop)
        flow("LLMEngine", f"streaming seq {seq_id[:8]}… queued; waiting on client asyncio.Queue")
        flow("LLMEngine", "(tokens produced asynchronously by requests_processing_loop thread)")
        
        try:
            while True:
                # Get next token from queue
                data = await queue.get()
                if data is None:  # End of stream
                    flow("LLMEngine", f"event_generator — end of stream for seq {seq_id[:8]}…")
                    break
                yield f"data: {data}\n\n"
        except Exception as e:
            print(f"Error in stream for sequence {seq_id}: {e}", flush=True)
        finally:
            # Clean up
            self.workload_manager.remove_finished_sequence(seq_id)
            flow("LLMEngine", f"event_generator — cleaned up seq {seq_id[:8]}…")

    def generate_vllm(self, prompts: List[str]) -> List[str]:
        """
        Generate text using vLLM for multiple prompts.
        
        Args:
            prompts: List of prompts to generate text for
            
        Returns:
            List of generated texts
        """
        flow("LLMEngine", f"generate_vllm({len(prompts)} prompts) — direct vLLM, bypasses custom stack")
        flow_skip("WorkloadManager", "not used")
        flow_skip("ModelExecutor / ModelWorker / ModelManager", "not used; vLLM owns inference")

        # Configure sampling parameters
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=self.max_tokens
        )
        
        flow("LLMEngine", "→ vLLM.generate()  [GPU via vLLM engine]")
        # Generate text for all prompts
        outputs = self.vllm_model.generate(prompts, sampling_params)
        
        # Extract generated text from outputs
        generated_texts = [output.outputs[0].text for output in outputs]
        flow("LLMEngine", f"← vLLM returned {len(generated_texts)} texts")
        
        return generated_texts
