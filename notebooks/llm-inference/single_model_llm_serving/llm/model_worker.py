import multiprocessing as mp
from typing import List, Dict, Any
from .model_manager import ModelManager
from .flow import flow, flow_skip
import torch
import traceback
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


def _prompt_text(p) -> str:
    if isinstance(p, dict):
        return p.get("prompt") or p.get("text")
    return p.prompt


def _request_id(p) -> str:
    if isinstance(p, dict):
        return p.get("id") or p.get("request_id")
    return p.id


class ModelWorker:
    def __init__(self, model_name: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        flow("ModelWorker", f"__init__ — device={self.device}, loading model via ModelManager")
        self.model, self.tokenizer = ModelManager().load_model(model_name)
        self.model.to(self.device)
        self.model.eval()
        flow("ModelWorker", f"model moved to {self.device}")
        # Initialize state for streaming
        self.stream_states = {}  # request_id -> (input_ids, attention_mask, past_key_values)
    
    def generate(self, prompts: List[Any]) -> List[Dict[str, Any]]:
        flow("ModelWorker", f"generate() — batch size={len(prompts)}")
        flow_skip("ModelManager", "model already loaded at worker startup; not reloaded per request")
        
        prompt_texts = [_prompt_text(p) for p in prompts]
        request_ids = [_request_id(p) for p in prompts]
        
        inputs = self.tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        flow("GPU", f"model.generate() on {self.device}  input_shape={tuple(inputs.input_ids.shape)}")
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=50,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        flow("GPU", "← model.generate() finished")
        
        generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        results = [
            {
                'request_id': request_id,
                'generated_text': generated_text
            }
            for request_id, generated_text in zip(request_ids, generated_texts)
        ]
        flow("ModelWorker", f"generate() — returning {len(results)} result(s)")
        
        return results

    def generate_forward_batch(self, prompts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate one token for each prompt in the batch."""
        flow("ModelWorker", f"generate_forward_batch() — {len(prompts)} prompt(s), one token each")
        flow_skip("ModelManager", "model already loaded; not reloaded per request")
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        encoded = self.tokenizer(
            [_prompt_text(p) for p in prompts],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        flow("GPU", f"model forward() on {self.device}  input_shape={tuple(encoded.input_ids.shape)}")
        with torch.no_grad():
            outputs = self.model(
                input_ids=encoded.input_ids,
                attention_mask=encoded.attention_mask,
                use_cache=False
            )
            
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.multinomial(
                torch.softmax(next_token_logits / 0.7, dim=-1),
                num_samples=1
            ).squeeze(-1)
            flow("GPU", "← forward() + sample finished")
            
            results = []
            for i, prompt_data in enumerate(prompts):
                token = self.tokenizer.decode(next_token[i].unsqueeze(0), skip_special_tokens=True)
                results.append({
                    'request_id': _request_id(prompt_data),
                    'token': token,
                    'is_finished': token == self.tokenizer.eos_token
                })

            flow("ModelWorker", f"generate_forward_batch() — returning {len(results)} token(s)")
            return results

    @staticmethod
    def run(model_name: str, task_queue: mp.Queue, result_queue: mp.Queue):
        # Force unbuffered-ish logging in spawned worker (Jupyter often hides child stdout)
        try:
            sys.stdout.reconfigure(line_buffering=True)
        except Exception:
            pass

        try:
            flow("ModelWorker", "run() — spawn worker started, loading model…")
            worker = ModelWorker(model_name)
            result_queue.put(("ready", None))
            flow("ModelWorker", "worker ready — entering task loop")
        except Exception:
            err = traceback.format_exc()
            print(err, flush=True)
            try:
                result_queue.put(("error", err))
            except Exception:
                pass
            return
        
        while True:
            try:
                batch_data = task_queue.get()

                if batch_data is None:  # Shutdown signal
                    flow("ModelWorker", "shutdown signal received")
                    break
                
                batch, is_streaming = batch_data
                flow("ModelWorker", f"task received — is_streaming={is_streaming}, batch_size={len(batch)}")
                
                if is_streaming:
                    result_queue.put(('stream', worker.generate_forward_batch(batch)))
                else:
                    result_queue.put(('complete', worker.generate(batch)))
                flow("ModelWorker", "result put on result_queue")
            except Exception:
                err = traceback.format_exc()
                print(err, flush=True)
                try:
                    result_queue.put(("error", err))
                except Exception:
                    pass
