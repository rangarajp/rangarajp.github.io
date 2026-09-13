import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .flow import flow

class ModelManager:
    def __init__(self):
        self.model_dir = "model_cache"
        flow("ModelManager", f"__init__ — model_dir={self.model_dir}")
    
    def load_model(self, model_name: str) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
        flow("ModelManager", f"load_model({model_name!r}) — called at worker startup (not per request)")
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)

        # Prefer local files when offline / MODEL_PATH points at a downloaded tree
        offline = os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        flow("ModelManager", f"from_pretrained(model) local_files_only={offline} dtype={dtype}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            local_files_only=offline,
            torch_dtype=dtype,
        )
        flow("ModelManager", "from_pretrained(tokenizer)…")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=offline,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        flow("ModelManager", "← model + tokenizer loaded")
        
        return model, tokenizer
