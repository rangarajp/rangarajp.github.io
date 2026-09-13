from typing import Any, Dict, Optional, Union
import torch
from abc import ABC, abstractmethod
from transformers import AutoModelForCausalLM, AutoTokenizer
from .store import get_models_dir


class ModelWorker(ABC):
    def __init__(self, model_metadata):
        self.model_metadata = model_metadata
        self.model: Optional[torch.nn.Module] = None
        self._load_model()

    @abstractmethod
    def _load_model(self):
        pass

    @abstractmethod
    def predict(self, input_data: Any) -> Dict[str, Any]:
        pass

    def unload(self):
        """Free model weights and GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class LLMCausalWorker(ModelWorker):
    """Causal LM worker for local chat / instruct checkpoints (Qwen, TinyLlama, …)."""

    def __init__(self, model_metadata):
        self.tokenizer: Optional[AutoTokenizer] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        super().__init__(model_metadata)

    def _load_model(self):
        path = self.model_metadata.resolved_path()
        model_path = str(path)
        print(
            f"[LLMCausalWorker] loading {self.model_metadata.name} "
            f"from {model_path} on {self.device} (MODELS_DIR={get_models_dir()})"
        )
        if not path.exists():
            raise FileNotFoundError(
                f"Model path not found: {model_path}. "
                f"Check MODELS_DIR={get_models_dir()} and config path={self.model_metadata.path!r}"
            )
        if not (path / "config.json").exists():
            raise FileNotFoundError(
                f"config.json missing under {model_path} — incomplete download?"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=self.dtype,
        )
        self.model.to(self.device)
        self.model.eval()
        print(f"[LLMCausalWorker] loaded {self.model_metadata.name}")

    def _format_prompt(self, prompt: str) -> str:
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt

    def _parse_input(self, input_data: Union[str, Dict[str, Any]]) -> tuple[str, int]:
        if isinstance(input_data, str):
            return input_data, 64
        if isinstance(input_data, dict):
            prompt = input_data.get("prompt") or input_data.get("text") or ""
            max_new_tokens = int(input_data.get("max_new_tokens", 64))
            return prompt, max_new_tokens
        raise ValueError("input_data must be a string prompt or a dict with 'prompt'")

    def predict(self, input_data: Any) -> Dict[str, Any]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model or tokenizer not initialized")

        prompt, max_new_tokens = self._parse_input(input_data)
        if not prompt:
            raise ValueError("Empty prompt")

        formatted = self._format_prompt(prompt)
        inputs = self.tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only newly generated tokens
        generated_ids = outputs[0][inputs["input_ids"].shape[-1] :]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return {
            "model_id": self.model_metadata.id,
            "model_name": self.model_metadata.name,
            "prompt": prompt,
            "generated_text": generated_text,
        }

    def unload(self):
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        super().unload()
