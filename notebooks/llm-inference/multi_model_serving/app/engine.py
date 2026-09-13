from typing import Optional
from .store import ModelMetadata
from .worker import ModelWorker, LLMCausalWorker


class ModelEngine:
    def __init__(self):
        self.workers = {}  # {model_id: worker}

    def get_worker(self, model_id: str) -> Optional[ModelWorker]:
        return self.workers.get(model_id)

    def create_worker(self, model_metadata: ModelMetadata) -> ModelWorker:
        if model_metadata.id not in self.workers:
            if model_metadata.framework == "transformers" and model_metadata.type == "llm":
                self.workers[model_metadata.id] = LLMCausalWorker(model_metadata)
            else:
                raise ValueError(
                    f"Unsupported framework/type: {model_metadata.framework}/{model_metadata.type}"
                )
        return self.workers[model_metadata.id]

    def delete_worker(self, model_id: str):
        worker = self.workers.pop(model_id, None)
        if worker is not None:
            worker.unload()
            print(f"[ModelEngine] unloaded worker for {model_id}")
