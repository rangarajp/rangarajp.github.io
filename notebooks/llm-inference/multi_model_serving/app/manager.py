from collections import OrderedDict
from typing import Dict, Optional
from .store import ModelStore
from .worker import ModelWorker
from .engine import ModelEngine


class ModelManager:
    def __init__(self, model_store: ModelStore, max_models: int = 2):
        self.model_store = model_store
        self.max_models = max_models
        # OrderedDict tracks LRU order: oldest at front, newest at end
        self.model_cache: OrderedDict[str, ModelWorker] = OrderedDict()
        self.model_engine = ModelEngine()

    def get_model_worker(self, model_id: str) -> Optional[ModelWorker]:
        if model_id in self.model_cache:
            self.model_cache.move_to_end(model_id)
            return self.model_engine.get_worker(model_id)

        model_metadata = self.model_store.get_model(model_id)
        if not model_metadata:
            return None

        # Evict least-recently used when cache is full
        if len(self.model_cache) >= self.max_models:
            evicted_id, _ = self.model_cache.popitem(last=False)
            print(f"[ModelManager] LRU eviction: unloading {evicted_id}")
            self.model_engine.delete_worker(evicted_id)

        worker = self.model_engine.create_worker(model_metadata)
        self.model_cache[model_id] = worker
        return worker

    def list_loaded_models(self) -> Dict[str, str]:
        return {
            model_id: worker.model_metadata.name
            for model_id, worker in self.model_cache.items()
        }

    def clear_cache(self) -> Dict[str, str]:
        """Unload all cached workers (for cold-start / LRU demos)."""
        evicted = {}
        while self.model_cache:
            model_id, worker = self.model_cache.popitem(last=False)
            evicted[model_id] = worker.model_metadata.name
            self.model_engine.delete_worker(model_id)
            print(f"[ModelManager] clear_cache: unloaded {model_id}")
        return evicted

    def set_max_models(self, max_models: int) -> None:
        if max_models < 1:
            raise ValueError("max_models must be >= 1")
        self.max_models = max_models
        # Shrink cache immediately if over capacity
        while len(self.model_cache) > self.max_models:
            evicted_id, _ = self.model_cache.popitem(last=False)
            print(f"[ModelManager] set_max_models: evicting {evicted_id}")
            self.model_engine.delete_worker(evicted_id)
