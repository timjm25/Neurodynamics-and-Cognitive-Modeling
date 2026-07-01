"""Dataset registry and caching."""
from pathlib import Path
from typing import Any, Callable, Dict, Optional


class DatasetManager:
    def __init__(self, cache_dir: str = "./data/cache"):
        self.cache_dir = Path(cache_dir)
        self._registry: Dict[str, Callable] = {}
        self._cache: Dict[str, Any] = {}

    def register(self, name: str, loader_fn: Callable) -> None:
        self._registry[name] = loader_fn

    def load(self, name: str, **kwargs) -> Any:
        cache_key = f"{name}_{hash(str(kwargs))}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        if name not in self._registry:
            raise KeyError(f"Dataset '{name}' not registered")
        data = self._registry[name](**kwargs)
        self._cache[cache_key] = data
        return data

    def list_datasets(self):
        return list(self._registry.keys())

    def clear_cache(self) -> None:
        self._cache.clear()
