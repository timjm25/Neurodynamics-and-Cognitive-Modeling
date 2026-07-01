from typing import Any, Dict, List, Optional, Type


class PluginRegistry:
    _instance: Optional["PluginRegistry"] = None

    def __new__(cls) -> "PluginRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._registry: Dict[str, Dict[str, Any]] = {}
        return cls._instance

    def register(self, name: str, cls: Type, category: str = "general") -> None:
        self._registry[name] = {"class": cls, "category": category}

    def get(self, name: str) -> Optional[Type]:
        entry = self._registry.get(name)
        return entry["class"] if entry else None

    def list_all(self) -> List[str]:
        return list(self._registry.keys())

    def list_by_category(self, category: str) -> List[str]:
        return [n for n, v in self._registry.items() if v["category"] == category]

    def clear(self) -> None:
        self._registry.clear()
