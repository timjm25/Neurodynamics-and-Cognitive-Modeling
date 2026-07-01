"""Plugin SDK — how to extend the Neurodynamics platform."""
import importlib.util
import sys
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


class PluginBase(ABC):
    name: str = "base_plugin"
    version: str = "0.1.0"
    category: str = "general"
    description: str = ""

    @abstractmethod
    def validate(self) -> bool:
        pass

    @abstractmethod
    def run(self, **kwargs) -> Any:
        pass

    def metadata(self) -> Dict[str, str]:
        return {
            "name": self.name,
            "version": self.version,
            "category": self.category,
            "description": self.description,
        }


class NeuronModelPlugin(PluginBase):
    category = "neuron_model"

    @abstractmethod
    def step(self, inputs: Any, state: Dict) -> Dict:
        pass

    @abstractmethod
    def simulate(self, duration: float, inputs: Any) -> Any:
        pass

    def validate(self) -> bool:
        return callable(getattr(self, "step", None)) and callable(getattr(self, "simulate", None))

    def run(self, **kwargs) -> Any:
        return self.simulate(**kwargs)


class CognitiveModulePlugin(PluginBase):
    category = "cognitive_module"

    @abstractmethod
    def process(self, input_data: Any) -> Any:
        pass

    def validate(self) -> bool:
        return callable(getattr(self, "process", None))

    def run(self, **kwargs) -> Any:
        return self.process(**kwargs)


class AnalysisPlugin(PluginBase):
    category = "analysis"

    @abstractmethod
    def analyze(self, data: Any) -> Any:
        pass

    def validate(self) -> bool:
        return callable(getattr(self, "analyze", None))

    def run(self, **kwargs) -> Any:
        return self.analyze(**kwargs)


def load_plugin(path: str) -> PluginBase:
    """Load a plugin from a Python file path."""
    spec = importlib.util.spec_from_file_location("plugin_module", path)
    if spec is None:
        raise ImportError(f"Cannot load plugin from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["plugin_module"] = module
    spec.loader.exec_module(module)
    # Find PluginBase subclass
    for name in dir(module):
        obj = getattr(module, name)
        try:
            if (isinstance(obj, type) and issubclass(obj, PluginBase)
                    and obj is not PluginBase):
                return obj()
        except TypeError:
            pass
    raise ValueError(f"No PluginBase subclass found in {path}")


def validate_plugin(plugin: PluginBase) -> Tuple[bool, List[str]]:
    errors = []
    if not hasattr(plugin, "name") or not plugin.name:
        errors.append("Plugin must have a non-empty 'name'")
    if not hasattr(plugin, "category"):
        errors.append("Plugin must have a 'category'")
    try:
        valid = plugin.validate()
        if not valid:
            errors.append("Plugin.validate() returned False")
    except Exception as e:
        errors.append(f"Plugin.validate() raised: {e}")
    return len(errors) == 0, errors
