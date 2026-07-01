from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class NeuronModel(ABC):
    name: str = "base_neuron"
    description: str = "Abstract neuron model"

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        self.parameters = parameters or {}

    def validate(self) -> bool:
        return True

    @abstractmethod
    def step(self, inputs: Any, state: Any) -> Any:
        pass

    @abstractmethod
    def simulate(self, duration: float, inputs: Any) -> Any:
        pass

    def run(self, **kwargs) -> Any:
        return self.simulate(**kwargs)


class NetworkModel(ABC):
    name: str = "base_network"
    description: str = "Abstract network model"

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        self.parameters = parameters or {}

    def validate(self) -> bool:
        return True

    @abstractmethod
    def run(self, **kwargs) -> Any:
        pass


class CognitiveModule(ABC):
    name: str = "base_cognitive"
    description: str = "Abstract cognitive module"

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        self.parameters = parameters or {}

    def validate(self) -> bool:
        return True

    @abstractmethod
    def process(self, input_data: Any) -> Any:
        pass

    def run(self, **kwargs) -> Any:
        return self.process(**kwargs)


class AnalysisModule(ABC):
    name: str = "base_analysis"
    description: str = "Abstract analysis module"

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        self.parameters = parameters or {}

    def validate(self) -> bool:
        return True

    @abstractmethod
    def analyze(self, data: Any) -> Any:
        pass

    def run(self, **kwargs) -> Any:
        return self.analyze(**kwargs)


class VisualizationModule(ABC):
    name: str = "base_visualization"
    description: str = "Abstract visualization module"

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        self.parameters = parameters or {}

    def validate(self) -> bool:
        return True

    @abstractmethod
    def visualize(self, data: Any) -> Any:
        pass

    def run(self, **kwargs) -> Any:
        return self.visualize(**kwargs)


class Plugin(ABC):
    name: str = "base_plugin"
    description: str = "Abstract plugin"
    category: str = "general"

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        self.parameters = parameters or {}

    @abstractmethod
    def validate(self) -> bool:
        pass

    @abstractmethod
    def run(self, **kwargs) -> Any:
        pass


class Experiment(ABC):
    name: str = "base_experiment"
    description: str = "Abstract experiment"

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        self.parameters = parameters or {}

    def validate(self) -> bool:
        return True

    @abstractmethod
    def reset(self) -> Any:
        pass

    @abstractmethod
    def step(self, action: Any) -> Any:
        pass

    def run(self, **kwargs) -> Any:
        return self.step(**kwargs)
