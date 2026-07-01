"""Cognitive architectures: Global Workspace, Hierarchical PP, ACT-R-like."""
from typing import Any, Dict, List, Optional

import numpy as np


class GlobalWorkspace:
    """Global Workspace Theory: specialized modules + broadcast workspace."""

    def __init__(self, n_modules: int = 6):
        self.n_modules = n_modules
        self.workspace: Optional[np.ndarray] = None
        self._modules: Dict[str, Any] = {}
        self._broadcasts: List[np.ndarray] = []

    def add_module(self, name: str, module: Any) -> None:
        self._modules[name] = module

    def compete_for_access(self, activations: Dict[str, np.ndarray]) -> Optional[str]:
        if not activations:
            return None
        winner = max(activations, key=lambda k: np.max(activations[k]))
        self.workspace = activations[winner].copy()
        self._broadcasts.append(self.workspace.copy())
        return winner

    def broadcast(self) -> Optional[np.ndarray]:
        return self.workspace.copy() if self.workspace is not None else None


class HierarchicalPredictiveProcessing:
    """Multi-level predictive processing stack."""

    def __init__(self, n_levels: int = 4, dim: int = 8):
        from .predictive_coding import PredictiveCodingLayer
        self.levels = [PredictiveCodingLayer(dim=dim) for _ in range(n_levels)]
        self.n_levels = n_levels

    def process(self, sensory_input: np.ndarray) -> List[np.ndarray]:
        errors = []
        signal = sensory_input.copy()
        for level in self.levels:
            pe = level.forward(signal)
            errors.append(pe)
            # Prediction error propagates up as next level's input
            signal = np.abs(pe)
        return errors

    @property
    def total_free_energy(self) -> float:
        return sum(l.free_energy for l in self.levels)


class ACTRLikeArchitecture:
    """Simplified ACT-R-inspired cognitive architecture."""

    def __init__(self):
        self.declarative_memory: Dict[str, float] = {}
        self.procedural_memory: List[Dict] = []
        self.goal_stack: List[str] = []
        self.activation_noise: float = 0.1

    def add_chunk(self, name: str, base_activation: float = 1.0) -> None:
        self.declarative_memory[name] = base_activation

    def retrieve(self, name: str, noise: float = None) -> Optional[str]:
        sigma = noise if noise is not None else self.activation_noise
        if name not in self.declarative_memory:
            return None
        activation = self.declarative_memory[name] + np.random.normal(0, sigma)
        return name if activation > 0 else None

    def set_goal(self, goal: str) -> None:
        self.goal_stack.append(goal)

    def pop_goal(self) -> Optional[str]:
        return self.goal_stack.pop() if self.goal_stack else None

    def add_production(self, condition: str, action: str) -> None:
        self.procedural_memory.append({"condition": condition, "action": action})

    def fire_production(self, state: str) -> Optional[str]:
        for prod in self.procedural_memory:
            if prod["condition"] == state:
                return prod["action"]
        return None
