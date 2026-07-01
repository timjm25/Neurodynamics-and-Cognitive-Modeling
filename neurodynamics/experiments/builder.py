"""Experiment builder — compose tasks into experiment pipelines."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ExperimentConfig:
    name: str
    task_class: str
    task_params: Dict[str, Any] = field(default_factory=dict)
    n_subjects: int = 1
    n_repetitions: int = 1
    record_fields: List[str] = field(default_factory=lambda: ["accuracy", "rt"])


class ExperimentBuilder:
    def __init__(self):
        self._configs: List[ExperimentConfig] = []

    def add_task(self, config: ExperimentConfig) -> "ExperimentBuilder":
        self._configs.append(config)
        return self

    def build(self) -> List[ExperimentConfig]:
        return self._configs.copy()

    def reset(self) -> None:
        self._configs.clear()
