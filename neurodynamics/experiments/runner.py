"""Experiment runner and recorder."""
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class TrialRecord:
    trial: int
    action: Any
    reward: float
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ExperimentRecord:
    experiment_name: str
    trials: List[TrialRecord] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({
            "experiment_name": self.experiment_name,
            "n_trials": len(self.trials),
            "trials": [asdict(t) for t in self.trials],
            "metadata": self.metadata,
        }, default=str)

    def accuracy(self) -> float:
        rewards = [t.reward for t in self.trials]
        if not rewards:
            return 0.0
        return sum(1 for r in rewards if r > 0) / len(rewards)


class ExperimentRunner:
    def run(self, task: Any, policy=None, max_steps: int = 500) -> ExperimentRecord:
        import numpy as np
        record = ExperimentRecord(experiment_name=type(task).__name__)
        obs = task.reset()
        step = 0
        done = False
        while not done and step < max_steps:
            if policy is not None:
                action = policy(obs)
            else:
                # Random policy
                action = int(np.random.randint(0, 2))
            result = task.step(action)
            if len(result) == 4:
                obs, reward, done, info = result
            else:
                obs, reward, done = result[:3]
                info = {}
            record.trials.append(TrialRecord(step, action, reward, done, info))
            step += 1
        return record
