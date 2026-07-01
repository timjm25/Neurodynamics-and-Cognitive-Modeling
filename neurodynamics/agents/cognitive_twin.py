from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .brain_regions import BrainNetwork


@dataclass
class CognitiveTwin:
    memory_capacity: int = 7
    attention_span: float = 1.0
    learning_rate: float = 0.1
    decision_threshold: float = 1.0
    plasticity: float = 1.0
    age_years: float = 30.0
    pathology: Optional[str] = None
    brain_network: Optional[BrainNetwork] = field(default=None, repr=False)


@dataclass
class TaskResult:
    accuracy: float
    mean_rt: float
    n_trials: int
    details: Dict[str, Any] = field(default_factory=dict)


class CognitiveTwinSimulator:
    def create(self, params: Dict[str, Any] = None) -> CognitiveTwin:
        p = params or {}
        twin = CognitiveTwin(
            memory_capacity=p.get("memory_capacity", 7),
            attention_span=p.get("attention_span", 1.0),
            learning_rate=p.get("learning_rate", 0.1),
            decision_threshold=p.get("decision_threshold", 1.0),
            plasticity=p.get("plasticity", 1.0),
            age_years=p.get("age_years", 30.0),
            pathology=p.get("pathology", None),
        )
        twin.brain_network = BrainNetwork()
        # Configure based on params
        pfc = twin.brain_network.regions["PFC"]
        pfc.wm.capacity = twin.memory_capacity
        pfc.ddm.threshold = twin.decision_threshold
        return twin

    def run_task(self, twin: CognitiveTwin, task: Any) -> TaskResult:
        obs = task.reset()
        n_correct = 0
        rts = []
        n_trials = 0
        done = False
        while not done:
            n_trials += 1
            stim = float(np.mean(obs)) if hasattr(obs, '__len__') else float(obs)
            pfc = twin.brain_network.regions["PFC"]
            choice, rt = pfc.decide(stim * twin.attention_span)
            result = task.step(choice)
            if len(result) == 4:
                obs, reward, done, info = result
            else:
                obs, reward, done = result[:3]
                info = {}
            if reward > 0:
                n_correct += 1
            rts.append(rt)
            if n_trials > 200:
                break
        accuracy = n_correct / max(n_trials, 1)
        return TaskResult(accuracy=accuracy, mean_rt=float(np.mean(rts)),
                          n_trials=n_trials)

    def apply_lesion(self, twin: CognitiveTwin, region: str,
                     severity: float) -> None:
        twin.brain_network.apply_lesion(region, severity)

    def simulate_aging(self, twin: CognitiveTwin, years: float) -> None:
        twin.age_years += years
        decay = years * 0.005
        twin.plasticity = max(0.1, twin.plasticity - decay)
        twin.memory_capacity = max(3, int(twin.memory_capacity - years * 0.02))
        if twin.brain_network:
            twin.brain_network.regions["PFC"].wm.capacity = twin.memory_capacity

    def simulate_drug(self, twin: CognitiveTwin, drug_name: str,
                      effect_map: Dict[str, float]) -> None:
        if twin.brain_network:
            bg = twin.brain_network.regions.get("BasalGanglia")
            if bg and "dopamine" in effect_map:
                bg.dopamine = effect_map["dopamine"]
            pfc = twin.brain_network.regions.get("PFC")
            if pfc and "control_gain" in effect_map:
                pfc.control_gain = effect_map["control_gain"]
        for attr, val in effect_map.items():
            if hasattr(twin, attr):
                setattr(twin, attr, val)
