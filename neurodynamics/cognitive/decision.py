from typing import Callable, List, Optional, Tuple

import numpy as np


class DriftDiffusionModel:
    """Drift Diffusion Model for two-alternative forced choice."""

    def __init__(self, drift: float = 1.0, noise: float = 1.0,
                 threshold: float = 1.0, bias: float = 0.0,
                 dt: float = 0.001, max_time: float = 5.0,
                 seed: int = 42):
        self.drift = drift
        self.noise = noise
        self.threshold = threshold
        self.bias = bias
        self.dt = dt
        self.max_time = max_time
        self.rng = np.random.default_rng(seed)

    def decide(self, stimulus_strength: float = None) -> Tuple[int, float]:
        v = stimulus_strength if stimulus_strength is not None else self.drift
        x = self.bias
        t = 0.0
        while t < self.max_time:
            x += v * self.dt + self.noise * np.sqrt(self.dt) * self.rng.standard_normal()
            t += self.dt
            if x >= self.threshold:
                return 1, t
            if x <= -self.threshold:
                return 0, t
        return (1 if x > 0 else 0), t

    def psychometric_curve(self, drift_values: np.ndarray,
                           n_trials: int = 100) -> np.ndarray:
        p_correct = []
        for v in drift_values:
            choices = [self.decide(v)[0] for _ in range(n_trials)]
            p_correct.append(np.mean(choices))
        return np.array(p_correct)


class BayesianDecisionMaker:
    """Bayesian decision maker with prior and likelihood update."""

    def __init__(self, n_hypotheses: int = 2,
                 prior: Optional[np.ndarray] = None):
        self.n = n_hypotheses
        if prior is None:
            self.posterior = np.ones(n_hypotheses) / n_hypotheses
        else:
            self.posterior = np.array(prior, dtype=float)
            self.posterior /= self.posterior.sum()

    def update(self, evidence: np.ndarray) -> np.ndarray:
        likelihood = np.asarray(evidence, dtype=float)
        likelihood = np.clip(likelihood, 1e-12, None)
        self.posterior = self.posterior * likelihood
        self.posterior /= self.posterior.sum()
        return self.posterior.copy()

    def decide(self) -> int:
        return int(np.argmax(self.posterior))

    def reset(self, prior: Optional[np.ndarray] = None) -> None:
        if prior is None:
            self.posterior = np.ones(self.n) / self.n
        else:
            self.posterior = np.array(prior, dtype=float)
            self.posterior /= self.posterior.sum()


class RaceModel:
    """Race model — multiple accumulators, first to threshold wins."""

    def __init__(self, n_alternatives: int = 2, threshold: float = 1.0,
                 noise: float = 0.1, dt: float = 0.001,
                 max_time: float = 5.0, seed: int = 42):
        self.n = n_alternatives
        self.threshold = threshold
        self.noise = noise
        self.dt = dt
        self.max_time = max_time
        self.rng = np.random.default_rng(seed)

    def compete(self, inputs: np.ndarray) -> Tuple[int, float]:
        assert len(inputs) == self.n
        x = np.zeros(self.n)
        t = 0.0
        while t < self.max_time:
            dx = inputs * self.dt + self.noise * np.sqrt(self.dt) * self.rng.standard_normal(self.n)
            x = np.maximum(0, x + dx)
            t += self.dt
            winners = np.where(x >= self.threshold)[0]
            if len(winners) > 0:
                return int(winners[0]), t
        return int(np.argmax(x)), t
