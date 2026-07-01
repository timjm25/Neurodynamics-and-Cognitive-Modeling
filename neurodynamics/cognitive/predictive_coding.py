from typing import Optional

import numpy as np


class PredictiveCodingLayer:
    """Single predictive coding layer — minimises prediction error."""

    def __init__(self, dim: int = 10, lr: float = 0.1, precision: float = 1.0):
        self.dim = dim
        self.lr = lr
        self.precision = precision
        self.mu = np.zeros(dim)          # beliefs/predictions
        self.epsilon = np.zeros(dim)      # prediction errors

    def forward(self, sensory_input: np.ndarray,
                n_iter: int = 10) -> np.ndarray:
        obs = np.asarray(sensory_input, dtype=float)
        for _ in range(n_iter):
            self.epsilon = obs - self.mu
            # Gradient descent on free energy (precision-weighted PE)
            grad = -self.precision * self.epsilon
            self.mu -= self.lr * grad
        return self.epsilon.copy()

    @property
    def free_energy(self) -> float:
        return 0.5 * self.precision * float(np.dot(self.epsilon, self.epsilon))

    def reset(self) -> None:
        self.mu[:] = 0.0
        self.epsilon[:] = 0.0


class ActiveInferenceAgent:
    """Active inference agent minimising expected free energy."""

    def __init__(self, n_states: int = 4, n_actions: int = 2,
                 lr: float = 0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = lr
        self.beliefs = np.ones(n_states) / n_states  # posterior over states

    def infer_state(self, observation: np.ndarray) -> np.ndarray:
        obs = np.asarray(observation, dtype=float)
        obs = np.clip(obs, 1e-12, None)
        obs /= obs.sum()
        # Bayes update: likelihood × prior
        self.beliefs = self.beliefs * obs[:len(self.beliefs)]
        s = self.beliefs.sum()
        if s > 0:
            self.beliefs /= s
        return self.beliefs.copy()

    def _expected_free_energy(self, action: int,
                               preferences: np.ndarray) -> float:
        """Approximate EFE = epistemic value + pragmatic value."""
        eps = 1e-12
        beliefs_shifted = np.roll(self.beliefs, action)
        beliefs_shifted = np.clip(beliefs_shifted, eps, None)
        beliefs_shifted /= beliefs_shifted.sum()
        prefs = np.clip(np.asarray(preferences), eps, None)
        prefs /= prefs.sum()
        # Pragmatic: log beliefs vs preferences (KL)
        kl = float(np.sum(beliefs_shifted * np.log(beliefs_shifted / prefs)))
        # Epistemic: entropy of beliefs (info gain proxy)
        H = float(-np.sum(beliefs_shifted * np.log(beliefs_shifted + eps)))
        return kl - H

    def select_action(self, preferences: np.ndarray) -> int:
        prefs = np.asarray(preferences, dtype=float)
        if len(prefs) < self.n_states:
            prefs = np.pad(prefs, (0, self.n_states - len(prefs)), constant_values=1.0)
        G = np.array([self._expected_free_energy(a, prefs)
                      for a in range(self.n_actions)])
        return int(np.argmin(G))

    def reset(self) -> None:
        self.beliefs = np.ones(self.n_states) / self.n_states
