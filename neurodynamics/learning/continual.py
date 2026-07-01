"""Continual and meta-learning utilities."""
import numpy as np


class EWCRegularizer:
    """Elastic Weight Consolidation for continual learning."""

    def __init__(self, lambda_: float = 100.0):
        self.lambda_ = lambda_
        self.theta_star: np.ndarray = None
        self.fisher: np.ndarray = None

    def compute_fisher(self, gradients: list) -> None:
        self.fisher = np.mean([g**2 for g in gradients], axis=0)

    def set_reference(self, params: np.ndarray) -> None:
        self.theta_star = params.copy()

    def penalty(self, params: np.ndarray) -> float:
        if self.theta_star is None or self.fisher is None:
            return 0.0
        return 0.5 * self.lambda_ * float(
            np.sum(self.fisher * (params - self.theta_star)**2))


class MAMLAgent:
    """Simplified Model-Agnostic Meta-Learning."""

    def __init__(self, n_params: int, lr_inner: float = 0.01,
                 lr_outer: float = 0.001, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.theta = self.rng.normal(0, 0.1, n_params)
        self.lr_inner = lr_inner
        self.lr_outer = lr_outer

    def inner_update(self, grad: np.ndarray) -> np.ndarray:
        return self.theta - self.lr_inner * grad

    def outer_update(self, meta_grad: np.ndarray) -> None:
        self.theta -= self.lr_outer * meta_grad
