from typing import List, Optional

import numpy as np


class HebbianRule:
    def update(self, W: np.ndarray, pre: np.ndarray,
               post: np.ndarray, lr: float = 0.01) -> np.ndarray:
        return W + lr * np.outer(post, pre)


class BCMRule:
    """Bienenstock-Cooper-Munro rule with sliding modification threshold."""

    def __init__(self, tau_theta: float = 1000.0):
        self.tau_theta = tau_theta
        self.theta = 0.0

    def update(self, W: np.ndarray, pre: np.ndarray,
               post: np.ndarray, lr: float = 0.01,
               dt: float = 1.0) -> np.ndarray:
        phi = post * (post - self.theta)
        dW = lr * np.outer(phi, pre)
        # Update sliding threshold
        self.theta += dt * (post.mean()**2 - self.theta) / self.tau_theta
        return W + dW


class STDPRule:
    """Spike-Timing Dependent Plasticity."""

    def update(self, W: np.ndarray, pre_spikes: np.ndarray,
               post_spikes: np.ndarray, dt: float = 1.0,
               A_plus: float = 0.01, A_minus: float = 0.012,
               tau_plus: float = 20.0, tau_minus: float = 20.0) -> np.ndarray:
        dW = np.zeros_like(W)
        for t_post in post_spikes:
            for t_pre in pre_spikes:
                delta_t = t_post - t_pre
                if delta_t > 0:
                    dW += A_plus * np.exp(-delta_t / tau_plus)
                elif delta_t < 0:
                    dW -= A_minus * np.exp(delta_t / tau_minus)
        return W + dW


class OjaRule:
    """Oja's learning rule — normalized Hebbian (extracts first PC)."""

    def update(self, W: np.ndarray, pre: np.ndarray,
               post: np.ndarray, lr: float = 0.01) -> np.ndarray:
        dW = lr * (np.outer(post, pre) - np.outer(post**2, W))
        return W + dW


class HopfieldNetwork:
    """Binary Hopfield associative memory."""

    def __init__(self, n: int):
        self.n = n
        self.W = np.zeros((n, n))

    def store(self, patterns: List[np.ndarray]) -> None:
        self.W = np.zeros((self.n, self.n))
        for p in patterns:
            p = np.asarray(p, dtype=float)
            p = np.where(p > 0, 1.0, -1.0)
            self.W += np.outer(p, p)
        self.W /= self.n
        np.fill_diagonal(self.W, 0)

    def recall(self, partial_pattern: np.ndarray,
               max_iter: int = 20) -> np.ndarray:
        state = np.where(np.asarray(partial_pattern) > 0, 1.0, -1.0).copy()
        for _ in range(max_iter):
            new_state = np.sign(self.W @ state)
            new_state[new_state == 0] = 1.0
            if np.allclose(new_state, state):
                break
            state = new_state
        return state

    @property
    def capacity(self) -> float:
        return 0.138 * self.n
