from typing import Any, List, Optional

import numpy as np


class WorkingMemoryModel:
    """Miller's 7±2 working memory with decay."""

    def __init__(self, capacity: int = 7, decay_rate: float = 0.05):
        self.capacity = capacity
        self.decay_rate = decay_rate
        self._items: List[Any] = []
        self._strengths: List[float] = []

    def encode(self, item: Any) -> bool:
        if len(self._items) >= self.capacity:
            return False
        self._items.append(item)
        self._strengths.append(1.0)
        return True

    def retrieve(self, cue: Any = None) -> Optional[Any]:
        if not self._items:
            return None
        if cue is None:
            return self._items[-1]
        try:
            idx = self._items.index(cue)
            return self._items[idx] if self._strengths[idx] > 0.1 else None
        except ValueError:
            return None

    def forget(self, dt: float = 1.0) -> None:
        self._strengths = [max(0.0, s - self.decay_rate * dt)
                           for s in self._strengths]
        # Remove forgotten items
        alive = [(item, s) for item, s in zip(self._items, self._strengths) if s > 0.0]
        self._items = [a[0] for a in alive]
        self._strengths = [a[1] for a in alive]

    @property
    def load(self) -> int:
        return len(self._items)

    def clear(self) -> None:
        self._items.clear()
        self._strengths.clear()


class ContinuousAttractorWM:
    """1D ring attractor network for spatial working memory."""

    def __init__(self, n: int = 100, J_plus: float = 1.8, J_minus: float = 1.2,
                 sigma: float = 0.1, tau: float = 10.0, I_0: float = 0.3):
        self.n = n
        self.J_plus = J_plus
        self.J_minus = J_minus
        self.sigma = sigma
        self.tau = tau
        self.I_0 = I_0
        self.phi = np.linspace(0, 2 * np.pi, n, endpoint=False)
        self.r = np.zeros(n)
        # Build connectivity kernel
        diff = self.phi[:, None] - self.phi[None, :]
        self.W = J_plus * np.exp(-0.5 * (diff / sigma)**2) - J_minus
        self.W /= n

    def _f(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)**2

    def set_memory(self, position: float, amplitude: float = 1.0) -> None:
        diff = self.phi - position
        self.r = amplitude * np.exp(-2 * (1 - np.cos(diff)))

    def drift(self, noise: float = 0.01, dt: float = 1.0,
              seed: Optional[int] = None) -> None:
        rng = np.random.default_rng(seed)
        I = self.W @ self.f_activity + self.I_0 + noise * rng.normal(size=self.n)
        dr = (-self.r + self._f(I)) / self.tau
        self.r = np.maximum(0, self.r + dt * dr)

    @property
    def f_activity(self) -> np.ndarray:
        return self._f(self.r)

    def read_out(self) -> float:
        """Population vector decoding → estimated position."""
        activity = self.f_activity
        if activity.sum() < 1e-10:
            return 0.0
        cos_comp = float(np.dot(activity, np.cos(self.phi)))
        sin_comp = float(np.dot(activity, np.sin(self.phi)))
        return float(np.arctan2(sin_comp, cos_comp)) % (2 * np.pi)
