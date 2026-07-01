"""Attractor detection and stability analysis."""
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Attractor:
    attractor_type: str  # "fixed_point", "limit_cycle", "strange"
    center: np.ndarray
    radius: float
    lyapunov_estimate: float


def classify_trajectory(trajectory: np.ndarray,
                        transient_frac: float = 0.5) -> Attractor:
    """Classify asymptotic dynamics from a trajectory."""
    n = len(trajectory)
    start = int(n * transient_frac)
    asymptotic = trajectory[start:]

    variance = float(np.var(asymptotic))
    center = asymptotic.mean(axis=0)
    radius = float(np.std(np.linalg.norm(asymptotic - center, axis=1) if asymptotic.ndim > 1
                          else np.abs(asymptotic - center)))

    if variance < 1e-6:
        return Attractor("fixed_point", center, 0.0, -1.0)

    # Check periodicity
    if asymptotic.ndim == 1:
        sig = asymptotic - asymptotic.mean()
        std = np.std(sig)
        if std > 0:
            sig /= std
            auto = np.correlate(sig, sig, mode="full")[len(sig) - 1:]
            auto /= auto[0]
            secondary = auto[3:len(auto)//2]
            if len(secondary) > 0 and np.max(secondary) > 0.5:
                return Attractor("limit_cycle", center, radius, 0.0)

    return Attractor("strange", center, radius, 0.1)


def detect_attractors(trajectories: List[np.ndarray]) -> List[Attractor]:
    return [classify_trajectory(traj) for traj in trajectories]


def stability_eigenvalues(jacobian: np.ndarray) -> np.ndarray:
    """Eigenvalues of Jacobian at fixed point — sign determines stability."""
    return np.linalg.eigvals(jacobian)
