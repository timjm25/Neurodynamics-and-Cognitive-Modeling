from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

import numpy as np

from ..dynamics.attractors import Attractor, classify_trajectory


@dataclass
class EmergentBehavior:
    behavior_type: str
    description: str
    evidence: Any
    confidence: float


@dataclass
class HypothesisCandidate:
    description: str
    score: float
    evidence: Any = None
    equation: str = ""


class HypothesisDiscoveryEngine:
    def detect_emergent_behaviors(self,
                                   time_series: np.ndarray) -> List[EmergentBehavior]:
        behaviors = []
        ts = np.asarray(time_series)
        # Detect oscillations
        if ts.ndim == 1:
            fft = np.abs(np.fft.rfft(ts - ts.mean()))
            peak = np.argmax(fft[1:]) + 1
            if fft[peak] > 3 * np.mean(fft):
                behaviors.append(EmergentBehavior(
                    "oscillation", f"Dominant frequency at bin {peak}",
                    fft, float(fft[peak] / np.mean(fft)) / 10))
        # Detect synchronization (low variance of phase difference)
        if ts.ndim == 2 and ts.shape[0] >= 2:
            fc = np.corrcoef(ts)
            mean_corr = float(np.mean(np.abs(fc - np.eye(ts.shape[0]))))
            if mean_corr > 0.6:
                behaviors.append(EmergentBehavior(
                    "synchronization", f"High mean FC: {mean_corr:.3f}",
                    fc, mean_corr))
        return behaviors

    def find_attractors(self, phase_trajectories: List[np.ndarray]) -> List[Attractor]:
        return [classify_trajectory(traj) for traj in phase_trajectories]

    def symbolic_regression(self, X: np.ndarray, y: np.ndarray,
                             max_complexity: int = 10) -> str:
        """Simple symbolic regression: tries polynomial bases."""
        X = np.asarray(X).reshape(-1, 1) if np.asarray(X).ndim == 1 else np.asarray(X)
        y = np.asarray(y)
        best_eq = "y = c"
        best_err = float("inf")
        for degree in range(1, min(max_complexity, 6)):
            poly = np.column_stack([X[:, 0]**d for d in range(degree + 1)])
            coeffs, _, _, _ = np.linalg.lstsq(poly, y, rcond=None)
            pred = poly @ coeffs
            err = float(np.mean((pred - y)**2))
            if err < best_err:
                best_err = err
                terms = [f"{coeffs[d]:.3f}*x^{d}" if d > 0 else f"{coeffs[0]:.3f}"
                         for d in range(degree + 1)]
                best_eq = "y = " + " + ".join(terms)
        return best_eq

    def scan_parameter_space(self, model: Callable,
                              param_grid: Dict[str, np.ndarray],
                              metric_fn: Callable) -> Dict[str, Any]:
        results = {}
        param_name = list(param_grid.keys())[0]
        values = param_grid[param_name]
        metrics = []
        for v in values:
            try:
                out = model(**{param_name: v})
                metrics.append(metric_fn(out))
            except Exception:
                metrics.append(float("nan"))
        results[param_name] = {"values": values, "metrics": np.array(metrics)}
        return results

    def generate_hypothesis(self,
                             observations: Any) -> List[HypothesisCandidate]:
        candidates = []
        obs = np.asarray(observations, dtype=float) if not isinstance(observations, np.ndarray) else observations
        if obs.ndim == 1:
            variance = float(np.var(obs))
            mean_val = float(np.mean(obs))
            candidates.append(HypothesisCandidate(
                f"Signal has mean={mean_val:.3f}, var={variance:.3f}",
                score=1.0 / (variance + 0.01)))
            if variance > 0.1:
                candidates.append(HypothesisCandidate(
                    "High variance suggests dynamic/chaotic regime",
                    score=variance))
            else:
                candidates.append(HypothesisCandidate(
                    "Low variance suggests near-equilibrium or limit cycle",
                    score=1.0 - variance))
        return sorted(candidates, key=lambda h: h.score, reverse=True)
