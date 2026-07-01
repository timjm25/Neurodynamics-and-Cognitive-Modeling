from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np


@dataclass
class BifurcationDiagram:
    param_values: np.ndarray
    asymptotic_states: List[np.ndarray]
    detected_types: List[str]  # "equilibrium", "limit_cycle", "chaos"


class BifurcationScanner:
    """Scan a parameter range and classify asymptotic dynamics."""

    def scan(self, model_fn: Callable, param_name: str,
             param_range: np.ndarray, initial_state: np.ndarray,
             dt: float = 0.1, transient: int = 5000,
             record: int = 500) -> BifurcationDiagram:
        states = []
        types = []
        for param_val in param_range:
            kwargs = {param_name: param_val}
            state = initial_state.copy()
            for _ in range(transient):
                state = model_fn(state, dt=dt, **kwargs)
            rec = []
            for _ in range(record):
                state = model_fn(state, dt=dt, **kwargs)
                rec.append(state[0] if hasattr(state, '__len__') else float(state))
            rec = np.array(rec)
            variance = np.var(rec)
            if variance < 1e-4:
                dtype = "equilibrium"
            elif self._is_periodic(rec):
                dtype = "limit_cycle"
            else:
                dtype = "chaos"
            states.append(rec)
            types.append(dtype)
        return BifurcationDiagram(
            param_values=param_range,
            asymptotic_states=states,
            detected_types=types,
        )

    @staticmethod
    def _is_periodic(signal: np.ndarray, threshold: float = 0.7) -> bool:
        """Check for periodicity via autocorrelation peak."""
        n = len(signal)
        if n < 20:
            return False
        sig = signal - signal.mean()
        std = np.std(sig)
        if std < 1e-10:
            return False
        sig /= std
        lags = min(n // 2, 200)
        auto = np.correlate(sig, sig, mode="full")[n - 1:]
        auto = auto[:lags] / auto[0]
        # Find secondary peak after initial drop
        if lags < 5:
            return False
        after_drop = auto[3:]
        if len(after_drop) == 0:
            return False
        peak = np.max(after_drop)
        return bool(peak > threshold)
