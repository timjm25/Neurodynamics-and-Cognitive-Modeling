"""Time-varying graph support."""
from typing import List

import numpy as np


class TimeVaryingGraph:
    def __init__(self, n_nodes: int, n_timepoints: int):
        self.n_nodes = n_nodes
        self.n_timepoints = n_timepoints
        self.adjacencies: List[np.ndarray] = [
            np.zeros((n_nodes, n_nodes)) for _ in range(n_timepoints)
        ]

    def set_adjacency(self, t: int, W: np.ndarray) -> None:
        self.adjacencies[t] = W.copy()

    def get_adjacency(self, t: int) -> np.ndarray:
        return self.adjacencies[t]

    def sliding_window_fc(self, time_series: np.ndarray,
                          window: int = 50, step: int = 10) -> "TimeVaryingGraph":
        n, T = time_series.shape
        n_windows = (T - window) // step + 1
        tvg = TimeVaryingGraph(n, n_windows)
        for i in range(n_windows):
            start = i * step
            segment = time_series[:, start:start + window]
            tvg.adjacencies[i] = np.corrcoef(segment)
        return tvg

    def mean_adjacency(self) -> np.ndarray:
        return np.mean(self.adjacencies, axis=0)

    def variability(self) -> np.ndarray:
        return np.std(self.adjacencies, axis=0)
