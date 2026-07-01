"""Topological data analysis utilities (lightweight, no external TDA deps)."""
import numpy as np


def persistent_homology_0d(point_cloud: np.ndarray,
                           max_r: float = None) -> list:
    """0D persistent homology (connected components) via single-linkage."""
    n = len(point_cloud)
    dists = np.sqrt(((point_cloud[:, None] - point_cloud[None, :])**2).sum(-1))
    if max_r is None:
        max_r = dists.max()
    # Union-Find
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
            return True
        return False
    # Sort edges by distance
    edges = [(dists[i, j], i, j) for i in range(n) for j in range(i+1, n)]
    edges.sort()
    pairs = []  # (birth, death)
    birth_times = {i: 0.0 for i in range(n)}
    for d, i, j in edges:
        if d > max_r:
            break
        if find(i) != find(j):
            ci, cj = find(i), find(j)
            # The younger component dies
            b = max(birth_times[ci], birth_times[cj])
            pairs.append((b, d))
            union(i, j)
    return pairs


def delay_embedding(time_series: np.ndarray, dim: int = 3,
                    tau: int = 1) -> np.ndarray:
    """Takens delay embedding of a 1D time series."""
    n = len(time_series)
    n_pts = n - (dim - 1) * tau
    if n_pts <= 0:
        raise ValueError("Time series too short for embedding")
    embedded = np.zeros((n_pts, dim))
    for d in range(dim):
        embedded[:, d] = time_series[d * tau: d * tau + n_pts]
    return embedded
