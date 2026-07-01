"""Whole-brain connectome utilities."""
import numpy as np


REGION_NAMES = [
    "PFC", "ACC", "M1", "S1", "V1", "V2", "Hippocampus", "Amygdala",
    "BG_Striatum", "Thalamus", "Cerebellum", "Brainstem",
]

N_REGIONS = len(REGION_NAMES)


def synthetic_connectome(seed: int = 0) -> np.ndarray:
    """Generate a synthetic whole-brain connectivity matrix."""
    rng = np.random.default_rng(seed)
    n = N_REGIONS
    W = rng.random((n, n)) * 0.3
    # Strengthen key pathways
    pairs = [(0, 1), (0, 6), (0, 7), (6, 7), (4, 5), (2, 9), (8, 9), (9, 6)]
    for i, j in pairs:
        W[i, j] = W[j, i] = rng.uniform(0.6, 0.9)
    np.fill_diagonal(W, 0)
    return W


def connectome_summary(W: np.ndarray) -> dict:
    from .analysis import clustering_coefficient, path_length
    return {
        "n_regions": W.shape[0],
        "density": float((W > 0).sum() / (W.shape[0]**2 - W.shape[0])),
        "mean_strength": float(W.mean()),
        "mean_clustering": float(clustering_coefficient((W > 0.1).astype(float)).mean()),
    }
