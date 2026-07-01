from collections import defaultdict
from typing import Dict, Optional

import numpy as np
from scipy import linalg


def graph_laplacian(W: np.ndarray) -> np.ndarray:
    D = np.diag(W.sum(axis=1))
    return D - W


def algebraic_connectivity(W: np.ndarray) -> float:
    L = graph_laplacian(W)
    eigvals = np.sort(np.real(linalg.eigvalsh(L)))
    return float(eigvals[1]) if len(eigvals) > 1 else 0.0


def spectral_clustering(W: np.ndarray, k: int = 2) -> np.ndarray:
    from scipy.linalg import eigh
    L = graph_laplacian(W)
    n = W.shape[0]
    _, vecs = eigh(L, subset_by_index=[0, min(k - 1, n - 1)])
    # K-means on eigenvectors
    return _kmeans(vecs, k)


def _kmeans(X: np.ndarray, k: int, max_iter: int = 100,
            seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    centers = X[rng.choice(n, k, replace=False)]
    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        dists = np.array([np.linalg.norm(X - c, axis=1) for c in centers])
        new_labels = np.argmin(dists, axis=0)
        if np.all(new_labels == labels):
            break
        labels = new_labels
        for i in range(k):
            mask = labels == i
            if mask.any():
                centers[i] = X[mask].mean(axis=0)
    return labels


def community_detection_louvain(W: np.ndarray) -> np.ndarray:
    """Greedy modularity optimization (simplified Louvain)."""
    n = W.shape[0]
    labels = np.arange(n)
    m = W.sum() / 2
    if m == 0:
        return labels

    def _modularity_gain(node, community, W, labels, m):
        k_i = W[node].sum()
        k_ic = W[node][labels == community].sum()
        sigma_tot = W[labels == community].sum()
        return k_ic / m - (sigma_tot * k_i) / (2 * m**2)

    improved = True
    while improved:
        improved = False
        for i in range(n):
            best_comm = labels[i]
            best_gain = 0.0
            current_comm = labels[i]
            neighbors = np.where(W[i] > 0)[0]
            candidate_comms = set(labels[neighbors])
            candidate_comms.discard(current_comm)
            for comm in candidate_comms:
                gain = _modularity_gain(i, comm, W, labels, m)
                gain -= _modularity_gain(i, current_comm, W, labels, m)
                if gain > best_gain:
                    best_gain = gain
                    best_comm = comm
            if best_comm != current_comm:
                labels[i] = best_comm
                improved = True
    # Remap to consecutive integers
    unique = np.unique(labels)
    remap = {v: i for i, v in enumerate(unique)}
    return np.array([remap[l] for l in labels])


def functional_connectivity(time_series: np.ndarray) -> np.ndarray:
    """Pearson correlation matrix of time series (n_nodes x n_timepoints)."""
    return np.corrcoef(time_series)


def effective_connectivity(time_series: np.ndarray,
                           method: str = "granger",
                           lag: int = 1) -> np.ndarray:
    """Simple pairwise Granger causality estimate (F-statistic proxy)."""
    n, T = time_series.shape
    EC = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            y = time_series[i, lag:]
            x1 = time_series[i, :-lag]  # restricted (AR of target)
            x2 = time_series[j, :-lag]  # unrestricted (adds source)
            X_r = np.column_stack([np.ones(len(x1)), x1])
            X_u = np.column_stack([np.ones(len(x1)), x1, x2])
            try:
                b_r = np.linalg.lstsq(X_r, y, rcond=None)[0]
                b_u = np.linalg.lstsq(X_u, y, rcond=None)[0]
                RSS_r = np.sum((y - X_r @ b_r)**2)
                RSS_u = np.sum((y - X_u @ b_u)**2)
                EC[j, i] = max(0, (RSS_r - RSS_u) / (RSS_u + 1e-10))
            except np.linalg.LinAlgError:
                EC[j, i] = 0.0
    return EC


def clustering_coefficient(W: np.ndarray) -> np.ndarray:
    n = W.shape[0]
    A = (W > 0).astype(float)
    np.fill_diagonal(A, 0)
    cc = np.zeros(n)
    for i in range(n):
        neighbors = np.where(A[i] > 0)[0]
        k = len(neighbors)
        if k < 2:
            cc[i] = 0.0
        else:
            sub = A[np.ix_(neighbors, neighbors)]
            cc[i] = sub.sum() / (k * (k - 1))
    return cc


def path_length(W: np.ndarray) -> float:
    """Average shortest path length (BFS-based, unweighted)."""
    n = W.shape[0]
    A = (W > 0).astype(int)
    total = 0
    count = 0
    for src in range(n):
        dist = [-1] * n
        dist[src] = 0
        queue = [src]
        idx = 0
        while idx < len(queue):
            u = queue[idx]; idx += 1
            for v in np.where(A[u] > 0)[0]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        for d in dist:
            if d > 0:
                total += d
                count += 1
    return total / count if count > 0 else float("inf")


def small_world_coefficient(W: np.ndarray, n_rand: int = 5,
                             seed: int = 0) -> float:
    """Small-world sigma = (C/C_rand) / (L/L_rand)."""
    rng = np.random.default_rng(seed)
    n = W.shape[0]
    C_real = float(clustering_coefficient(W).mean())
    L_real = path_length(W)
    C_rands, L_rands = [], []
    k = int(W.sum() / n)
    for _ in range(n_rand):
        p = W.sum() / (n * (n - 1))
        W_rand = (rng.random((n, n)) < p).astype(float)
        W_rand = np.triu(W_rand, 1)
        W_rand = W_rand + W_rand.T
        np.fill_diagonal(W_rand, 0)
        C_rands.append(clustering_coefficient(W_rand).mean())
        L_rands.append(path_length(W_rand))
    C_rand = float(np.mean(C_rands)) if C_rands else 1.0
    L_rand = float(np.mean(L_rands)) if L_rands else 1.0
    if C_rand == 0 or L_rand == 0:
        return 0.0
    return (C_real / C_rand) / (L_real / L_rand)
