import numpy as np


def build_small_world(n: int, k: int = 4, p: float = 0.1,
                      seed: int = 0) -> np.ndarray:
    """Watts-Strogatz small-world network."""
    rng = np.random.default_rng(seed)
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(1, k // 2 + 1):
            W[i, (i + j) % n] = 1
            W[(i + j) % n, i] = 1
    # Rewire
    for i in range(n):
        for j in range(1, k // 2 + 1):
            if rng.random() < p:
                target = (i + j) % n
                W[i, target] = 0
                W[target, i] = 0
                new_target = rng.integers(0, n)
                while new_target == i or W[i, new_target]:
                    new_target = rng.integers(0, n)
                W[i, new_target] = 1
                W[new_target, i] = 1
    return W


def build_scale_free(n: int, m: int = 2, seed: int = 0) -> np.ndarray:
    """Barabasi-Albert scale-free network via preferential attachment."""
    rng = np.random.default_rng(seed)
    W = np.zeros((n, n))
    # Start with m+1 fully connected nodes
    for i in range(min(m + 1, n)):
        for j in range(i + 1, min(m + 1, n)):
            W[i, j] = 1
            W[j, i] = 1
    degrees = W.sum(axis=1)
    for i in range(m + 1, n):
        degree_sum = degrees.sum()
        if degree_sum == 0:
            probs = np.ones(i) / i
        else:
            probs = degrees[:i] / degree_sum
        targets = rng.choice(i, size=min(m, i), replace=False, p=probs)
        for t in targets:
            W[i, t] = 1
            W[t, i] = 1
            degrees[i] += 1
            degrees[t] += 1
    return W


def build_random(n: int, p: float = 0.1, seed: int = 0) -> np.ndarray:
    """Erdos-Renyi random graph."""
    rng = np.random.default_rng(seed)
    W = (rng.random((n, n)) < p).astype(float)
    W = np.triu(W, 1)
    W = W + W.T
    np.fill_diagonal(W, 0)
    return W


def build_hierarchical(levels: int = 3, branching_factor: int = 3) -> np.ndarray:
    """Hierarchical network with intra-level connections."""
    n = branching_factor ** levels
    W = np.zeros((n, n))
    for level in range(levels):
        group_size = branching_factor ** (levels - level)
        n_groups = n // group_size
        for g in range(n_groups):
            start = g * group_size
            end = start + group_size
            # Connect within each group at this level
            sub = np.ones((group_size, group_size))
            np.fill_diagonal(sub, 0)
            W[start:end, start:end] = np.maximum(W[start:end, start:end], sub * (0.5 ** level))
    np.fill_diagonal(W, 0)
    return W


def build_modular(n_modules: int = 4, n_per_module: int = 25,
                  p_intra: float = 0.4, p_inter: float = 0.02,
                  seed: int = 0) -> np.ndarray:
    """Modular (community) network."""
    rng = np.random.default_rng(seed)
    n = n_modules * n_per_module
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            same_module = (i // n_per_module) == (j // n_per_module)
            p = p_intra if same_module else p_inter
            if rng.random() < p:
                W[i, j] = 1
                W[j, i] = 1
    return W


def build_cortical_column(n_layers: int = 6,
                          neurons_per_layer: int = 20) -> np.ndarray:
    """Layered cortical column with feedforward + feedback connections."""
    n = n_layers * neurons_per_layer
    W = np.zeros((n, n))
    rng = np.random.default_rng(0)
    for layer in range(n_layers):
        start = layer * neurons_per_layer
        end = start + neurons_per_layer
        # Recurrent within layer
        sub = (rng.random((neurons_per_layer, neurons_per_layer)) < 0.3).astype(float)
        np.fill_diagonal(sub, 0)
        W[start:end, start:end] = sub
        # Feedforward to next layer
        if layer < n_layers - 1:
            next_start = end
            next_end = next_start + neurons_per_layer
            ff = (rng.random((neurons_per_layer, neurons_per_layer)) < 0.2).astype(float)
            W[start:end, next_start:next_end] = ff
        # Feedback to previous layer
        if layer > 0:
            prev_start = (layer - 1) * neurons_per_layer
            prev_end = prev_start + neurons_per_layer
            fb = (rng.random((neurons_per_layer, neurons_per_layer)) < 0.1).astype(float)
            W[start:end, prev_start:prev_end] = fb
    return W
