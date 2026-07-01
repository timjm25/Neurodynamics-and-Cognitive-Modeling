import numpy as np


def shannon_entropy(prob_dist: np.ndarray) -> float:
    p = np.asarray(prob_dist, dtype=float)
    p = p[p > 0]
    p /= p.sum()
    return float(-np.sum(p * np.log2(p)))


def conditional_entropy(joint_dist: np.ndarray) -> float:
    """H(Y|X) from joint distribution P(X,Y)."""
    joint = np.asarray(joint_dist, dtype=float)
    joint /= joint.sum()
    px = joint.sum(axis=1)
    H_YX = 0.0
    for i in range(joint.shape[0]):
        if px[i] > 0:
            py_given_x = joint[i] / px[i]
            py_given_x = py_given_x[py_given_x > 0]
            H_YX += px[i] * (-np.sum(py_given_x * np.log2(py_given_x)))
    return float(H_YX)


def mutual_information(x: np.ndarray, y: np.ndarray, bins: int = 20) -> float:
    """Estimate I(X;Y) via histogram."""
    x, y = np.asarray(x), np.asarray(y)
    joint_hist, _, _ = np.histogram2d(x, y, bins=bins)
    joint_hist = joint_hist / joint_hist.sum()
    px = joint_hist.sum(axis=1)
    py = joint_hist.sum(axis=0)
    mi = 0.0
    for i in range(bins):
        for j in range(bins):
            if joint_hist[i, j] > 0 and px[i] > 0 and py[j] > 0:
                mi += joint_hist[i, j] * np.log2(joint_hist[i, j] / (px[i] * py[j]))
    return float(max(0.0, mi))


def transfer_entropy(source: np.ndarray, target: np.ndarray,
                     lag: int = 1, bins: int = 10) -> float:
    """Schreiber transfer entropy TE(source→target)."""
    s = np.asarray(source)
    t = np.asarray(target)
    n = min(len(s), len(t))
    s, t = s[:n], t[:n]
    # Discretize
    s_d = np.digitize(s, np.linspace(s.min(), s.max() + 1e-10, bins)) - 1
    t_d = np.digitize(t, np.linspace(t.min(), t.max() + 1e-10, bins)) - 1
    N = n - lag
    t_n = t_d[lag:]
    t_past = t_d[:N]
    s_past = s_d[:N]

    def _prob(*args):
        counts = {}
        for vals in zip(*args):
            counts[vals] = counts.get(vals, 0) + 1
        return {k: v / N for k, v in counts.items()}

    p_tns_tp = _prob(t_n, s_past, t_past)
    p_tp = _prob(t_past,)
    p_tn_tp = _prob(t_n, t_past)
    p_s_tp = _prob(s_past, t_past)

    te = 0.0
    for (tn, sp, tp), p in p_tns_tp.items():
        p_tp_v = p_tp.get((tp,), 1e-12)
        p_tn_tp_v = p_tn_tp.get((tn, tp), 1e-12)
        p_s_tp_v = p_s_tp.get((sp, tp), 1e-12)
        te += p * np.log2(p * p_tp_v / (p_tn_tp_v * p_s_tp_v + 1e-12) + 1e-12)
    return float(max(0.0, te))


def integrated_information(connectivity_matrix: np.ndarray) -> float:
    """Simplified Phi estimate: reduction in mutual information by partition."""
    W = np.asarray(connectivity_matrix, dtype=float)
    n = W.shape[0]
    if n < 2:
        return 0.0
    whole_info = float(np.sum(np.abs(W)))
    # Bipartition at midpoint
    half = n // 2
    W_part1 = np.sum(np.abs(W[:half, :half]))
    W_part2 = np.sum(np.abs(W[half:, half:]))
    cross = np.sum(np.abs(W[:half, half:])) + np.sum(np.abs(W[half:, :half]))
    phi = cross / (whole_info + 1e-12)
    return float(phi)


def complexity(time_series: np.ndarray) -> float:
    """Lempel-Ziv complexity (normalized)."""
    ts = np.asarray(time_series)
    median = np.median(ts)
    binary = "".join("1" if x > median else "0" for x in ts)
    n = len(binary)
    if n == 0:
        return 0.0
    # LZ76 algorithm
    i, k, l = 0, 1, 1
    complexity_count = 1
    while True:
        if binary[i + k - 1] == binary[l + k - 1]:
            k += 1
            if l + k > n:
                complexity_count += 1
                break
        else:
            if k > l - i:
                complexity_count += 1
                i = 0
                l += 1
                k = 1
                if l == n:
                    break
            else:
                i += 1
                k = 1
    return complexity_count / (n / np.log2(n + 1))
