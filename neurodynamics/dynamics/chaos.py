from typing import Callable, List

import numpy as np
from scipy.integrate import solve_ivp


def lorenz_system(t, state, sigma: float = 10.0, rho: float = 28.0,
                  beta: float = 8/3):
    x, y, z = state
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]


def rossler_system(t, state, a: float = 0.2, b: float = 0.2, c: float = 5.7):
    x, y, z = state
    return [-y - z, x + a * y, b + z * (x - c)]


def simulate_lorenz(duration: float = 50.0, dt: float = 0.01,
                    init: List[float] = None, **kwargs) -> np.ndarray:
    y0 = init or [1.0, 0.0, 0.0]
    t_eval = np.arange(0, duration, dt)
    sol = solve_ivp(lorenz_system, (0, duration), y0, t_eval=t_eval,
                    args=(kwargs.get("sigma", 10), kwargs.get("rho", 28),
                          kwargs.get("beta", 8/3)), method="RK45")
    return sol.y.T


def lyapunov_exponent_1d(f: Callable, x0: float, n_steps: int = 10000,
                          dt: float = 0.01) -> float:
    """Largest Lyapunov exponent for a 1D map/ODE."""
    x = x0
    log_sum = 0.0
    eps = 1e-8
    for _ in range(n_steps):
        df = abs((f(x + eps) - f(x - eps)) / (2 * eps))
        if df > 0:
            log_sum += np.log(df)
        x = f(x)
    return log_sum / n_steps


def lyapunov_spectrum(trajectory: np.ndarray, dt: float = 0.01) -> np.ndarray:
    """Estimate Lyapunov spectrum from trajectory using QR decomposition."""
    T, n = trajectory.shape
    Q = np.eye(n)
    exponents = np.zeros(n)
    step = max(1, T // 100)
    count = 0
    for t in range(step, T, step):
        # Finite-difference Jacobian approximation
        if t + 1 < T:
            J = np.zeros((n, n))
            for i in range(n):
                delta = trajectory[t + 1] - trajectory[t]
                J[:, i] = delta / (dt + 1e-12)
            M = J @ Q
            Q, R = np.linalg.qr(M)
            exponents += np.log(np.abs(np.diag(R)) + 1e-12)
            count += 1
    if count > 0:
        exponents /= (count * step * dt)
    return exponents


def correlation_dimension(trajectory: np.ndarray,
                          r_range: np.ndarray = None,
                          n_pairs: int = 2000) -> float:
    """Estimate correlation dimension D2 via Grassberger-Procaccia."""
    rng = np.random.default_rng(0)
    T = len(trajectory)
    idx = rng.choice(T, size=min(n_pairs, T), replace=False)
    pts = trajectory[idx]
    dists = []
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(np.linalg.norm(pts[i] - pts[j]))
    dists = np.sort(dists)
    if r_range is None:
        r_range = np.logspace(np.log10(np.percentile(dists, 5) + 1e-10),
                              np.log10(np.percentile(dists, 95) + 1e-10), 20)
    Cr = np.array([np.mean(dists < r) for r in r_range])
    Cr = np.clip(Cr, 1e-10, None)
    valid = Cr > 0
    if valid.sum() < 2:
        return 0.0
    log_r = np.log(r_range[valid])
    log_C = np.log(Cr[valid])
    slope = np.polyfit(log_r, log_C, 1)[0]
    return float(np.clip(slope, 0, 10))


def sample_entropy(time_series: np.ndarray, m: int = 2,
                   r: float = 0.2) -> float:
    """Sample entropy SampEn(m, r, N)."""
    N = len(time_series)
    r_abs = r * np.std(time_series)

    def _count_templates(m):
        count = 0
        for i in range(N - m):
            template = time_series[i:i + m]
            for j in range(N - m):
                if i != j:
                    match = np.max(np.abs(time_series[j:j + m] - template))
                    if match < r_abs:
                        count += 1
        return count

    A = _count_templates(m + 1)
    B = _count_templates(m)
    if B == 0:
        return float("inf")
    return -np.log(A / B)


def approximate_entropy(time_series: np.ndarray, m: int = 2,
                        r: float = 0.2) -> float:
    """Approximate entropy ApEn(m, r)."""
    N = len(time_series)
    r_abs = r * np.std(time_series)

    def _phi(m_val):
        count = np.zeros(N - m_val + 1)
        for i in range(N - m_val + 1):
            template = time_series[i:i + m_val]
            for j in range(N - m_val + 1):
                if np.max(np.abs(time_series[j:j + m_val] - template)) < r_abs:
                    count[i] += 1
        return np.mean(np.log(count / (N - m_val + 1) + 1e-12))

    return float(_phi(m) - _phi(m + 1))


def is_chaotic(lyapunov_exponents: np.ndarray) -> bool:
    return bool(np.max(lyapunov_exponents) > 0)
