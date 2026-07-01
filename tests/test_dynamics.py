import numpy as np
import pytest

from neurodynamics.dynamics import (
    lorenz_system, simulate_lorenz, lyapunov_spectrum, is_chaotic,
    shannon_entropy, mutual_information, transfer_entropy,
    BifurcationScanner, sample_entropy, correlation_dimension,
)


def test_lorenz_is_chaotic():
    traj = simulate_lorenz(duration=50.0, dt=0.01)
    les = lyapunov_spectrum(traj, dt=0.01)
    assert is_chaotic(les)


def test_fixed_point_not_chaotic():
    # A decaying system — all LE should be negative
    # Use stable linear system x' = -x
    rng = np.random.default_rng(1)
    T = 1000
    traj = np.zeros((T, 2))
    traj[0] = rng.normal(size=2)
    for i in range(1, T):
        traj[i] = 0.95 * traj[i - 1]
    les = lyapunov_spectrum(traj, dt=1.0)
    assert not is_chaotic(les)


def test_shannon_entropy_uniform():
    p = np.ones(8) / 8
    H = shannon_entropy(p)
    assert abs(H - 3.0) < 0.01  # log2(8) = 3 bits


def test_mutual_information_independent():
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 1000)
    y = rng.normal(0, 1, 1000)
    mi = mutual_information(x, y, bins=10)
    assert mi < 0.3  # Independent → near 0


def test_transfer_entropy_causal():
    rng = np.random.default_rng(0)
    n = 500
    source = rng.normal(size=n)
    target = np.zeros(n)
    for i in range(1, n):
        target[i] = 0.7 * source[i - 1] + 0.1 * rng.normal()
    te_fwd = transfer_entropy(source, target, lag=1, bins=5)
    te_bwd = transfer_entropy(target, source, lag=1, bins=5)
    assert te_fwd > te_bwd


def test_bifurcation_scanner_detects_transition():
    def logistic_step(state, dt=1.0, r=3.0):
        x = state[0] if hasattr(state, '__len__') else float(state)
        return np.array([r * x * (1 - x)])

    scanner = BifurcationScanner()
    param_range = np.linspace(2.5, 3.9, 10)
    diag = scanner.scan(logistic_step, "r", param_range,
                        np.array([0.5]), dt=1.0, transient=200, record=100)
    types = diag.detected_types
    # At r<3: equilibrium; at r>3: limit cycle or chaos
    assert "equilibrium" in types or "limit_cycle" in types


def test_correlation_dimension_lorenz():
    traj = simulate_lorenz(duration=30.0, dt=0.05)
    D2 = correlation_dimension(traj[:500], n_pairs=200)
    assert 1.0 < D2 < 5.0


def test_sample_entropy_random_vs_sine():
    rng = np.random.default_rng(0)
    random_ts = rng.normal(size=200)
    t = np.linspace(0, 20 * np.pi, 200)
    sine_ts = np.sin(t)
    se_random = sample_entropy(random_ts, m=2, r=0.2)
    se_sine = sample_entropy(sine_ts, m=2, r=0.2)
    # Random should have higher entropy than regular sine
    assert se_random > se_sine
