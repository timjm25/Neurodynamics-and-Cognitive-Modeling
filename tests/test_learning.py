import numpy as np
import pytest

from neurodynamics.learning import HebbianRule, STDPRule, OjaRule, HopfieldNetwork


def test_hebbian_increases_weight():
    hebb = HebbianRule()
    W = np.zeros((3, 3))
    pre = np.array([1.0, 0.0, 1.0])
    post = np.array([1.0, 1.0, 0.0])
    W_new = hebb.update(W, pre, post, lr=0.1)
    assert W_new.sum() > 0


def test_stdp_potentiation():
    stdp = STDPRule()
    W = np.zeros((2, 2))
    pre_spikes = np.array([10.0, 50.0])
    post_spikes = np.array([15.0, 55.0])  # post after pre
    W_new = stdp.update(W, pre_spikes, post_spikes, dt=0.1)
    assert W_new.sum() > 0  # net potentiation


def test_stdp_depression():
    stdp = STDPRule()
    W = np.zeros((2, 2))
    pre_spikes = np.array([15.0, 55.0])
    post_spikes = np.array([10.0, 50.0])  # post before pre → LTD
    W_new = stdp.update(W, pre_spikes, post_spikes, dt=0.1,
                         A_plus=0.0, A_minus=0.02)
    assert W_new.sum() < 0


def test_hopfield_pattern_recall():
    n = 20
    net = HopfieldNetwork(n)
    rng = np.random.default_rng(0)
    pattern = rng.choice([-1, 1], size=n).astype(float)
    net.store([pattern])
    # Corrupt 3 bits
    corrupted = pattern.copy()
    corrupted[:3] *= -1
    recalled = net.recall(corrupted)
    overlap = float(np.dot(recalled, pattern)) / n
    assert overlap > 0.6


def test_hopfield_capacity():
    n = 50
    net = HopfieldNetwork(n)
    assert net.capacity == pytest.approx(0.138 * n, rel=0.01)


def test_oja_rule_extracts_pca():
    oja = OjaRule()
    rng = np.random.default_rng(0)
    # Data with clear first PC along [1, 0]
    data = rng.normal(size=(1000, 2))
    data[:, 0] *= 5  # strong variance in dim 0
    W = np.zeros((1, 2))
    W[0] = rng.normal(size=2)
    W[0] /= np.linalg.norm(W[0])
    for x in data:
        post = W @ x
        W = oja.update(W, x, post, lr=0.01)
        W[0] /= np.linalg.norm(W[0]) + 1e-10
    # W should align with first PC [1, 0]
    assert abs(W[0, 0]) > 0.8
