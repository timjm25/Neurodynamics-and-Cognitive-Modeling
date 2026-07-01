import numpy as np
import pytest

from neurodynamics.connectivity import (
    build_small_world, build_scale_free, build_random, build_modular,
    functional_connectivity, graph_laplacian, community_detection_louvain,
    clustering_coefficient,
)


def test_small_world_has_correct_node_count():
    W = build_small_world(n=20, k=4, p=0.1)
    assert W.shape == (20, 20)
    assert np.all(np.diag(W) == 0)


def test_scale_free_has_power_law_degree():
    W = build_scale_free(n=50, m=2)
    degrees = W.sum(axis=1)
    # Scale-free: variance should be high (skewed degree distribution)
    assert np.std(degrees) > 1.0
    assert degrees.max() > degrees.mean()


def test_modular_intra_gt_inter():
    W = build_modular(n_modules=3, n_per_module=10,
                       p_intra=0.6, p_inter=0.02)
    n_per = 10
    intra_edges = 0
    inter_edges = 0
    n = W.shape[0]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if (i // n_per) == (j // n_per):
                intra_edges += W[i, j]
            else:
                inter_edges += W[i, j]
    assert intra_edges > inter_edges


def test_functional_connectivity_self_correlation_is_one():
    rng = np.random.default_rng(0)
    ts = rng.normal(0, 1, (5, 200))
    FC = functional_connectivity(ts)
    np.testing.assert_allclose(np.diag(FC), 1.0, atol=1e-10)


def test_laplacian_is_symmetric():
    W = build_small_world(n=10, k=4, p=0.1)
    L = graph_laplacian(W)
    np.testing.assert_allclose(L, L.T, atol=1e-10)


def test_community_detection_finds_modules():
    W = build_modular(n_modules=2, n_per_module=10,
                       p_intra=0.8, p_inter=0.01)
    labels = community_detection_louvain(W)
    assert len(labels) == 20
    # First 10 should mostly share a community label
    first_half = labels[:10]
    second_half = labels[10:]
    # Most nodes in each half should have same label
    assert np.bincount(first_half).max() >= 7
    assert np.bincount(second_half).max() >= 7


def test_clustering_coefficient_range():
    W = build_small_world(n=20, k=4, p=0.1)
    cc = clustering_coefficient(W)
    assert np.all(cc >= 0.0)
    assert np.all(cc <= 1.0)
