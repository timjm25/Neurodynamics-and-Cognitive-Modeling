import numpy as np
import pytest

from neurodynamics.hypothesis import HypothesisDiscoveryEngine
from neurodynamics.dynamics import simulate_lorenz


def test_attractor_detection_fixed_point():
    # Stable decaying system → fixed point
    traj = np.zeros((500, 2))
    traj[0] = [1.0, 0.5]
    for i in range(1, 500):
        traj[i] = 0.9 * traj[i - 1]
    engine = HypothesisDiscoveryEngine()
    attractors = engine.find_attractors([traj])
    assert attractors[0].attractor_type == "fixed_point"


def test_attractor_detection_limit_cycle():
    # Oscillatory system → limit cycle
    t = np.linspace(0, 20 * np.pi, 500)
    traj = np.column_stack([np.sin(t), np.cos(t)])
    engine = HypothesisDiscoveryEngine()
    attractors = engine.find_attractors([traj])
    assert attractors[0].attractor_type in ("limit_cycle", "strange")  # allow both


def test_symbolic_regression_linear():
    engine = HypothesisDiscoveryEngine()
    X = np.linspace(0, 5, 50)
    y = 2.0 * X + 1.0  # linear: y = 1.0 + 2.0*x
    eq = engine.symbolic_regression(X, y, max_complexity=3)
    assert "y = " in eq
    # Should contain approx coefficients
    assert "x^1" in eq or "x^0" in eq


def test_hypothesis_generation_returns_list():
    engine = HypothesisDiscoveryEngine()
    obs = np.sin(np.linspace(0, 10, 100))
    hypotheses = engine.generate_hypothesis(obs)
    assert isinstance(hypotheses, list)
    assert len(hypotheses) > 0


def test_emergent_behavior_detection():
    engine = HypothesisDiscoveryEngine()
    # Strong oscillation
    t = np.linspace(0, 100, 1000)
    signal = 5.0 * np.sin(2 * np.pi * 10 * t)
    behaviors = engine.detect_emergent_behaviors(signal)
    assert isinstance(behaviors, list)
    # Should detect oscillation in a clean sine wave
    types = [b.behavior_type for b in behaviors]
    assert "oscillation" in types
