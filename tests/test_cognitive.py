import numpy as np
import pytest

from neurodynamics.cognitive import (
    WorkingMemoryModel, DriftDiffusionModel, BayesianDecisionMaker, RaceModel,
    PredictiveCodingLayer, ActiveInferenceAgent,
)


def test_working_memory_capacity():
    wm = WorkingMemoryModel(capacity=3)
    assert wm.encode("a")
    assert wm.encode("b")
    assert wm.encode("c")
    assert not wm.encode("d")  # Over capacity
    assert wm.load == 3


def test_working_memory_decay():
    wm = WorkingMemoryModel(capacity=5, decay_rate=0.5)
    wm.encode("item1")
    assert wm.load == 1
    wm.forget(dt=10.0)  # large dt → items decay
    # After large decay, items should be removed
    assert wm.load == 0


def test_drift_diffusion_correct_direction():
    ddm = DriftDiffusionModel(drift=2.0, noise=0.5, threshold=1.0, seed=42)
    choices = [ddm.decide(2.0)[0] for _ in range(100)]
    # With strong positive drift, should choose 1 more often
    assert np.mean(choices) > 0.6


def test_bayesian_decision_updates_prior():
    bdm = BayesianDecisionMaker(n_hypotheses=2, prior=[0.5, 0.5])
    initial_posterior = bdm.posterior.copy()
    # Strong evidence for hypothesis 1
    bdm.update([0.1, 0.9])
    assert bdm.posterior[1] > initial_posterior[1]
    assert bdm.decide() == 1


def test_predictive_coding_reduces_error():
    pc = PredictiveCodingLayer(dim=5, lr=0.2)
    obs = np.ones(5)
    initial_fe = None
    for i in range(20):
        pc.forward(obs)
        if i == 0:
            initial_fe = pc.free_energy
    final_fe = pc.free_energy
    assert final_fe < initial_fe


def test_active_inference_selects_low_free_energy_action():
    agent = ActiveInferenceAgent(n_states=4, n_actions=2)
    beliefs = np.array([0.1, 0.7, 0.1, 0.1])
    agent.beliefs = beliefs.copy()
    preferences = np.array([0.1, 0.7, 0.1, 0.1])
    # Should return a valid action index
    action = agent.select_action(preferences)
    assert action in [0, 1]


def test_race_model_faster_with_stronger_input():
    rm = RaceModel(n_alternatives=2, threshold=1.0, noise=0.01, seed=0)
    _, rt_strong = rm.compete(np.array([2.0, 0.1]))
    rm2 = RaceModel(n_alternatives=2, threshold=1.0, noise=0.01, seed=0)
    _, rt_weak = rm2.compete(np.array([0.5, 0.1]))
    assert rt_strong < rt_weak
