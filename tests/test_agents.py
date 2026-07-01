import numpy as np
import pytest

from neurodynamics.agents import (
    MessageBus, Message, BrainNetwork,
    PrefrontalCortex, Hippocampus, BasalGanglia,
    CognitiveTwin, CognitiveTwinSimulator,
)


def test_message_bus_delivery():
    bus = MessageBus()
    bus.register("A")
    bus.register("B")
    received = []
    bus.subscribe("B", lambda msg: received.append(msg))
    bus.send(Message("A", "B", "test", {"data": 42}))
    assert len(received) == 1
    assert received[0].payload == {"data": 42}


def test_prefrontal_cortex_holds_memory():
    bus = MessageBus()
    pfc = PrefrontalCortex(bus, wm_capacity=5)
    bus.send(Message("test", "PFC", "encode", "item_1"))
    bus.send(Message("test", "PFC", "encode", "item_2"))
    state = pfc.process()
    assert state["wm_load"] == 2


def test_hippocampus_pattern_completion():
    bus = MessageBus()
    hippo = Hippocampus(bus, capacity=20)
    rng = np.random.default_rng(0)
    pattern = rng.choice([-1.0, 1.0], size=20)
    hippo.store_pattern(pattern)
    corrupted = pattern.copy()
    corrupted[:4] *= -1
    recalled = hippo.recall(corrupted)
    assert len(recalled) == 20


def test_basal_ganglia_action_selection():
    bus = MessageBus()
    bg = BasalGanglia(bus, n_actions=4)
    inputs = np.array([0.1, 0.9, 0.2, 0.3])
    action = bg.select_action(inputs)
    assert action == 1  # highest input should win (dopamine=1, no noise)


def test_cognitive_twin_creation():
    sim = CognitiveTwinSimulator()
    twin = sim.create({"memory_capacity": 5, "age_years": 25.0})
    assert twin.memory_capacity == 5
    assert twin.age_years == 25.0
    assert twin.brain_network is not None


def test_lesion_impairs_function():
    sim = CognitiveTwinSimulator()
    twin = sim.create({"memory_capacity": 7})
    pfc = twin.brain_network.regions["PFC"]
    initial_capacity = pfc.wm.capacity
    sim.apply_lesion(twin, "PFC", severity=0.5)
    assert pfc.wm.capacity < initial_capacity


def test_aging_reduces_memory_capacity():
    sim = CognitiveTwinSimulator()
    twin = sim.create({"memory_capacity": 7, "age_years": 30.0})
    initial_cap = twin.memory_capacity
    sim.simulate_aging(twin, years=20.0)
    assert twin.memory_capacity <= initial_cap
    assert twin.age_years == 50.0
