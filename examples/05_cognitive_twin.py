"""Example 5: Cognitive Digital Twin — Aging Simulation."""
from neurodynamics.agents import CognitiveTwinSimulator
from neurodynamics.experiments import NBackTask

sim = CognitiveTwinSimulator()

# Create healthy young adult twin
twin = sim.create({
    "memory_capacity": 7,
    "attention_span": 1.0,
    "decision_threshold": 0.8,
    "age_years": 25.0,
})

# Baseline N-back performance
task = NBackTask(n=2, seq_length=30, seed=42)
result_young = sim.run_task(twin, task)
print(f"Young adult (age {twin.age_years}):")
print(f"  N-back accuracy: {result_young.accuracy:.2f}")
print(f"  Mean RT: {result_young.mean_rt*1000:.1f} ms")
print(f"  WM capacity: {twin.memory_capacity}")

# Simulate aging by 40 years
sim.simulate_aging(twin, years=40.0)

# Post-aging N-back performance
task2 = NBackTask(n=2, seq_length=30, seed=42)
result_aged = sim.run_task(twin, task2)
print(f"\nAged adult (age {twin.age_years}):")
print(f"  N-back accuracy: {result_aged.accuracy:.2f}")
print(f"  Mean RT: {result_aged.mean_rt*1000:.1f} ms")
print(f"  WM capacity: {twin.memory_capacity}")
