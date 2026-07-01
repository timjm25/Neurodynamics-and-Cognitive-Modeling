"""Example 1: Leaky Integrate-and-Fire Neuron Simulation."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from neurodynamics.neural_dynamics import LeakyIntegrateAndFire

lif = LeakyIntegrateAndFire(tau_m=20.0, v_rest=-70.0, v_thresh=-55.0,
                              v_reset=-75.0, r_m=10.0)

# Step current: off for 100ms, on for 300ms, off for 100ms
duration = 500.0
dt = 0.1

def step_current(t):
    return 3.0 if 100.0 <= t <= 400.0 else 0.0

result = lif.simulate(duration, current=0.0, dt=dt, current_fn=step_current)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(result.t, result.V, "navy", lw=0.8, label="Membrane Voltage")
for st in result.spike_times:
    ax.axvline(st, color="red", alpha=0.5, lw=0.5)
ax.axhline(lif.v_thresh, color="orange", ls="--", lw=1, label="Threshold")
ax.axhline(lif.v_rest, color="gray", ls=":", lw=1, label="Rest")
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Voltage (mV)")
ax.set_title(f"LIF Neuron — {len(result.spike_times)} spikes in {duration} ms")
ax.legend()
plt.tight_layout()
plt.savefig("lif_simulation.png", dpi=150)
print(f"Spikes: {len(result.spike_times)}")
print(f"Saved: lif_simulation.png")
