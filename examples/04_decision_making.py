"""Example 4: Drift Diffusion Model Psychometric Curve."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from neurodynamics.cognitive import DriftDiffusionModel

drift_values = np.linspace(-2.0, 2.0, 15)
ddm = DriftDiffusionModel(noise=1.0, threshold=1.0, seed=0)

p_correct = ddm.psychometric_curve(drift_values, n_trials=200)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(drift_values, p_correct, "o-", color="steelblue", lw=2)
ax.axhline(0.5, color="gray", ls="--", lw=1)
ax.set_xlabel("Stimulus Strength (drift rate)")
ax.set_ylabel("P(Choice = 1)")
ax.set_title("Drift Diffusion Model — Psychometric Curve")
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig("psychometric_curve.png", dpi=150)
print("Saved: psychometric_curve.png")
