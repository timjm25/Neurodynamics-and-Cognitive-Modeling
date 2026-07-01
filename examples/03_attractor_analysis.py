"""Example 3: Lorenz Attractor Analysis."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from neurodynamics.dynamics import simulate_lorenz, lyapunov_spectrum, is_chaotic

traj = simulate_lorenz(duration=50.0, dt=0.01)
print(f"Lorenz trajectory: {traj.shape} points")

les = lyapunov_spectrum(traj, dt=0.01)
print(f"Lyapunov exponents (approx): {les}")
print(f"System is chaotic: {is_chaotic(les)}")

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], lw=0.3, alpha=0.8, color="navy")
ax.set_title("Lorenz Strange Attractor")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.tight_layout()
plt.savefig("lorenz_attractor.png", dpi=150)
print("Saved: lorenz_attractor.png")
