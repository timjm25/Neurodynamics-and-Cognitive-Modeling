"""Example 2: Hopfield Network Associative Memory."""
import numpy as np
from neurodynamics.learning import HopfieldNetwork

rng = np.random.default_rng(42)
n = 50
net = HopfieldNetwork(n)

patterns = [rng.choice([-1.0, 1.0], size=n) for _ in range(5)]
net.store(patterns)

print(f"Storing {len(patterns)} patterns in Hopfield network (N={n})")
print(f"Theoretical capacity: {net.capacity:.1f} patterns")

for i, pattern in enumerate(patterns):
    noise_frac = 0.2
    corrupted = pattern.copy()
    n_corrupt = int(n * noise_frac)
    flip_idx = rng.choice(n, size=n_corrupt, replace=False)
    corrupted[flip_idx] *= -1

    recalled = net.recall(corrupted)
    overlap = float(np.dot(recalled, pattern)) / n
    print(f"Pattern {i+1}: overlap after recall = {overlap:.3f} "
          f"({'SUCCESS' if overlap > 0.8 else 'FAIL'})")
