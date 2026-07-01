"""Neural ODE wrapper — requires PyTorch. Handles ImportError gracefully."""
from typing import Any, Callable, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class NeuralODE:
    """Wraps a PyTorch neural network as an ODE vector field."""

    def __init__(self, hidden_dim: int = 64, input_dim: int = 2):
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch required for NeuralODE. pip install torch")
        import torch.nn as nn
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, t, y):
        import torch
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32)
        return self.net(y)

    def simulate_euler(self, y0: np.ndarray, duration: float,
                       dt: float = 0.01) -> np.ndarray:
        import torch
        y = torch.tensor(y0, dtype=torch.float32)
        n_steps = int(duration / dt)
        traj = [y.detach().numpy()]
        for i in range(n_steps):
            t = torch.tensor(i * dt)
            dy = self.forward(t, y)
            y = y + dt * dy
            traj.append(y.detach().numpy())
        return np.array(traj)


def is_torch_available() -> bool:
    return _TORCH_AVAILABLE
