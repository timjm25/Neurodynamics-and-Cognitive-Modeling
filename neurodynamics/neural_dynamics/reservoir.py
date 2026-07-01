from typing import Optional

import numpy as np

from .neuron_models import LeakyIntegrateAndFire


class EchoStateNetwork:
    name = "esn"
    description = "Echo State Network (reservoir computing)"

    def __init__(self, input_dim: int = 1, reservoir_dim: int = 100,
                 output_dim: int = 1, spectral_radius: float = 0.9,
                 sparsity: float = 0.1, input_scaling: float = 1.0,
                 ridge_alpha: float = 1e-4, seed: int = 42):
        self.input_dim = input_dim
        self.reservoir_dim = reservoir_dim
        self.output_dim = output_dim
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.input_scaling = input_scaling
        self.ridge_alpha = ridge_alpha
        self.rng = np.random.default_rng(seed)
        self.W_in: Optional[np.ndarray] = None
        self.W_res: Optional[np.ndarray] = None
        self.W_out: Optional[np.ndarray] = None
        self.initialize()

    def initialize(self) -> None:
        self.W_in = self.rng.uniform(-1, 1, (self.reservoir_dim, self.input_dim)) * self.input_scaling
        W = self.rng.uniform(-1, 1, (self.reservoir_dim, self.reservoir_dim))
        mask = self.rng.random((self.reservoir_dim, self.reservoir_dim)) > self.sparsity
        W[mask] = 0.0
        eigvals = np.linalg.eigvals(W)
        rho = np.max(np.abs(eigvals))
        if rho > 0:
            W = W * (self.spectral_radius / rho)
        self.W_res = W

    def run(self, input_sequence: np.ndarray,
            washout: int = 100) -> np.ndarray:
        T = len(input_sequence)
        states = np.zeros((T, self.reservoir_dim))
        x = np.zeros(self.reservoir_dim)
        inp = np.atleast_2d(input_sequence)
        if inp.shape[0] == 1:
            inp = inp.T

        for t in range(T):
            u = inp[t]
            x = np.tanh(self.W_res @ x + self.W_in @ u)
            states[t] = x
        return states[washout:]

    def train_readout(self, states: np.ndarray, targets: np.ndarray) -> None:
        targets = np.atleast_2d(targets)
        if targets.shape[0] == 1:
            targets = targets.T
        if targets.shape[0] != states.shape[0]:
            targets = targets[:states.shape[0]]
        # Ridge regression: W_out = (S^T S + alpha I)^-1 S^T T
        A = states.T @ states + self.ridge_alpha * np.eye(self.reservoir_dim)
        B = states.T @ targets
        self.W_out = np.linalg.solve(A, B).T

    def predict(self, input_sequence: np.ndarray, washout: int = 0) -> np.ndarray:
        states = self.run(input_sequence, washout=washout)
        if self.W_out is None:
            raise RuntimeError("Call train_readout before predict")
        return (self.W_out @ states.T).T


class LiquidStateMachine:
    name = "lsm"
    description = "Liquid State Machine with LIF reservoir"

    def __init__(self, input_dim: int = 1, reservoir_dim: int = 50,
                 output_dim: int = 1, seed: int = 42):
        self.input_dim = input_dim
        self.reservoir_dim = reservoir_dim
        self.output_dim = output_dim
        self.rng = np.random.default_rng(seed)
        self.neurons = [LeakyIntegrateAndFire() for _ in range(reservoir_dim)]
        self.W_in = self.rng.uniform(-2, 2, (reservoir_dim, input_dim))
        W = self.rng.uniform(-1, 1, (reservoir_dim, reservoir_dim))
        mask = self.rng.random((reservoir_dim, reservoir_dim)) > 0.1
        W[mask] = 0.0
        self.W_res = W * 0.5
        self.W_out: Optional[np.ndarray] = None
        self._states: list = [{"V": -70.0, "dt": 0.1}] * reservoir_dim

    def run(self, input_sequence: np.ndarray) -> np.ndarray:
        T = len(input_sequence)
        liquid_states = np.zeros((T, self.reservoir_dim))
        inp = np.atleast_2d(input_sequence)
        if inp.shape[0] == 1:
            inp = inp.T
        states = [{"V": -70.0, "dt": 0.1}] * self.reservoir_dim
        spikes = np.zeros(self.reservoir_dim)

        for t in range(T):
            u = inp[t]
            new_spikes = np.zeros(self.reservoir_dim)
            for j, neuron in enumerate(self.neurons):
                I = float(self.W_in[j] @ u) + float(self.W_res[j] @ spikes)
                state = neuron.step(I, states[j])
                states[j] = state
                new_spikes[j] = float(state["spiked"])
            spikes = new_spikes
            liquid_states[t] = spikes
        return liquid_states

    def train_readout(self, states: np.ndarray, targets: np.ndarray) -> None:
        targets = np.atleast_2d(targets)
        if targets.shape[0] == 1:
            targets = targets.T
        A = states.T @ states + 1e-4 * np.eye(self.reservoir_dim)
        B = states.T @ targets[:states.shape[0]]
        self.W_out = np.linalg.solve(A, B).T

    def predict(self, input_sequence: np.ndarray) -> np.ndarray:
        states = self.run(input_sequence)
        if self.W_out is None:
            raise RuntimeError("Call train_readout before predict")
        return (self.W_out @ states.T).T
