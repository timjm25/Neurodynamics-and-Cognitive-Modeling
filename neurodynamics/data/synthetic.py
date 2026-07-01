"""Synthetic dataset generators."""
import numpy as np


def generate_spike_trains(n_neurons: int = 10, duration: float = 1000.0,
                          mean_rate: float = 20.0, seed: int = 0) -> list:
    """Generate Poisson spike trains."""
    from ..neural_dynamics.spiking import SpikeTrain
    rng = np.random.default_rng(seed)
    trains = []
    for i in range(n_neurons):
        rate = rng.uniform(mean_rate * 0.5, mean_rate * 1.5)
        n_spikes = int(rng.poisson(rate * duration / 1000.0))
        times = np.sort(rng.uniform(0, duration, n_spikes))
        trains.append(SpikeTrain(times, duration, i))
    return trains


def generate_oscillatory_signal(duration: float = 1000.0, dt: float = 1.0,
                                 freq: float = 10.0, noise: float = 0.1,
                                 seed: int = 0) -> np.ndarray:
    """Generate oscillatory signal (e.g., alpha wave)."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration, dt)
    signal = np.sin(2 * np.pi * freq * t / 1000.0) + noise * rng.normal(size=len(t))
    return signal


def generate_eeg_like(n_channels: int = 64, duration: float = 10000.0,
                       dt: float = 1.0, seed: int = 0) -> np.ndarray:
    """Generate synthetic EEG-like multichannel data."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration, dt)
    n = len(t)
    data = np.zeros((n_channels, n))
    for ch in range(n_channels):
        for freq, amp in [(10, 2.0), (20, 0.5), (40, 0.3)]:
            phase = rng.uniform(0, 2 * np.pi)
            data[ch] += amp * np.sin(2 * np.pi * freq * t / 1000.0 + phase)
        data[ch] += rng.normal(0, 0.5, n)
    return data


def generate_connectivity_timeseries(n_nodes: int = 10, n_time: int = 500,
                                      W: np.ndarray = None, seed: int = 0) -> np.ndarray:
    """Generate time series with given connectivity structure."""
    rng = np.random.default_rng(seed)
    if W is None:
        W = rng.random((n_nodes, n_nodes)) * 0.1
        np.fill_diagonal(W, 0)
    data = np.zeros((n_nodes, n_time))
    data[:, 0] = rng.normal(size=n_nodes)
    for t in range(1, n_time):
        data[:, t] = 0.9 * data[:, t - 1] + W @ data[:, t - 1] + rng.normal(0, 0.1, n_nodes)
    return data
