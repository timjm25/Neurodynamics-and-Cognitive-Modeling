from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class SpikeTrain:
    timestamps: np.ndarray
    duration: float
    neuron_id: int = 0

    def __post_init__(self):
        self.timestamps = np.sort(np.asarray(self.timestamps, dtype=float))

    @property
    def n_spikes(self) -> int:
        return len(self.timestamps)

    def firing_rate(self) -> float:
        return self.n_spikes / (self.duration / 1000.0) if self.duration > 0 else 0.0


def inter_spike_interval(spike_train: SpikeTrain) -> np.ndarray:
    if len(spike_train.timestamps) < 2:
        return np.array([])
    return np.diff(spike_train.timestamps)


def firing_rate(spike_train: SpikeTrain, window: float = 50.0) -> np.ndarray:
    if len(spike_train.timestamps) == 0:
        return np.zeros(int(spike_train.duration / window))
    bins = np.arange(0, spike_train.duration + window, window)
    counts, _ = np.histogram(spike_train.timestamps, bins=bins)
    return counts / (window / 1000.0)


def fano_factor(spike_train: SpikeTrain, window: float = 50.0) -> float:
    rates = firing_rate(spike_train, window)
    if rates.mean() == 0:
        return float("nan")
    return float(np.var(rates) / np.mean(rates))


def coefficient_of_variation(spike_train: SpikeTrain) -> float:
    isi = inter_spike_interval(spike_train)
    if len(isi) < 2:
        return float("nan")
    return float(np.std(isi) / np.mean(isi))


def stdp_update(pre_spikes: np.ndarray, post_spikes: np.ndarray,
                W: np.ndarray, A_plus: float = 0.01, A_minus: float = 0.012,
                tau_plus: float = 20.0, tau_minus: float = 20.0) -> np.ndarray:
    """STDP weight update.
    pre before post (dt > 0) → potentiation (LTP).
    post before pre (dt < 0) → depression (LTD).
    """
    dW = np.zeros_like(W)
    for t_pre in pre_spikes:
        for t_post in post_spikes:
            dt = t_post - t_pre  # positive = pre before post → LTP
            if 0 < dt < 5 * tau_plus:
                dW += A_plus * np.exp(-dt / tau_plus)
            elif -5 * tau_minus < dt < 0:
                dW -= A_minus * np.exp(dt / tau_minus)
    return dW


class SpikingNetwork:
    def __init__(self, n_neurons: int, weights: Optional[np.ndarray] = None,
                 tau_m: float = 20.0, v_thresh: float = -55.0,
                 v_reset: float = -75.0, v_rest: float = -70.0,
                 r_m: float = 10.0):
        self.n = n_neurons
        self.W = weights if weights is not None else np.zeros((n_neurons, n_neurons))
        self.tau_m = tau_m
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.v_rest = v_rest
        self.r_m = r_m

    def simulate(self, duration: float, external_current: np.ndarray,
                 dt: float = 0.1) -> List[SpikeTrain]:
        n_steps = int(duration / dt)
        V = np.full(self.n, self.v_rest)
        spike_times = [[] for _ in range(self.n)]

        for i in range(n_steps):
            t = i * dt
            I_ext = external_current[:, i] if external_current.ndim == 2 else external_current
            I_net = I_ext + self.W @ (V > self.v_thresh).astype(float)
            dV = (-(V - self.v_rest) + self.r_m * I_net) / self.tau_m
            V = V + dt * dV
            spiked = V >= self.v_thresh
            for j in np.where(spiked)[0]:
                spike_times[j].append(t)
            V[spiked] = self.v_reset

        return [SpikeTrain(np.array(st), duration, i)
                for i, st in enumerate(spike_times)]
