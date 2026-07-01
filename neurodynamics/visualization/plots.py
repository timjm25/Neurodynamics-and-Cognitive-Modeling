"""Visualization routines using matplotlib."""
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


def raster_plot(spike_trains: list, duration: float, title: str = "Raster Plot"):
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, st in enumerate(spike_trains):
        times = st.timestamps if hasattr(st, "timestamps") else np.array(st)
        ax.scatter(times, np.full_like(times, i), marker="|", s=50, color="black")
    ax.set_xlim(0, duration)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Neuron")
    ax.set_title(title)
    plt.tight_layout()
    return fig


def firing_rate_histogram(spike_trains: list, bin_size: float = 50.0,
                           title: str = "Firing Rate"):
    fig, ax = plt.subplots(figsize=(8, 3))
    all_times = []
    duration = 0.0
    for st in spike_trains:
        times = st.timestamps if hasattr(st, "timestamps") else np.array(st)
        all_times.extend(times)
        if hasattr(st, "duration"):
            duration = max(duration, st.duration)
    if duration == 0 and all_times:
        duration = max(all_times) + bin_size
    bins = np.arange(0, duration + bin_size, bin_size)
    counts, _ = np.histogram(all_times, bins=bins)
    ax.bar(bins[:-1], counts / (bin_size / 1000) / len(spike_trains),
           width=bin_size * 0.9, color="steelblue")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Firing Rate (Hz)")
    ax.set_title(title)
    plt.tight_layout()
    return fig


def connectivity_matrix_plot(W: np.ndarray, labels: List[str] = None,
                              title: str = "Connectivity Matrix"):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(W, cmap="RdBu_r", aspect="auto")
    plt.colorbar(im, ax=ax)
    if labels:
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(title)
    plt.tight_layout()
    return fig


def phase_portrait_2d(trajectory: np.ndarray, title: str = "Phase Portrait",
                       xlabel: str = "x", ylabel: str = "y"):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(trajectory[:, 0], trajectory[:, 1], lw=0.5, color="navy", alpha=0.7)
    ax.scatter(trajectory[0, 0], trajectory[0, 1], color="green", s=50, zorder=5, label="start")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    return fig


def time_series_plot(t: np.ndarray, signals: np.ndarray,
                     labels: List[str] = None, title: str = "Time Series"):
    fig, ax = plt.subplots(figsize=(10, 3))
    signals = np.atleast_2d(signals)
    for i, sig in enumerate(signals):
        lbl = labels[i] if labels and i < len(labels) else f"Signal {i}"
        ax.plot(t[:len(sig)], sig, label=lbl, lw=0.8)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    if labels:
        ax.legend(fontsize=8)
    plt.tight_layout()
    return fig


def state_space_3d(trajectory: np.ndarray, title: str = "3D State Space"):
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
            lw=0.3, alpha=0.7, color="navy")
    ax.set_title(title)
    plt.tight_layout()
    return fig


def attractor_plot(trajectory: np.ndarray, title: str = "Attractor"):
    if trajectory.shape[1] >= 3:
        return state_space_3d(trajectory, title)
    return phase_portrait_2d(trajectory[:, :2], title)
