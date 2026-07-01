from .synthetic import (
    generate_spike_trains, generate_oscillatory_signal,
    generate_eeg_like, generate_connectivity_timeseries,
)
from .loaders import load_csv_timeseries, load_npy, BIDSLoader, NWBLoader
from .dataset_manager import DatasetManager

__all__ = [
    "generate_spike_trains", "generate_oscillatory_signal",
    "generate_eeg_like", "generate_connectivity_timeseries",
    "load_csv_timeseries", "load_npy", "BIDSLoader", "NWBLoader",
    "DatasetManager",
]
