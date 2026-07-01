"""Data loaders for various neuroscience data formats."""
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def load_csv_timeseries(path: str, delimiter: str = ",") -> np.ndarray:
    return np.loadtxt(path, delimiter=delimiter)


def load_npy(path: str) -> np.ndarray:
    return np.load(path)


class BIDSLoader:
    """Stub BIDS loader — loads from directory structure."""
    def __init__(self, bids_root: str):
        self.root = Path(bids_root)

    def list_subjects(self):
        return [d.name for d in self.root.glob("sub-*") if d.is_dir()]

    def load_eeg(self, subject: str, task: str) -> Optional[np.ndarray]:
        # Stub: in real use, parse .vhdr or .edf files
        return None


class NWBLoader:
    """Stub NWB loader."""
    def __init__(self, filepath: str):
        self.filepath = filepath

    def load_spike_trains(self) -> Dict[str, Any]:
        return {}

    def load_lfp(self) -> Optional[np.ndarray]:
        return None


class AllenBrainAtlasLoader:
    """Stub Allen Brain Atlas loader (requires allensdk)."""
    def get_structure_names(self):
        return ["VISp", "VISl", "VISrl", "VISal", "VISpm", "VISam"]

    def get_connectivity_matrix(self) -> np.ndarray:
        n = 6
        rng = np.random.default_rng(0)
        W = rng.random((n, n))
        np.fill_diagonal(W, 0)
        return W
