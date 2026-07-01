"""Attention models."""
import numpy as np


class SpotlightAttention:
    """Spatial spotlight attention with Gaussian focus."""

    def __init__(self, n_locations: int = 20, sigma: float = 2.0):
        self.n = n_locations
        self.sigma = sigma
        self.focus = n_locations // 2

    def attend(self, stimuli: np.ndarray) -> np.ndarray:
        locs = np.arange(self.n)
        weights = np.exp(-0.5 * ((locs - self.focus) / self.sigma)**2)
        weights /= weights.sum()
        return stimuli * weights

    def shift(self, target: int) -> None:
        self.focus = int(np.clip(target, 0, self.n - 1))


class FeatureBasedAttention:
    """Feature-based attention modulating a feature map."""

    def __init__(self, n_features: int = 10):
        self.n = n_features
        self.weights = np.ones(n_features) / n_features

    def set_target_feature(self, feature_idx: int, gain: float = 2.0) -> None:
        self.weights = np.ones(self.n)
        self.weights[feature_idx] = gain
        self.weights /= self.weights.sum()

    def apply(self, feature_map: np.ndarray) -> np.ndarray:
        return feature_map * self.weights
