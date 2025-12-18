"""
Regime Distance Strategy

Uses the latent space geometry to measure "distance from calm".
The SpectralDeepONet learns that calm regimes have lower latent norms.
"""
import numpy as np
from .base import ManifoldStrategy, StrategyState


class RegimeDistanceStrategy(ManifoldStrategy):
    """
    Trade based on distance from the "calm" regime centroid in latent space.

    Intuition:
    - Compute rolling centroid of latent vectors during stable periods
    - Current distance from centroid indicates crisis severity
    - High distance -> reduce exposure (go to cash)
    - Low distance -> full exposure (stay long)

    The signal is continuous, allowing gradual de-risking as markets destabilize.
    """

    def __init__(self,
                 distance_threshold: float = 0.3,
                 sensitivity: float = 5.0,
                 lookback: int = 252):
        super().__init__("Regime Distance", lookback=lookback)
        self.distance_threshold = distance_threshold
        self.sensitivity = sensitivity
        self._calm_centroid = None

    def compute_signal(self, state: StrategyState) -> float:
        if len(state.latent_history) < 60:
            return 1.0  # Default to long during warmup

        # Compute calm centroid from history
        # Use samples with below-median latent norm as "calm"
        norms = np.linalg.norm(state.latent_history, axis=1)
        median_norm = np.median(norms)
        calm_mask = norms < median_norm

        if calm_mask.sum() < 10:
            calm_centroid = np.mean(state.latent_history, axis=0)
        else:
            calm_centroid = np.mean(state.latent_history[calm_mask], axis=0)

        # Current distance from calm
        current_distance = np.linalg.norm(state.latent - calm_centroid)

        # Normalize by historical scale
        historical_distances = np.linalg.norm(
            state.latent_history - calm_centroid, axis=1
        )
        distance_std = np.std(historical_distances) + 1e-6
        normalized_distance = current_distance / distance_std

        # Signal: tanh mapping from distance to position
        # High distance -> signal approaches 0 (cash)
        # Low distance -> signal approaches 1 (long)
        signal = 1.0 - np.tanh(self.sensitivity * (normalized_distance - self.distance_threshold))
        signal = np.clip(signal, 0.0, 1.0)

        return signal
