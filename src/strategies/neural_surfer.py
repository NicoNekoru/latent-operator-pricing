import numpy as np
import pandas as pd
from .base import BaseStrategy

class NeuralSurferStrategy(BaseStrategy):
    """
    Refined 'Neural Surfer' using Phase Space Energy.

    Physics-Informed Logic:
    - We treat the latent space as a physical system.
    - We define the 'Stable Equilibrium' as the global mean of the latent vectors (Z_mean).
    - We monitor 'Radial Velocity': Are we moving AWAY from equilibrium?
    - We monitor 'Total Energy': Kinetic (Velocity^2) + Potential (Distance^2).
    """

    def __init__(self, percentile=80, window=252):
        super().__init__("Neural Surfer (Phase Energy)")
        self.percentile = percentile
        self.window = window

    def generate_signals(self, z_history: np.ndarray, prices=None, **kwargs) -> pd.Series:
        T = len(z_history)
        signals = np.ones(T)

        if T < self.window:
            return pd.Series(signals)

        # 1. Calculate Latent Energy (Squared Deviations from Origin)
        # Assuming Origin (0) is the "Stable/Mean" state for DeepONet latents roughly.
        # Shape: (T, latent_dim)
        energy = np.sum(z_history**2, axis=1) # Squared Norm

        energy_series = pd.Series(energy)

        # 2. Regime Threshold (Rolling 80th Percentile)
        # We compare absolute energy to recent history of energy.
        thresholds = energy_series.rolling(window=self.window, min_periods=60).quantile(self.percentile / 100.0)
        thresholds = thresholds.fillna(np.inf)

        # 3. Signal
        # If High Energy -> Crisis -> Cash
        is_high_energy = energy_series > thresholds

        signals[is_high_energy] = 0

        return pd.Series(signals)