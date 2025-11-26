import numpy as np
import pandas as pd
from .base import BaseStrategy

class NeuralSurferStrategy(BaseStrategy):
    """
    Trading strategy that 'surfs' the latent manifold of the Neural Operator.

    Logic:
    - The latent space Z captures the compressed state of market dynamics.
    - We monitor the 'Latent Velocity' (dZ/dt) - how fast the market state is changing.
    - High Velocity -> Regime Shift / Instability -> Go Defensive (Cash).
    - Low Velocity -> Stable Regime -> Go Long.
    """

    def __init__(self, velocity_threshold_percentile=80, window=5):
        super().__init__("Neural Surfer (Latent Velocity)")
        self.percentile = velocity_threshold_percentile
        self.window = window

    def generate_signals(self, z_history: np.ndarray, prices=None, **kwargs) -> pd.Series:
        """
        Args:
            z_history: (T, latent_dim) array of latent vectors.
        """
        # 1. Calculate Latent Velocity (Euclidean distance between consecutive Z states)
        # dZ = ||Z_t - Z_{t-1}||
        z_diff = np.diff(z_history, axis=0)
        velocity = np.linalg.norm(z_diff, axis=1)

        # Pad the first value to match length
        velocity = np.insert(velocity, 0, 0)

        # 2. Smooth Velocity
        vol_series = pd.Series(velocity).rolling(window=self.window).mean().fillna(0)

        # 3. Determine Dynamic Threshold (Rolling Percentile to adapt to long-term regimes)
        # We use a long window (e.g., 252 days) to define 'normal' velocity for that year
        thresholds = vol_series.rolling(window=252, min_periods=60).quantile(self.percentile / 100.0)

        signals = []
        for i in range(len(vol_series)):
            # If current velocity is higher than the 80th percentile of the past year -> Crisis Mode
            if vol_series[i] > thresholds.iloc[i]:
                signals.append(0) # Cash
            else:
                signals.append(1) # Long

        return pd.Series(signals)
