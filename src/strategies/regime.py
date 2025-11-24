from .base import BaseStrategy
import pandas as pd
import numpy as np

class RegimeStrategy(BaseStrategy):
    def __init__(self, threshold_percentile=80):
        super().__init__("Latent Regime Filter")
        self.threshold_percentile = threshold_percentile

    def generate_signals(self, z_history: np.ndarray) -> pd.Series:
        # Calculate Latent Norm (Volatility Proxy)
        latent_norm = np.linalg.norm(z_history, axis=1)

        # Determine Threshold (e.g., 80th percentile of history)
        # In a real scenario, this should be a rolling window to avoid lookahead bias.
        # For this benchmark, we'll use a rolling window of 30 days.

        signals = []
        window = 30

        for i in range(len(latent_norm)):
            if i < window:
                signals.append(1) # Default Long
                continue

            past_norms = latent_norm[i-window:i]
            threshold = np.percentile(past_norms, self.threshold_percentile)

            current_vol = latent_norm[i]

            if current_vol > threshold:
                signals.append(0) # Cash (High Vol Regime)
            else:
                signals.append(1) # Long (Safe Regime)

        return pd.Series(signals)
