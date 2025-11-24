from .base import BaseStrategy
import pandas as pd
import numpy as np

class MeanReversionStrategy(BaseStrategy):
    def __init__(self, z_score_threshold=2.0, window=60):
        super().__init__("Latent Mean Reversion")
        self.z_score_threshold = z_score_threshold
        self.window = window

    def generate_signals(self, z_history: np.ndarray) -> pd.Series:
        latent_norm = np.linalg.norm(z_history, axis=1)

        signals = []

        for i in range(len(latent_norm)):
            if i < self.window:
                signals.append(1)
                continue

            # Rolling Z-Score
            window_data = latent_norm[i-self.window:i]
            mu = np.mean(window_data)
            sigma = np.std(window_data)

            if sigma == 0:
                z_score = 0
            else:
                z_score = (latent_norm[i] - mu) / sigma

            if z_score > self.z_score_threshold:
                # Extreme Volatility -> Oversold -> Buy the Dip
                signals.append(1)
            elif z_score < -1.0:
                # Very Low Volatility -> Complacency -> Risk of Spike -> Cash
                signals.append(0)
            else:
                # Normal Regime -> Long
                signals.append(1)

        return pd.Series(signals)
