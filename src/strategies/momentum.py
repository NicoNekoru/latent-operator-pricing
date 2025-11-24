from .base import BaseStrategy
import pandas as pd
import numpy as np

class MomentumStrategy(BaseStrategy):
    def __init__(self, lookback=5):
        super().__init__("Latent Momentum")
        self.lookback = lookback

    def generate_signals(self, z_history: np.ndarray) -> pd.Series:
        # Calculate Latent Norm
        latent_norm = np.linalg.norm(z_history, axis=1)

        signals = []

        for i in range(len(latent_norm)):
            if i < self.lookback:
                signals.append(1)
                continue

            current_vol = latent_norm[i]
            past_vol = latent_norm[i-self.lookback]

            # Change in Volatility
            delta_vol = current_vol - past_vol

            if delta_vol > 0:
                # Volatility is increasing -> Panic -> Cash
                signals.append(0)
            else:
                # Volatility is decreasing -> Calming -> Long
                signals.append(1)

        return pd.Series(signals)
