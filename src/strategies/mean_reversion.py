from .base import BaseStrategy
import pandas as pd
import numpy as np

class MeanReversionStrategy(BaseStrategy):
    def __init__(self, z_score_threshold=2.0, window=60):
        super().__init__("Latent Mean Reversion")
        self.z_score_threshold = z_score_threshold
        self.window = window

    def generate_signals(self, z_history: np.ndarray, prices=None) -> pd.Series:
        vol_proxy = np.linalg.norm(z_history, axis=1)
        mean_vol = np.mean(vol_proxy)
        std_vol = np.std(vol_proxy)

        z_scores = (vol_proxy - mean_vol) / (std_vol + 1e-9)

        signals = []
        for z in z_scores:
            if z > self.z_score_threshold:
                # Extreme Vol -> Expect Reversion -> Long
                signals.append(1)
            elif z < -self.z_score_threshold:
                # Extreme Calm -> Expect Spike -> Cash
                signals.append(0)
            else:
                signals.append(1)

        return pd.Series(signals)
