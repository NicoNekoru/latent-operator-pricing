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
from .base import BaseStrategy
import pandas as pd
import numpy as np

class RegimeStrategy(BaseStrategy):
    def __init__(self, threshold_percentile=80):
        super().__init__("Latent Regime Filter")
        self.threshold_percentile = threshold_percentile

    def generate_signals(self, z_history: np.ndarray, prices=None) -> pd.Series:
        # Calculate volatility (norm of z)
        vol_proxy = np.linalg.norm(z_history, axis=1)
        threshold = np.percentile(vol_proxy, self.threshold_percentile)

        signals = []
        for v in vol_proxy:
            if v > threshold:
                signals.append(0) # Cash
            else:
                signals.append(1) # Long

        return pd.Series(signals)
