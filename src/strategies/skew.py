import numpy as np
import pandas as pd
from .base import BaseStrategy

class NeuralSkewStrategy(BaseStrategy):
    """
    Refined Neural Skew using Adaptive Z-Scoring.

    Logic:
    - Extracts the implied skew from the Neural Operator's predicted surface.
    - Instead of a hard threshold, computes the Rolling Z-Score of the skew.
    - Signals a crash when Skew spikes > X std devs above its recent baseline.
    - This adapts to different market regimes (e.g., low-skew bull markets vs high-skew bear markets).
    """

    def __init__(self, z_threshold=1.5, window=126):
        super().__init__("Neural Skew (Adaptive)")
        self.z_threshold = z_threshold
        self.window = window

    def generate_signals(self, z_history, prices=None, **kwargs):
        if prices is None:
            return pd.Series(np.ones(len(z_history)))

        T = len(prices)
        raw_skew = np.zeros(T)

        # 1. Vectorized Skew Calculation
        # Assume prices shape is (T, 21) where indices are sorted by strike
        otm_put = prices[:, 0]   # Deep OTM Put
        atm = prices[:, 10]      # ATM
        otm_call = prices[:, -1] # Deep OTM Call

        # Avoid div by zero
        atm = np.where(atm < 1e-4, 1e-4, atm)

        # Skew = (Put - Call) / ATM
        # Note: We want "Crash" skew, which is usually high Put prices relative to Calls.
        raw_skew = (otm_put - otm_call) / atm

        # 2. Adaptive Signal Generation
        signals = []
        skew_series = pd.Series(raw_skew)

        # Calculate rolling stats (shifted by 1 to avoid lookahead bias!)
        rolling_mean = skew_series.rolling(window=self.window).mean().shift(1)
        rolling_std = skew_series.rolling(window=self.window).std().shift(1)

        # Z-Score
        z_scores = (skew_series - rolling_mean) / (rolling_std + 1e-9)

        for t in range(T):
            if t < self.window:
                signals.append(1)
                continue

            # If Skew is statistically significant spike
            if z_scores[t] > self.z_threshold:
                signals.append(0) # Cash
            else:
                # Optional Hysteresis: Stay out if skew is still moderately high
                if len(signals) > 0 and signals[-1] == 0 and z_scores[t] > 0.0:
                    signals.append(0)
                else:
                    signals.append(1)

        return pd.Series(signals)