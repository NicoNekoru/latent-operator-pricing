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

    def __init__(self, percentile=95, window=252):
        super().__init__("Neural Skew (Adaptive)")
        self.percentile = percentile
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

        skew_series = pd.Series(raw_skew)

        # 2. Threshold (Rolling 95th Percentile)
        # We want to catch the "Tail" events (Crises) where skew explodes.
        # We use a long window (252) to capture the "Yearly High" skew context.
        thresholds = skew_series.rolling(window=self.window, min_periods=60).quantile(self.percentile / 100.0)
        thresholds = thresholds.fillna(np.inf)

        signals = np.ones(T)

        # 3. Signal
        # If Skew > Threshold -> Cash
        is_high_skew = skew_series > thresholds

        signals[is_high_skew] = 0

        return pd.Series(signals)