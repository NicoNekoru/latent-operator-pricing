from .base import BaseStrategy
import pandas as pd
import numpy as np
from .base import BaseStrategy
import pandas as pd
import numpy as np

class MomentumStrategy(BaseStrategy):
    def __init__(self, lookback=5):
        super().__init__("Latent Momentum")
        self.lookback = lookback

    def generate_signals(self, z_history: np.ndarray, prices=None) -> pd.Series:
        vol_proxy = np.linalg.norm(z_history, axis=1)
        vol_series = pd.Series(vol_proxy)

        # Simple MA crossover on Volatility
        ma = vol_series.rolling(window=self.lookback).mean()

        signals = []
        for i in range(len(vol_series)):
            if i < self.lookback:
                signals.append(1)
                continue

            if vol_series[i] > ma[i]:
                # Volatility is rising -> Defensive
                signals.append(0)
            else:
                # Volatility is falling -> Long
                signals.append(1)

        return pd.Series(signals)
