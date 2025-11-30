import numpy as np
import pandas as pd
from .base import BaseStrategy

class BSMVolStrategy(BaseStrategy):
    """
    Traditional Parametric Baseline Strategy.

    Logic:
    - Monitors Implied Volatility (VIX) or Realized Volatility.
    - If Volatility is High (> Threshold), assume "Crisis" -> Cash.
    - Else -> Long.

    This serves as the "Parametric" equivalent to the Neural Surfer.
    """

    def __init__(self, threshold_percentile=80, window=252):
        super().__init__("BSM Baseline (Vol Regime)")
        self.percentile = threshold_percentile
        self.window = window

    def generate_signals(self, z_history: np.ndarray, prices=None, **kwargs) -> pd.Series:
        """
        Args:
            z_history: Not used (this is a parametric baseline).
            prices: Not used.
            kwargs: Must contain 'market_data' with 'VIX' or 'RealizedVol'.
        """
        market_data = kwargs.get('market_data')
        if market_data is None:
            # Fallback to all ones if no data provided
            return pd.Series(np.ones(len(z_history)))

        # Prefer VIX, fallback to RealizedVol
        if 'VIX' in market_data.columns:
            vol_series = market_data['VIX']
        elif 'RealizedVol' in market_data.columns:
            vol_series = market_data['RealizedVol']
        else:
            return pd.Series(np.ones(len(z_history)))

        # Align length
        # The backtest passes sliced data, but we might need history for rolling.
        # Assuming market_data is the full dataframe aligned with z_history.
        if len(vol_series) != len(z_history):
            # Try to slice or pad?
            # Usually backtest passes the relevant slice.
            pass

        # Calculate Dynamic Threshold
        # We use a rolling window to define "High" relative to recent history
        thresholds = vol_series.rolling(window=self.window, min_periods=60).quantile(self.percentile / 100.0)

        signals = []
        for i in range(len(vol_series)):
            # If current Vol > Threshold -> Crisis -> Cash
            if vol_series.iloc[i] > thresholds.iloc[i]:
                signals.append(0)
            else:
                signals.append(1)

        # Align with z_history (which starts after start_idx)
        # We assume z_history corresponds to the end of the market_data
        if len(signals) > len(z_history):
            signals = signals[-len(z_history):]

        return pd.Series(signals)
