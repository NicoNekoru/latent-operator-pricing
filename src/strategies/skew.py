import numpy as np
import pandas as pd
from .base import BaseStrategy

class NeuralSkewStrategy(BaseStrategy):
    """
    Trading strategy based on the 'Neural Skew' of the predicted option surface.

    Logic:
    - The Neural Operator predicts the option price surface (21 points) from market history.
    - We calculate the Implied Skew: (Price_OTM_Put - Price_OTM_Call) / Price_ATM.
    - High Skew -> The model predicts a 'crash-phobic' pricing structure -> Go Defensive.
    - Low Skew -> The model predicts a relaxed pricing structure -> Go Long.

    This strategy directly utilizes the high-dimensional output of the Neural Operator.
    """

    def __init__(self, skew_threshold=8.0):
        super().__init__("Neural Skew (Surface Structure)")
        self.skew_threshold = skew_threshold

    def generate_signals(self, z_history, prices=None, **kwargs):
        """
        Args:
            z_history: Unused.
            prices: (T, 21) array of predicted option prices.
        """
        if prices is None:
            return pd.Series(np.zeros(len(z_history)))

        signals = []

        for t in range(len(prices)):
            price_curve = prices[t]

            # Index 0: Deep OTM Put (Low Strike)
            # Index 10: ATM
            # Index 20: Deep OTM Call (High Strike)

            otm_put_price = price_curve[0]
            otm_call_price = price_curve[-1]
            atm_price = price_curve[10] + 1e-9

            # Skew Metric
            skew = (otm_put_price - otm_call_price) / atm_price

            if skew > self.skew_threshold:
                # High Skew -> Fear -> Cash
                signals.append(0)
            else:
                # Normal Skew -> Long
                signals.append(1)

        return pd.Series(signals)
