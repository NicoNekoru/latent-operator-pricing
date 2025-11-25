import numpy as np
import pandas as pd
from .base import BaseStrategy

class SkewStrategy(BaseStrategy):
    """
    Trading strategy based on the Implied Skew of the predicted option surface.

    Logic:
    - Calculate Skew = Price(OTM Put) - Price(OTM Call)
    - High Skew (Steep Smile) -> Market fears a crash -> Go Short/Defensive
    - Low Skew (Flat Smile) -> Market is complacent -> Go Long
    """

    def __init__(self, skew_threshold=0.05):
        super().__init__("Neural Skew")
        self.skew_threshold = skew_threshold

    def generate_signals(self, z_history, prices=None):
        """
        Generates trading signals based on Implied Skew.

        Args:
            z_history: Latent vectors (unused here, but kept for interface consistency)
            prices: Array of shape (T, 21) containing predicted option prices.
                    We assume:
                    - Index 0-9: Puts (Deep OTM to ATM)
                    - Index 10: ATM
                    - Index 11-20: Calls (ATM to Deep OTM)

                    Specifically, let's assume a symmetric grid.
                    - OTM Put: Index 0 (Lowest Strike)
                    - OTM Call: Index 20 (Highest Strike)

        Returns:
            pd.Series: Trading signals (-1, 0, 1)
        """
        if prices is None:
            # Fallback if no prices provided (shouldn't happen in updated backtest)
            return pd.Series(np.zeros(len(z_history)))

        signals = []

        for t in range(len(prices)):
            price_curve = prices[t]

            # Calculate Skew: Price of OTM Put vs OTM Call
            # Assuming standard ordering: Low Strike -> High Strike
            # OTM Put is at Low Strike (Index 0)
            # OTM Call is at High Strike (Index 20)

            otm_put_price = price_curve[0]
            otm_call_price = price_curve[-1]

            # Normalize by ATM price to get relative skew
            atm_price = price_curve[10] + 1e-9

            skew = (otm_put_price - otm_call_price) / atm_price

            if skew > self.skew_threshold:
                # Steep Skew -> Fear -> Short
                signals.append(-1)
            else:
                # Flat/Normal Skew -> Complacency -> Long
                signals.append(1)

        return pd.Series(signals)
