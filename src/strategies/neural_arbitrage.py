import pandas as pd
import numpy as np
from .base import BaseStrategy


class NeuralArbitrageStrategy(BaseStrategy):
    """
    Trading strategy that exploits the divergence between the Neural Operator's
    physics-informed pricing and the standard Black-Scholes model.

    Concept:
    - The Neural Operator learns the "True" market physics (Heston/Stochastic Vol).
    - Black-Scholes assumes constant volatility and log-normal returns.
    - Divergence = P_Neural - P_BSM.

    Logic:
    - If P_Neural >> P_BSM: The Neural model detects hidden risk/value that BSM misses.
      This implies the market (if following BSM logic) is UNDERPRICING the option.
      However, if we are trading the UNDERLYING based on this info:
      - High Call Price -> Bullish Sentiment? OR
      - High Put Price -> Bearish Sentiment?

    Simplified Logic for Directional Trading:
    - We look at the "Neural Premium" on ATM Puts.
    - High Neural Put Premium -> Model predicts higher downside risk than BSM -> Go Defensive (Cash).
    - Low/Negative Neural Put Premium -> Model predicts lower risk -> Go Long.
    """

    def __init__(self, premium_threshold=8.0):
        super().__init__("Neural Arbitrage (BSM Divergence)")
        self.threshold = premium_threshold

    def generate_signals(self, z_history: np.ndarray, prices=None, **kwargs) -> pd.Series:
        """
        Args:
            z_history: Unused.
            prices: (T, 21) array of predicted option prices.
            kwargs: Must contain 'model' and ideally 'features' or 'market_data' if available.
                    However, since we don't have easy access to S/K/r/sigma here without refactoring backtest,
                    we will use the 'Neural Put Premium' logic but make it more robust.

        Refined Logic:
        - We assume the Neural Model captures "True" physics (fat tails, stochastic vol).
        - BSM assumes normal distribution.
        - If Neural ATM Put > BSM ATM Put, the market is pricing in higher risk than a random walk implies.

        Since we can't easily compute BSM without exact inputs (S, K, r, T, sigma), we will use a
        simplified proxy for "BSM-like" pricing:
        - BSM pricing for ATM options is roughly 0.4 * sigma * sqrt(T).
        - We can estimate 'sigma' from the past volatility (which the model sees).
        - But wait! The model output IS the price.

        Let's stick to the "Tail Risk Premium" as a proxy for "Non-BSM" behavior,
        as BSM tails are thin.

        New Logic:
        - Calculate "Implied Kurtosis" or "Fat Tail Ratio".
        - Ratio = Price(Deep OTM Put) / Price(ATM Put).
        - In BSM, this ratio is very small.
        - In Neural/Heston, it's larger.
        - If Ratio > Threshold -> High Tail Risk -> Defensive.
        """
        if prices is None:
            return pd.Series(np.zeros(len(z_history)))

        signals = []

        for t in range(len(prices)):
            price_curve = prices[t]

            # ATM Put (Index 10)
            atm_put_neural = price_curve[10]

            # Deep OTM Put (Index 0) - The "Crash" Put
            otm_put_neural = price_curve[0]

            # Ratio of OTM/ATM Put Price
            # This captures the "Fatness" of the left tail.
            # High ratio = Market fears a crash more than usual.
            tail_ratio = otm_put_neural / (atm_put_neural + 1e-9)

            # If the tail ratio is high, the Neural model is pricing in a crash.
            # We lower the threshold slightly to be more sensitive.
            if tail_ratio > self.threshold:
                signals.append(0) # Cash
            else:
                signals.append(1) # Long

        return pd.Series(signals)
