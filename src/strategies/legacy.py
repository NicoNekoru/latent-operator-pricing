"""
Legacy strategy adapters for baseline comparison.

Wraps the old NeuralSurfer and NeuralSkew strategies to work with the new interface.
"""
import numpy as np
import pandas as pd
from .base import ManifoldStrategy, StrategyState


class LegacySurferStrategy(ManifoldStrategy):
    """
    Adapter for the original Neural Surfer strategy.
    Uses latent "energy" (squared norm) as crisis indicator.
    """

    def __init__(self, percentile: float = 80, lookback: int = 252):
        super().__init__("Legacy Surfer", lookback=lookback)
        self.percentile = percentile

    def compute_signal(self, state: StrategyState) -> float:
        if len(state.latent_history) < 60:
            return 1.0

        # Energy = squared norm of latent
        current_energy = np.sum(state.latent ** 2)
        historical_energy = np.sum(state.latent_history ** 2, axis=1)

        # Threshold = rolling percentile
        threshold = np.percentile(historical_energy, self.percentile)

        # Binary signal: cash if high energy
        return 0.0 if current_energy > threshold else 1.0


class LegacySkewStrategy(ManifoldStrategy):
    """
    Adapter for the original Neural Skew strategy.
    Uses put-call price difference as crash indicator.
    """

    def __init__(self, percentile: float = 95, lookback: int = 252):
        super().__init__("Legacy Skew", lookback=lookback)
        self.percentile = percentile

    def compute_signal(self, state: StrategyState) -> float:
        if len(state.surface_history) < 60:
            return 1.0

        # Compute skew from surface
        # Grid: 7 strikes per maturity, 3 maturities
        # 1M slice is indices 0-6
        otm_put = state.surface[0]   # 0.8 strike (deep OTM put)
        atm = state.surface[3]       # 1.0 strike
        otm_call = state.surface[6]  # 1.2 strike (deep OTM call)

        atm = max(atm, 1e-4)
        current_skew = (otm_put - otm_call) / atm

        # Historical skew
        historical_skew = []
        for t in range(len(state.surface_history)):
            s = state.surface_history[t]
            if len(s) >= 7:
                put = s[0]
                a = max(s[3], 1e-4)
                call = s[6]
                historical_skew.append((put - call) / a)

        if len(historical_skew) < 10:
            return 1.0

        threshold = np.percentile(historical_skew, self.percentile)

        # Binary signal: cash if high skew
        return 0.0 if current_skew > threshold else 1.0
