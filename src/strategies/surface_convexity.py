"""
Surface Convexity Strategy

Uses the curvature of the predicted IV smile as a tail risk indicator.
High convexity = market pricing in tail events = reduce exposure.
"""
import numpy as np
from .base import ManifoldStrategy, StrategyState


class SurfaceConvexityStrategy(ManifoldStrategy):
    """
    Trade based on the convexity of the predicted volatility surface.

    Intuition:
    - The IV smile curvature reflects tail risk pricing
    - High convexity (steep smile) = market fears jumps
    - Low convexity (flat smile) = calm market expectations

    We focus on short-dated options (1M maturity) for sensitivity.

    Grid layout (21 points):
    - 3 maturities: [1M, 3M, 6M] (indices 0-6, 7-13, 14-20)
    - 7 strikes: [0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2]
    """

    def __init__(self,
                 convexity_threshold: float = 2.0,
                 sensitivity: float = 3.0,
                 lookback: int = 252):
        super().__init__("Surface Convexity", lookback=lookback)
        self.convexity_threshold = convexity_threshold
        self.sensitivity = sensitivity

        # Strike grid for computing curvature
        self.strikes = np.array([0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2])

    def compute_signal(self, state: StrategyState) -> float:
        if len(state.surface_history) < 60:
            return 1.0  # Default to long during warmup

        # Extract 1M slice (first 7 points, indices 0-6)
        current_smile = state.surface[:7]

        if len(current_smile) < 7:
            return 1.0

        # Compute second derivative (convexity) using finite differences
        # Focus on ATM region (indices 2-4 for strikes 0.95, 1.0, 1.05)
        atm_idx = 3  # Strike = 1.0

        # Second derivative at ATM
        d2_sigma = (current_smile[atm_idx + 1] - 2 * current_smile[atm_idx] +
                    current_smile[atm_idx - 1])

        # Normalize by ATM vol level
        atm_vol = current_smile[atm_idx] + 1e-6
        convexity = abs(d2_sigma) / atm_vol

        # Historical convexity for normalization
        historical_convexities = []
        for t in range(len(state.surface_history)):
            smile = state.surface_history[t, :7]
            if len(smile) >= 7:
                d2 = smile[atm_idx + 1] - 2 * smile[atm_idx] + smile[atm_idx - 1]
                atm = smile[atm_idx] + 1e-6
                historical_convexities.append(abs(d2) / atm)

        if len(historical_convexities) < 10:
            return 1.0

        convexity_mean = np.mean(historical_convexities)
        convexity_std = np.std(historical_convexities) + 1e-6

        # Z-score of current convexity
        z_convexity = (convexity - convexity_mean) / convexity_std

        # Signal: high convexity = defensive
        signal = 1.0 - np.tanh(self.sensitivity * (z_convexity - self.convexity_threshold))
        signal = np.clip(signal, 0.0, 1.0)

        return signal
