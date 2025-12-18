"""
Manifold Momentum Strategy

Instead of absolute distance from calm, this strategy uses:
1. Relative position: Z-score of current norm vs recent history
2. Momentum: Direction of movement in latent space
3. Market context: Combine with surface skew for confirmation

The key insight: the latent space captures regime information, but the
signal comes from CHANGES in the latent state, not absolute position.
"""
import numpy as np
from .base import ManifoldStrategy, StrategyState


class ManifoldMomentumStrategy(ManifoldStrategy):
    """
    Trade based on momentum and volatility in the latent manifold.

    Logic:
    1. Track rolling mean and std of latent norm
    2. Compute z-score: how unusual is current state?
    3. Combine with momentum: is the market getting MORE unusual?
    4. High z-score + increasing momentum = reduce exposure

    This should work even when train/val distributions overlap.
    """

    def __init__(self,
                 z_threshold: float = 1.5,   # Z-score threshold
                 momentum_window: int = 5,   # Window for momentum
                 lookback: int = 60):
        super().__init__("Manifold Momentum", lookback=lookback)
        self.z_threshold = z_threshold
        self.momentum_window = momentum_window

    def compute_signal(self, state: StrategyState) -> float:
        if len(state.latent_history) < self.lookback:
            return 1.0  # Default long during warmup

        # 1. Compute latent norms
        current_norm = np.linalg.norm(state.latent)
        historical_norms = np.linalg.norm(state.latent_history, axis=1)

        # 2. Z-score: how unusual is current state?
        norm_mean = np.mean(historical_norms)
        norm_std = np.std(historical_norms) + 1e-6
        z_score = (current_norm - norm_mean) / norm_std

        # 3. Momentum: rate of change in norm
        if len(historical_norms) >= self.momentum_window:
            recent_norms = historical_norms[-self.momentum_window:]
            momentum = (recent_norms[-1] - recent_norms[0]) / self.momentum_window
            momentum_z = momentum / norm_std  # Normalize momentum
        else:
            momentum_z = 0.0

        # 4. Combined signal
        # High z-score = unusual state (could be opportunity OR risk)
        # Positive momentum_z = getting more unusual = likely risk

        # Risk indicator: z-score weighted by momentum direction
        risk = z_score * (1 + np.tanh(momentum_z * 2))

        # Map to signal: high risk = low exposure
        # Threshold at z_threshold (e.g., z > 1.5 starts reducing exposure)
        signal = 1.0 - np.tanh((risk - self.z_threshold) / 2.0)
        signal = np.clip(signal, 0.0, 1.0)

        return signal


class VolatilityRegimeStrategy(ManifoldStrategy):
    """
    Simpler approach: trade based on surface volatility level.

    When predicted ATM vol is high relative to history, reduce exposure.
    This is the most direct use of the model's output.
    """

    def __init__(self,
                 vol_percentile: float = 75,  # Above this = reduce exposure
                 lookback: int = 60):
        super().__init__("Vol Regime", lookback=lookback)
        self.vol_percentile = vol_percentile

    def compute_signal(self, state: StrategyState) -> float:
        if len(state.surface_history) < 30:
            return 1.0

        # Extract ATM vol (strike index 3, 1M maturity)
        current_atm = state.surface[3]

        # Historical ATM vols
        historical_atm = state.surface_history[:, 3]

        # Percentile rank of current vol
        percentile = np.mean(historical_atm < current_atm) * 100

        # Signal: gradual reduction above threshold
        if percentile > self.vol_percentile:
            # Linear decay from 1.0 at percentile to 0.0 at 100th percentile
            signal = 1.0 - (percentile - self.vol_percentile) / (100 - self.vol_percentile)
        else:
            signal = 1.0

        return np.clip(signal, 0.0, 1.0)


class AdaptiveRiskStrategy(ManifoldStrategy):
    """
    Combines latent and surface signals with adaptive thresholds.

    Uses exponentially weighted moving averages for smoother signals.
    """

    def __init__(self,
                 ema_span: int = 20,
                 risk_threshold: float = 1.0,
                 lookback: int = 60):
        super().__init__("Adaptive Risk", lookback=lookback)
        self.ema_span = ema_span
        self.risk_threshold = risk_threshold
        self._ema_norm = None
        self._ema_var = None
        self._alpha = 2 / (ema_span + 1)

    def compute_signal(self, state: StrategyState) -> float:
        if len(state.latent_history) < 30:
            return 1.0

        current_norm = np.linalg.norm(state.latent)

        # Initialize EMAs
        if self._ema_norm is None:
            historical_norms = np.linalg.norm(state.latent_history, axis=1)
            self._ema_norm = np.mean(historical_norms)
            self._ema_var = np.var(historical_norms)

        # Update EMAs
        self._ema_norm = self._alpha * current_norm + (1 - self._alpha) * self._ema_norm
        deviation = (current_norm - self._ema_norm) ** 2
        self._ema_var = self._alpha * deviation + (1 - self._alpha) * self._ema_var

        # Current z-score relative to EMA
        ema_std = np.sqrt(self._ema_var + 1e-6)
        z = (current_norm - self._ema_norm) / ema_std

        # Also look at surface skew
        skew = (state.surface[0] - state.surface[6]) / (state.surface[3] + 1e-6)

        # Combined risk: high z-score OR high skew
        risk = max(z, skew * 2)  # Skew scaled to be comparable

        # Signal
        signal = 1.0 - np.tanh((risk - self.risk_threshold) / 1.5)
        return np.clip(signal, 0.0, 1.0)
