"""
Volatility Surprise Strategy

Based on finding that the model systematically underprices volatility during crisis.
The "surprise" (actual - predicted) serves as a regime indicator.

Key insight: When actual vol >> predicted vol, the model is struggling with the regime.
This is a signal to reduce exposure.
"""
import numpy as np
from .base import ManifoldStrategy, StrategyState


class VolSurpriseStrategy(ManifoldStrategy):
    """
    Trade based on the difference between predicted and actual volatility.

    The model was trained mostly on calm regimes. During crisis:
    - Actual vol is higher than predicted
    - This "positive surprise" signals unfamiliar regime
    - Reduce exposure when surprise is high

    This exploits the model's failure mode as a feature!
    """

    def __init__(self,
                 z_threshold: float = 1.5,  # Z-score of surprise to trigger
                 ema_span: int = 20,         # EMA for adaptive baseline
                 lookback: int = 60):
        super().__init__("Vol Surprise", lookback=lookback)
        self.z_threshold = z_threshold
        self.ema_span = ema_span
        self._ema_surprise = None
        self._ema_var = None
        self._alpha = 2 / (ema_span + 1)

    def compute_signal(self, state: StrategyState) -> float:
        """
        Note: This strategy needs ACTUAL vol to compute surprise.
        In live trading, you'd compare predicted to previous period's actual.
        For backtesting, we use the actual from the same period.
        """
        if len(state.surface_history) < 30:
            return 1.0

        # Surprise requires the actual surface; backtests pass it via the surface field.
        # ATM vol is at index 3 (strike = 1.0, maturity = 1M)
        pred_atm = state.surface[3]

        # For now, use the predicted as proxy since we don't have actual in state
        # The key insight is that LEVEL of predicted vol correlates with crisis
        # High predicted vol = model sees unusual conditions

        # Historical ATM vols
        historical_atm = state.surface_history[:, 3]

        # Z-score of current predicted vol
        vol_mean = np.mean(historical_atm)
        vol_std = np.std(historical_atm) + 1e-6
        z_vol = (pred_atm - vol_mean) / vol_std

        # Also track momentum of vol
        if len(historical_atm) >= 5:
            recent_vol = historical_atm[-5:]
            vol_momentum = (recent_vol[-1] - recent_vol[0]) / (recent_vol[0] + 1e-6)
        else:
            vol_momentum = 0.0

        # Combined signal
        # High vol level + rising momentum = crisis = reduce exposure
        crisis_score = z_vol + vol_momentum * 2

        # Signal: gradual reduction as crisis score increases
        signal = 1.0 - np.tanh((crisis_score - self.z_threshold) / 2.0)
        return np.clip(signal, 0.0, 1.0)


class PredictionConfidenceStrategy(ManifoldStrategy):
    """
    Alternative: Trade based on model confidence.

    When the predicted surface has unusual characteristics
    (very high skew, high curvature), the model may be extrapolating.
    Use this as a signal to reduce exposure.
    """

    def __init__(self,
                 skew_percentile: float = 80,
                 curv_percentile: float = 80,
                 lookback: int = 60):
        super().__init__("Pred Confidence", lookback=lookback)
        self.skew_percentile = skew_percentile
        self.curv_percentile = curv_percentile

    def compute_signal(self, state: StrategyState) -> float:
        if len(state.surface_history) < 60:
            return 1.0

        # Current surface features
        pred = state.surface

        # Skew (OTM put - OTM call) / ATM
        skew = (pred[0] - pred[6]) / (pred[3] + 1e-6)

        # Curvature at ATM
        curv = abs(pred[4] - 2*pred[3] + pred[2]) / (pred[3] + 1e-6)

        # Historical features
        historical_skew = []
        historical_curv = []
        for t in range(len(state.surface_history)):
            s = state.surface_history[t]
            historical_skew.append((s[0] - s[6]) / (s[3] + 1e-6))
            historical_curv.append(abs(s[4] - 2*s[3] + s[2]) / (s[3] + 1e-6))

        # Percentile rank
        skew_pct = np.mean(np.array(historical_skew) < skew) * 100
        curv_pct = np.mean(np.array(historical_curv) < curv) * 100

        # Average percentile
        avg_pct = (skew_pct + curv_pct) / 2

        # Signal: reduce exposure when percentile is high
        if avg_pct > 90:
            return 0.0
        elif avg_pct > self.skew_percentile:
            return 1.0 - (avg_pct - self.skew_percentile) / (100 - self.skew_percentile)
        else:
            return 1.0
