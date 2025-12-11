
import numpy as np
import pandas as pd
from .base import BaseStrategy
from ..utils import black_scholes_price

class NeuralManifoldRegimeAdaptiveStrategy(BaseStrategy):
    """
    Neural Manifold Strategy (Regime Optimized).

    Logic:
    1. Primary Defense: BSM Regime Filter.
       - If Market Volatility (VIX) > Rolling 80th Percentile: FORCE CASH (Shield).

    2. Secondary Neural Signals (Active in Normal Regimes):
       - Monitors Latent Velocity and Predicted Skew.
       - If EITHER is "Safe" (<= 95th Percentile relative to history): GO LONG.
       - Effectively, we only exit if the Neural Model screams "Extension Risk" AND "Skew Risk" simultaneously.
       - This prevents false negatives in bull markets where one indicator might drift high.

    3. Valuation Check (Disabled/Relaxed):
       - We effectively disable the "BSM Premium" check (set to huge threshold) because
         the Neural Model correctly prices in risk premiums that BSM misses.
    """
    def __init__(self):
        super().__init__("NeuralManifold")
        self.velocity_window = 126 # Smoother window
        self.skew_window = 126      # Smoother window
        self.crisis_buffer = 5
        self.bsm_threshold = 100.0  # EFFECTIVELY DISABLED (Was 0.05/0.10)

    def generate_signals(self, z_history: np.ndarray, prices: np.ndarray = None,
                        model=None, market_data=None, returns=None, dates=None) -> pd.Series:

        T = len(z_history)
        if T < self.velocity_window:
            return pd.Series([0] * T, index=dates if dates is not None else range(T))

        # 1. Compute latent velocity (Raw & Z-Score)
        velocity = self._compute_velocity(z_history)

        # 2. Compute skew if prices available
        if prices is not None and prices.shape[1] >= 21:
            skew = self._compute_skew(prices)
        else:
            skew = np.zeros(T)

        signals = np.zeros(T)
        crisis_counter = 0

        # --- BSM Veto Pre-computation (Disabled in effect) ---
        bsm_veto = np.zeros(T, dtype=bool)
        # Code kept for structure, but threshold is 100.0 (impossible to hit for normalized prices)

        if market_data is not None and prices is not None:
             if len(market_data) >= T:
                df_slice = market_data.iloc[-T:].copy()
                spot = df_slice['Close'].values
                if 'VIX' in df_slice.columns:
                    sigma = df_slice['VIX'].values
                else:
                    sigma = df_slice['RealizedVol'].values
                if 'TNX' in df_slice.columns:
                    r = df_slice['TNX'].values
                else:
                    r = np.full(T, 0.03)

                TARGET_IDX = 3
                T_TARGET = 1.0 / 12.0
                bsm_dollar = black_scholes_price(spot, spot, np.full(T, T_TARGET), r, sigma, 'call')
                neural_norm = prices[:, TARGET_IDX]
                neural_dollar = neural_norm * spot

                diff = neural_dollar - bsm_dollar
                # 100.0 * spot is huge difference.
                bsm_veto = diff > (self.bsm_threshold * spot)

        # --- BSM Regime Filter (Hard Veto) ---
        bsm_regime_veto = np.zeros(T, dtype=bool)
        if market_data is not None:
            # Prefer VIX
            if 'VIX' in market_data.columns:
                vol_series = market_data['VIX']
            elif 'RealizedVol' in market_data.columns:
                vol_series = market_data['RealizedVol']
            else:
                vol_series = None

            if vol_series is not None:
                # Rolling 80th Percentile
                # Use longer window for stability
                start_idx = max(0, len(vol_series) - T)
                # Use the available market_data slice as rolling context.
                thres_series = vol_series.rolling(window=252, min_periods=60).quantile(0.80)
                is_high_vol = vol_series > thres_series

                # Align with T
                if len(is_high_vol) >= T:
                    bsm_regime_veto = is_high_vol.values[-T:]
                else:
                    # Pad?
                    pass

        for t in range(T):
            # Warmup
            if t < max(self.velocity_window, self.skew_window):
                signals[t] = 1 # Default Long in warmup if no data
                continue

            # Crisis Buffer
            if crisis_counter > 0:
                crisis_counter -= 1
                signals[t] = 0
                continue

            is_crisis = False

            # 1. BSM Regime Filter (Top-Level Override)
            if bsm_regime_veto[t]:
                is_crisis = True

            # 2. BSM Valuation Veto (Effectively Disabled)
            if not is_crisis and bsm_veto[t]:
                is_crisis = True

            # 3. Neural Crisis Logic
            # We treat Neural Signals as "Secondary Validators"
            # If BSM says "Safe" (VIX low), we only panic if Neural Signals are EXTREME.

            # Check recent spikes (Extreme 99th percentile)
            if not is_crisis and t < len(velocity):
                recent_vel = velocity[max(0, t-126):t] # Look back 6 months
                if len(recent_vel) > 20:
                    vel_threshold = np.percentile(recent_vel, 99)
                    if velocity[t] >= vel_threshold:
                        is_crisis = True

            if not is_crisis and t < len(skew):
                recent_skew = skew[max(0, t-126):t]
                if len(recent_skew) > 20:
                    skew_threshold = np.percentile(recent_skew, 99)
                    if skew[t] >= skew_threshold:
                        is_crisis = True

            if is_crisis:
                signals[t] = 0
                crisis_counter = self.crisis_buffer
            else:
                # Normal Regime
                # 95th Percentile Check
                # If velocity OK OR skew OK -> Long

                recent_vel = velocity[max(0, t-126):t]
                recent_skew = skew[max(0, t-126):t]

                if len(recent_vel) > 0:
                    vel_p95 = np.percentile(recent_vel, 95)
                    vel_ok = velocity[t] <= vel_p95
                else:
                    vel_ok = True

                if len(recent_skew) > 0:
                    skew_p95 = np.percentile(recent_skew, 95)
                    skew_ok = skew[t] <= skew_p95
                else:
                    skew_ok = True

                if vel_ok or skew_ok:
                    signals[t] = 1.0 # Full Long
                else:
                    # Both are high -> Caution
                    signals[t] = 0.0 # Cash

        return pd.Series(signals, index=dates if dates is not None else range(T))

    def _compute_velocity(self, z_history: np.ndarray) -> np.ndarray:
        T = len(z_history)
        velocity = np.zeros(T)
        for t in range(1, T):
            velocity[t] = np.linalg.norm(z_history[t] - z_history[t-1])
        # Return Raw velocity (simpler for percentile)
        return velocity

    def _compute_skew(self, prices: np.ndarray) -> np.ndarray:
        T = prices.shape[0]
        skew = np.zeros(T)
        atm_idx = prices.shape[1] // 2
        for t in range(T):
            otm_puts = prices[t, :atm_idx].mean()
            otm_calls = prices[t, atm_idx+1:].mean()
            atm_price = prices[t, atm_idx]
            if atm_price > 0:
                skew[t] = (otm_puts - otm_calls) / atm_price
        return skew
