import numpy as np
import pandas as pd
from .base import BaseStrategy
from ..heston_utils import calibrate_heston_to_surface

class HestonRegimeStrategy(BaseStrategy):
    """
    Calibrated Heston Strategy (Physical Regime Detection).

    Logic:
    - Daily Calibration: Fits Heston Parameters (kappa, theta, sigma, rho, v0)
      to the option price surface predicted by the neural model.
    - Physical Signal: Compares Spot Variance (v0) to Long-Run Variance (theta).
    - If v0 > theta * 1.0 (Current Vol > Equilibrium Vol), classify as high/unstable and move to cash.
    - Otherwise, classify as mean reverting/stable and stay long.

    This avoids arbitrary rolling windows by using the asset's own intrinsic equilibrium level (theta).
    """

    def __init__(self):
        super().__init__("Heston Regime (Calibrated)")
        # Reconstruct Grid (must match models.get_standard_grid)
        self.maturities = [1/12, 3/12, 6/12]
        self.moneyness = [0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2]

        # Flattened Grid definition
        self.grid_k = []
        self.grid_t = []
        for m in self.maturities:
            for k in self.moneyness:
                self.grid_k.append(k)
                self.grid_t.append(m)

        self.grid_k = np.array(self.grid_k)
        self.grid_t = np.array(self.grid_t)

        # Cache for calibration warm start
        # [kappa, theta, sigma, rho, v0]
        self.last_params = [2.0, 0.04, 0.3, -0.7, 0.04]

    def generate_signals(self, z_history, prices=None, **kwargs):
        T = len(z_history)
        signals = np.ones(T)

        if prices is None:
            return pd.Series(signals)

        # Prices shape: (T, 21). Calibrate every day so plots show the full path.

        params_history = []

        for t in range(T):
            # Get surface slice
            market_prices = prices[t] # (21,)

            # Calibrate
            # Warm start: use last params
            try:
                # We use a very fast calibration config (defined in utils options)
                # But here we pass the initial guess
                # Note: S0=1.0 because prices are normalized (Moneyness K/S)
                fitted = calibrate_heston_to_surface(
                    self.grid_k,
                    self.grid_t,
                    market_prices,
                    S0=1.0,
                    # The calibrator currently owns its initial guess.

                    # Add x0 support there if warm-start calibration becomes necessary.
                )

                # Check outcome
                # params: kappa, theta, sigma, rho, v0
                kappa, theta, vol_vol, rho, v0 = fitted

                # Signal Logic
                # If v0 (Current Variance) > theta (Long Run Variance)
                if v0 > theta:
                    signals[t] = 0.0 # Cash
                else:
                    signals[t] = 1.0 # Long

            except Exception as e:
                # If calibration fails, stay previous signal or Long
                # print(f"Calibration failed at {t}: {e}")
                signals[t] = 1.0

        return pd.Series(signals)
