import numpy as np
import pandas as pd
from .base import BaseStrategy

class NeuralProjectionStrategy(BaseStrategy):
    """
    Trading strategy that uses the Latent Space Velocity Field to project future market states.
    (Naive Pricing / Projection)

    Concept:
    - We calculate the current velocity vector v_t = z_t - z_{t-1}.
    - We project the future state z_{t+1} = z_t + v_t.
    - We decode z_{t+1} (using the model passed via backtest) to get Predicted Prices.
    - We compare Predicted Volatility (via ATM Put Price) to Current Volatility.

    Logic:
    - If Predicted Vol > Current Vol: Market is moving towards instability -> Defensive/Short.
    - If Predicted Vol < Current Vol: Market is calming -> Long.

    Modes:
    - Aggressive=False: Long (1) or Cash (0).
    - Aggressive=True: Long (1) or Short (-1).
    """

    def __init__(self, aggressive=False, lookahead=1):
        mode_str = "Aggressive" if aggressive else "Conservative"
        super().__init__(f"Neural Projection ({mode_str})")
        self.aggressive = aggressive
        self.lookahead = lookahead

    def generate_signals(self, z_history: np.ndarray, prices=None, model=None) -> pd.Series:
        """
        Args:
            z_history: (T, latent_dim) array of latent vectors.
            prices: (T, 21) array of current decoded prices.
            model: The trained NeuralOperator model (required for projection).
        """
        if len(z_history) < 2 or model is None:
            return pd.Series(np.zeros(len(z_history)))

        signals = []

        # We need torch to use the model decoder
        import torch
        device = next(model.parameters()).device

        for t in range(len(z_history)):
            if t < 1:
                signals.append(0)
                continue

            # 1. Calculate Velocity
            z_t = z_history[t]
            z_prev = z_history[t-1]
            velocity = z_t - z_prev

            # 2. Project Future State
            # z_{t+1} = z_t + velocity * lookahead
            z_next = z_t + velocity * self.lookahead

            # 3. Decode Future State
            # We need to reshape for the decoder: (Batch, Latent)
            z_next_tensor = torch.tensor(z_next.reshape(1, -1), dtype=torch.float32).to(device)

            with torch.no_grad():
                # model.decoder takes z and returns prices
                pred_prices_tensor = model.decoder(z_next_tensor)
                pred_prices = pred_prices_tensor.cpu().numpy().flatten()

            # 4. Compare Volatility
            # We use ATM Put (Index 10) as a proxy for Volatility/Risk
            current_atm_put = prices[t][10]
            pred_atm_put = pred_prices[10]

            # 5. Generate Signal
            # If predicted risk is higher, we want to be out or short
            if pred_atm_put > current_atm_put:
                if self.aggressive:
                    signals.append(-1) # Short
                else:
                    signals.append(0)  # Cash
            else:
                signals.append(1)      # Long

        return pd.Series(signals)
