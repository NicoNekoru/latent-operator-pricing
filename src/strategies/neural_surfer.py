import numpy as np
import pandas as pd
from .base import BaseStrategy

class NeuralSurferStrategy(BaseStrategy):
    """
    Refined 'Neural Surfer' using Phase Space Energy.

    Physics-Informed Logic:
    - We treat the latent space as a physical system.
    - We define the 'Stable Equilibrium' as the global mean of the latent vectors (Z_mean).
    - We monitor 'Radial Velocity': Are we moving AWAY from equilibrium?
    - We monitor 'Total Energy': Kinetic (Velocity^2) + Potential (Distance^2).

    Signal:
    - Exit (Cash) when Energy is High AND Radial Velocity is positive (Ejection phase).
    - Re-enter when Energy dissipates.
    """

    def __init__(self, z_score_threshold=2.0, window=60):
        super().__init__("Neural Surfer (Phase Energy)")
        self.threshold = z_score_threshold
        self.window = window

    def generate_signals(self, z_history: np.ndarray, prices=None, **kwargs) -> pd.Series:
        # 1. Define Equilibrium (Global Mean of the training portion is best,
        # but here we use expanding mean to prevent lookahead bias)
        # For simplicity in backtest, we can use the running mean.

        T, D = z_history.shape
        signals = np.ones(T) # Default Long

        # Pre-compute velocity vectors
        z_vel = np.zeros_like(z_history)
        z_vel[1:] = z_history[1:] - z_history[:-1]

        # Calculate scalar Speed (Kinetic Energy proxy)
        speed = np.linalg.norm(z_vel, axis=1)

        # We need a warm-up period for the rolling stats
        warmup = 60

        for t in range(warmup, T):
            # 1. Estimate current Equilibrium (running mean of past history)
            # Using a sliding window to adapt to long-term regime shifts (e.g. 2010s vs 2020s)
            local_history = z_history[t-self.window:t]
            equilibrium = np.mean(local_history, axis=0)

            # 2. Current Position relative to Equilibrium
            displacement = z_history[t] - equilibrium
            dist = np.linalg.norm(displacement)

            # 3. Radial Velocity: Dot product of Velocity and Displacement direction
            # If > 0, we are moving AWAY from center. If < 0, we are reverting.
            if dist > 1e-6:
                radial_vel = np.dot(z_vel[t], displacement) / dist
            else:
                radial_vel = 0

            # 4. Construct the "Instability Score" (Energy)
            # We combine distance (potential) and speed (kinetic).
            # We normalize speed by local volatility to get a Z-score-like metric.
            local_speed_mu = np.mean(speed[t-self.window:t])
            local_speed_std = np.std(speed[t-self.window:t]) + 1e-9

            speed_z = (speed[t] - local_speed_mu) / local_speed_std

            # SIGNAL LOGIC:
            # Condition A: We are moving unusually fast (Speed Z > Threshold)
            # Condition B: We are moving AWAY from safety (Radial Vel > 0)
            # Condition C: We are already somewhat far out (Distance check - optional, implicit in Energy)

            is_ejection = (speed_z > self.threshold) and (radial_vel > 0)

            if is_ejection:
                signals[t] = 0 # CASH
            else:
                # Hysteresis: Don't re-enter immediately if we are still highly volatile
                if signals[t-1] == 0 and speed_z > (self.threshold * 0.5):
                    signals[t] = 0 # Stay in Cash
                else:
                    signals[t] = 1 # Long

        return pd.Series(signals)