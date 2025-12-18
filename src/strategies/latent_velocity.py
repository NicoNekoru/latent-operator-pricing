"""
Latent Velocity Strategy

Tracks the rate of change in latent space to detect regime transitions.
Fast movement toward crisis region triggers defensive positioning.
"""
import numpy as np
from .base import ManifoldStrategy, StrategyState


class LatentVelocityStrategy(ManifoldStrategy):
    """
    Trade based on velocity in the latent manifold.

    Intuition:
    - Smooth market transitions = slow latent movement
    - Rapid regime shifts = fast latent movement (high "velocity")
    - We want to reduce exposure when velocity is high AND directed away from calm

    The strategy uses both:
    1. Speed (magnitude of velocity)
    2. Direction (toward/away from origin, which correlates with calm)
    """

    def __init__(self,
                 velocity_window: int = 5,
                 speed_threshold: float = 2.0,
                 lookback: int = 252):
        super().__init__("Latent Velocity", lookback=lookback)
        self.velocity_window = velocity_window
        self.speed_threshold = speed_threshold

    def compute_signal(self, state: StrategyState) -> float:
        if len(state.latent_history) < self.velocity_window + 10:
            return 1.0  # Default to long during warmup

        # Compute velocity (change over window)
        recent = state.latent_history[-self.velocity_window:]
        velocity = recent[-1] - recent[0]  # Direction vector
        speed = np.linalg.norm(velocity)

        # Normalize speed by historical volatility
        historical_speeds = []
        for i in range(self.velocity_window, len(state.latent_history)):
            v = state.latent_history[i] - state.latent_history[i - self.velocity_window]
            historical_speeds.append(np.linalg.norm(v))

        if len(historical_speeds) < 10:
            return 1.0

        speed_std = np.std(historical_speeds) + 1e-6
        normalized_speed = speed / speed_std

        # Direction: are we moving toward or away from origin?
        # Positive radial_velocity = moving away from origin (toward crisis)
        current_norm = np.linalg.norm(state.latent)
        prev_norm = np.linalg.norm(recent[0])
        radial_velocity = current_norm - prev_norm

        # Signal logic:
        # - High speed + moving away from origin = defensive (low signal)
        # - High speed + moving toward origin = could be recovery
        # - Low speed = stay the course

        # Crisis detector: high speed AND moving away from calm
        crisis_indicator = normalized_speed * max(0, radial_velocity)

        # Map to signal
        signal = 1.0 - np.tanh(crisis_indicator / self.speed_threshold)
        signal = np.clip(signal, 0.0, 1.0)

        return signal
