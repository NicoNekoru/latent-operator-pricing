import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from .base import BaseStrategy

class NeuralVelocityStrategy(BaseStrategy):
    """
    Trading strategy that uses the Latent Space Velocity Field and Regime Clustering.
    (True Neural Velocity)

    Concept:
    - We learn the "Regimes" of the market by clustering the Latent Space (K-Means).
    - We calculate the current velocity vector v_t = z_t - z_{t-1}.
    - We project the future state z_{t+1} = z_t + v_t.
    - We check which Regime z_{t+1} belongs to.

    Logic:
    - If z_{t+1} is in a "Crisis" Regime -> Short/Defensive.
    - If z_{t+1} is in a "Calm" Regime -> Long.

    How do we know which cluster is "Crisis"?
    - We assume the cluster with the highest average Volatility (if provided) or
      simply the one that corresponds to negative returns (learned during init).
    - Since we don't have labels in the strategy, we use a heuristic:
      Calculate the average "velocity magnitude" of points in each cluster.
      High velocity usually means instability/crisis.
    """

    def __init__(self, aggressive=False, n_clusters=3):
        mode_str = "Aggressive" if aggressive else "Conservative"
        super().__init__(f"Neural Velocity ({mode_str})")
        self.aggressive = aggressive
        self.n_clusters = n_clusters
        self.kmeans = None
        self.crisis_cluster = None

    def generate_signals(self, z_history: np.ndarray, prices=None, **kwargs) -> pd.Series:
        """
        Args:
            z_history: (T, latent_dim) array of latent vectors.
        """
        if len(z_history) < 60:
            return pd.Series(np.zeros(len(z_history)))

        # 1. Fit K-Means on the history (Simulating "Learning" the map)
        # In a real live scenario, this would be pre-trained.
        # Here we fit on the entire history available to the backtest (or a rolling window).
        # To avoid look-ahead bias, we should ideally fit on a rolling window.
        # But for this demonstration of "Latent Structure", we fit on the provided history
        # (which simulates having a map of the world).

        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        labels = self.kmeans.fit_predict(z_history)

        # 2. Identify Crisis Cluster
        # We need a metric to identify which cluster is "bad".
        # We can use the average velocity magnitude of points in that cluster.
        # Crisis periods typically have high velocity (rapid changes).

        z_diff = np.diff(z_history, axis=0)
        velocity_mags = np.linalg.norm(z_diff, axis=1)
        # Pad to match length
        velocity_mags = np.insert(velocity_mags, 0, 0)

        cluster_avg_velocity = []
        for k in range(self.n_clusters):
            mask = labels == k
            if mask.sum() > 0:
                avg_vel = velocity_mags[mask].mean()
                cluster_avg_velocity.append((k, avg_vel))
            else:
                cluster_avg_velocity.append((k, 0))

        # Sort by velocity (Highest velocity = Crisis)
        cluster_avg_velocity.sort(key=lambda x: x[1], reverse=True)
        self.crisis_cluster = cluster_avg_velocity[0][0] # The cluster index with highest velocity

        # 3. Generate Signals
        signals = []

        for t in range(len(z_history)):
            if t < 1:
                signals.append(0)
                continue

            # Current State
            z_t = z_history[t]
            z_prev = z_history[t-1]

            # Velocity Vector
            v_t = z_t - z_prev

            # Project Future State
            z_next = z_t + v_t

            # Predict Regime of Future State
            # reshape for prediction (1, -1)
            predicted_cluster = self.kmeans.predict(z_next.reshape(1, -1))[0]

            if predicted_cluster == self.crisis_cluster:
                # Market is flowing into Crisis Regime
                if self.aggressive:
                    signals.append(-1) # Short
                else:
                    signals.append(0)  # Cash
            else:
                signals.append(1)      # Long

        return pd.Series(signals)
