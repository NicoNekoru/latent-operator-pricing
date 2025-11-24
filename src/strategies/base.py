from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class BaseStrategy(ABC):
    def __init__(self, name: str):
        self._name = name

    @property
    def name(self):
        return self._name

    @abstractmethod
    def generate_signals(self, z_history: np.ndarray) -> pd.Series:
        """
        Generate trading signals based on latent space history.

        Args:
            z_history (np.ndarray): Array of shape (T, latent_dim) containing latent vectors.

        Returns:
            pd.Series: Series of binary signals (1 for Long, 0 for Cash) indexed by time.
        """
        pass
