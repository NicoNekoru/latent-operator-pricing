from .base import BaseStrategy
import pandas as pd
import numpy as np

class BenchmarkStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("Buy & Hold (S&P 500)")

    def generate_signals(self, z_history: np.ndarray) -> pd.Series:
        # Always Long
        return pd.Series(np.ones(len(z_history)))
