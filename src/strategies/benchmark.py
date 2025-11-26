from .base import BaseStrategy
import pandas as pd
from .base import BaseStrategy
import pandas as pd
import numpy as np

class BenchmarkStrategy(BaseStrategy):
    def __init__(self, ticker="Benchmark"):
        super().__init__(f"Buy & Hold ({ticker})")

    def generate_signals(self, z_history: np.ndarray, prices=None, **kwargs) -> pd.Series:
        # Buy and Hold: Always 1
        return pd.Series(np.ones(len(z_history)))
