from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Optional
import torch

class BaseStrategy(ABC):
    def __init__(self, name: str):
        self._name = name
        self.signals_history = []

    @property
    def name(self):
        return self._name

    @abstractmethod
    def generate_signals(
        self,
        z_history: np.ndarray,
        prices: Optional[np.ndarray] = None,
        model: Optional[torch.nn.Module] = None,
        market_data: Optional[pd.DataFrame] = None,
        returns: Optional[np.ndarray] = None,
        dates: Optional[pd.DatetimeIndex] = None
    ) -> pd.Series:
        """
        Enhanced base class for trading strategies.

        Args:
            z_history (np.ndarray): Array of shape (T, latent_dim) containing latent vectors.
            prices (np.ndarray, optional): Array of shape (T, 21) containing predicted option prices.
            model (torch.nn.Module, optional): The trained model for additional computations.
            market_data (pd.DataFrame, optional): Raw market data.
            returns (np.ndarray, optional): Asset returns for the period.
            dates (pd.DatetimeIndex, optional): Dates for the signals.

        Returns:
            pd.Series: Series of signals (1 for Long, 0 for Cash, -1 for Short) indexed by time.
        """
        pass

    def get_performance_metrics(self, signals: pd.Series, returns: np.ndarray) -> Dict:
        """Calculate performance metrics for the strategy."""
        if len(signals) != len(returns):
            signals = signals.iloc[:len(returns)]

        strategy_returns = signals.shift(1).fillna(0).values * returns

        if len(strategy_returns) == 0:
            return {
                'sharpe': 0.0,
                'cum_return': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'volatility': 0.0
            }

        # Calculate metrics
        sharpe = np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-9) * np.sqrt(252)
        cum_return = np.prod(1 + strategy_returns) - 1

        # Max drawdown
        cumulative = np.cumprod(1 + strategy_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)

        # Win rate
        win_rate = np.mean(strategy_returns > 0) if len(strategy_returns) > 0 else 0

        metrics = {
            'sharpe': sharpe,
            'cum_return': cum_return,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'volatility': np.std(strategy_returns) * np.sqrt(252),
            'returns': strategy_returns
        }

        return metrics