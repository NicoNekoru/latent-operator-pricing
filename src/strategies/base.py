"""
Manifold-aware trading strategies base class.

These strategies leverage the latent space geometry learned by the SpectralDeepONet.
"""
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class StrategyState:
    """Container for strategy inputs at each timestep."""
    latent: np.ndarray           # Current latent vector (d,)
    latent_history: np.ndarray   # Historical latents (T, d)
    surface: np.ndarray          # Current IV surface (21,)
    surface_history: np.ndarray  # Historical surfaces (T, 21)
    market_return: float         # Current period return
    date: pd.Timestamp           # Current date


class ManifoldStrategy(ABC):
    """
    Base class for strategies that exploit the learned manifold geometry.

    The SpectralDeepONet learns a latent space where:
    - Calm regimes cluster at low latent norms (~0.2)
    - Crisis regimes have higher latent norms (~0.4)
    - Transitions are smooth (continuous manifold)

    Strategies should exploit this geometry for regime-aware trading.
    """

    def __init__(self, name: str, lookback: int = 252):
        self.name = name
        self.lookback = lookback
        self._signals = []
        self._metrics_cache = {}

    @abstractmethod
    def compute_signal(self, state: StrategyState) -> float:
        """
        Compute trading signal for current state.

        Args:
            state: StrategyState containing current and historical data

        Returns:
            Signal in [-1, 1]:
                1.0 = Full long
                0.0 = Cash/neutral
               -1.0 = Full short
        """
        pass

    def generate_signals(self,
                         latent_history: np.ndarray,
                         surface_history: np.ndarray,
                         returns: np.ndarray,
                         dates: pd.DatetimeIndex) -> pd.Series:
        """
        Generate signals for entire history.

        Args:
            latent_history: (T, d) array of latent vectors
            surface_history: (T, 21) array of IV surfaces
            returns: (T,) array of market returns
            dates: DatetimeIndex of length T

        Returns:
            pd.Series of signals indexed by date
        """
        T = len(latent_history)
        signals = np.zeros(T)

        for t in range(self.lookback, T):
            state = StrategyState(
                latent=latent_history[t],
                latent_history=latent_history[max(0, t-self.lookback):t],
                surface=surface_history[t],
                surface_history=surface_history[max(0, t-self.lookback):t],
                market_return=returns[t] if t < len(returns) else 0.0,
                date=dates[t]
            )
            signals[t] = self.compute_signal(state)

        self._signals = signals
        return pd.Series(signals, index=dates)

    def compute_metrics(self, signals: pd.Series, returns: np.ndarray) -> Dict:
        """Compute performance metrics."""
        # Lag signals by 1 period (trade next day)
        strategy_returns = signals.shift(1).fillna(0).values * returns

        if len(strategy_returns) == 0 or np.std(strategy_returns) < 1e-9:
            return {
                'sharpe': 0.0,
                'cum_return': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'volatility': 0.0,
                'strategy_returns': strategy_returns
            }

        sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
        cum_return = np.prod(1 + strategy_returns) - 1

        # Max drawdown
        cumulative = np.cumprod(1 + strategy_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / (running_max + 1e-9)
        max_drawdown = np.min(drawdown)

        # Win rate
        win_rate = np.mean(strategy_returns > 0)

        return {
            'sharpe': sharpe,
            'cum_return': cum_return,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'volatility': np.std(strategy_returns) * np.sqrt(252),
            'strategy_returns': strategy_returns
        }


class BuyAndHold(ManifoldStrategy):
    """Simple buy-and-hold baseline."""

    def __init__(self):
        super().__init__("Buy & Hold", lookback=0)

    def compute_signal(self, state: StrategyState) -> float:
        return 1.0