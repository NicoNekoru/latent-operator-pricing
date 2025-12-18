"""
Trading Strategies Module

Manifold-aware strategies that exploit the SpectralDeepONet's latent space geometry.
"""

from .base import ManifoldStrategy, StrategyState, BuyAndHold
from .regime_distance import RegimeDistanceStrategy
from .latent_velocity import LatentVelocityStrategy
from .surface_convexity import SurfaceConvexityStrategy
from .manifold_momentum import ManifoldMomentumStrategy, VolatilityRegimeStrategy, AdaptiveRiskStrategy
from .vol_surprise import VolSurpriseStrategy, PredictionConfidenceStrategy
from .legacy import LegacySurferStrategy, LegacySkewStrategy

__all__ = [
    'ManifoldStrategy',
    'StrategyState',
    'BuyAndHold',
    'RegimeDistanceStrategy',
    'LatentVelocityStrategy',
    'SurfaceConvexityStrategy',
    'ManifoldMomentumStrategy',
    'VolatilityRegimeStrategy',
    'AdaptiveRiskStrategy',
    'VolSurpriseStrategy',
    'PredictionConfidenceStrategy',
    'LegacySurferStrategy',
    'LegacySkewStrategy',
]
