# python/strategies/__init__.py
"""
Strategy execution and backtesting framework.
"""

from .executor import StrategyExecutor, StrategyConfig
from .definitions import AVAILABLE_STRATEGIES
from .adaptive_strategies import (
    RegimeConfig,
    RegimeStats,
    AdaptiveStrategy,
    AdaptiveMeanReversion,
    AdaptiveMomentum,
    AdaptiveStatArb
)
# from .meanrev import MeanReversionStrategy  # meanrev.py contains only functions, not classes
from .sparse_meanrev import (
    sparse_pca,
    box_tao_decomposition,
    hurst_exponent,
    sparse_cointegration,
    generate_sparse_meanrev_signals,
    compute_risk_metrics
)

# Aliases for backward compatibility
HMMRegimeStrategy = AdaptiveStrategy
MeanReversionRegimeStrategy = AdaptiveMeanReversion
MomentumRegimeStrategy = AdaptiveMomentum
PairTradingRegimeStrategy = AdaptiveStatArb

__all__ = [
    "StrategyExecutor", 
    "StrategyConfig", 
    "AVAILABLE_STRATEGIES",
    'RegimeConfig',
    'RegimeStats',
    'AdaptiveStrategy',
    'AdaptiveMeanReversion',
    'AdaptiveMomentum',
    'AdaptiveStatArb',
    'HMMRegimeStrategy',
    'MeanReversionRegimeStrategy',
    'MomentumRegimeStrategy',
    'PairTradingRegimeStrategy',
    'sparse_pca',
    'box_tao_decomposition',
    'hurst_exponent',
    'sparse_cointegration',
    'generate_sparse_meanrev_signals',
    'compute_risk_metrics'
]
