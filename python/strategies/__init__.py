# python/strategies/__init__.py
"""
Strategy execution and backtesting framework.
"""

from .executor import StrategyExecutor, StrategyConfig
from .definitions import AVAILABLE_STRATEGIES
from .adaptive_strategies import (
    HMMRegimeStrategy,
    RegimeConfig,
    MeanReversionRegimeStrategy,
    MomentumRegimeStrategy,
    PairTradingRegimeStrategy
)
from .meanrev import MeanReversionStrategy
from .sparse_meanrev import (
    sparse_portfolio_selection,
    box_tao_decomposition,
    box_tao_decomposition_rust
)

__all__ = [
    "StrategyExecutor", 
    "StrategyConfig", 
    "AVAILABLE_STRATEGIES",
    'HMMRegimeStrategy',
    'RegimeConfig',
    'MeanReversionRegimeStrategy',
    'MomentumRegimeStrategy',
    'PairTradingRegimeStrategy',
    'MeanReversionStrategy',
    'sparse_portfolio_selection',
    'box_tao_decomposition',
    'box_tao_decomposition_rust'
]
