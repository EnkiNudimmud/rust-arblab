"""
Core utilities and base classes for the trading system.

This module provides:
- Type definitions and protocols
- Base classes for strategies and models
- Common utility functions
- Type conversion helpers
"""

from .types import ArrayLike, to_numpy_array, validate_array
from .errors import TradingError, ValidationError, DataError
from .base import BaseStrategy, BaseModel, BaseOptimizer

__all__ = [
    'ArrayLike',
    'to_numpy_array',
    'validate_array',
    'TradingError',
    'ValidationError',
    'DataError',
    'BaseStrategy',
    'BaseModel',
    'BaseOptimizer',
]
