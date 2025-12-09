"""
Custom exceptions for the trading system.

Provides clear error hierarchy for better error handling.
"""


class TradingError(Exception):
    """Base exception for trading system errors."""
    pass


class ValidationError(TradingError):
    """Raised when input validation fails."""
    pass


class DataError(TradingError):
    """Raised when there are data-related issues."""
    pass


class OptimizationError(TradingError):
    """Raised when optimization fails."""
    pass


class StrategyError(TradingError):
    """Raised when strategy execution fails."""
    pass


class ConnectorError(TradingError):
    """Raised when market connector fails."""
    pass
