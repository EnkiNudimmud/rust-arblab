"""
Factory classes for creating strategies and data fetchers.

Implements the Factory pattern to simplify object creation and promote loose coupling.
"""

from typing import Dict, Any, Optional, Type
import logging

logger = logging.getLogger(__name__)


class StrategyFactory:
    """Factory for creating trading strategy instances."""
    
    _strategies: Dict[str, Type] = {}
    
    @classmethod
    def register(cls, name: str, strategy_class: Type) -> None:
        """Register a strategy class with a name."""
        cls._strategies[name] = strategy_class
        logger.info(f"Registered strategy: {name}")
    
    @classmethod
    def create(cls, name: str, **kwargs: Any):
        """Create a strategy instance by name."""
        if name not in cls._strategies:
            available = ', '.join(cls._strategies.keys())
            raise ValueError(
                f"Unknown strategy: {name}. "
                f"Available strategies: {available}"
            )
        
        strategy_class = cls._strategies[name]
        try:
            return strategy_class(**kwargs)
        except Exception as e:
            logger.error(f"Failed to create strategy {name}: {e}")
            raise
    
    @classmethod
    def list_strategies(cls) -> list[str]:
        """List all registered strategy names."""
        return list(cls._strategies.keys())


class DataFetcherFactory:
    """Factory for creating data fetcher instances."""
    
    _fetchers: Dict[str, Type] = {}
    
    @classmethod
    def register(cls, name: str, fetcher_class: Type) -> None:
        """Register a data fetcher class with a name."""
        cls._fetchers[name] = fetcher_class
        logger.info(f"Registered data fetcher: {name}")
    
    @classmethod
    def create(cls, name: str, api_key: Optional[str] = None, **kwargs: Any):
        """Create a data fetcher instance by name."""
        if name not in cls._fetchers:
            available = ', '.join(cls._fetchers.keys())
            raise ValueError(
                f"Unknown data fetcher: {name}. "
                f"Available fetchers: {available}"
            )
        
        fetcher_class = cls._fetchers[name]
        try:
            if api_key:
                return fetcher_class(api_key=api_key, **kwargs)
            return fetcher_class(**kwargs)
        except Exception as e:
            logger.error(f"Failed to create data fetcher {name}: {e}")
            raise
    
    @classmethod
    def list_fetchers(cls) -> list[str]:
        """List all registered data fetcher names."""
        return list(cls._fetchers.keys())


# Auto-register strategies
def _register_strategies():
    """Auto-register all available strategies."""
    try:
        from python.strategies.adaptive_strategies import (
            AdaptiveStrategy,
            AdaptiveMeanReversion,
            AdaptiveMomentum,
            AdaptiveStatArb
        )
        
        StrategyFactory.register('adaptive', AdaptiveStrategy)
        StrategyFactory.register('adaptive_meanrev', AdaptiveMeanReversion)
        StrategyFactory.register('adaptive_momentum', AdaptiveMomentum)
        StrategyFactory.register('adaptive_statarb', AdaptiveStatArb)
    except ImportError as e:
        logger.warning(f"Could not auto-register strategies: {e}")


# Auto-register data fetchers
def _register_fetchers():
    """Auto-register all available data fetchers."""
    # Data fetchers are module-based with functions, not classes
    # Factory pattern works best with classes, so this is left as placeholder
    # for future class-based fetcher implementations
    pass


# Auto-register on module import
_register_strategies()
_register_fetchers()
