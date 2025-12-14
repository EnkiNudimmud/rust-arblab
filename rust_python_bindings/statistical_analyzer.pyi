# Type stubs for hft_py.statistical_analyzer module
from typing import Any, Dict, List, Optional

class StatisticalAnalyzer:
    """
    High-performance statistical analyzer implemented in Rust.
    Provides parallel processing for correlation analysis, cointegration testing,
    and mean-reverting basket construction.
    """
    def __init__(self) -> None:
        """Initialize the statistical analyzer."""
        ...
    
    def filter_valid_symbols(self, data: Dict[str, Any]) -> Any:
        """Filter symbols with sufficient data quality."""
        ...
    
    def compute_correlation_matrix(self, data: Dict[str, Any]) -> Any:
        """Compute correlation matrix for all symbols."""
        ...
    
    def find_cointegrated_pairs(
        self,
        data: Dict[str, Any],
        correlation_matrix: List[List[float]],
        threshold: float = 0.6
    ) -> List[CointegrationResult]:
        """Find cointegrated pairs using Engle-Granger test."""
        ...
    
    def build_mean_reverting_baskets(
        self,
        correlation_matrix: List[List[float]],
        threshold: float = 0.7
    ) -> List[BasketResult]:
        """Build mean-reverting baskets from highly correlated symbols."""
        ...
    
    def compute_volatility_rankings(self, data: Dict[str, Any]) -> List[VolatilityRanking]:
        """Compute volatility rankings for all symbols."""
        ...
    
    def analyze_portfolio_drift(
        self,
        data: Dict[str, Any],
        correlation_matrix: List[List[float]],
        threshold: float = 0.7
    ) -> List[Any]:
        """Analyze portfolio drift and stability."""
        ...

class CointegrationResult:
    """Result of cointegration test between two symbols."""
    symbol1: str
    symbol2: str
    correlation: float
    cointegration_pvalue: float
    hedge_ratio: float
    half_life: float

class BasketResult:
    """Result of basket construction."""
    symbols: List[str]
    weights: List[float]
    avg_correlation: float

class VolatilityRanking:
    """Volatility ranking for a symbol."""
    symbol: str
    volatility: float
    rank: int

class ProgressTracker:
    """Progress tracker for long-running operations."""
    def __init__(self, total: int, desc: str) -> None: ...
    def update(self, n: int = 1) -> None: ...
    def close(self) -> None: ...

__all__ = [
    'StatisticalAnalyzer',
    'CointegrationResult',
    'BasketResult',
    'VolatilityRanking',
    'ProgressTracker',
]
