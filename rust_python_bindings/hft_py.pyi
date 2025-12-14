# Type stubs for hft_py module
from typing import Any, Dict, List, Tuple, Optional

class PyAggregator:
    def __init__(self) -> None: ...
    def subscribe(self, connector: str, symbol: str) -> None: ...
    def unsubscribe(self, connector: str, symbol: str) -> None: ...
    def stop_connector(self, connector: str) -> bool: ...

# Statistical Analyzer module
class statistical_analyzer:
    class StatisticalAnalyzer:
        def __init__(self) -> None: ...
        def filter_valid_symbols(self, data: Dict[str, Any]) -> Any: ...
        def compute_correlation_matrix(self, data: Dict[str, Any]) -> Any: ...
        def find_cointegrated_pairs(
            self, data: Dict[str, Any], correlation_matrix: List[List[float]], threshold: float = 0.6
        ) -> List[Any]: ...
        def build_mean_reverting_baskets(
            self, correlation_matrix: List[List[float]], threshold: float = 0.7
        ) -> List[Any]: ...
        def compute_volatility_rankings(self, data: Dict[str, Any]) -> List[Any]: ...
        def analyze_portfolio_drift(
            self, data: Dict[str, Any], correlation_matrix: List[List[float]], threshold: float = 0.7
        ) -> List[Any]: ...
    
    class CointegrationResult:
        symbol1: str
        symbol2: str
        correlation: float
        cointegration_pvalue: float
        hedge_ratio: float
        half_life: float
        
    class BasketResult:
        symbols: List[str]
        weights: List[float]
        avg_correlation: float
        
    class VolatilityRanking:
        symbol: str
        volatility: float
        rank: int
    
    class ProgressTracker:
        def __init__(self, total: int, desc: str) -> None: ...
        def update(self, n: int = 1) -> None: ...
        def close(self) -> None: ...

# Superspace module
class superspace:
    class PyGrassmannNumber:
        def __init__(self, real: float, epsilon: float) -> None: ...
        
    class PyGhostFieldParams:
        def __init__(self, mass: float, coupling: float, field_strength: float) -> None: ...
        
    class PyChernSimonsCalculator:
        def __init__(self, level: int) -> None: ...
        def compute_invariant(self, data: List[float]) -> float: ...
        
    class PyAnomalyDetector:
        def __init__(self, dimension: int, threshold: float) -> None: ...
        def detect_anomalies(self, data: List[List[float]]) -> List[Tuple[int, float]]: ...
        def get_anomaly_report(self) -> Dict[str, Any]: ...

# Signature module
class signature:
    def compute_signature(path: List[List[float]], depth: int) -> List[float]: ...
    def compute_log_signature(path: List[List[float]], depth: int) -> List[float]: ...
    def expected_signature(path: List[List[float]], depth: int) -> List[float]: ...
    def signature_kernel(path1: List[List[float]], path2: List[List[float]], depth: int) -> float: ...

# Rough Heston module
class rough_heston:
    class PyRoughHestonParams:
        def __init__(
            self,
            hurst: float,
            lambda_: float,
            nu: float,
            rho: float,
            v0: float,
            theta: float,
        ) -> None: ...
        
    class PyRoughHestonCharFunc:
        def __init__(self, params: PyRoughHestonParams) -> None: ...
        def characteristic_function(self, u: complex, tau: float) -> complex: ...

# Flat file module
class flat_file:
    class PyS3Config:
        def __init__(
            self,
            bucket: str,
            region: Optional[str] = None,
            access_key: Optional[str] = None,
            secret_key: Optional[str] = None,
        ) -> None: ...
    
    def download_and_process_flat_file(
        s3_config: PyS3Config,
        s3_key: str,
        local_output_path: str,
        chunk_size: int = 10000,
        compression: Optional[str] = None,
    ) -> Tuple[int, int, int, bool]: ...
    
    def process_local_flat_file(
        input_path: str,
        output_path: str,
        chunk_size: int = 10000,
        compression: Optional[str] = None,
    ) -> Tuple[int, int, int, bool]: ...

# Analytics module
class analytics:
    ...

# Options module
class options:
    ...

# Portfolio drift module
class portfolio_drift:
    ...
