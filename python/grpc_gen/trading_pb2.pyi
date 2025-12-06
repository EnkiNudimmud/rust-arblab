"""Type stubs for trading_pb2 generated protobuf module."""

from typing import Any, List, Dict, Optional, Iterable
from google.protobuf.message import Message
from google.protobuf.internal.containers import RepeatedScalarFieldContainer, RepeatedCompositeFieldContainer

class MeanReversionRequest(Message):
    prices: RepeatedScalarFieldContainer[float]
    threshold: float
    lookback: int
    def __init__(self, prices: Any = ..., threshold: float = ..., lookback: int = ...) -> None: ...

class MeanReversionResponse(Message):
    signal: float
    zscore: float
    entry_signal: bool
    exit_signal: bool
    metrics: Dict[str, float]
    def __init__(self, signal: float = ..., zscore: float = ..., entry_signal: bool = ..., exit_signal: bool = ..., metrics: Optional[Dict[str, float]] = ...) -> None: ...

class PriceVector(Message):
    symbol: str
    prices: RepeatedScalarFieldContainer[float]
    def __init__(self, symbol: str = ..., prices: Optional[List[float]] = ...) -> None: ...

class PortfolioOptimizationRequest(Message):
    prices: RepeatedCompositeFieldContainer[PriceVector]
    method: str
    parameters: Dict[str, float]
    def __init__(self, prices: Optional[List[PriceVector]] = ..., method: str = ..., parameters: Optional[Dict[str, float]] = ...) -> None: ...

class PortfolioOptimizationResponse(Message):
    weights: RepeatedScalarFieldContainer[float]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    metrics: Dict[str, float]
    def __init__(self, weights: Optional[List[float]] = ..., expected_return: float = ..., volatility: float = ..., sharpe_ratio: float = ..., metrics: Optional[Dict[str, float]] = ...) -> None: ...

class RegimeDetectionRequest(Message):
    returns: RepeatedScalarFieldContainer[float]
    n_regimes: int
    n_iterations: int
    def __init__(self, returns: Any = ..., n_regimes: int = ..., n_iterations: int = ...) -> None: ...

class RegimeDetectionResponse(Message):
    current_regime: int
    regime_probs: RepeatedScalarFieldContainer[float]
    transition_matrix: RepeatedScalarFieldContainer[float]
    metrics: Dict[str, float]
    def __init__(self, current_regime: int = ..., regime_probs: Optional[List[float]] = ..., transition_matrix: Optional[List[float]] = ..., metrics: Optional[Dict[str, float]] = ...) -> None: ...

class StreamRequest(Message):
    symbols: RepeatedScalarFieldContainer[str]
    exchange: str
    interval_ms: int
    def __init__(self, symbols: Any = ..., exchange: str = ..., interval_ms: int = ...) -> None: ...

class MarketDataUpdate(Message):
    symbol: str
    price: float
    volume: float
    timestamp: int
    def __init__(self, symbol: str = ..., price: float = ..., volume: float = ..., timestamp: int = ...) -> None: ...

class OrderBookRequest(Message):
    symbol: str
    depth: int
    def __init__(self, symbol: str = ..., depth: int = ...) -> None: ...

class OrderBookResponse(Message):
    bids: RepeatedScalarFieldContainer[float]
    asks: RepeatedScalarFieldContainer[float]
    bid_volumes: RepeatedScalarFieldContainer[float]
    ask_volumes: RepeatedScalarFieldContainer[float]
    def __init__(self, bids: Optional[List[float]] = ..., asks: Optional[List[float]] = ..., bid_volumes: Optional[List[float]] = ..., ask_volumes: Optional[List[float]] = ...) -> None: ...

class HMMRequest(Message):
    observations: RepeatedScalarFieldContainer[float]
    n_states: int
    max_iterations: int
    tolerance: float
    def __init__(self, observations: Any = ..., n_states: int = ..., max_iterations: int = ..., tolerance: float = ...) -> None: ...

class HMMResponse(Message):
    states: RepeatedScalarFieldContainer[int]
    transition_matrix: RepeatedScalarFieldContainer[float]
    emission_means: RepeatedScalarFieldContainer[float]
    emission_stds: RepeatedScalarFieldContainer[float]
    log_likelihood: float
    def __init__(self, states: Optional[List[int]] = ..., transition_matrix: Optional[List[float]] = ..., emission_means: Optional[List[float]] = ..., emission_stds: Optional[List[float]] = ..., log_likelihood: float = ...) -> None: ...

class ParameterSpace(Message):
    name: str
    min: float
    max: float
    def __init__(self, name: str = ..., min: float = ..., max: float = ...) -> None: ...

class MCMCRequest(Message):
    parameters: RepeatedCompositeFieldContainer[ParameterSpace]
    n_iterations: int
    burn_in: int
    step_size: float
    def __init__(self, parameters: Any = ..., n_iterations: int = ..., burn_in: int = ..., step_size: float = ...) -> None: ...

class MCMCResponse(Message):
    samples: RepeatedScalarFieldContainer[float]
    acceptance_rate: float
    best_params: RepeatedScalarFieldContainer[float]
    best_value: float
    def __init__(self, samples: Optional[List[float]] = ..., acceptance_rate: float = ..., best_params: Optional[List[float]] = ..., best_value: float = ...) -> None: ...

class SparsePortfolioRequest(Message):
    prices: RepeatedCompositeFieldContainer[PriceVector]
    method: str
    alpha: float
    def __init__(self, prices: Any = ..., method: str = ..., **kwargs: Any) -> None: ...

class SparsePortfolioResponse(Message):
    weights: RepeatedScalarFieldContainer[float]
    sparsity: float
    non_zero_count: int
    variance_explained: float
    metrics: Dict[str, float]
    def __init__(self, weights: Optional[List[float]] = ..., sparsity: float = ..., non_zero_count: int = ..., variance_explained: float = ..., metrics: Optional[Dict[str, float]] = ...) -> None: ...

class BoxTaoRequest(Message):
    prices: RepeatedCompositeFieldContainer[PriceVector]
    def __init__(self, prices: Any = ..., **kwargs: Any) -> None: ...

class BoxTaoResponse(Message):
    weights: RepeatedScalarFieldContainer[float]
    rank: int
    sparsity: float
    converged: bool
    iterations: int
    def __init__(self, weights: Optional[List[float]] = ..., rank: int = ..., sparsity: float = ..., converged: bool = ..., iterations: int = ...) -> None: ...
