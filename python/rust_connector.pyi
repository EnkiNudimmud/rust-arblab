# Type stub for rust_connector module
# This helps Pylance understand the Rust module's exports

from typing import Any, List, Tuple

class OrderBook:
    bids: List[Tuple[float, float]]
    asks: List[Tuple[float, float]]
    def __init__(self, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]]) -> None: ...

class OrderBookLevel:
    price: float
    quantity: float
    def __init__(self, price: float, quantity: float) -> None: ...

class OrderBookSnapshot:
    timestamp: str
    symbol: str
    last_update_id: int
    exchange: str
    bids: List[Tuple[float, float]]
    asks: List[Tuple[float, float]]
    def __init__(
        self,
        timestamp: str,
        symbol: str,
        last_update_id: int,
        exchange: str,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]]
    ) -> None: ...

class OrderBookUpdate:
    timestamp: str
    symbol: str
    first_update_id: int
    final_update_id: int
    bids: List[Tuple[float, float]]
    asks: List[Tuple[float, float]]
    def __init__(
        self,
        timestamp: str,
        symbol: str,
        first_update_id: int,
        final_update_id: int,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]]
    ) -> None: ...

class LOBAnalytics:
    timestamp: str
    symbol: str
    best_bid: float
    best_ask: float
    spread_abs: float
    spread_bps: float
    mid_price: float
    bid_depth_1: float
    ask_depth_1: float
    bid_depth_5: float
    ask_depth_5: float
    bid_depth_10: float
    ask_depth_10: float
    volume_imbalance: float
    price_imbalance: float
    depth_imbalance_1: float
    depth_imbalance_5: float
    bid_levels: int
    ask_levels: int
    total_bid_volume: float
    total_ask_volume: float
    effective_spread_bps: float
    market_impact_10k: float

class ExchangeConnector:
    name: str
    def __init__(self, name: str) -> None: ...
    def list_symbols(self) -> List[str]: ...
    def fetch_orderbook_sync(self, symbol: str) -> OrderBook: ...
    def start_stream(self, symbol: str, callback: Any) -> None: ...
    def latest_snapshot(self) -> OrderBook | None: ...
    def set_api_credentials(self, api_key: str, api_secret: str) -> None: ...
    def set_api_key(self, api_key: str) -> None: ...

def list_connectors() -> List[str]: ...
def get_connector(name: str) -> ExchangeConnector: ...
def compute_dex_cex_arbitrage(ob_cex: Any, ob_dex: Any, fee_cex: float, fee_dex: float) -> dict[str, float]: ...
def uniswap_get_reserves(rpc_url: str, pair_address: str) -> Tuple[float, float]: ...

# LOB functions
def calculate_lob_analytics(snapshot: OrderBookSnapshot) -> LOBAnalytics: ...
def apply_orderbook_update(snapshot: OrderBookSnapshot, update: OrderBookUpdate, max_levels: int) -> OrderBookSnapshot: ...
def parse_binance_orderbook(data: dict[str, Any], symbol: str, exchange: str) -> OrderBookSnapshot: ...

# Mean reversion functions
def compute_pca_rust(returns: List[List[float]], n_components: int) -> dict[str, Any]: ...
def estimate_ou_process_rust(prices: List[float]) -> dict[str, float]: ...
def cointegration_test_rust(y: List[float], x: List[float]) -> dict[str, Any]: ...
def backtest_strategy_rust(prices: List[float], z_scores: List[float], entry_threshold: float, exit_threshold: float) -> dict[str, Any]: ...
def cara_optimal_weights_rust(returns: List[List[float]], risk_aversion: float) -> List[float]: ...
def sharpe_optimal_weights_rust(returns: List[List[float]], risk_free_rate: float) -> List[float]: ...
def backtest_with_costs_rust(prices: List[float], z_scores: List[float], entry_threshold: float, exit_threshold: float, transaction_cost: float, holding_cost: float) -> dict[str, Any]: ...
def optimal_thresholds_rust(prices: List[float], z_scores: List[float], transaction_cost: float) -> dict[str, float]: ...
def multiperiod_optimize_rust(returns: List[List[float]], risk_aversion: float, transaction_cost: float, periods: int) -> dict[str, Any]: ...
