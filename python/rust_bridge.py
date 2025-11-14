# python/rust_bridge.py
# Async-aware bridge that prefers the Rust implementation when available.
# Presents the same API used by the Streamlit UI and notebooks.

from typing import Any, List
import logging

logger = logging.getLogger(__name__)
try:
    import rust_connector  # type: ignore
    RUST_AVAILABLE = True
    logger.info("rust_connector available")
except Exception as e:
    rust_connector = None
    RUST_AVAILABLE = False
    logger.warning("rust_connector not available: %s", e)

def list_connectors() -> List[str]:
    if RUST_AVAILABLE and hasattr(rust_connector, "list_connectors"):
        try:
            return rust_connector.list_connectors()
        except Exception:
            logger.exception("rust list_connectors failed")
    return ["binance", "coinbase", "uniswap", "mock"]

def get_connector(name: str):
    if RUST_AVAILABLE and hasattr(rust_connector, "get_connector"):
        try:
            return rust_connector.get_connector(name)
        except Exception:
            logger.exception("rust get_connector failed; falling back to Python adapter")
    # Fallback adapter with same methods
    class _Fallback:
        def __init__(self, n):
            self.name = n
            self.symbols = ["BTC-USD", "ETH-USD"]
        def list_symbols(self):
            return self.symbols
        def fetch_orderbook_sync(self, symbol):
            return {"bids":[[100.0,1.0]], "asks":[[100.2,1.0]]}
        def start_stream(self, symbol, cb):
            raise NotImplementedError("Fallback connector does not support streaming")
        def latest_snapshot(self):
            return None
    return _Fallback(name)

def compute_dex_cex_arbitrage(ob_cex: Any, ob_dex: Any, fee_cex: float = 0.001, fee_dex: float = 0.002):
    if RUST_AVAILABLE and hasattr(rust_connector, "compute_dex_cex_arbitrage"):
        try:
            return rust_connector.compute_dex_cex_arbitrage(ob_cex, ob_dex, fee_cex, fee_dex)
        except Exception:
            logger.exception("rust compute_dex_cex_arbitrage failed")
    # fallback python
    try:
        cex_bid = ob_cex["bids"][0][0] if isinstance(ob_cex, dict) else ob_cex.bids[0][0]
        dex_price = ob_dex["bids"][0][0] if isinstance(ob_dex, dict) else ob_dex.bids[0][0]
        gross = cex_bid / dex_price - 1.0 if dex_price != 0 else 0.0
        net = (1 - fee_dex) * (1 + gross) * (1 - fee_cex) - 1.0
        return {"gross": gross, "net": net}
    except Exception:
        return {"gross": 0.0, "net": 0.0}