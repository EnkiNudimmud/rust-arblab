"""gRPC bridge exposing legacy `rust_connector` API via explicit gRPC client.

This module prefers calling the Rust gRPC backend through
`python.grpc_client.TradingGrpcClient`. If the gRPC backend or client
isn't available, it falls back to the local compatibility shim
`rust_connector` (root-level `rust_connector.py`).

Call sites can either do ``from python.rust_grpc_bridge import compute_pca_rust``
or ``from python.rust_grpc_bridge import rust_connector as rust_connector```
to maintain previous usage patterns.
"""
from typing import Any, Dict, List, Optional
import sys
import warnings
import os

_HAS_GRPC = False
_GRPC_CLIENT_CLASS = None
try:
    from python.grpc_client import TradingGrpcClient
    _HAS_GRPC = True
    _GRPC_CLIENT_CLASS = TradingGrpcClient
except Exception:
    _HAS_GRPC = False

# Fallback shim - must be available for all functions
try:
    # Add the repo root to path to find rust_connector shim
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    import rust_connector as _shim  # root shim
except Exception:
    _shim = None


def _get_client(timeout: Optional[float] = None):
    if not _HAS_GRPC or _GRPC_CLIENT_CLASS is None:
        return None
    try:
        c = _GRPC_CLIENT_CLASS()
        c.connect(timeout=timeout) if hasattr(c, 'connect') else c.connect()
        return c
    except Exception:
        return None


# --- Exported functions: try gRPC first, then shim ---
def compute_pca_rust(returns: List[List[float]], n_components: int) -> Dict[str, Any]:
    client = _get_client()
    if client is not None and hasattr(client, 'compute_pca'):
        try:
            res = client.compute_pca(returns, n_components)
            client.close()
            return res
        except Exception:
            try:
                client.close()
            except Exception:
                pass
    if _shim is not None and hasattr(_shim, 'compute_pca_rust'):
        return _shim.compute_pca_rust(returns, n_components)
    raise RuntimeError('No implementation available for compute_pca_rust')


def estimate_ou_process_rust(prices: List[float]) -> Dict[str, Any]:
    client = _get_client()
    if client is not None and hasattr(client, 'estimate_ou'):
        try:
            res = client.estimate_ou(prices)
            client.close()
            return res
        except Exception:
            try:
                client.close()
            except Exception:
                pass
    if _shim is not None:
        return _shim.estimate_ou_process_rust(prices)
    raise RuntimeError('No implementation available for estimate_ou_process_rust')


def compute_correlation_matrix(returns):
    client = _get_client()
    if client is not None and hasattr(client, 'compute_correlation_matrix'):
        try:
            res = client.compute_correlation_matrix(returns)
            client.close()
            return res
        except Exception:
            try:
                client.close()
            except Exception:
                pass
    if _shim is not None:
        return _shim.compute_correlation_matrix(returns)
    raise RuntimeError('No implementation available for compute_correlation_matrix')


def compute_covariance_matrix(returns, unbiased: bool = True):
    if _shim is not None:
        return _shim.compute_covariance_matrix(returns, unbiased=unbiased)
    # small local fallback
    import numpy as _np
    arr = _np.asarray(returns)
    return _np.cov(arr.T)


def compute_rolling_mean(series, window: int):
    if _shim is not None:
        return _shim.compute_rolling_mean(series, window)
    import pandas as _pd
    return _pd.Series(series).rolling(window).mean().to_numpy()


def compute_rolling_zscores(series, window: int):
    if _shim is not None:
        return _shim.compute_rolling_zscores(series, window)
    import pandas as _pd
    s = _pd.Series(series)
    return ((s - s.rolling(window).mean()) / (s.rolling(window).std() + 1e-12)).to_numpy()


def compute_mean(series):
    if _shim is not None:
        return _shim.compute_mean(series)
    import numpy as _np
    return float(_np.mean(series))


def compute_std(series, ddof: int = 1):
    if _shim is not None:
        return _shim.compute_std(series, ddof=ddof)
    import numpy as _np
    return float(_np.std(series, ddof=ddof))


def compute_variance(series, ddof: int = 1):
    if _shim is not None:
        return _shim.compute_variance(series, ddof=ddof)
    import numpy as _np
    return float(_np.var(series, ddof=ddof))


def compute_skewness(series):
    if _shim is not None:
        return _shim.compute_skewness(series)
    import numpy as _np
    s = _np.asarray(series)
    m = _np.mean(s)
    std = _np.std(s, ddof=1)
    if std == 0:
        return 0.0
    return float(_np.mean(((s - m) / std) ** 3))


def compute_kurtosis(series, excess: bool = True):
    if _shim is not None:
        return _shim.compute_kurtosis(series, excess=excess)
    import numpy as _np
    s = _np.asarray(series)
    m = _np.mean(s)
    std = _np.std(s, ddof=1)
    if std == 0:
        return -3.0 if excess else 0.0
    raw = float(_np.mean(((s - m) / std) ** 4))
    return raw - 3.0 if excess else raw


def compute_rolling_correlation(x, y, window: int):
    if _shim is not None:
        return _shim.compute_rolling_correlation(x, y, window)
    import pandas as _pd
    return _pd.Series(x).rolling(window).corr(_pd.Series(y)).to_numpy()


def simulate_ou_strategy_rust(*args, **kwargs):
    if _shim is not None:
        return _shim.simulate_ou_strategy_rust(*args, **kwargs)
    raise RuntimeError('simulate_ou_strategy_rust not available')


def backtest_strategy_rust(*args, **kwargs):
    if _shim is not None:
        return _shim.backtest_strategy_rust(*args, **kwargs)
    raise RuntimeError('backtest_strategy_rust not available')


def cointegration_test_rust(prices1, prices2):
    if _shim is not None:
        return _shim.cointegration_test_rust(prices1, prices2)
    raise RuntimeError('cointegration_test_rust not available')


def cara_optimal_weights_rust(expected_returns, covariance, gamma: float = 2.0):
    if _shim is not None:
        return _shim.cara_optimal_weights_rust(expected_returns, covariance, gamma)
    raise RuntimeError('cara_optimal_weights_rust not available')


def sharpe_optimal_weights_rust(expected_returns, covariance, risk_free_rate: float = 0.0):
    if _shim is not None:
        return _shim.sharpe_optimal_weights_rust(expected_returns, covariance, risk_free_rate)
    raise RuntimeError('sharpe_optimal_weights_rust not available')


def backtest_with_costs_rust(prices, z_scores, entry_z=2.0, exit_z=0.5, transaction_cost=0.001, slippage=0.0):
    if _shim is not None:
        return _shim.backtest_with_costs_rust(prices, z_scores, entry_z, exit_z, transaction_cost, slippage)
    raise RuntimeError('backtest_with_costs_rust not available')


def optimal_thresholds_rust(theta, mu, sigma, transaction_cost):
    if _shim is not None:
        return _shim.optimal_thresholds_rust(theta, mu, sigma, transaction_cost)
    raise RuntimeError('optimal_thresholds_rust not available')


def multiperiod_optimize_rust(returns_data, covariance, gamma, transaction_cost, n_periods):
    if _shim is not None:
        return _shim.multiperiod_optimize_rust(returns_data, covariance, gamma, transaction_cost, n_periods)
    raise RuntimeError('multiperiod_optimize_rust not available')


# Expose a module-like object for callers wanting ``from python.rust_grpc_bridge import rust_connector``
import types
rust_connector = types.SimpleNamespace(**{name: globals()[name] for name in list(globals().keys()) if name.startswith(('compute_', 'estimate_', 'simulate_', 'backtest_', 'cointegration_', 'cara_', 'sharpe_', 'optimal_', 'multiperiod_'))})

__all__ = list(rust_connector.__dict__.keys()) + ["rust_connector"]
