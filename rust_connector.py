"""
Compatibility shim for `rust_connector` Python module.

Provides a lightweight pure-Python implementation of commonly-used
analytics and mean-reversion functions. When the gRPC server is
available, it will delegate heavy operations to the Rust gRPC backend
via `python.grpc_client.TradingGrpcClient`. Otherwise, it falls back to
numpy / pandas-based implementations so the codebase runs without the
PyO3 extension during and after the gRPC migration.

This shim intentionally exposes the function names used throughout the
codebase (e.g. `compute_pca_rust`, `estimate_ou_process_rust`,
`compute_correlation_matrix`, etc.).
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import math
import warnings

# Try to import the gRPC client
_GRPC_AVAILABLE = False
_grpc_client = None
try:
    from python.grpc_client import TradingGrpcClient
    _GRPC_AVAILABLE = True
except Exception:
    _GRPC_AVAILABLE = False
    TradingGrpcClient = None  # type: ignore


def _get_grpc_client() -> Optional['TradingGrpcClient']:
    global _grpc_client
    if not _GRPC_AVAILABLE or TradingGrpcClient is None:
        return None
    if _grpc_client is None:
        _grpc_client = TradingGrpcClient()
        try:
            _grpc_client.connect()
        except Exception:
            # If connection fails, we silently ignore and fall back
            _grpc_client = None
            return None
    return _grpc_client


# ------------------------- Analytics helpers -------------------------

def compute_correlation_matrix(returns: np.ndarray) -> np.ndarray:
    """Compute correlation matrix for returns (n_obs x n_assets)."""
    arr = np.asarray(returns)
    if arr.ndim != 2:
        raise ValueError("returns must be 2D array-like (n_obs x n_assets)")
    return np.corrcoef(arr.T)


def compute_covariance_matrix(returns: np.ndarray, unbiased: bool = True) -> np.ndarray:
    arr = np.asarray(returns)
    if arr.ndim != 2:
        raise ValueError("returns must be 2D array-like (n_obs x n_assets)")
    return np.cov(arr.T, ddof=1 if unbiased else 0)


def compute_rolling_mean(series: np.ndarray, window: int) -> np.ndarray:
    s = pd.Series(series)
    return s.rolling(window).mean().to_numpy()


def compute_rolling_zscores(series: np.ndarray, window: int) -> np.ndarray:
    s = pd.Series(series)
    rolling_mean = s.rolling(window).mean()
    rolling_std = s.rolling(window).std(ddof=0)
    z = (s - rolling_mean) / (rolling_std + 1e-12)
    return z.to_numpy()


def compute_mean(series: np.ndarray) -> float:
    return float(np.mean(series))


def compute_std(series: np.ndarray, ddof: int = 1) -> float:
    return float(np.std(series, ddof=ddof))


def compute_variance(series: np.ndarray, ddof: int = 1) -> float:
    return float(np.var(series, ddof=ddof))


def compute_skewness(series: np.ndarray) -> float:
    # Fisher-Pearson standardized moment
    s = np.asarray(series)
    m = np.mean(s)
    std = np.std(s, ddof=1)
    if std == 0:
        return 0.0
    return float(np.mean(((s - m) / std) ** 3))


def compute_kurtosis(series: np.ndarray, excess: bool = True) -> float:
    s = np.asarray(series)
    m = np.mean(s)
    std = np.std(s, ddof=1)
    if std == 0:
        return -3.0 if excess else 0.0
    raw = float(np.mean(((s - m) / std) ** 4))
    if excess:
        return raw - 3.0
    return raw


def compute_rolling_correlation(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    sx = pd.Series(x)
    sy = pd.Series(y)
    return sx.rolling(window).corr(sy).to_numpy()


# ------------------------- PCA & Mean-reversion -------------------------

def compute_pca_rust(returns: List[List[float]], n_components: int) -> Dict[str, Any]:
    """Compute PCA components and explained variance.

    Returns a dict matching the legacy Rust output shape:
      { 'components': [[...], ...], 'explained_variance': [...] }
    """
    arr = np.asarray(returns)
    # Expect shape (n_obs, n_assets)
    if arr.ndim != 2:
        raise ValueError("returns must be 2D array-like")

    # center columns (assets)
    X = arr - np.mean(arr, axis=0)
    # SVD on centered data: X = U S Vt
    try:
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
    except Exception:
        # Fallback to sklearn PCA if available
        from sklearn.decomposition import PCA
        p = PCA(n_components=min(n_components, arr.shape[1]))
        p.fit(arr)
        comps = p.components_.tolist()
        explained = p.explained_variance_ratio_.tolist()
        return {"components": comps, "explained_variance": explained}

    # components are rows of Vt
    comps = Vt[:n_components].tolist()
    # explained variance ratio approx: S**2 / (n_obs-1) normalized
    var_explained = (S ** 2) / (max(arr.shape[0] - 1, 1))
    explained_ratio = (var_explained / np.sum(var_explained))[:n_components].tolist()
    return {"components": comps, "explained_variance": explained_ratio}


def estimate_ou_process_rust(prices: List[float]) -> Dict[str, Any]:
    """Estimate OU parameters (theta, mu, sigma, half_life).

    Uses a simple discrete-time regression approximation (MLE/OLS fallback).
    """
    x = np.asarray(prices, dtype=float)
    if x.size < 3:
        return {"theta": float('nan'), "mu": float('nan'), "sigma": float('nan'), "half_life": float('nan')}

    # Work on log-prices or raw? Assume input is price series; convert to log-price
    try:
        y = np.log(x)
    except Exception:
        y = x

    y_lag = y[:-1]
    y_curr = y[1:]
    n = len(y_lag)
    # Regress y_curr = a + b * y_lag + eps
    A = np.vstack([np.ones(n), y_lag]).T
    sol, *_ = np.linalg.lstsq(A, y_curr, rcond=None)
    a, b = sol[0], sol[1]

    # Discrete-time OU parameters
    # theta = -log(b) (if 0<b<1), mu = a/(1-b), sigma ~ std(residual) * sqrt(2*theta/(1-b**2))
    theta = -math.log(b) if (b > 0 and b < 1) else 0.0
    mu = a / (1 - b) if (1 - b) != 0 else float('nan')
    resid = y_curr - (a + b * y_lag)
    sigma = float(np.std(resid, ddof=1))
    half_life = math.log(2) / theta if theta > 0 else float('inf')
    return {"theta": float(theta), "mu": float(mu), "sigma": float(sigma), "half_life": float(half_life)}


# ------------------------- gRPC delegations (when available) -------------------------
def _try_grpc_client():
    """Return a connected TradingGrpcClient or None."""
    if not _GRPC_AVAILABLE:
        return None
    try:
        client = TradingGrpcClient()
        client.connect()
        return client
    except Exception:
        return None


def calculate_mean_reversion(prices, threshold: float = 2.0, lookback: int = 20) -> Dict:
    """Delegate mean-reversion calculation to gRPC server when available."""
    client = _try_grpc_client()
    if client is not None:
        try:
            res = client.calculate_mean_reversion(prices, threshold=threshold, lookback=lookback)
            client.close()
            return res
        except Exception:
            try:
                client.close()
            except Exception:
                pass

    # Fallback: compute simple z-score-based signals locally
    arr = np.asarray(prices, dtype=float)
    if arr.ndim != 1:
        arr = arr.ravel()
    if len(arr) < lookback:
        return {'signal': 0, 'zscore': [], 'entry_signal': [], 'exit_signal': [], 'metrics': {}}

    import pandas as _pd
    s = _pd.Series(arr)
    rol_mean = s.rolling(window=lookback).mean()
    rol_std = s.rolling(window=lookback).std()
    z = ((s - rol_mean) / (rol_std + 1e-12)).fillna(0).tolist()
    entry = [abs(x) > threshold for x in z]
    exit = [abs(x) < (threshold/2) for x in z]
    return {'signal': 0, 'zscore': z, 'entry_signal': entry, 'exit_signal': exit, 'metrics': {}}


def optimize_portfolio(prices, method: str = 'markowitz', parameters: Optional[Dict[str, float]] = None) -> Dict:
    """Delegate portfolio optimization to gRPC server when available."""
    client = _try_grpc_client()
    if client is not None:
        try:
            res = client.optimize_portfolio(prices, method=method, parameters=parameters)
            client.close()
            return res
        except Exception:
            try:
                client.close()
            except Exception:
                pass

    # Fallback: simple equal-weight or Markowitz using numpy
    parameters = parameters or {}
    if method in ('equal_weight', 'equal'):
        n = len(prices[0]) if isinstance(prices, list) and len(prices) > 0 else 1
        weights = np.ones(n) / n
        return {'weights': weights, 'expected_return': 0.0, 'volatility': 0.0, 'sharpe_ratio': 0.0, 'metrics': {}}

    # Basic Markowitz (requires covariance matrix)
    try:
        # prices: list of PriceVector-like -> convert to returns matrix
        arr = np.asarray(prices)
        # If shape (n_assets, n_obs) transpose
        if arr.ndim == 2 and arr.shape[0] > arr.shape[1]:
            arr = arr.T
        # compute mean returns and cov
        rets = np.diff(np.log(arr), axis=0)
        mu = np.nanmean(rets, axis=0)
        cov = np.nan_to_num(np.cov(rets.T))
        inv = np.linalg.pinv(cov + np.eye(len(mu)) * 1e-8)
        w = inv @ mu
        if np.sum(np.abs(w)) > 0:
            w = w / np.sum(np.abs(w))
        expected_return = float(w @ mu)
        volatility = float(np.sqrt(w @ cov @ w))
        sharpe = float(expected_return / (volatility + 1e-12))
        return {'weights': w.tolist(), 'expected_return': expected_return, 'volatility': volatility, 'sharpe_ratio': sharpe, 'metrics': {}}
    except Exception:
        n = arr.shape[1] if arr.ndim == 2 else 1
        return {'weights': [1.0 / n] * n, 'expected_return': 0.0, 'volatility': 0.0, 'sharpe_ratio': 0.0, 'metrics': {}}


# ------------------------- Strategy simulation / backtest -------------------------

def simulate_ou_strategy_rust(weights: List[float], prices: List[List[float]], entry_z: float = 1.5, exit_z: float = 0.5, notional: float = 1.0) -> Dict[str, Any]:
    """Simple mean-reversion backtester compatible with legacy Rust signature.

    weights: list of asset weights (length n_assets)
    prices: 2D list or array-like of shape (n_obs, n_assets)
    Returns: dict with keys 'equity', 'positions', 'z_scores', 'sharpe', 'pnl'
    """
    P = np.asarray(prices)
    weights_arr = np.asarray(weights)
    if P.ndim == 1:
        # Single asset series
        P = P.reshape(-1, 1)
    n_obs = P.shape[0]
    portfolio = P.dot(weights_arr)
    logp = np.log(portfolio)
    z = (logp - pd.Series(logp).rolling(20).mean().to_numpy()) / (pd.Series(logp).rolling(20).std().to_numpy() + 1e-12)

    position = 0.0
    cash = 0.0
    equity = []
    positions = []
    for t in range(len(z)):
        zz = z[t]
        price = portfolio[t]
        if position == 0:
            if zz > entry_z:
                position = -notional / price
                cash = notional
            elif zz < -entry_z:
                position = notional / price
                cash = -notional
        else:
            if abs(zz) < exit_z:
                cash = cash + position * price
                position = 0.0
        equity.append(cash + position * price)
        positions.append(position)

    equity_arr = np.array(equity)
    returns = np.nan_to_num(pd.Series(equity_arr).pct_change().to_numpy())
    sharpe = float(np.sqrt(252) * np.mean(returns[~np.isnan(returns)]) / (np.std(returns[~np.isnan(returns)]) + 1e-12))
    pnl = equity_arr.tolist()
    # Calculate max drawdown
    running_max = pd.Series(equity_arr).expanding().max().to_numpy()
    drawdowns = (equity_arr - running_max) / (running_max + 1e-12)
    max_drawdown = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0
    return {
        "equity": equity_arr.tolist(), 
        "positions": positions, 
        "z_scores": z.tolist(), 
        "sharpe": sharpe, 
        "pnl": pnl,
        "returns": returns.tolist(),
        "max_drawdown": max_drawdown
    }


def backtest_strategy_rust(prices: List[float], entry_z: float = 2.0, exit_z: float = 0.5, notional: float = 1.0) -> Dict[str, Any]:
    # Single-asset wrapper for convenience
    return simulate_ou_strategy_rust([1.0], np.array(prices).reshape(-1, 1), entry_z, exit_z, notional)


# ------------------------- Cointegration / statistical tests -------------------------

try:
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import adfuller
    _STATS_AVAILABLE = True
except Exception:
    _STATS_AVAILABLE = False


def cointegration_test_rust(prices1: List[float], prices2: List[float]) -> Dict[str, Any]:
    x = np.asarray(prices1)
    y = np.asarray(prices2)
    if x.size != y.size:
        minlen = min(x.size, y.size)
        x = x[:minlen]
        y = y[:minlen]

    # Regress y on x
    n = len(x)
    A = np.vstack([np.ones(n), x]).T
    try:
        sol, *_ = np.linalg.lstsq(A, y, rcond=None)
        intercept, slope = sol[0], sol[1]
        resid = y - (intercept + slope * x)
    except Exception:
        resid = y - x

    if _STATS_AVAILABLE:
        try:
            adf = adfuller(resid, autolag='AIC')
            stat = float(adf[0])
            pval = float(adf[1])
            is_cointegrated = pval < 0.05
            return {"statistic": stat, "p_value": pval, "is_cointegrated": bool(is_cointegrated)}
        except Exception:
            pass

    # Fallback: use simple heuristic
    stat = float(np.nan)
    pval = 1.0
    is_cointegrated = False
    return {"statistic": stat, "p_value": pval, "is_cointegrated": is_cointegrated}


# ------------------------- Optimization placeholders -------------------------

def cara_optimal_weights_rust(expected_returns: List[float], covariance: List[List[float]], gamma: float = 2.0) -> Dict[str, Any]:
    mu = np.asarray(expected_returns, dtype=float)
    cov = np.asarray(covariance, dtype=float)
    try:
        cov_inv = np.linalg.pinv(cov + np.eye(cov.shape[0]) * 1e-8)
        weights = (1.0 / gamma) * (cov_inv @ mu)
        expected_return = float(weights @ mu)
        expected_variance = float(weights @ cov @ weights)
        return {"weights": weights.tolist(), "expected_return": expected_return, "expected_variance": expected_variance}
    except Exception:
        n = len(mu)
        return {"weights": [1.0 / n] * n, "expected_return": 0.0, "expected_variance": 0.0}


def sharpe_optimal_weights_rust(expected_returns: List[float], covariance: List[List[float]], risk_free_rate: float = 0.0) -> Dict[str, Any]:
    mu = np.asarray(expected_returns, dtype=float)
    cov = np.asarray(covariance, dtype=float)
    try:
        excess = mu - risk_free_rate
        cov_inv = np.linalg.pinv(cov + np.eye(cov.shape[0]) * 1e-8)
        w = cov_inv @ excess
        if np.sum(np.abs(w)) > 0:
            w = w / np.sum(w)
        else:
            w = np.ones_like(w) / len(w)
        expected_return = float(w @ mu)
        expected_std = float(np.sqrt(w @ cov @ w))
        sharpe = float((expected_return - risk_free_rate) / (expected_std + 1e-12))
        return {"weights": w.tolist(), "sharpe_ratio": sharpe, "expected_return": expected_return, "expected_std": expected_std}
    except Exception:
        n = len(mu)
        return {"weights": [1.0 / n] * n, "sharpe_ratio": 0.0, "expected_return": 0.0, "expected_std": 0.0}


def backtest_with_costs_rust(prices: List[float], z_scores: List[float], entry_z: float = 2.0, exit_z: float = 0.5, transaction_cost: float = 0.001, slippage: float = 0.0) -> Dict[str, Any]:
    """Simple mean-reversion backtest with transaction costs."""
    prices_arr = np.asarray(prices, dtype=float)
    z_scores_arr = np.asarray(z_scores, dtype=float)
    
    position = 0.0
    cash = 1.0  # Start with 1 unit of cash
    equity = []
    positions_list = []
    total_costs = 0.0
    
    for t in range(len(z_scores_arr)):
        z = z_scores_arr[t]
        price = prices_arr[t]
        
        if position == 0:
            if z > entry_z:
                # Go short
                cost = abs(position - (-1.0)) * price * (transaction_cost + slippage)
                position = -1.0
                cash = 1.0 + cost
                total_costs += cost
            elif z < -entry_z:
                # Go long
                cost = abs(position - 1.0) * price * (transaction_cost + slippage)
                position = 1.0
                cash = 1.0 - cost
                total_costs += cost
        else:
            if abs(z) < exit_z:
                # Close position
                cost = abs(position) * price * (transaction_cost + slippage)
                cash = cash + position * price - cost
                total_costs += cost
                position = 0.0
        
        equity.append(cash + position * price)
        positions_list.append(position)
    
    equity_arr = np.array(equity)
    returns = np.diff(equity_arr) / (equity_arr[:-1] + 1e-12)
    sharpe = float(np.sqrt(252) * np.mean(returns) / (np.std(returns) + 1e-12)) if len(returns) > 0 else 0.0
    
    running_max = np.max(equity_arr)
    max_drawdown = float((np.min(equity_arr) - running_max) / (running_max + 1e-12)) if running_max > 0 else 0.0
    
    return {
        "pnl": equity_arr.tolist(),
        "positions": positions_list,
        "z_scores": z_scores_arr.tolist(),
        "sharpe": sharpe,
        "total_costs": total_costs,
        "max_drawdown": max_drawdown
    }


def optimal_thresholds_rust(theta: float, mu: float, sigma: float, transaction_cost: float) -> Dict[str, Any]:
    """Compute optimal entry/exit thresholds for mean-reversion strategy."""
    if theta <= 0:
        theta = 0.01
    # Simple heuristic: higher theta => tighter thresholds
    optimal_entry = float(2.0 / (1.0 + theta))
    optimal_exit = float(0.5 / (1.0 + theta))
    expected_holding_period = float(1.0 / max(theta, 0.001)) if theta > 0 else 10.0
    
    return {
        "optimal_entry": optimal_entry,
        "optimal_exit": optimal_exit,
        "expected_holding_period": expected_holding_period
    }


def multiperiod_optimize_rust(returns_data: List[List[float]], covariance: List[List[float]], gamma: float, transaction_cost: float, n_periods: int) -> Dict[str, Any]:
    """Multi-period portfolio optimization with rebalancing."""
    mu = np.mean(np.asarray(returns_data, dtype=float), axis=0)
    cov = np.asarray(covariance, dtype=float)
    n_assets = len(mu)
    
    # Simple equal-weight portfolio across periods
    weights_sequence = []
    rebalance_times = []
    
    for p in range(n_periods):
        # CARA utility maximization
        try:
            cov_inv = np.linalg.pinv(cov + np.eye(cov.shape[0]) * 1e-8)
            w = (1.0 / gamma) * (cov_inv @ mu)
            if np.sum(np.abs(w)) > 0:
                w = w / np.sum(w)
            else:
                w = np.ones(n_assets) / n_assets
        except Exception:
            w = np.ones(n_assets) / n_assets
        
        weights_sequence.append(w.tolist())
        rebalance_times.append(p)
    
    expected_return = float(np.mean(mu))
    expected_utility = float(expected_return - (gamma / 2.0) * np.mean(np.diag(cov)))
    
    return {
        "weights_sequence": weights_sequence,
        "rebalance_times": rebalance_times,
        "expected_utility": expected_utility
    }


# Expose module-level name for debugging
__all__ = [
    'compute_correlation_matrix', 'compute_covariance_matrix', 'compute_rolling_mean', 'compute_rolling_zscores',
    'compute_mean', 'compute_std', 'compute_variance', 'compute_skewness', 'compute_kurtosis', 'compute_rolling_correlation',
    'compute_pca_rust', 'estimate_ou_process_rust', 'simulate_ou_strategy_rust', 'backtest_strategy_rust',
    'cointegration_test_rust', 'cara_optimal_weights_rust', 'sharpe_optimal_weights_rust',
    'backtest_with_costs_rust', 'optimal_thresholds_rust', 'multiperiod_optimize_rust'
]
