"""
Mean-Reversion Portfolio Analysis Module

This module provides high-performance mean-reversion portfolio analysis functions,
using gRPC client to connect to Rust backend service.
Includes fallback to pure Python implementations when gRPC is unavailable.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Any
import logging
import asyncio

logger = logging.getLogger(__name__)

# Try to import gRPC client
try:
    from python.grpc_client import MeanRevClient
    GRPC_AVAILABLE = True
    logger.info("✓ gRPC backend available for mean-reversion analysis")
except ImportError:
    GRPC_AVAILABLE = False
    logger.warning("⚠ gRPC backend unavailable - using Python fallback")

# Global gRPC client instance (lazy-initialized)
_grpc_client = None

def get_grpc_client():
    """Get or create gRPC client instance"""
    global _grpc_client
    if not GRPC_AVAILABLE:
        return None
    if _grpc_client is None:
        try:
            _grpc_client = MeanRevClient(host="localhost", port=50051)
        except Exception as e:
            logger.warning(f"Failed to create gRPC client: {e}")
            return None
    return _grpc_client

def _async_call(func):
    """Helper to run async gRPC calls synchronously"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(func)


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute log returns from price data.
    
    Args:
        prices: DataFrame with prices (timestamps x assets)
    
    Returns:
        DataFrame with log returns
    """
    return np.log(prices / prices.shift(1)).dropna()


def pca_portfolios(returns: pd.DataFrame, n_components: int = 10) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Compute PCA on returns and extract principal components.
    
    Args:
        returns: Returns matrix (T x N)
        n_components: Number of components to extract
    
    Returns:
        Tuple of (components, info_dict)
        - components: (n_components x N) matrix of principal components
        - info_dict: Dictionary with "explained_variance_ratio_"
    """
    if GRPC_AVAILABLE:
        try:
            client = get_grpc_client()
            if client:
                result = _async_call(client.pca_portfolios(returns.values.tolist(), n_components))
                components = np.array(result['components'])
                explained_var = np.array(result['variance_explained'])
                return components, {'explained_variance_ratio_': explained_var}
        except Exception as e:
            logger.warning(f"gRPC PCA failed, falling back to Python: {e}")
    
    # Python fallback
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)
    pca.fit(returns)
    
    return pca.components_, {'explained_variance_ratio_': pca.explained_variance_ratio_}


def estimate_ou_params(prices: np.ndarray) -> Dict[str, float]:
    """
    Estimate Ornstein-Uhlenbeck process parameters.
    
    Args:
        prices: Price series
    
    Returns:
        Dictionary with keys: theta, mu, sigma, half_life
    """
    if GRPC_AVAILABLE:
        try:
            client = get_grpc_client()
            if client:
                prices_list = prices.tolist() if isinstance(prices, (np.ndarray, pd.Series)) else list(prices)
                result = _async_call(client.estimate_ou_params(prices_list))
                return result
        except Exception as e:
            logger.warning(f"gRPC OU estimation failed, falling back to Python: {e}")
    
    # Python fallback
    prices_arr = np.array(prices)
    n = len(prices_arr)
    
    # Estimate mu as mean
    mu = prices_arr.mean()
    
    # Estimate theta and sigma via regression
    diffs = np.diff(prices_arr)
    lags = prices_arr[:-1]
    
    mean_lag = lags.mean()
    mean_diff = diffs.mean()
    
    num = np.sum((lags - mean_lag) * (diffs - mean_diff))
    den = np.sum((lags - mean_lag) ** 2)
    
    slope = num / den if den > 1e-10 else 0.0
    theta = -slope
    
    intercept = mean_diff - slope * mean_lag
    residuals = diffs - (intercept + slope * lags)
    sigma = np.sqrt(np.mean(residuals ** 2))
    
    half_life = np.log(2) / theta if theta > 1e-10 else np.inf
    
    return {
        'theta': float(theta),
        'mu': float(mu),
        'sigma': float(sigma),
        'half_life': float(half_life)
    }


def cara_optimal_weights(expected_returns: np.ndarray, covariance: np.ndarray, 
                         gamma: float = 2.0) -> Dict[str, Any]:
    """
    Compute CARA utility-maximizing portfolio weights.
    
    Args:
        expected_returns: Expected returns vector (N,)
        covariance: Covariance matrix (N x N)
        gamma: Risk aversion parameter
    
    Returns:
        Dictionary with keys: weights, expected_return, expected_variance
    """
    if GRPC_AVAILABLE:
        try:
            client = get_grpc_client()
            if client:
                returns_list = expected_returns.tolist() if isinstance(expected_returns, np.ndarray) else list(expected_returns)
                cov_list = covariance.tolist() if isinstance(covariance, np.ndarray) else [list(row) for row in covariance]
                result = _async_call(client.cara_optimal_weights(returns_list, cov_list, gamma))
                return result
        except Exception as e:
            logger.warning(f"gRPC CARA optimization failed, falling back to Python: {e}")
    
    # Python fallback
    expected_returns = np.array(expected_returns)
    covariance = np.array(covariance)
    n = len(expected_returns)
    
    # Add regularization to avoid singularity
    sigma_reg = covariance + 1e-8 * np.eye(n)
    
    try:
        sigma_inv = np.linalg.inv(sigma_reg)
        weights = sigma_inv @ expected_returns / gamma
    except np.linalg.LinAlgError:
        weights = np.ones(n) / n
    
    expected_return = weights @ expected_returns
    expected_variance = weights @ covariance @ weights
    
    return {
        'weights': weights.tolist(),
        'expected_return': float(expected_return),
        'expected_variance': float(expected_variance)
    }


def sharpe_optimal_weights(expected_returns: np.ndarray, covariance: np.ndarray,
                           risk_free_rate: float = 0.02) -> Dict[str, Any]:
    """
    Compute Sharpe ratio-maximizing portfolio weights.
    
    Args:
        expected_returns: Expected returns vector (N,)
        covariance: Covariance matrix (N x N)
        risk_free_rate: Risk-free rate
    
    Returns:
        Dictionary with keys: weights, sharpe_ratio, expected_return, expected_std
    """
    if GRPC_AVAILABLE:
        try:
            client = get_grpc_client()
            if client:
                returns_list = expected_returns.tolist() if isinstance(expected_returns, np.ndarray) else list(expected_returns)
                cov_list = covariance.tolist() if isinstance(covariance, np.ndarray) else [list(row) for row in covariance]
                result = _async_call(client.sharpe_optimal_weights(returns_list, cov_list, risk_free_rate))
                return result
        except Exception as e:
            logger.warning(f"gRPC Sharpe optimization failed, falling back to Python: {e}")
    
    # Python fallback
    expected_returns = np.array(expected_returns)
    covariance = np.array(covariance)
    n = len(expected_returns)
    
    # Excess returns
    excess_returns = expected_returns - risk_free_rate
    
    # Optimize
    sigma_reg = covariance + 1e-8 * np.eye(n)
    
    try:
        sigma_inv = np.linalg.inv(sigma_reg)
        weights = sigma_inv @ excess_returns
        total = weights.sum()
        weights = weights / total if total != 0 else np.ones(n) / n
    except np.linalg.LinAlgError:
        weights = np.ones(n) / n
    
    expected_return = weights @ expected_returns
    expected_std = np.sqrt(weights @ covariance @ weights)
    sharpe_ratio = (expected_return - risk_free_rate) / expected_std if expected_std > 1e-10 else 0.0
    
    return {
        'weights': weights.tolist(),
        'sharpe_ratio': float(sharpe_ratio),
        'expected_return': float(expected_return),
        'expected_std': float(expected_std)
    }


def optimal_thresholds(theta: float, mu: float, sigma: float,
                       transaction_cost: float = 0.001) -> Dict[str, float]:
    """
    Compute optimal entry/exit thresholds for mean-reversion trading.
    
    Args:
        theta: Mean reversion speed
        mu: Long-term mean
        sigma: Volatility
        transaction_cost: Transaction cost as proportion
    
    Returns:
        Dictionary with keys: optimal_entry, optimal_exit, expected_holding_period
    """
    if GRPC_AVAILABLE:
        try:
            client = get_grpc_client()
            if client:
                result = _async_call(client.optimal_thresholds(theta, mu, sigma, transaction_cost))
                return result
        except Exception as e:
            logger.warning(f"gRPC threshold optimization failed, falling back to Python: {e}")
    
    # Python fallback
    if theta <= 0 or sigma <= 0:
        return {
            'optimal_entry': 2.0,
            'optimal_exit': 0.5,
            'expected_holding_period': 10.0
        }
    
    half_life = np.log(2) / theta
    cost_adjustment = np.sqrt(1 + 100 * transaction_cost)
    
    return {
        'optimal_entry': float(1.5 * cost_adjustment),
        'optimal_exit': float(0.3 * (cost_adjustment ** 0.5)),
        'expected_holding_period': float(half_life * 0.5)
    }


def backtest_with_costs(prices: np.ndarray, entry_z: float = 2.0, exit_z: float = 0.5,
                        transaction_cost: float = 0.001) -> Dict[str, Any]:
    """
    Backtest a mean-reversion strategy with transaction costs.
    
    Args:
        prices: Price series
        entry_z: Entry threshold (z-score)
        exit_z: Exit threshold (z-score)
        transaction_cost: Transaction cost as proportion
    
    Returns:
        Dictionary with keys: returns, positions, pnl, sharpe, max_drawdown, total_costs
    """
    if GRPC_AVAILABLE:
        try:
            client = get_grpc_client()
            if client:
                prices_list = prices.tolist() if isinstance(prices, (np.ndarray, pd.Series)) else list(prices)
                result = _async_call(client.backtest_with_costs(prices_list, entry_z, exit_z, transaction_cost))
                return result
        except Exception as e:
            logger.warning(f"gRPC backtesting failed, falling back to Python: {e}")
    
    # Python fallback
    prices_arr = np.array(prices)
    n = len(prices_arr)
    window = max(20, n // 4)
    
    positions = np.zeros(n, dtype=int)
    pnl = np.zeros(n)
    returns = np.zeros(n)
    
    current_position = 0
    cash = 100000.0
    portfolio_value = cash
    peak_value = cash
    max_dd = 0.0
    total_costs = 0.0
    
    for i in range(window, n):
        window_prices = prices_arr[i-window:i]
        mean = window_prices.mean()
        std = window_prices.std()
        
        if std < 1e-10:
            positions[i] = current_position
            continue
        
        z_score = (prices_arr[i] - mean) / std
        prev_position = current_position
        
        # Trading logic
        if z_score < -entry_z and current_position == 0:
            current_position = 1
        elif z_score > entry_z and current_position == 0:
            current_position = -1
        elif abs(z_score) < exit_z and current_position != 0:
            current_position = 0
        
        positions[i] = current_position
        
        # Compute returns with costs
        if i > 0:
            price_return = (prices_arr[i] - prices_arr[i-1]) / prices_arr[i-1]
            
            cost = 0.0
            if prev_position != current_position:
                position_change = abs(prev_position - current_position)
                cost = transaction_cost * prices_arr[i] * position_change
                total_costs += cost
            
            returns[i] = price_return * prev_position - (cost / portfolio_value)
            portfolio_value *= 1 + returns[i]
            pnl[i] = portfolio_value - cash
            
            if portfolio_value > peak_value:
                peak_value = portfolio_value
            
            drawdown = (peak_value - portfolio_value) / peak_value
            if drawdown > max_dd:
                max_dd = drawdown
    
    # Compute Sharpe
    mean_return = returns.mean()
    return_std = returns.std()
    sharpe = (mean_return / return_std * np.sqrt(252)) if return_std > 1e-10 else 0.0
    
    return {
        'returns': returns.tolist(),
        'positions': positions.tolist(),
        'pnl': pnl.tolist(),
        'sharpe': float(sharpe),
        'max_drawdown': float(max_dd),
        'total_costs': float(total_costs)
    }


def multiperiod_optimize(returns_history: pd.DataFrame, covariance: np.ndarray,
                        gamma: float = 2.0, transaction_cost: float = 0.001,
                        n_periods: int = 10) -> Dict[str, Any]:
    """
    Optimize portfolio over multiple periods with rebalancing.
    
    Args:
        returns_history: Historical returns (T x N)
        covariance: Covariance matrix (N x N)
        gamma: Risk aversion
        transaction_cost: Rebalancing cost
        n_periods: Number of rebalancing periods
    
    Returns:
        Dictionary with keys: weights_sequence, rebalance_times, expected_utility
    """
    if GRPC_AVAILABLE:
        try:
            client = get_grpc_client()
            if client:
                returns_list = returns_history.values.tolist()
                cov_list = covariance.tolist() if isinstance(covariance, np.ndarray) else [list(row) for row in covariance]
                result = _async_call(client.multiperiod_optimize(returns_list, cov_list, gamma, transaction_cost, n_periods))
                return result
        except Exception as e:
            logger.warning(f"gRPC multiperiod optimization failed, falling back to Python: {e}")
    
    # Python fallback
    if isinstance(returns_history, pd.DataFrame):
        returns_arr = returns_history.values
    else:
        returns_arr = np.array(returns_history)
    
    t_total, n_assets = returns_arr.shape
    period_length = max(1, t_total // n_periods)
    
    weights_sequence = []
    rebalance_times = []
    expected_utility = 0.0
    
    for p in range(n_periods):
        start_idx = p * period_length
        end_idx = min((p + 1) * period_length, t_total)
        
        if start_idx >= t_total:
            break
        
        period_returns = returns_arr[start_idx:end_idx]
        avg_returns = period_returns.mean(axis=0)
        
        # CARA optimization
        sigma_reg = covariance + 1e-8 * np.eye(n_assets)
        try:
            sigma_inv = np.linalg.inv(sigma_reg)
            weights = sigma_inv @ avg_returns / gamma
        except np.linalg.LinAlgError:
            weights = np.ones(n_assets) / n_assets
        
        # Apply transaction cost penalty
        if p > 0:
            prev_weights = np.array(weights_sequence[-1])
            change = np.abs(weights - prev_weights)
            weights -= transaction_cost * np.sign(weights - prev_weights) * change
        
        weights_sequence.append(weights.tolist())
        rebalance_times.append(start_idx)
        
        # Compute utility
        period_return = weights @ avg_returns
        period_var = weights @ covariance @ weights
        utility = np.exp(-gamma * (period_return - 0.5 * gamma * period_var))
        expected_utility += utility
    
    if n_periods > 0:
        expected_utility /= n_periods
    
    return {
        'weights_sequence': weights_sequence,
        'rebalance_times': rebalance_times,
        'expected_utility': float(expected_utility)
    }
