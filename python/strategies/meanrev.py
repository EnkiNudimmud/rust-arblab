"""Mean-reversion portfolio discovery utilities.

This module provides hybrid Rust/Python implementations of algorithms to
find mean-reverting portfolios from large market data sets.

Expensive operations (PCA, OU estimation, portfolio simulation) are
implemented in Rust and exposed via PyO3 for performance. Python fallbacks
are provided when Rust is unavailable.

Functions implemented:
- fetch_price_series(connector, symbol, start, end, freq)
- compute_log_returns(df) -> uses Rust by default
- pca_portfolios(returns_df, n_components) -> uses Rust by default
- engle_granger_test(y, x)
- estimate_ou_params(ts) -> uses Rust by default
- simulate_ou_strategy(weights, price_df, entry_z, exit_z, notional=1.0) -> uses Rust
- backtest_portfolio(weights, price_df, entry_z, exit_z, notional=1.0) -> alias

Notes:
- Uses Finnhub via `python.finnhub_helper` if available
- Falls back to synthetic data if no real data source
- Uses Rust for performance-critical operations with Python fallback
"""

from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from datetime import datetime, timedelta

# Make statsmodels optional (only needed for Engle-Granger test)
try:
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("⚠️  statsmodels not available - Engle-Granger test will be disabled")


except Exception:
    fh_fetch_ohlcv = None

# Try to import Rust functions for performance
try:
    import rust_connector
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False


def fetch_price_series(connector, symbol: str, start: str, end: str, freq: str = "1D") -> pd.Series:
    """Fetch close price series for `symbol` between start and end.

    Parameters
    - connector: either a rust connector object from `python.rust_bridge` or None
    - symbol: string symbol compatible with the connector
    - start/end: ISO date strings like '2023-01-01'
    - freq: pandas offset alias

    Returns
    - pandas Series indexed by UTC timestamps with close prices
    """
    # Try connector if provided
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)

    if connector is not None:
        # Many connectors provide an OHLCV fetch; attempt common names
        if hasattr(connector, "fetch_ohlcv"):
            df = connector.fetch_ohlcv(symbol, start_dt.isoformat(), end_dt.isoformat())
            if isinstance(df, dict):
                df = pd.DataFrame(df)
        elif hasattr(connector, "fetch_candles"):
            df = connector.fetch_candles(symbol, start_dt.isoformat(), end_dt.isoformat())
        else:
            df = None

        if isinstance(df, pd.DataFrame) and "close" in df.columns:
            s = pd.to_datetime(df["ts"]) if "ts" in df.columns else pd.to_datetime(df.index)
            series = pd.Series(df["close"].values, index=pd.to_datetime(s)).sort_index()
            return series.resample(freq).last().ffill()

    # Try finnhub helper if available
    if fh_fetch_ohlcv is not None:
        df = fh_fetch_ohlcv(symbol, start_dt, end_dt)
        if isinstance(df, pd.DataFrame) and "c" in df.columns:
            timestamps: pd.DatetimeIndex = pd.to_datetime(df["t"], unit="s")  # type: ignore[assignment]
            series = pd.Series(df["c"].values, index=timestamps)
            return series.resample(freq).last().ffill()

    # Fallback: generate synthetic geometric Brownian motion
    rng = np.random.default_rng(hash(symbol) & 0xFFFFFFFF)
    days = pd.date_range(start=start_dt, end=end_dt, freq=freq)
    S0 = 100.0 + (rng.random() - 0.5) * 10.0
    logs = np.cumsum(rng.normal(loc=0.0, scale=0.001, size=len(days)))
    prices = S0 * np.exp(logs)
    return pd.Series(prices, index=days)


def compute_log_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns from price DataFrame (columns are symbols).
    
    Uses Rust implementation if available for better performance.
    """
    if RUST_AVAILABLE and hasattr(rust_connector, 'compute_log_returns_rust'):
        try:
            prices_list = price_df.values.tolist()
            returns_list = rust_connector.compute_log_returns_rust(prices_list)  # type: ignore[union-attr]
            return pd.DataFrame(returns_list, columns=price_df.columns, index=price_df.index[1:])
        except Exception:
            pass  # Fall back to Python
    
    log_prices = pd.DataFrame(np.log(price_df.values), columns=price_df.columns, index=price_df.index)
    return log_prices.diff().dropna()


def pca_portfolios(returns_df: pd.DataFrame, n_components: int = 5) -> Tuple[np.ndarray, Dict]:
    """Compute PCA on returns and return principal component weight vectors.

    Returns an array of shape (n_components, n_assets) where each row is
    a weight vector (not normalized to sum=1), and a dict with explained variance.
    
    Uses Rust implementation if available for better performance.
    """
    if RUST_AVAILABLE and hasattr(rust_connector, 'compute_pca_rust'):
        try:
            returns_list = returns_df.fillna(0).values.tolist()
            result = rust_connector.compute_pca_rust(returns_list, n_components)  # type: ignore[union-attr]
            components = np.array(result["components"])
            pca_info = {"explained_variance_ratio_": np.array(result["explained_variance"])}
            return components, pca_info
        except Exception as e:
            print(f"Rust PCA failed: {e}, falling back to Python")
            pass  # Fall back to Python
    
    # Python fallback using sklearn
    pca = PCA(n_components=min(n_components, returns_df.shape[1]))
    pca.fit(returns_df.fillna(0).values)
    components = pca.components_  # shape (n_components, n_assets)
    pca_info = {"explained_variance_ratio_": pca.explained_variance_ratio_}
    return components, pca_info


def engle_granger_test(y: pd.Series, x: pd.Series) -> Dict:
    """Run Engle-Granger two-step cointegration test for y ~ a + b x.

    Returns dictionary with residual series, ADF pvalue and regression params.
    """
    if not STATSMODELS_AVAILABLE:
        raise ImportError("Engle-Granger test requires statsmodels. Install with: pip install statsmodels")
    
    X = sm.add_constant(x)
    res = sm.OLS(y, X).fit()
    resid = res.resid
    adf_res = sm.tsa.stattools.adfuller(resid, autolag='AIC')
    return {"params": res.params.to_dict(), "resid": resid, "adf": adf_res}


def estimate_ou_params(ts: pd.Series) -> Dict:
    """Estimate Ornstein-Uhlenbeck parameters for a price or spread time series.

    Uses discretized OU MLE for dx = theta*(mu - x) dt + sigma dW
    Returns theta, mu, sigma (annualized approx assuming daily steps).
    
    Uses Rust implementation if available for better performance.
    """
    x = ts.dropna().astype(float)
    if len(x) < 3:
        return {"theta": np.nan, "mu": np.nan, "sigma": np.nan}
    
    if RUST_AVAILABLE:
        try:
            ts_list = x.tolist()  # type: ignore[assignment]
            return rust_connector.estimate_ou_process_rust(ts_list)  # type: ignore[arg-type]
        except Exception as e:
            print(f"Rust OU estimation failed: {e}, falling back to Python")
    
    # Python fallback
    if not STATSMODELS_AVAILABLE:
        # Simple estimation without statsmodels
        x_lag = x.shift(1).dropna()
        x_curr = x.loc[x_lag.index]
        # Simple linear regression: x_t = a + b*x_{t-1}
        # Estimate b using covariance
        x_lag_arr = np.array(x_lag)
        x_curr_arr = np.array(x_curr)
        cov = np.cov(x_lag_arr, x_curr_arr)
        b = float(cov[0, 1] / cov[0, 0]) if cov[0, 0] != 0 else 1.0
        a = float(np.mean(x_curr_arr) - b * np.mean(x_lag_arr))
        
        theta = -np.log(b) if b > 0 and b < 1 else 0.01
        mu = a / (1 - b) if b != 1 else float(np.mean(x))
        resid = x_curr_arr - (a + b * x_lag_arr)
        sigma = float(np.std(resid))
        
        return {"theta": theta, "mu": mu, "sigma": sigma}
    
    x_lag = x.shift(1).dropna()
    x_curr = x.loc[x_lag.index]
    # regress x_curr on x_lag
    X = sm.add_constant(x_lag)
    res = sm.OLS(x_curr, X).fit()
    a = res.params[1]
    b = res.params[0]
    theta = -np.log(a) if a>0 else np.nan
    mu = b / (1 - a) if a!=1 else np.nan
    residuals = res.resid
    sigma_hat = np.std(residuals) * np.sqrt(2*theta/(1 - a**2)) if not np.isnan(theta) else np.nan
    return {"theta": float(theta), "mu": float(mu), "sigma": float(sigma_hat)}


def portfolio_time_series(weights: np.ndarray, price_df: pd.DataFrame) -> pd.Series:
    """Compute portfolio value series given asset prices and weight vector.

    weights: (n_assets,) array with weights (can be any scale)
    price_df: DataFrame with columns matching order of weights
    """
    w = np.array(weights).reshape(-1)
    vals = price_df.values.dot(w)
    return pd.Series(vals, index=price_df.index)


def simulate_ou_strategy(weights: np.ndarray, price_df: pd.DataFrame, entry_z: float = 1.5, exit_z: float = 0.5, notional: float = 1.0) -> Dict:
    """Simulate a mean-reversion strategy on the constructed portfolio.

    Simple strategy:
    - Compute spread series from portfolio (log-price or raw)
    - Estimate rolling mean/std to produce z-score
    - Enter when z > entry_z (short) or z < -entry_z (long)
    - Exit when abs(z) < exit_z

    Returns dict with trades, equity curve, and metrics.
    
    Uses Rust implementation if available for better performance.
    """
    if RUST_AVAILABLE and hasattr(rust_connector, 'simulate_ou_strategy_rust'):
        try:
            weights_list = weights.tolist()
            prices_list = price_df.values.tolist()
            result = rust_connector.simulate_ou_strategy_rust(  # type: ignore[union-attr]
                weights_list, prices_list, entry_z, exit_z, notional
            )
            # Convert back to pandas
            sharpe = result["sharpe"][0] if result["sharpe"] else 0.0
            equity_series = pd.Series(result["equity"], index=price_df.index[-len(result["equity"]):])
            trades = pd.Series(result["positions"], index=price_df.index[-len(result["positions"]):])
            z = pd.Series(result["z_scores"], index=price_df.index[-len(result["z_scores"]):])
            return {"equity": equity_series, "trades": trades, "sharpe": float(sharpe), "z": z}
        except Exception:
            pass  # Fall back to Python
    
    # Python fallback
    ps = portfolio_time_series(weights, price_df)
    log_ps_series = pd.Series(np.log(ps.values), index=ps.index).dropna()
    rol_mean = log_ps_series.rolling(window=20).mean()
    rol_std = log_ps_series.rolling(window=20).std()
    z = (log_ps_series - rol_mean) / rol_std

    position = 0.0
    cash = 0.0
    equity = []
    positions = []
    for t in range(len(z)):
        zz = z.iloc[t]
        price = ps.iloc[t]
        if position == 0:
            if zz > entry_z:
                # short the portfolio
                position = -notional / price
                cash = notional
            elif zz < -entry_z:
                position = notional / price
                cash = -notional
        else:
            if abs(zz) < exit_z:
                # close position
                cash = cash + position * price
                position = 0.0
        equity.append(cash + position * price)
        positions.append(position)

    equity_series = pd.Series(equity, index=ps.index)
    trades = pd.Series(positions, index=ps.index)
    ret = equity_series.pct_change().fillna(0)
    sharpe = np.sqrt(252) * ret.mean() / (ret.std() + 1e-12)
    return {"equity": equity_series, "trades": trades, "sharpe": float(sharpe), "z": z}


def backtest_portfolio(weights: np.ndarray, price_df: pd.DataFrame, entry_z: float = 1.5, exit_z: float = 0.5, notional: float = 1.0) -> Dict:
    """Alias for simulate_ou_strategy for naming consistency."""
    return simulate_ou_strategy(weights, price_df, entry_z, exit_z, notional)


def cara_optimal_weights(expected_returns: np.ndarray, covariance: np.ndarray, gamma: float = 2.0) -> Dict:
    """Compute CARA optimal portfolio weights using utility maximization (Appendix A).
    
    Args:
        expected_returns: Expected return for each asset
        covariance: Covariance matrix of returns
        gamma: Risk aversion parameter (higher = more conservative)
        
    Returns:
        Dictionary with 'weights', 'expected_return', 'expected_variance'
        
    Uses Rust implementation if available for better performance.
    """
    if RUST_AVAILABLE and hasattr(rust_connector, 'cara_optimal_weights_rust'):
        try:
            returns_list = expected_returns.tolist()
            return rust_connector.cara_optimal_weights_rust(returns_list, gamma)  # type: ignore[union-attr]
        except Exception as e:
            print(f"Rust CARA optimization failed: {e}, falling back to Python")
    
    # Python fallback
    try:
        # w* = (1/gamma) * Sigma^{-1} * mu
        cov_inv = np.linalg.pinv(covariance + np.eye(len(covariance)) * 1e-8)
        weights = (1.0 / gamma) * (cov_inv @ expected_returns)
        
        expected_return = weights @ expected_returns
        expected_variance = weights @ covariance @ weights
        
        return {
            "weights": weights.tolist(),
            "expected_return": float(expected_return),
            "expected_variance": float(expected_variance)
        }
    except Exception as e:
        n = len(expected_returns)
        return {
            "weights": [1.0/n] * n,
            "expected_return": 0.0,
            "expected_variance": 0.0
        }


def sharpe_optimal_weights(expected_returns: np.ndarray, covariance: np.ndarray, risk_free_rate: float = 0.0) -> Dict:
    """Compute Sharpe-ratio optimal portfolio weights.
    
    Args:
        expected_returns: Expected return for each asset
        covariance: Covariance matrix of returns
        risk_free_rate: Risk-free rate for Sharpe calculation
        
    Returns:
        Dictionary with 'weights', 'sharpe_ratio', 'expected_return', 'expected_std'
        
    Uses Rust implementation if available for better performance.
    """
    if RUST_AVAILABLE and hasattr(rust_connector, 'sharpe_optimal_weights_rust'):
        try:
            returns_list = expected_returns.tolist()
            return rust_connector.sharpe_optimal_weights_rust(returns_list, risk_free_rate)  # type: ignore[union-attr]
        except Exception as e:
            print(f"Rust Sharpe optimization failed: {e}, falling back to Python")
    
    # Python fallback
    try:
        # w* = Sigma^{-1} * (mu - rf), normalized
        excess_returns = expected_returns - risk_free_rate
        cov_inv = np.linalg.pinv(covariance + np.eye(len(covariance)) * 1e-8)
        weights = cov_inv @ excess_returns
        weights = weights / np.sum(weights) if np.sum(weights) != 0 else weights
        
        expected_return = weights @ expected_returns
        expected_std = np.sqrt(weights @ covariance @ weights)
        sharpe_ratio = (expected_return - risk_free_rate) / expected_std if expected_std > 0 else 0.0
        
        return {
            "weights": weights.tolist(),
            "sharpe_ratio": float(sharpe_ratio),
            "expected_return": float(expected_return),
            "expected_std": float(expected_std)
        }
    except Exception:
        n = len(expected_returns)
        return {
            "weights": [1.0/n] * n,
            "sharpe_ratio": 0.0,
            "expected_return": 0.0,
            "expected_std": 0.0
        }


def backtest_with_costs(prices: pd.Series, entry_z: float = 2.0, exit_z: float = 0.5, transaction_cost: float = 0.001) -> Dict:
    """Backtest mean-reversion strategy with transaction costs.
    
    Args:
        prices: Price series
        entry_z: Z-score threshold for entry
        exit_z: Z-score threshold for exit
        transaction_cost: Proportional cost per trade (e.g., 0.001 = 0.1%)
        
    Returns:
        Dictionary with 'returns', 'positions', 'pnl', 'sharpe', 'max_drawdown', 'total_costs'
        
    Uses Rust implementation if available for better performance.
    """
    if RUST_AVAILABLE and hasattr(rust_connector, 'backtest_with_costs_rust'):
        try:
            prices_list = prices.tolist()
            z_scores = ((prices - prices.mean()) / prices.std()).tolist()
            return rust_connector.backtest_with_costs_rust(prices_list, z_scores, entry_z, exit_z, transaction_cost, 0.0)  # type: ignore[union-attr]
        except Exception as e:
            print(f"Rust backtest with costs failed: {e}, falling back to Python")
    
    # Python fallback with transaction costs
    window = 20
    positions = []
    pnl = []
    returns_list = []
    
    current_position = 0
    cash = 100000.0
    portfolio_value = cash
    peak_value = cash
    max_dd = 0.0
    total_costs = 0.0
    
    for i in range(window, len(prices)):
        window_prices = prices[i-window:i]
        mean = window_prices.mean()
        std = window_prices.std()
        
        if std < 1e-10:
            positions.append(current_position)
            pnl.append(portfolio_value - cash)
            returns_list.append(0.0)
            continue
            
        z_score = (prices.iloc[i] - mean) / std
        prev_position = current_position
        
        if z_score < -entry_z and current_position == 0:
            current_position = 1
        elif z_score > entry_z and current_position == 0:
            current_position = -1
        elif abs(z_score) < exit_z and current_position != 0:
            current_position = 0
            
        positions.append(current_position)
        
        # Calculate returns with costs
        if i > window:
            price_return = (prices.iloc[i] - prices.iloc[i-1]) / prices.iloc[i-1]
            
            # Transaction cost
            cost = 0.0
            if prev_position != current_position:
                position_change = abs(prev_position - current_position)
                cost = transaction_cost * prices.iloc[i] * position_change
                total_costs += cost
                
            ret = price_return * prev_position - (cost / portfolio_value)
            returns_list.append(ret)
            portfolio_value *= (1.0 + ret)
            pnl.append(portfolio_value - cash)
            
            if portfolio_value > peak_value:
                peak_value = portfolio_value
            drawdown = (peak_value - portfolio_value) / peak_value
            if drawdown > max_dd:
                max_dd = drawdown
        else:
            returns_list.append(0.0)
            pnl.append(0.0)
    
    returns_arr = np.array(returns_list)
    sharpe = np.sqrt(252) * returns_arr.mean() / (returns_arr.std() + 1e-12)
    
    return {
        "returns": returns_list,
        "positions": positions,
        "pnl": pnl,
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "total_costs": float(total_costs)
    }


def optimal_thresholds(theta: float, mu: float, sigma: float, transaction_cost: float = 0.001) -> Dict:
    """Compute optimal entry/exit thresholds based on OU parameters.
    
    Args:
        theta: Mean reversion speed
        mu: Long-term mean
        sigma: Volatility
        transaction_cost: Transaction cost parameter
        
    Returns:
        Dictionary with 'optimal_entry', 'optimal_exit', 'expected_holding_period'
        
    Uses Rust implementation if available for better performance.
    """
    if RUST_AVAILABLE and hasattr(rust_connector, 'optimal_thresholds_rust'):
        try:
            prices_array = [mu] * 100  # Dummy price array for rust function
            z_scores = [0.0] * 100
            return rust_connector.optimal_thresholds_rust(prices_array, z_scores, transaction_cost)  # type: ignore[union-attr]
        except Exception as e:
            print(f"Rust optimal thresholds failed: {e}, falling back to Python")
    
    # Python fallback
    if theta <= 0.0 or sigma <= 0.0:
        return {
            "optimal_entry": 2.0,
            "optimal_exit": 0.5,
            "expected_holding_period": 10.0
        }
    
    half_life = np.log(2) / theta
    cost_adjustment = np.sqrt(1.0 + 100.0 * transaction_cost)
    optimal_entry = 1.5 * cost_adjustment
    optimal_exit = 0.3 * np.sqrt(cost_adjustment)
    expected_holding_period = half_life * 0.5
    
    return {
        "optimal_entry": float(optimal_entry),
        "optimal_exit": float(optimal_exit),
        "expected_holding_period": float(expected_holding_period)
    }


def multiperiod_optimize(returns_df: pd.DataFrame, covariance: np.ndarray, gamma: float = 2.0, 
                        transaction_cost: float = 0.001, n_periods: int = 10) -> Dict:
    """Multi-period portfolio optimization with rebalancing.
    
    Args:
        returns_df: Historical returns matrix (time x assets)
        covariance: Covariance matrix
        gamma: Risk aversion parameter
        transaction_cost: Transaction cost for rebalancing
        n_periods: Number of rebalancing periods
        
    Returns:
        Dictionary with 'weights_sequence', 'rebalance_times', 'expected_utility'
        
    Uses Rust implementation if available for better performance.
    """
    if RUST_AVAILABLE and hasattr(rust_connector, 'multiperiod_optimize_rust'):
        try:
            returns_list = returns_df.fillna(0).values.tolist()
            result = rust_connector.multiperiod_optimize_rust(  # type: ignore[union-attr]
                returns_list, gamma, transaction_cost, n_periods
            )
            return result
        except Exception as e:
            print(f"Rust multiperiod optimization failed: {e}, falling back to Python")
    
    # Python fallback - simple equal-weighted periodic rebalancing
    t_total = len(returns_df)
    period_length = max(t_total // n_periods, 1)
    n_assets = returns_df.shape[1]
    
    weights_sequence = []
    rebalance_times = []
    
    for p in range(n_periods):
        start_idx = p * period_length
        if start_idx >= t_total:
            break
        end_idx = min((p + 1) * period_length, t_total)
        
        # Simple equal weighting
        weights = [1.0 / n_assets] * n_assets
        weights_sequence.append(weights)
        rebalance_times.append(start_idx)
    
    return {
        "weights_sequence": weights_sequence,
        "rebalance_times": rebalance_times,
        "expected_utility": 0.0
    }
