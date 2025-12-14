"""
Shared Utilities for Multi-Page App
====================================

Common functions used across multiple pages:
- Data fetching and caching
- Portfolio calculations
- Visualization helpers
- Performance metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import streamlit as st

# Performance metrics
def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02, periods_per_year: int = 252) -> float:
    """Calculate annualized Sharpe ratio"""
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    
    mean_return = returns.mean() * periods_per_year
    std_return = returns.std() * np.sqrt(periods_per_year)
    
    sharpe = (mean_return - risk_free_rate) / std_return
    return sharpe

def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02, periods_per_year: int = 252) -> float:
    """Calculate Sortino ratio (uses downside deviation)"""
    if len(returns) < 2:
        return 0.0
    
    mean_return = returns.mean() * periods_per_year
    downside_returns = returns[returns < 0]
    
    if len(downside_returns) == 0:
        return float('inf')
    
    downside_std = downside_returns.std() * np.sqrt(periods_per_year)
    
    if downside_std == 0:
        return float('inf')
    
    sortino = (mean_return - risk_free_rate) / downside_std
    return sortino

def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """Calculate maximum drawdown"""
    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve - running_max) / running_max
    return drawdown.min()

def calculate_calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Calculate Calmar ratio (annual return / max drawdown)"""
    if len(returns) < 2:
        return 0.0
    
    annual_return = returns.mean() * periods_per_year
    
    # Calculate equity curve for drawdown
    equity = (1 + returns).cumprod()
    max_dd = abs(calculate_max_drawdown(equity))
    
    if max_dd == 0:
        return float('inf')
    
    return annual_return / max_dd

def calculate_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Calculate Value at Risk"""
    if len(returns) == 0:
        return 0.0
    return np.percentile(returns, (1 - confidence) * 100)

def calculate_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """Calculate Conditional Value at Risk (Expected Shortfall)"""
    if len(returns) == 0:
        return 0.0
    var = calculate_var(returns, confidence)
    return returns[returns <= var].mean()

# Portfolio calculations
def calculate_portfolio_weights(returns: pd.DataFrame, method: str = 'equal') -> np.ndarray:
    """Calculate portfolio weights using various methods"""
    n_assets = returns.shape[1]
    
    if method == 'equal':
        return np.ones(n_assets) / n_assets
    
    elif method == 'inverse_volatility':
        vols = returns.std()
        inv_vols = 1 / vols
        weights = inv_vols / inv_vols.sum()
        return weights.values
    
    elif method == 'min_variance':
        cov_matrix = returns.cov()
        inv_cov = np.linalg.inv(cov_matrix)
        ones = np.ones(n_assets)
        weights = inv_cov @ ones / (ones @ inv_cov @ ones)
        return weights
    
    elif method == 'max_sharpe':
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        inv_cov = np.linalg.inv(cov_matrix)
        weights = inv_cov @ mean_returns
        weights = weights / weights.sum()
        return weights
    
    else:
        return np.ones(n_assets) / n_assets

def rebalance_portfolio(
    current_weights: Dict[str, float],
    target_weights: Dict[str, float],
    threshold: float = 0.05
) -> Dict[str, float]:
    """Calculate rebalancing trades if deviation exceeds threshold"""
    
    trades = {}
    
    for symbol in target_weights:
        current = current_weights.get(symbol, 0.0)
        target = target_weights[symbol]
        diff = target - current
        
        if abs(diff) > threshold:
            trades[symbol] = diff
    
    return trades

# Data validation
def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> Tuple[bool, str]:
    """Validate DataFrame has required columns and data"""
    
    if df is None or df.empty:
        return False, "DataFrame is empty"
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        return False, f"Missing columns: {', '.join(missing_cols)}"
    
    # Check for NaN
    if df[required_columns].isnull().any().any():
        return False, "DataFrame contains NaN values"
    
    return True, "Valid"

def clean_price_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean price data: remove NaN, outliers, etc."""
    
    # Remove NaN
    df = df.dropna()
    
    # Remove zero or negative prices
    if 'close' in df.columns:
        df = df[df['close'] > 0]  # type: ignore[assignment]
    
    # Remove outliers (prices that jump more than 50% in one period)
    if 'close' in df.columns and len(df) > 1:
        returns = df['close'].pct_change()
        df = df[abs(returns) < 0.5]  # type: ignore[assignment]
    
    return df

# Caching helpers
@st.cache_data(ttl=3600)
def cached_data_fetch(symbols: List[str], start: str, end: str, interval: str, source: str) -> pd.DataFrame:
    """Cached wrapper for data fetching"""
    from python.data.data_fetcher import fetch_intraday_data
    return fetch_intraday_data(symbols, start, end, interval, source)

# Formatting helpers
def format_currency(value: float, precision: int = 2) -> str:
    """Format value as currency"""
    return f"${value:,.{precision}f}"

def format_percentage(value: float, precision: int = 2) -> str:
    """Format value as percentage"""
    return f"{value:.{precision}f}%"

def format_number(value: float, precision: int = 2) -> str:
    """Format number with thousands separator"""
    return f"{value:,.{precision}f}"

# Date helpers
def get_trading_days_between(start: datetime, end: datetime) -> int:
    """Calculate number of trading days between two dates"""
    # Approximate: ~252 trading days per year
    days = (end - start).days
    trading_days = int(days * 252 / 365)
    return max(1, trading_days)

# State management
def initialize_session_state(defaults: Dict):
    """Initialize session state with default values"""
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def reset_session_state(keys: List[str]):
    """Reset specific session state keys"""
    for key in keys:
        if key in st.session_state:
            del st.session_state[key]

# Logging and alerts
def log_trade(trade_data: Dict):
    """Log a trade to session state"""
    if 'trade_log' not in st.session_state:
        st.session_state.trade_log = []
    
    trade_data['timestamp'] = datetime.now()
    st.session_state.trade_log.append(trade_data)

def log_error(error_message: str, context: str = ""):
    """Log an error to session state"""
    if 'error_log' not in st.session_state:
        st.session_state.error_log = []
    
    st.session_state.error_log.append({
        'timestamp': datetime.now(),
        'message': error_message,
        'context': context
    })

# Plotting helpers
def create_candlestick_chart(df: pd.DataFrame, symbol: str):
    """Create candlestick chart"""
    import plotly.graph_objects as go
    
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name=symbol
    )])
    
    fig.update_layout(
        title=f'{symbol} Price Chart',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_dark',
        height=400
    )
    
    return fig

def create_returns_distribution(returns: pd.Series, bins: int = 50):
    """Create returns distribution histogram"""
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=returns * 100,
        nbinsx=bins,
        name='Returns',
        marker_color='steelblue'
    ))
    
    # Add normal distribution overlay
    mean = returns.mean() * 100
    std = returns.std() * 100
    x = np.linspace(returns.min() * 100, returns.max() * 100, 100)
    from scipy.stats import norm
    y = norm.pdf(x, mean, std) * len(returns) * (returns.max() - returns.min()) / bins * 100
    
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='lines',
        name='Normal Distribution',
        line={'color': 'red', 'dash': 'dash'}
    ))
    
    fig.update_layout(
        title='Returns Distribution',
        xaxis_title='Return (%)',
        yaxis_title='Frequency',
        template='plotly_dark',
        height=300
    )
    
    return fig

# Risk management
def calculate_position_size(
    capital: float,
    risk_per_trade: float,
    entry_price: float,
    stop_loss: float
) -> float:
    """Calculate position size based on risk management"""
    
    if entry_price <= 0 or stop_loss <= 0:
        return 0.0
    
    risk_amount = capital * risk_per_trade
    price_risk = abs(entry_price - stop_loss)
    
    if price_risk == 0:
        return 0.0
    
    position_size = risk_amount / price_risk
    return position_size

def check_risk_limits(
    portfolio_value: float,
    position_value: float,
    max_position_size: float = 0.2,
    max_leverage: float = 2.0
) -> Tuple[bool, str]:
    """Check if position meets risk limits"""
    
    position_pct = position_value / portfolio_value
    
    if position_pct > max_position_size:
        return False, f"Position size {position_pct:.1%} exceeds limit {max_position_size:.1%}"
    
    # Add more risk checks as needed
    
    return True, "OK"
