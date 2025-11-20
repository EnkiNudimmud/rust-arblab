"""Enhanced data fetching for intraday mean-reversion analysis.

Supports multiple data sources:
- Finnhub (primary)
- Yahoo Finance (fallback)
- Synthetic data (testing)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time

try:
    from python.finnhub_helper import fetch_ohlcv as fh_fetch_ohlcv
except ImportError:
    fh_fetch_ohlcv = None

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False


def fetch_intraday_data(
    symbols: List[str],
    start: str,
    end: str,
    interval: str = "1h",  # 1m, 5m, 15m, 30m, 1h, 1d
    source: str = "auto"
) -> pd.DataFrame:
    """Fetch intraday OHLCV data for multiple symbols.
    
    Args:
        symbols: List of ticker symbols
        start: Start date (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)
        end: End date
        interval: Data interval (1m, 5m, 15m, 30m, 1h, 1d)
        source: Data source ('finnhub', 'yfinance', 'auto', 'synthetic')
        
    Returns:
        DataFrame with MultiIndex (timestamp, symbol) and columns [open, high, low, close, volume]
    """
    
    if source == "auto":
        # Try sources in order of preference
        if fh_fetch_ohlcv is not None and interval in ["1h", "1d"]:
            source = "finnhub"
        elif YF_AVAILABLE:
            source = "yfinance"
        else:
            source = "synthetic"
    
    if source == "finnhub":
        return _fetch_finnhub(symbols, start, end, interval)
    elif source == "yfinance":
        return _fetch_yfinance(symbols, start, end, interval)
    else:
        return _generate_synthetic(symbols, start, end, interval)


def _fetch_yfinance(symbols: List[str], start: str, end: str, interval: str) -> pd.DataFrame:
    """Fetch data from Yahoo Finance."""
    if not YF_AVAILABLE:
        raise ImportError("yfinance not installed. Run: pip install yfinance")
    
    # Map interval names
    yf_interval_map = {
        "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
        "1h": "1h", "1d": "1d"
    }
    yf_interval = yf_interval_map.get(interval, "1h")
    
    all_data = []
    
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start, end=end, interval=yf_interval)
            
            if not df.empty:
                df = df.reset_index()
                df.columns = [c.lower() for c in df.columns]
                df['symbol'] = symbol
                df = df.rename(columns={'date': 'timestamp', 'datetime': 'timestamp'})
                all_data.append(df[['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']])
        except Exception as e:
            print(f"Warning: Failed to fetch {symbol} from yfinance: {e}")
    
    if not all_data:
        raise ValueError("No data fetched from yfinance")
    
    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.set_index(['timestamp', 'symbol']).sort_index()
    return combined


def _fetch_finnhub(symbols: List[str], start: str, end: str, interval: str) -> pd.DataFrame:
    """Fetch data from Finnhub."""
    if fh_fetch_ohlcv is None:
        raise ImportError("Finnhub helper not available")
    
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    
    all_data = []
    
    for symbol in symbols:
        try:
            df = fh_fetch_ohlcv(symbol, start_dt, end_dt)
            
            if isinstance(df, pd.DataFrame) and not df.empty:
                # Finnhub returns: t (timestamp), o, h, l, c, v
                df = df.rename(columns={'t': 'timestamp', 'o': 'open', 'h': 'high', 
                                       'l': 'low', 'c': 'close', 'v': 'volume'})
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df['symbol'] = symbol
                all_data.append(df[['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']])
            
            time.sleep(0.1)  # Rate limiting
        except Exception as e:
            print(f"Warning: Failed to fetch {symbol} from Finnhub: {e}")
    
    if not all_data:
        raise ValueError("No data fetched from Finnhub")
    
    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.set_index(['timestamp', 'symbol']).sort_index()
    return combined


def _generate_synthetic(symbols: List[str], start: str, end: str, interval: str) -> pd.DataFrame:
    """Generate synthetic intraday data for testing."""
    
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    
    # Map interval to frequency
    freq_map = {
        "1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min",
        "1h": "1h", "1d": "1d"
    }
    freq = freq_map.get(interval, "1h")
    
    # Generate timestamps (business hours only for intraday)
    if interval != "1d":
        # Business days, 9:30 AM to 4:00 PM ET
        timestamps = pd.date_range(start=start_dt, end=end_dt, freq=freq)
        timestamps = timestamps[(timestamps.hour >= 9) & (timestamps.hour < 16)]
    else:
        timestamps = pd.date_range(start=start_dt, end=end_dt, freq=freq)
    
    all_data = []
    
    for symbol in symbols:
        rng = np.random.default_rng(hash(symbol) & 0xFFFFFFFF)
        
        n_periods = len(timestamps)
        S0 = 100.0 + rng.normal(0, 20)
        
        # Generate correlated returns with mean reversion
        drift = 0.0001
        vol = 0.002
        mean_rev_speed = 0.01
        
        prices = [S0]
        for i in range(1, n_periods):
            # OU-like dynamics
            mean_rev = -mean_rev_speed * (prices[-1] - S0)
            shock = rng.normal(drift + mean_rev, vol)
            prices.append(prices[-1] * np.exp(shock))
        
        prices = np.array(prices)
        
        # Generate OHLC from close prices
        opens = prices * (1 + rng.normal(0, 0.0005, n_periods))
        highs = np.maximum(opens, prices) * (1 + abs(rng.normal(0, 0.001, n_periods)))
        lows = np.minimum(opens, prices) * (1 - abs(rng.normal(0, 0.001, n_periods)))
        volumes = rng.integers(1000000, 10000000, n_periods)
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'symbol': symbol,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': prices,
            'volume': volumes
        })
        
        all_data.append(df)
    
    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.set_index(['timestamp', 'symbol']).sort_index()
    return combined


def get_close_prices(data: pd.DataFrame) -> pd.DataFrame:
    """Extract close prices from multi-index OHLCV data.
    
    Returns:
        DataFrame with timestamps as index and symbols as columns
    """
    close_df = data['close'].unstack('symbol')
    return close_df.ffill()


def get_universe_symbols(universe: str = "sp500_tech") -> List[str]:
    """Get predefined symbol universes for testing.
    
    Args:
        universe: One of 'sp500_tech', 'crypto', 'forex', 'test'
        
    Returns:
        List of ticker symbols
    """
    universes = {
        "sp500_tech": [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO",
            "ORCL", "ADBE", "CRM", "CSCO", "ACN", "AMD", "IBM", "INTC",
            "QCOM", "TXN", "NOW", "INTU", "AMAT", "MU", "ADI", "LRCX",
            "KLAC", "SNPS", "CDNS", "MRVL", "FTNT", "PANW"
        ],
        "sp500_finance": [
            "JPM", "BAC", "WFC", "C", "GS", "MS", "BLK", "SCHW",
            "AXP", "USB", "PNC", "TFC", "COF", "BK", "STT"
        ],
        "crypto": [
            "BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD",
            "SOL-USD", "DOGE-USD", "DOT-USD", "MATIC-USD", "AVAX-USD"
        ],
        "forex": [
            "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X"
        ],
        "test": [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"
        ]
    }
    
    return universes.get(universe, universes["test"])


def resample_to_period(data: pd.DataFrame, period: str) -> pd.DataFrame:
    """Resample OHLCV data to different period.
    
    Args:
        data: DataFrame with MultiIndex (timestamp, symbol)
        period: Resample period ('5min', '15min', '1h', '1d')
        
    Returns:
        Resampled DataFrame
    """
    result = []
    
    for symbol in data.index.get_level_values('symbol').unique():
        symbol_data = data.xs(symbol, level='symbol')
        
        resampled = symbol_data.resample(period).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        resampled['symbol'] = symbol
        resampled = resampled.reset_index().set_index(['timestamp', 'symbol'])
        result.append(resampled)
    
    combined = pd.concat(result).sort_index()
    return combined
