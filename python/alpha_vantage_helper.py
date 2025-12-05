# python/alpha_vantage_helper.py
"""
Helper functions for fetching market data from Alpha Vantage API.

Free tier limitations:
- 25 API requests per day
- 5 API requests per minute
- No real-time data (15-20 minute delay)

Use this for:
- Historical stock data (daily, intraday)
- Forex data
- Crypto data
- Fundamental data (earnings, financial statements)
"""

import time
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Try to import API keys utility
try:
    from python.api_keys import get_alpha_vantage_key
    API_KEYS_AVAILABLE = True
except ImportError:
    API_KEYS_AVAILABLE = False
    logger.warning("Could not import api_keys module")

# Free tier rate limits
FREE_TIER_DAILY_LIMIT = 25
FREE_TIER_PER_MINUTE_LIMIT = 5
MIN_SECONDS_BETWEEN_CALLS = 12  # 60 seconds / 5 calls = 12 seconds


class RateLimiter:
    """Track API call count and enforce rate limits."""
    
    def __init__(self):
        self.call_timestamps: List[datetime] = []
        self.daily_calls = 0
        self.last_reset = datetime.now().date()
    
    def can_make_call(self) -> Tuple[bool, str]:
        """Check if we can make an API call without exceeding rate limits."""
        now = datetime.now()
        
        # Reset daily counter if new day
        if now.date() > self.last_reset:
            self.daily_calls = 0
            self.last_reset = now.date()
            self.call_timestamps.clear()
        
        # Check daily limit
        if self.daily_calls >= FREE_TIER_DAILY_LIMIT:
            return False, f"Daily limit reached ({FREE_TIER_DAILY_LIMIT} calls/day)"
        
        # Check per-minute limit (remove calls older than 1 minute)
        one_minute_ago = now - timedelta(minutes=1)
        self.call_timestamps = [ts for ts in self.call_timestamps if ts > one_minute_ago]
        
        if len(self.call_timestamps) >= FREE_TIER_PER_MINUTE_LIMIT:
            wait_seconds = (self.call_timestamps[0] + timedelta(minutes=1) - now).total_seconds()
            return False, f"Rate limit: wait {int(wait_seconds)}s (5 calls/minute max)"
        
        return True, ""
    
    def record_call(self):
        """Record that an API call was made."""
        now = datetime.now()
        self.call_timestamps.append(now)
        self.daily_calls += 1
    
    def get_remaining_calls(self) -> Tuple[int, int]:
        """Get remaining calls (daily, per_minute)."""
        daily_remaining = FREE_TIER_DAILY_LIMIT - self.daily_calls
        minute_remaining = FREE_TIER_PER_MINUTE_LIMIT - len(self.call_timestamps)
        return daily_remaining, minute_remaining


# Global rate limiter instance
_rate_limiter = RateLimiter()


def get_alpha_vantage_api_key() -> Optional[str]:
    """Get Alpha Vantage API key from api_keys.properties file."""
    if API_KEYS_AVAILABLE:
        return get_alpha_vantage_key()
    return None


def check_rate_limit() -> Tuple[bool, str]:
    """Check if we can make an API call. Returns (can_call, message)."""
    return _rate_limiter.can_make_call()


def get_remaining_calls() -> Tuple[int, int]:
    """Get remaining API calls. Returns (daily_remaining, per_minute_remaining)."""
    return _rate_limiter.get_remaining_calls()


def fetch_intraday(
    symbol: str,
    interval: str = "5min",
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch intraday time series data from Alpha Vantage.
    
    Args:
        symbol: Stock symbol (e.g., "AAPL", "IBM")
        interval: Time interval (1min, 5min, 15min, 30min, 60min)
        api_key: Alpha Vantage API key (auto-loaded if None)
    
    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    if api_key is None:
        api_key = get_alpha_vantage_api_key()
    
    if not api_key:
        logger.error("Alpha Vantage API key not found")
        raise ValueError("Alpha Vantage API key required. Check api_keys.properties")
    
    # Check rate limit
    can_call, message = check_rate_limit()
    if not can_call:
        logger.warning(f"Rate limit check failed: {message}")
        raise RuntimeError(f"Rate limit exceeded: {message}")
    
    try:
        import requests
        
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": interval,
            "apikey": api_key,
            "outputsize": "compact",  # Last 100 data points
            "datatype": "json"
        }
        
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        # Record successful API call
        _rate_limiter.record_call()
        
        # Check for error messages
        if "Error Message" in data:
            logger.error(f"API error: {data['Error Message']}")
            return pd.DataFrame()
        
        if "Note" in data:
            logger.warning(f"API note: {data['Note']}")
            return pd.DataFrame()
        
        # Find the time series key (varies by function)
        time_series_key = None
        for key in data.keys():
            if "Time Series" in key:
                time_series_key = key
                break
        
        if not time_series_key or time_series_key not in data:
            logger.error(f"No time series data found in response")
            return pd.DataFrame()
        
        # Parse time series data
        time_series = data[time_series_key]
        rows = []
        
        for timestamp_str, values in time_series.items():
            rows.append({
                'timestamp': pd.to_datetime(timestamp_str),
                'open': float(values['1. open']),
                'high': float(values['2. high']),
                'low': float(values['3. low']),
                'close': float(values['4. close']),
                'volume': float(values['5. volume'])
            })
        
        df = pd.DataFrame(rows)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Fetched {len(df)} intraday data points for {symbol}")
        return df
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed for {symbol}: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Failed to fetch intraday data for {symbol}: {e}")
        return pd.DataFrame()


def fetch_daily(
    symbol: str,
    api_key: Optional[str] = None,
    outputsize: str = "compact"
) -> pd.DataFrame:
    """
    Fetch daily time series data from Alpha Vantage.
    
    Args:
        symbol: Stock symbol (e.g., "AAPL", "IBM")
        api_key: Alpha Vantage API key (auto-loaded if None)
        outputsize: "compact" (last 100 days) or "full" (20+ years)
    
    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    if api_key is None:
        api_key = get_alpha_vantage_api_key()
    
    if not api_key:
        logger.error("Alpha Vantage API key not found")
        raise ValueError("Alpha Vantage API key required. Check api_keys.properties")
    
    # Check rate limit
    can_call, message = check_rate_limit()
    if not can_call:
        logger.warning(f"Rate limit check failed: {message}")
        raise RuntimeError(f"Rate limit exceeded: {message}")
    
    try:
        import requests
        
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "apikey": api_key,
            "outputsize": outputsize,
            "datatype": "json"
        }
        
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        # Record successful API call
        _rate_limiter.record_call()
        
        # Check for error messages
        if "Error Message" in data:
            logger.error(f"API error: {data['Error Message']}")
            return pd.DataFrame()
        
        if "Note" in data:
            logger.warning(f"API note: {data['Note']}")
            return pd.DataFrame()
        
        # Parse time series data
        if "Time Series (Daily)" not in data:
            logger.error(f"No daily time series data found")
            return pd.DataFrame()
        
        time_series = data["Time Series (Daily)"]
        rows = []
        
        for date_str, values in time_series.items():
            rows.append({
                'timestamp': pd.to_datetime(date_str),
                'open': float(values['1. open']),
                'high': float(values['2. high']),
                'low': float(values['3. low']),
                'close': float(values['4. close']),
                'volume': float(values['5. volume'])
            })
        
        df = pd.DataFrame(rows)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Fetched {len(df)} daily data points for {symbol}")
        return df
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed for {symbol}: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Failed to fetch daily data for {symbol}: {e}")
        return pd.DataFrame()


def fetch_quote(symbol: str, api_key: Optional[str] = None) -> Optional[Dict]:
    """
    Fetch real-time quote from Alpha Vantage.
    
    Args:
        symbol: Stock symbol (e.g., "AAPL", "IBM")
        api_key: Alpha Vantage API key (auto-loaded if None)
    
    Returns:
        Dict with quote data or None if failed
    """
    if api_key is None:
        api_key = get_alpha_vantage_api_key()
    
    if not api_key:
        logger.error("Alpha Vantage API key not found")
        return None
    
    # Check rate limit
    can_call, message = check_rate_limit()
    if not can_call:
        logger.warning(f"Rate limit check failed: {message}")
        return None
    
    try:
        import requests
        
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": symbol,
            "apikey": api_key,
            "datatype": "json"
        }
        
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        # Record successful API call
        _rate_limiter.record_call()
        
        # Check for error messages
        if "Error Message" in data:
            logger.error(f"API error: {data['Error Message']}")
            return None
        
        if "Note" in data:
            logger.warning(f"API note: {data['Note']}")
            return None
        
        # Parse quote data
        if "Global Quote" not in data or not data["Global Quote"]:
            logger.error(f"No quote data found for {symbol}")
            return None
        
        quote = data["Global Quote"]
        
        return {
            'symbol': quote.get('01. symbol', symbol),
            'price': float(quote.get('05. price', 0)),
            'volume': float(quote.get('06. volume', 0)),
            'open': float(quote.get('02. open', 0)),
            'high': float(quote.get('03. high', 0)),
            'low': float(quote.get('04. low', 0)),
            'previous_close': float(quote.get('08. previous close', 0)),
            'change': float(quote.get('09. change', 0)),
            'change_percent': quote.get('10. change percent', '0%'),
            'latest_trading_day': quote.get('07. latest trading day', '')
        }
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed for {symbol}: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to fetch quote for {symbol}: {e}")
        return None


def fetch_forex_intraday(
    from_symbol: str,
    to_symbol: str = "USD",
    interval: str = "5min",
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch forex intraday data from Alpha Vantage.
    
    Args:
        from_symbol: From currency (e.g., "EUR", "GBP")
        to_symbol: To currency (e.g., "USD")
        interval: Time interval (1min, 5min, 15min, 30min, 60min)
        api_key: Alpha Vantage API key (auto-loaded if None)
    
    Returns:
        DataFrame with columns: timestamp, open, high, low, close
    """
    if api_key is None:
        api_key = get_alpha_vantage_api_key()
    
    if not api_key:
        logger.error("Alpha Vantage API key not found")
        raise ValueError("Alpha Vantage API key required. Check api_keys.properties")
    
    # Check rate limit
    can_call, message = check_rate_limit()
    if not can_call:
        logger.warning(f"Rate limit check failed: {message}")
        raise RuntimeError(f"Rate limit exceeded: {message}")
    
    try:
        import requests
        
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "FX_INTRADAY",
            "from_symbol": from_symbol,
            "to_symbol": to_symbol,
            "interval": interval,
            "apikey": api_key,
            "outputsize": "compact",
            "datatype": "json"
        }
        
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        # Record successful API call
        _rate_limiter.record_call()
        
        # Check for error messages
        if "Error Message" in data:
            logger.error(f"API error: {data['Error Message']}")
            return pd.DataFrame()
        
        # Find time series key
        time_series_key = None
        for key in data.keys():
            if "Time Series FX" in key:
                time_series_key = key
                break
        
        if not time_series_key:
            logger.error("No forex time series data found")
            return pd.DataFrame()
        
        time_series = data[time_series_key]
        rows = []
        
        for timestamp_str, values in time_series.items():
            rows.append({
                'timestamp': pd.to_datetime(timestamp_str),
                'open': float(values['1. open']),
                'high': float(values['2. high']),
                'low': float(values['3. low']),
                'close': float(values['4. close'])
            })
        
        df = pd.DataFrame(rows)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Fetched {len(df)} forex data points for {from_symbol}/{to_symbol}")
        return df
        
    except Exception as e:
        logger.error(f"Failed to fetch forex data: {e}")
        return pd.DataFrame()


def fetch_crypto_daily(
    symbol: str,
    market: str = "USD",
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch crypto daily data from Alpha Vantage.
    
    Args:
        symbol: Crypto symbol (e.g., "BTC", "ETH")
        market: Market currency (e.g., "USD", "EUR")
        api_key: Alpha Vantage API key (auto-loaded if None)
    
    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume, market_cap
    """
    if api_key is None:
        api_key = get_alpha_vantage_api_key()
    
    if not api_key:
        logger.error("Alpha Vantage API key not found")
        raise ValueError("Alpha Vantage API key required. Check api_keys.properties")
    
    # Check rate limit
    can_call, message = check_rate_limit()
    if not can_call:
        logger.warning(f"Rate limit check failed: {message}")
        raise RuntimeError(f"Rate limit exceeded: {message}")
    
    try:
        import requests
        
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "DIGITAL_CURRENCY_DAILY",
            "symbol": symbol,
            "market": market,
            "apikey": api_key,
            "datatype": "json"
        }
        
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        # Record successful API call
        _rate_limiter.record_call()
        
        # Check for error messages
        if "Error Message" in data:
            logger.error(f"API error: {data['Error Message']}")
            return pd.DataFrame()
        
        # Parse crypto data
        if "Time Series (Digital Currency Daily)" not in data:
            logger.error("No crypto time series data found")
            return pd.DataFrame()
        
        time_series = data["Time Series (Digital Currency Daily)"]
        rows = []
        
        for date_str, values in time_series.items():
            rows.append({
                'timestamp': pd.to_datetime(date_str),
                'open': float(values[f'1a. open ({market})']),
                'high': float(values[f'2a. high ({market})']),
                'low': float(values[f'3a. low ({market})']),
                'close': float(values[f'4a. close ({market})']),
                'volume': float(values['5. volume']),
                'market_cap': float(values[f'6. market cap ({market})'])
            })
        
        df = pd.DataFrame(rows)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Fetched {len(df)} crypto daily data points for {symbol}")
        return df
        
    except Exception as e:
        logger.error(f"Failed to fetch crypto data: {e}")
        return pd.DataFrame()
