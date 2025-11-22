"""Yahoo Finance data fetching helper with caching and optimization.

Provides utilities for optimal yfinance usage with:
- Smart caching to reduce redundant API calls
- Date range validation for intraday data
- Batch fetching optimization
- Rate limiting and retry logic

Yahoo Finance Data Limitations:
- 1m data: Last 7 days only
- 5m/15m/30m data: Last 60 days only
- 1h+ data: Multiple years available
- Rate limits apply (respect their servers!)

Examples:
    >>> from python.yfinance_helper import fetch_stocks, fetch_crypto
    >>> 
    >>> # Fetch stock data
    >>> df = fetch_stocks(['AAPL', 'MSFT'], days=30, interval='1h')
    >>> 
    >>> # Fetch crypto data
    >>> df = fetch_crypto(['BTC', 'ETH'], days=7, interval='1m')
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import time
import logging
from pathlib import Path
import json
import hashlib

logger = logging.getLogger(__name__)

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False
    logger.warning("yfinance not installed. Install with: pip install yfinance")


# Cache configuration
CACHE_DIR = Path.home() / ".cache" / "rust-hft-arbitrage-lab" / "yfinance"
CACHE_EXPIRY_MINUTES = {
    "1m": 5,      # 1-minute data expires after 5 minutes
    "5m": 15,     # 5-minute data expires after 15 minutes
    "15m": 30,    # 15-minute data expires after 30 minutes
    "30m": 60,    # 30-minute data expires after 1 hour
    "1h": 180,    # 1-hour data expires after 3 hours
    "1d": 1440,   # Daily data expires after 24 hours
}

# Data limitations
INTERVAL_LIMITS = {
    "1m": 7,      # days
    "5m": 60,     # days
    "15m": 60,    # days
    "30m": 60,    # days
    "1h": None,   # unlimited
    "2h": None,   # unlimited
    "1d": None,   # unlimited
    "1wk": None,  # unlimited
    "1mo": None,  # unlimited
}


def _get_cache_key(symbols: List[str], start: str, end: str, interval: str) -> str:
    """Generate a unique cache key for the request."""
    key_str = f"{sorted(symbols)}_{start}_{end}_{interval}"
    return hashlib.md5(key_str.encode()).hexdigest()


def _get_cached_data(cache_key: str, interval: str) -> Optional[pd.DataFrame]:
    """Retrieve cached data if available and not expired."""
    if not CACHE_DIR.exists():
        return None
    
    cache_file = CACHE_DIR / f"{cache_key}.parquet"
    metadata_file = CACHE_DIR / f"{cache_key}.json"
    
    if not cache_file.exists() or not metadata_file.exists():
        return None
    
    try:
        # Check if cache is expired
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        cache_time = datetime.fromisoformat(metadata['cached_at'])
        expiry_minutes = CACHE_EXPIRY_MINUTES.get(interval, 60)
        
        if datetime.now() - cache_time > timedelta(minutes=expiry_minutes):
            logger.info(f"Cache expired for key {cache_key}")
            return None
        
        # Load cached data
        df = pd.read_parquet(cache_file)
        logger.info(f"✅ Loaded {len(df)} rows from cache for {metadata['symbols']}")
        return df
    
    except Exception as e:
        logger.warning(f"Failed to load cache: {e}")
        return None


def _save_to_cache(df: pd.DataFrame, cache_key: str, symbols: List[str], 
                   start: str, end: str, interval: str) -> None:
    """Save data to cache."""
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
        cache_file = CACHE_DIR / f"{cache_key}.parquet"
        metadata_file = CACHE_DIR / f"{cache_key}.json"
        
        # Save data
        df.to_parquet(cache_file, index=False)
        
        # Save metadata
        metadata = {
            "symbols": symbols,
            "start": start,
            "end": end,
            "interval": interval,
            "cached_at": datetime.now().isoformat(),
            "rows": len(df)
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"💾 Cached {len(df)} rows for {symbols}")
    
    except Exception as e:
        logger.warning(f"Failed to save cache: {e}")


def clear_cache(older_than_hours: Optional[int] = None) -> int:
    """Clear cached data.
    
    Args:
        older_than_hours: If specified, only clear cache older than this many hours.
                         If None, clears all cache.
    
    Returns:
        Number of cache files removed.
    """
    if not CACHE_DIR.exists():
        return 0
    
    removed = 0
    cutoff_time = datetime.now() - timedelta(hours=older_than_hours) if older_than_hours else None
    
    for metadata_file in CACHE_DIR.glob("*.json"):
        try:
            if cutoff_time:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                cache_time = datetime.fromisoformat(metadata['cached_at'])
                
                if cache_time > cutoff_time:
                    continue
            
            # Remove both data and metadata files
            cache_key = metadata_file.stem
            data_file = CACHE_DIR / f"{cache_key}.parquet"
            
            if data_file.exists():
                data_file.unlink()
            metadata_file.unlink()
            removed += 1
        
        except Exception as e:
            logger.warning(f"Failed to remove cache file {metadata_file}: {e}")
    
    logger.info(f"🗑️  Removed {removed} cache files")
    return removed


def validate_date_range(interval: str, start: str, end: str) -> Tuple[bool, Optional[str]]:
    """Validate if date range is supported for the given interval.
    
    Returns:
        (is_valid, warning_message)
    """
    limit_days = INTERVAL_LIMITS.get(interval)
    
    if limit_days is None:
        return True, None
    
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    days_range = (end_dt - start_dt).days
    
    if days_range > limit_days:
        return False, (
            f"⚠️  Yahoo Finance: {interval} data limited to {limit_days} days.\n"
            f"   Requested: {days_range} days. Data may be incomplete.\n"
            f"   Suggestion: Use higher timeframe (1h, 1d) or shorter date range."
        )
    
    return True, None


def fetch_stocks(
    symbols: List[str],
    days: int = 30,
    interval: str = "1h",
    use_cache: bool = True
) -> pd.DataFrame:
    """Fetch stock market data with optimal settings.
    
    Args:
        symbols: List of stock tickers (e.g., ['AAPL', 'MSFT', 'TSLA'])
        days: Number of days of historical data
        interval: Data interval ('1m', '5m', '15m', '30m', '1h', '1d')
        use_cache: Whether to use cached data if available
    
    Returns:
        DataFrame with columns: timestamp, symbol, open, high, low, close, volume
    
    Examples:
        >>> df = fetch_stocks(['AAPL', 'MSFT'], days=30, interval='1h')
        >>> df = fetch_stocks(['TSLA'], days=7, interval='1m')  # Last 7 days only
    """
    if not YF_AVAILABLE:
        raise ImportError(
            "❌ yfinance not installed!\n"
            "Install it with: pip install yfinance"
        )
    
    end = datetime.now()
    start = end - timedelta(days=days)
    
    return _fetch_yfinance_optimized(
        symbols=symbols,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval=interval,
        use_cache=use_cache
    )


def fetch_crypto(
    coins: List[str],
    days: int = 7,
    interval: str = "1h",
    use_cache: bool = True
) -> pd.DataFrame:
    """Fetch cryptocurrency data from Yahoo Finance.
    
    Args:
        coins: List of coin symbols WITHOUT '-USD' suffix (e.g., ['BTC', 'ETH'])
        days: Number of days of historical data
        interval: Data interval ('1m', '5m', '15m', '30m', '1h', '1d')
        use_cache: Whether to use cached data if available
    
    Returns:
        DataFrame with columns: timestamp, symbol, open, high, low, close, volume
    
    Note:
        For better crypto coverage, consider using CCXT instead.
        See python/ccxt_helper.py
    
    Examples:
        >>> df = fetch_crypto(['BTC', 'ETH'], days=7, interval='1m')
        >>> df = fetch_crypto(['BTC'], days=30, interval='1h')
    """
    if not YF_AVAILABLE:
        raise ImportError(
            "❌ yfinance not installed!\n"
            "Install it with: pip install yfinance"
        )
    
    # Convert coin symbols to Yahoo Finance format
    yf_symbols = [f"{coin}-USD" for coin in coins]
    
    end = datetime.now()
    start = end - timedelta(days=days)
    
    df = _fetch_yfinance_optimized(
        symbols=yf_symbols,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval=interval,
        use_cache=use_cache
    )
    
    # Convert symbol back to coin format (remove -USD suffix)
    if not df.empty:
        df['symbol'] = df['symbol'].str.replace('-USD', '')
    
    return df


def _fetch_yfinance_optimized(
    symbols: List[str],
    start: str,
    end: str,
    interval: str,
    use_cache: bool = True
) -> pd.DataFrame:
    """Optimized internal yfinance fetching with caching and retry logic."""
    
    # Check cache first
    if use_cache:
        cache_key = _get_cache_key(symbols, start, end, interval)
        cached_data = _get_cached_data(cache_key, interval)
        if cached_data is not None:
            return cached_data
    
    # Validate date range
    is_valid, warning = validate_date_range(interval, start, end)
    if warning:
        print(warning)
    
    all_data = []
    failed_symbols = []
    
    print(f"📊 Fetching {len(symbols)} symbols from Yahoo Finance...")
    
    for i, symbol in enumerate(symbols, 1):
        print(f"  [{i}/{len(symbols)}] {symbol}...", end=" ")
        
        retry_count = 0
        max_retries = 3
        success = False
        
        while retry_count < max_retries and not success:
            try:
                ticker = yf.Ticker(symbol)
                
                # Fetch data with appropriate period
                start_dt = pd.to_datetime(start)
                end_dt = pd.to_datetime(end)
                days_range = (end_dt - start_dt).days
                
                if interval in ["1m", "5m", "15m", "30m", "1h"]:
                    # For intraday data, use period parameter for efficiency
                    if days_range <= 7:
                        df = ticker.history(period="7d", interval=interval)
                    elif days_range <= 60:
                        df = ticker.history(period="60d", interval=interval)
                    else:
                        df = ticker.history(start=start, end=end, interval=interval)
                else:
                    df = ticker.history(start=start, end=end, interval=interval)
                
                if not df.empty:
                    # Filter to exact date range (handle timezone-aware timestamps)
                    if df.index.tz is not None:
                        # Make comparison timestamps timezone-aware
                        start_dt_tz = start_dt.tz_localize('UTC').tz_convert(df.index.tz)
                        end_dt_tz = end_dt.tz_localize('UTC').tz_convert(df.index.tz)
                        df = df[(df.index >= start_dt_tz) & (df.index <= end_dt_tz)]
                    else:
                        df = df[(df.index >= start_dt) & (df.index <= end_dt)]
                    
                    if not df.empty:
                        df = df.reset_index()
                        df.columns = [c.lower() for c in df.columns]
                        df['symbol'] = symbol
                        
                        # Normalize timestamp column
                        if 'date' in df.columns:
                            df = df.rename(columns={'date': 'timestamp'})
                        elif 'datetime' in df.columns:
                            df = df.rename(columns={'datetime': 'timestamp'})
                        
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        
                        # Select required columns
                        df = df[['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
                        all_data.append(df)
                        
                        print(f"✅ {len(df)} bars")
                        success = True
                    else:
                        print(f"⚠️  No data in range")
                        failed_symbols.append((symbol, "No data in range"))
                        success = True
                else:
                    print(f"⚠️  No data")
                    failed_symbols.append((symbol, "No data available"))
                    success = True
                
                # Rate limiting
                time.sleep(0.3)
            
            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    wait_time = retry_count * 2
                    print(f"⚠️  Retry {retry_count}/{max_retries} in {wait_time}s...", end=" ")
                    time.sleep(wait_time)
                else:
                    print(f"❌ Failed: {str(e)[:50]}")
                    failed_symbols.append((symbol, str(e)))
    
    if not all_data:
        raise ValueError(
            f"❌ Failed to fetch data for all symbols: {symbols}\n\n"
            f"Failed symbols:\n" + 
            "\n".join([f"  • {sym}: {reason}" for sym, reason in failed_symbols[:10]]) +
            "\n\n💡 Tip: Verify symbols at finance.yahoo.com"
        )
    
    result = pd.concat(all_data, ignore_index=True)
    
    # Save to cache
    if use_cache:
        _save_to_cache(result, cache_key, symbols, start, end, interval)
    
    print(f"\n✅ Fetched {len(all_data)}/{len(symbols)} symbols, {len(result)} total bars")
    
    return result


def get_cache_info() -> Dict:
    """Get information about the current cache."""
    if not CACHE_DIR.exists():
        return {
            "cache_dir": str(CACHE_DIR),
            "exists": False,
            "files": 0,
            "size_mb": 0
        }
    
    cache_files = list(CACHE_DIR.glob("*.parquet"))
    total_size = sum(f.stat().st_size for f in cache_files)
    
    return {
        "cache_dir": str(CACHE_DIR),
        "exists": True,
        "files": len(cache_files),
        "size_mb": round(total_size / (1024 * 1024), 2)
    }


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("Yahoo Finance Helper - Example Usage")
    print("=" * 60)
    
    # Check cache info
    cache_info = get_cache_info()
    print(f"\n📦 Cache Info:")
    print(f"   Directory: {cache_info['cache_dir']}")
    print(f"   Files: {cache_info['files']}")
    print(f"   Size: {cache_info['size_mb']} MB")
    
    # Fetch stock data
    print(f"\n{'='*60}")
    print("Test 1: Fetch stock data (AAPL, MSFT)")
    print("=" * 60)
    try:
        df = fetch_stocks(['AAPL', 'MSFT'], days=7, interval='1h')
        print(f"\n📊 Result:")
        print(f"   Rows: {len(df)}")
        print(f"   Symbols: {df['symbol'].unique().tolist()}")
        print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"\n   First few rows:")
        print(df.head(3))
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Fetch crypto data
    print(f"\n{'='*60}")
    print("Test 2: Fetch crypto data (BTC, ETH)")
    print("=" * 60)
    try:
        df = fetch_crypto(['BTC', 'ETH'], days=3, interval='1h')
        print(f"\n📊 Result:")
        print(f"   Rows: {len(df)}")
        print(f"   Symbols: {df['symbol'].unique().tolist()}")
        print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"\n   First few rows:")
        print(df.head(3))
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Check cache again
    cache_info = get_cache_info()
    print(f"\n📦 Cache Info (after tests):")
    print(f"   Files: {cache_info['files']}")
    print(f"   Size: {cache_info['size_mb']} MB")
