"""
CCXT Helper Module - Free Crypto Market Data Provider
======================================================

This module provides access to historical cryptocurrency data from multiple exchanges
using the CCXT library. No API keys required for public market data!

Supported Exchanges:
- Binance: Most liquid, extensive historical data
- Kraken: Reliable, good for BTC/EUR pairs
- Coinbase: US-based, good for regulated trading
- Bybit, OKX, Bitfinex, and 100+ more exchanges

Key Benefits:
- FREE access to historical OHLCV data
- Second-level, minute-level, and higher timeframes
- No API key required for public data
- Consistent unified API across all exchanges
- Better than Finnhub for crypto data
"""

import ccxt
import pandas as pd
import time
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# Recommended exchanges for different use cases
RECOMMENDED_EXCHANGES = {
    'binance': {
        'name': 'Binance',
        'description': 'Largest crypto exchange, best liquidity',
        'has_ohlcv': True,
        'max_limit': 1000,
        'rate_limit': 1200,  # ms between requests
        'timeframes': ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M'],
        'best_for': 'Most crypto pairs, high frequency data'
    },
    'kraken': {
        'name': 'Kraken',
        'description': 'Reliable, regulated exchange',
        'has_ohlcv': True,
        'max_limit': 720,
        'rate_limit': 3000,
        'timeframes': ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w', '15d'],
        'best_for': 'BTC pairs, regulatory compliance'
    },
    'coinbase': {
        'name': 'Coinbase Pro',
        'description': 'US-based, highly regulated',
        'has_ohlcv': True,
        'max_limit': 300,
        'rate_limit': 1000,
        'timeframes': ['1m', '5m', '15m', '1h', '6h', '1d'],
        'best_for': 'US trading, major pairs'
    },
    'bybit': {
        'name': 'Bybit',
        'description': 'Good for perpetual futures',
        'has_ohlcv': True,
        'max_limit': 200,
        'rate_limit': 1000,
        'timeframes': ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d', '1w', '1M'],
        'best_for': 'Derivatives, perpetual contracts'
    },
    'okx': {
        'name': 'OKX',
        'description': 'Comprehensive product range',
        'has_ohlcv': True,
        'max_limit': 300,
        'rate_limit': 1000,
        'timeframes': ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d', '1w', '1M'],
        'best_for': 'Wide variety of altcoins'
    }
}

# Map common interval formats to CCXT timeframes
INTERVAL_MAP = {
    '1m': '1m',
    '3m': '3m',
    '5m': '5m',
    '15m': '15m',
    '30m': '30m',
    '1h': '1h',
    '2h': '2h',
    '4h': '4h',
    '6h': '6h',
    '8h': '8h',
    '12h': '12h',
    '1d': '1d',
    '3d': '3d',
    '1w': '1w',
    '1M': '1M',
}


def get_available_exchanges() -> Dict[str, dict]:
    """
    Get list of recommended exchanges with their capabilities.
    
    Returns:
        Dict of exchange info keyed by exchange id
    """
    return RECOMMENDED_EXCHANGES


def create_exchange(exchange_id: str = 'binance', params: Optional[Dict] = None) -> ccxt.Exchange:
    """
    Create a CCXT exchange instance.
    
    Args:
        exchange_id: ID of the exchange (e.g., 'binance', 'kraken')
        params: Additional parameters for exchange initialization
        
    Returns:
        Initialized exchange instance
        
    Raises:
        ValueError: If exchange is not supported
    """
    if not hasattr(ccxt, exchange_id):
        available = ', '.join(RECOMMENDED_EXCHANGES.keys())
        raise ValueError(
            f"❌ Exchange '{exchange_id}' not found!\\n"
            f"Recommended exchanges: {available}\\n"
            f"All CCXT exchanges: {', '.join(ccxt.exchanges)}"
        )
    
    params = params or {}
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({
        'enableRateLimit': True,  # Respect rate limits
        'timeout': 30000,  # 30 seconds timeout
        **params
    })
    
    logger.info(f"Created {exchange.name} exchange instance (Rate limit: {exchange.rateLimit}ms)")
    return exchange


def fetch_ohlcv_data(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str = '1h',
    since: Optional[int] = None,
    limit: int = 1000
) -> pd.DataFrame:
    """
    Fetch OHLCV data from exchange.
    
    Args:
        exchange: CCXT exchange instance
        symbol: Trading pair symbol (e.g., 'BTC/USDT')
        timeframe: Candle timeframe ('1m', '5m', '1h', '1d', etc.)
        since: Start timestamp in milliseconds (optional)
        limit: Maximum number of candles to fetch
        
    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
        
    Raises:
        ValueError: If symbol or timeframe not supported
    """
    # Load markets if not already loaded
    if not exchange.markets:
        exchange.load_markets()
    
    # Validate symbol
    if symbol not in exchange.markets:
        similar = [s for s in exchange.markets if symbol.split('/')[0] in s]
        raise ValueError(
            f"❌ Symbol '{symbol}' not found on {exchange.name}!\\n"
            f"Similar symbols: {similar[:5] if similar else 'None'}\\n"
            f"Example format: 'BTC/USDT', 'ETH/BTC'"
        )
    
    # Validate timeframe
    if not exchange.has.get('fetchOHLCV'):
        raise ValueError(f"❌ {exchange.name} does not support OHLCV data fetching")
    
    if timeframe not in exchange.timeframes:
        raise ValueError(
            f"❌ Timeframe '{timeframe}' not supported by {exchange.name}\\n"
            f"Available timeframes: {list(exchange.timeframes.keys())}"
        )
    
    try:
        logger.info(f"Fetching {symbol} {timeframe} data from {exchange.name} (limit={limit})")
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
        
        if not ohlcv:
            raise ValueError(f"No data returned from {exchange.name} for {symbol}")
        
        # Convert to DataFrame
        df = pd.DataFrame(
            ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        logger.info(f"Fetched {len(df)} candles from {df['timestamp'].min()} to {df['timestamp'].max()}")
        return df
        
    except ccxt.NetworkError as e:
        raise ValueError(
            f"🌐 Network error connecting to {exchange.name}: {str(e)}\\n"
            f"Please check your internet connection and try again."
        )
    except ccxt.ExchangeError as e:
        raise ValueError(
            f"❌ Exchange error from {exchange.name}: {str(e)}\\n"
            f"The exchange may be experiencing issues or rate limits."
        )
    except Exception as e:
        raise ValueError(f"Unexpected error fetching data from {exchange.name}: {str(e)}")


def fetch_ohlcv_range(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    start_date: datetime,
    end_date: datetime,
    max_per_request: Optional[int] = None
) -> pd.DataFrame:
    """
    Fetch OHLCV data for a date range by making multiple requests if needed.
    
    Args:
        exchange: CCXT exchange instance
        symbol: Trading pair symbol
        timeframe: Candle timeframe
        start_date: Start date
        end_date: End date
        max_per_request: Maximum candles per request (uses exchange default if None)
        
    Returns:
        DataFrame with complete date range data
    """
    if max_per_request is None:
        # Use exchange-specific limits
        exchange_id = exchange.id.lower()
        max_per_request = RECOMMENDED_EXCHANGES.get(exchange_id, {}).get('max_limit', 500)
    
    # Calculate timeframe duration in milliseconds
    timeframe_duration_ms = exchange.parse_timeframe(timeframe) * 1000
    
    # Convert dates to timestamps
    since_ms = int(start_date.timestamp() * 1000)
    end_ms = int(end_date.timestamp() * 1000)
    
    all_data = []
    current_since = since_ms
    
    logger.info(f"Fetching data from {start_date} to {end_date} in chunks of {max_per_request}")
    
    max_iterations = 100  # Safety limit to prevent infinite loops
    iteration = 0
    
    while current_since < end_ms and iteration < max_iterations:
        iteration += 1
        try:
            df_chunk = fetch_ohlcv_data(exchange, symbol, timeframe, current_since, max_per_request or 1000)
            
            if df_chunk.empty:
                logger.warning(f"No more data available from {current_since}")
                break
            
            all_data.append(df_chunk)
            
            # Move to next chunk
            last_timestamp_ms = int(df_chunk['timestamp'].iloc[-1].timestamp() * 1000)
            current_since = last_timestamp_ms + timeframe_duration_ms
            
            # Respect rate limits - reduced sleep time for faster fetching
            time.sleep(min(0.5, exchange.rateLimit / 2000))  # Faster but still respectful
            
            # Log progress
            progress_pct = min(100, ((current_since - since_ms) / (end_ms - since_ms)) * 100)
            logger.info(f"Progress: {progress_pct:.1f}% - fetched {len(df_chunk)} candles")
            
        except Exception as e:
            logger.error(f"Error fetching chunk starting at {current_since}: {e}")
            break
    
    if not all_data:
        raise ValueError(f"Failed to fetch any data for {symbol} from {start_date} to {end_date}")
    
    # Combine all chunks
    df_combined = pd.concat(all_data, ignore_index=True)
    
    # Remove duplicates and sort
    df_combined = df_combined.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    
    # Filter to exact date range
    df_combined = df_combined[
        (df_combined['timestamp'] >= start_date) &
        (df_combined['timestamp'] <= end_date)
    ]
    
    logger.info(f"Successfully fetched {len(df_combined)} total candles")
    # Ensure DataFrame return type
    if isinstance(df_combined, pd.Series):
        return pd.DataFrame(df_combined)
    return df_combined


def fetch_multiple_symbols(
    exchange_id: str,
    symbols: List[str],
    timeframe: str,
    start_date: datetime,
    end_date: datetime
) -> pd.DataFrame:
    """
    Fetch data for multiple symbols and combine into a single DataFrame.
    
    Args:
        exchange_id: Exchange identifier
        symbols: List of trading pair symbols
        timeframe: Candle timeframe
        start_date: Start date
        end_date: End date
        
    Returns:
        DataFrame with MultiIndex (timestamp, symbol) and OHLCV columns
    """
    exchange = create_exchange(exchange_id)
    all_symbol_data = []
    
    for symbol in symbols:
        try:
            logger.info(f"Fetching {symbol}...")
            df = fetch_ohlcv_range(exchange, symbol, timeframe, start_date, end_date)
            df['symbol'] = symbol
            all_symbol_data.append(df)
            
            # Small delay between symbols to respect rate limits
            time.sleep(exchange.rateLimit / 1000 * 2)
            
        except Exception as e:
            logger.error(f"Failed to fetch {symbol}: {e}")
            continue
    
    if not all_symbol_data:
        raise ValueError(f"Failed to fetch data for any symbols: {symbols}")
    
    # Combine all symbol data
    combined = pd.concat(all_symbol_data, ignore_index=True)
    
    # Set multi-index
    combined = combined.set_index(['timestamp', 'symbol']).sort_index()
    
    logger.info(f"Successfully fetched data for {len(all_symbol_data)}/{len(symbols)} symbols")
    return combined


# Convenience function for quick testing
def quick_fetch(
    symbol: str = 'BTC/USDT',
    exchange_id: str = 'binance',
    timeframe: str = '1h',
    days_back: int = 7
) -> pd.DataFrame:
    """
    Quick fetch function for testing - fetches last N days of data.
    
    Example:
        >>> df = quick_fetch('ETH/USDT', 'binance', '5m', days_back=1)
        >>> print(df.head())
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    exchange = create_exchange(exchange_id)
    return fetch_ohlcv_range(exchange, symbol, timeframe, start_date, end_date)
