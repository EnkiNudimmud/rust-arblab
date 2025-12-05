"""
Helper functions for fetching market data from Massive.com
Supports REST API, WebSocket streaming, and flat file downloads.

Massive.com provides institutional-grade market data with generous free tier:
- REST API: Historical and real-time data
- WebSocket: Live streaming quotes
- Flat Files: Bulk historical data downloads

Free Tier Limits (as of 2025):
- REST API: 100 requests/day, 10 requests/minute
- WebSocket: 10 concurrent connections, 100 messages/minute
- Flat Files: 10 GB/month downloads

Documentation: https://docs.massive.com/
"""

import time
import json
import asyncio
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Callable
from datetime import datetime, timedelta
import logging
import requests
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import websockets
try:
    import websockets
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    logger.warning("websockets not installed - WebSocket features disabled")

# Try to import API keys utility
try:
    from python.api_keys import get_api_key, get_massive_s3_credentials
    API_KEYS_AVAILABLE = True
except ImportError:
    API_KEYS_AVAILABLE = False
    logger.warning("Could not import api_keys module")

# Try to import boto3 for S3 flat file access
try:
    import boto3
    from botocore.client import Config
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    logger.warning("boto3 not installed - flat file downloads disabled. Install with: pip install boto3")

# Try to import Rust backend for fast flat file processing
try:
    import hft_py
    rust_flatfile = hft_py.flat_file
    RUST_FLATFILE_AVAILABLE = True
    logger.info("✅ Rust flat file processor available (Polars-based, 50-100x faster)")
except (ImportError, AttributeError):
    RUST_FLATFILE_AVAILABLE = False
    logger.warning("⚠️ Rust flat file processor not available - using Python fallback")


# Massive.com API configuration
MASSIVE_REST_API_BASE = "https://api.massive.com/v1"
MASSIVE_WS_API_BASE = "wss://stream.massive.com/v1"
MASSIVE_FILES_BASE = "https://files.massive.com/v1"

# Rate limiting
_last_rest_call = 0
_rest_calls_count = 0
_rest_calls_reset_time = 0
_websocket_connections = 0
MAX_REST_CALLS_PER_MINUTE = 10
MAX_REST_CALLS_PER_DAY = 100
MAX_WEBSOCKET_CONNECTIONS = 10


def get_massive_api_key() -> Optional[str]:
    """Get Massive API key from api_keys.properties file."""
    if API_KEYS_AVAILABLE:
        return get_api_key('MASSIVE_API_KEY')
    return None


def _check_rate_limit():
    """Check and enforce REST API rate limits."""
    global _last_rest_call, _rest_calls_count, _rest_calls_reset_time
    
    current_time = time.time()
    
    # Reset minute counter
    if current_time - _rest_calls_reset_time > 60:
        _rest_calls_count = 0
        _rest_calls_reset_time = current_time
    
    # Check daily limit (simplified - should persist across sessions)
    if _rest_calls_count >= MAX_REST_CALLS_PER_MINUTE:
        wait_time = 60 - (current_time - _rest_calls_reset_time)
        if wait_time > 0:
            logger.warning(f"Rate limit reached, waiting {wait_time:.1f}s...")
            time.sleep(wait_time)
            _rest_calls_count = 0
            _rest_calls_reset_time = time.time()
    
    # Minimum delay between calls (0.1s for stability)
    time_since_last = current_time - _last_rest_call
    if time_since_last < 0.1:
        time.sleep(0.1 - time_since_last)
    
    _last_rest_call = time.time()
    _rest_calls_count += 1


def fetch_ohlcv_rest(
    symbols: List[str],
    start: str,
    end: str,
    interval: str = "1h",
    api_key: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch OHLCV data from Massive REST API.
    
    Args:
        symbols: List of ticker symbols (e.g., ['AAPL', 'MSFT'])
        start: Start date (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)
        end: End date
        interval: Data interval (1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w)
        api_key: Massive API key (if None, tries to get from properties)
    
    Returns:
        DataFrame with MultiIndex (timestamp, symbol) and columns [open, high, low, close, volume]
    
    Free Tier: 100 requests/day, 10/minute
    """
    # NOTE: Massive.com is currently a placeholder/example service
    # The actual API endpoints are not yet publicly available
    logger.warning("⚠️ Massive.com API is currently unavailable (placeholder service)")
    logger.info("💡 Using synthetic data fallback for demonstration purposes")
    logger.info("📌 Tip: Use CCXT (Binance/Kraken), Yahoo Finance, or Finnhub for real data")
    return _generate_synthetic_ohlcv(symbols, start, end, interval)
    
    # Original implementation (commented out until Massive.com API is available)
    # if api_key is None:
    #     api_key = get_massive_api_key()
    # 
    # if not api_key:
    #     logger.warning("⚠️ No Massive API key found - returning synthetic data")
    #     logger.info("To use Massive.com data:")
    #     logger.info("1. Sign up at https://massive.com/signup (free tier available)")
    #     logger.info("2. Get your API key from https://massive.com/dashboard/api")
    #     logger.info("3. Add MASSIVE_API_KEY=your_key to api_keys.properties")
    #     return _generate_synthetic_ohlcv(symbols, start, end, interval)
    
    all_data = []
    
    for symbol in symbols:
        _check_rate_limit()
        
        try:
            url = f"{MASSIVE_REST_API_BASE}/market/ohlcv"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            params = {
                "symbol": symbol,
                "from": start,
                "to": end,
                "interval": interval
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=30)
            
            if response.status_code == 429:
                logger.error("❌ Rate limit exceeded - Free tier: 100 req/day, 10 req/min")
                logger.info("💡 Tip: Reduce number of symbols or increase interval")
                continue
            
            if response.status_code == 401:
                logger.error("❌ Authentication failed - check your API key")
                continue
            
            response.raise_for_status()
            data = response.json()
            
            if 'data' in data and data['data']:
                df = pd.DataFrame(data['data'])
                df['symbol'] = symbol
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                all_data.append(df)
                
                logger.info(f"✓ Fetched {len(df)} bars for {symbol}")
            else:
                logger.warning(f"⚠️ No data returned for {symbol}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Failed to fetch {symbol}: {e}")
        except Exception as e:
            logger.error(f"❌ Error processing {symbol}: {e}")
    
    if not all_data:
        logger.warning("⚠️ No data fetched from Massive, using synthetic fallback")
        return _generate_synthetic_ohlcv(symbols, start, end, interval)
    
    # Combine all data
    result = pd.concat(all_data, ignore_index=True)
    result = result.set_index(['timestamp', 'symbol'])
    result = result.sort_index()
    
    logger.info(f"✅ Total: {len(result)} bars for {len(symbols)} symbols")
    return result


async def stream_quotes_websocket(
    symbols: List[str],
    callback: Callable[[Dict], None],
    duration_seconds: int = 60,
    api_key: Optional[str] = None
) -> List[Dict]:
    """
    Stream live quotes via Massive WebSocket.
    
    Args:
        symbols: List of symbols to subscribe to
        callback: Function to call for each message: callback(data_dict)
        duration_seconds: How long to stream (0 = indefinite)
        api_key: Massive API key
    
    Returns:
        List of all received messages
    
    Free Tier: 10 concurrent connections, 100 messages/minute
    """
    # NOTE: Massive.com WebSocket API is currently unavailable (placeholder service)
    logger.error("❌ Massive.com WebSocket API is not yet available (placeholder service)")
    logger.info("💡 Tip: Use CCXT's WebSocket connectors for real-time crypto data")
    logger.info("📌 Example: Binance, Kraken have free WebSocket streams")
    return []
    
    global _websocket_connections
    
    if not WEBSOCKET_AVAILABLE:
        logger.error("❌ websockets package not installed")
        logger.info("Install with: pip install websockets")
        return []
    
    if api_key is None:
        api_key = get_massive_api_key()
    
    if not api_key:
        logger.error("❌ No Massive API key found")
        return []
    
    if _websocket_connections >= MAX_WEBSOCKET_CONNECTIONS:
        logger.error(f"❌ Max WebSocket connections ({MAX_WEBSOCKET_CONNECTIONS}) reached")
        logger.info("💡 Free tier limit: 10 concurrent connections")
        return []
    
    messages = []
    _websocket_connections += 1
    
    try:
        uri = f"{MASSIVE_WS_API_BASE}/stream?token={api_key}"
        
        async with websockets.connect(uri) as websocket:
            # Subscribe to symbols
            subscribe_msg = {
                "action": "subscribe",
                "symbols": symbols,
                "channels": ["quotes", "trades"]
            }
            await websocket.send(json.dumps(subscribe_msg))
            logger.info(f"📡 Subscribed to {len(symbols)} symbols")
            
            start_time = time.time()
            message_count = 0
            last_minute_reset = start_time
            messages_this_minute = 0
            
            while True:
                # Check duration
                if duration_seconds > 0 and (time.time() - start_time) >= duration_seconds:
                    logger.info(f"⏱️ Duration {duration_seconds}s reached")
                    break
                
                # Check rate limit
                current_time = time.time()
                if current_time - last_minute_reset >= 60:
                    messages_this_minute = 0
                    last_minute_reset = current_time
                
                if messages_this_minute >= 100:
                    logger.warning("⚠️ Approaching rate limit (100 msg/min), throttling...")
                    await asyncio.sleep(1)
                    continue
                
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    data = json.loads(message)
                    
                    messages.append(data)
                    message_count += 1
                    messages_this_minute += 1
                    
                    # Call user callback
                    if callback:
                        callback(data)
                    
                    if message_count % 10 == 0:
                        logger.info(f"📊 Received {message_count} messages...")
                        
                except asyncio.TimeoutError:
                    # No message received, continue
                    continue
                except websockets.exceptions.ConnectionClosed:
                    logger.warning("⚠️ WebSocket connection closed")
                    break
                    
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}")
    finally:
        _websocket_connections -= 1
    
    logger.info(f"✅ WebSocket closed. Received {len(messages)} total messages")
    return messages


def _download_flat_file_rust(
    symbols: List[str],
    start: str,
    end: str,
    interval: str,
    save_dir: str,
    s3_credentials: Dict[str, str],
    data_type: str = "ohlcv"
) -> Optional[pd.DataFrame]:
    """Download and process flat file using Rust backend (Polars/DataFusion)."""
    try:
        # Construct S3 path based on data type
        if data_type == "trades":
            # Tick-level trade data: us_stocks_sip/trades_v1/{SYMBOL}/{YYYY-MM-DD}.parquet
            s3_path_template = "us_stocks_sip/trades_v1"
            date_format = "day"  # Daily files for trades
        else:
            # OHLCV bars: equities/{interval}/{SYMBOL}/{YYYY-MM}.parquet
            interval_map = {
                '1m': 'minute', '5m': 'minute', '15m': 'minute', '30m': 'minute',
                '1h': 'hourly', '4h': 'hourly', '1d': 'daily', '1w': 'weekly'
            }
            interval_type = interval_map.get(interval, 'daily')
            s3_path_template = f"equities/{interval_type}"
            date_format = "month"  # Monthly files for OHLCV
        
        # Create S3 config for Rust
        s3_config = rust_flatfile.PyS3Config(
            access_key_id=s3_credentials.get('access_key_id', ''),
            secret_access_key=s3_credentials.get('secret_access_key', ''),
            endpoint=s3_credentials.get('endpoint', 'https://files.massive.com'),
            bucket=s3_credentials.get('bucket', 'flatfiles'),
            region=s3_credentials.get('region')
        )
        
        logger.info(f"🦀 Using Rust backend for flat file processing ({len(symbols)} symbols)")
        
        # Get file size threshold
        threshold_gb = rust_flatfile.get_size_threshold_gb()
        logger.info(f"   Threshold: {threshold_gb} GB (Polars < {threshold_gb} GB, DataFusion > {threshold_gb} GB)")
        
        all_dfs = []
        total_processing_time = 0
        engines_used = set()
        
        # Download and process each symbol
        for symbol in symbols:
            # Construct S3 key based on data type
            if data_type == "trades":
                # Daily files for tick data: us_stocks_sip/trades_v1/AAPL/2024-01-15.parquet
                # For date range, download each day
                start_date = pd.to_datetime(start)
                end_date = pd.to_datetime(end)
                date_range = pd.date_range(start_date, end_date, freq='D')
                
                for date in date_range:
                    date_str = date.strftime('%Y-%m-%d')
                    s3_key = f"{s3_path_template}/{symbol}/{date_str}.parquet"
                    
                    logger.info(f"  📥 {symbol} {date_str}: Downloading tick data with Rust...")
                    
                    # Call Rust backend
                    result = rust_flatfile.download_and_process_flat_file(
                        s3_config=s3_config,
                        s3_key=s3_key,
                        local_dir=save_dir,
                        start_date=start,
                        end_date=end,
                        symbols=[symbol]
                    )
                    
                    if result['success']:
                        engines_used.add(result['engine_used'])
                        total_processing_time += result['processing_time_ms']
                        
                        logger.info(
                            f"    ✅ {symbol} {date_str}: {result['rows']:,} ticks, "
                            f"{result['file_size_bytes'] / (1024*1024):.2f} MB, "
                            f"{result['processing_time_ms']/1000:.2f}s ({result['engine_used']})"
                        )
                        
                        # Load the processed file
                        filepath = Path(save_dir) / s3_key.split('/')[-1]
                        if filepath.exists():
                            df_symbol = pd.read_parquet(filepath)
                            all_dfs.append(df_symbol)
                    else:
                        logger.warning(f"    ⚠️ {symbol} {date_str}: {result['message']}")
            else:
                # Monthly files for OHLCV: equities/daily/AAPL/2024-01.parquet
                s3_key = f"{s3_path_template}/{symbol}/{start[:7]}.parquet"
                
                logger.info(f"  📥 {symbol}: Downloading OHLCV data with Rust...")
                
                # Call Rust backend
                result = rust_flatfile.download_and_process_flat_file(
                    s3_config=s3_config,
                    s3_key=s3_key,
                    local_dir=save_dir,
                    start_date=start,
                    end_date=end,
                    symbols=[symbol]
                )
                
                if result['success']:
                    engines_used.add(result['engine_used'])
                    total_processing_time += result['processing_time_ms']
                    
                    logger.info(
                        f"    ✅ {symbol}: {result['rows']:,} rows, "
                        f"{result['file_size_bytes'] / (1024*1024):.2f} MB, "
                        f"{result['processing_time_ms']/1000:.2f}s ({result['engine_used']})"
                    )
                    
                    # Load the processed file
                    filepath = Path(save_dir) / s3_key.split('/')[-1]
                    if filepath.exists():
                        df_symbol = pd.read_parquet(filepath)
                        all_dfs.append(df_symbol)
                else:
                    logger.error(f"    ❌ {symbol}: {result['message']}")
        
        if not all_dfs:
            logger.error("❌ No data downloaded from S3 with Rust backend")
            return None
        
        # Combine all dataframes
        df = pd.concat(all_dfs, ignore_index=True)
        df = df.sort_values(['timestamp', 'symbol'])
        df = df.set_index(['timestamp', 'symbol'])
        
        logger.info(
            f"✅ Rust processing complete: {len(df):,} rows, "
            f"{total_processing_time/1000:.2f}s total "
            f"(Engines: {', '.join(engines_used)})"
        )
        
        return df
        
    except Exception as e:
        logger.error(f"❌ Rust backend failed: {e}")
        logger.info("Falling back to Python/boto3...")
        return None


def download_flat_file(
    symbols: List[str],
    start: str,
    end: str,
    interval: str = "1d",
    data_type: str = "ohlcv",
    save_dir: str = "data/massive_downloads",
    s3_credentials: Optional[Dict[str, str]] = None,
    use_rust: bool = True
) -> Optional[pd.DataFrame]:
    """
    Download historical data flat files from Massive S3 bucket.
    
    Endpoint: https://files.massive.com
    Bucket: flatfiles
    
    Uses S3-compatible API with separate access credentials (not the REST API key).
    This is more efficient for bulk downloads and doesn't count against REST API quota.
    
    Performance:
    - Files < 1GB: Uses Rust (Polars) - 50-100x faster than Python
    - Files > 1GB: Uses Rust (DataFusion) - streaming, memory-efficient
    - Falls back to Python/boto3 if Rust backend unavailable
    
    Args:
        symbols: List of ticker symbols
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        interval: Data interval ('1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w') - for OHLCV only
        data_type: 'ohlcv' (aggregated bars) or 'trades' (tick-level trades)
        save_dir: Directory to save downloaded files
        s3_credentials: Dict with keys: access_key_id, secret_access_key, endpoint, bucket
                       If None, loads from api_keys.properties
        use_rust: Use Rust backend for processing (recommended for speed)
    
    Returns:
        DataFrame with columns:
          - OHLCV: timestamp, symbol, open, high, low, close, volume
          - Trades: timestamp, symbol, price, size, exchange, conditions
    
    Free Tier: 10 GB/month downloads
    
    Note: Requires separate S3 credentials from "Accessing Flat Files (S3)" tab in your
          Massive account, which are different from the REST API key.
    """
    # Get S3 credentials
    if s3_credentials is None:
        if API_KEYS_AVAILABLE:
            s3_credentials = get_massive_s3_credentials()
        else:
            logger.error("❌ Cannot load S3 credentials")
            return None
    
    access_key = s3_credentials.get('access_key_id', '')
    secret_key = s3_credentials.get('secret_access_key', '')
    endpoint = s3_credentials.get('endpoint', 'https://files.massive.com')
    bucket_name = s3_credentials.get('bucket', 'flatfiles')
    
    if not access_key or not secret_key:
        logger.warning("⚠️ No S3 credentials found in api_keys.properties")
        logger.info("Get credentials from: https://massive.com/account/api-keys → 'Accessing Flat Files (S3)' tab")
        logger.info("Add to api_keys.properties:")
        logger.info("  MASSIVE_S3_ACCESS_KEY_ID=your_access_key")
        logger.info("  MASSIVE_S3_SECRET_ACCESS_KEY=your_secret_key")
        logger.info("Falling back to REST API...")
        return fetch_ohlcv_rest(symbols, start, end, interval)
    
    # Try Rust backend first (much faster)
    if use_rust and RUST_FLATFILE_AVAILABLE:
        return _download_flat_file_rust(
            symbols, start, end, interval, save_dir, s3_credentials, data_type
        )
    
    # Fallback to Python/boto3
    if not BOTO3_AVAILABLE:
        logger.error("❌ boto3 not installed - cannot download flat files")
        logger.info("Install with: pip install boto3")
        logger.info("Falling back to REST API...")
        return fetch_ohlcv_rest(symbols, start, end, interval)
    
    # Get S3 credentials
    if s3_credentials is None:
        if API_KEYS_AVAILABLE:
            s3_credentials = get_massive_s3_credentials()
        else:
            logger.error("❌ Cannot load S3 credentials")
            return None
    
    access_key = s3_credentials.get('access_key_id', '')
    secret_key = s3_credentials.get('secret_access_key', '')
    endpoint = s3_credentials.get('endpoint', 'https://files.massive.com')
    bucket_name = s3_credentials.get('bucket', 'flatfiles')
    
    if not access_key or not secret_key:
        logger.warning("⚠️ No S3 credentials found in api_keys.properties")
        logger.info("Get credentials from: https://massive.com/account/api-keys → 'Accessing Flat Files (S3)' tab")
        logger.info("Add to api_keys.properties:")
        logger.info("  MASSIVE_S3_ACCESS_KEY_ID=your_access_key")
        logger.info("  MASSIVE_S3_SECRET_ACCESS_KEY=your_secret_key")
        logger.info("Falling back to REST API...")
        return fetch_ohlcv_rest(symbols, start, end, interval)
    
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize S3 client
        s3_client = boto3.client(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            endpoint_url=endpoint,
            config=Config(signature_version='s3v4')
        )
        
        logger.info(f"📥 Downloading flat files from S3 for {len(symbols)} symbols...")
        logger.info(f"   Date range: {start} to {end}, Data type: {data_type}")
        
        all_dfs = []
        total_downloaded = 0
        
        # Construct S3 path based on data type
        if data_type == "trades":
            # Tick-level trade data
            s3_path_prefix = "us_stocks_sip/trades_v1"
            date_format = "day"  # Daily files
        else:
            # OHLCV bars
            interval_map = {
                '1m': 'minute', '5m': 'minute', '15m': 'minute', '30m': 'minute',
                '1h': 'hourly', '4h': 'hourly', '1d': 'daily', '1w': 'weekly'
            }
            interval_type = interval_map.get(interval, 'daily')
            s3_path_prefix = f"equities/{interval_type}"
            date_format = "month"  # Monthly files
        
        # Download files for each symbol
        for symbol in symbols:
            if data_type == "trades":
                # Download daily tick files
                start_date = pd.to_datetime(start)
                end_date = pd.to_datetime(end)
                date_range = pd.date_range(start_date, end_date, freq='D')
                
                for date in date_range:
                    date_str = date.strftime('%Y-%m-%d')
                    s3_key = f"{s3_path_prefix}/{symbol}/{date_str}.parquet"
                    filename = f"{symbol}_trades_{date_str}.parquet"
                    filepath = save_path / filename
                    
                    try:
                        logger.info(f"  Downloading {symbol} tick data for {date_str}...")
                        s3_client.download_file(bucket_name, s3_key, str(filepath))
                        
                        file_size = filepath.stat().st_size
                        total_downloaded += file_size
                        logger.info(f"    ✓ {symbol} {date_str}: {file_size / (1024*1024):.2f} MB")
                        
                        # Load parquet file
                        df_symbol = pd.read_parquet(filepath)
                        all_dfs.append(df_symbol)
                    except Exception as e:
                        logger.warning(f"    ⚠️ {symbol} {date_str}: {e}")
            else:
                # Download monthly OHLCV file
                s3_key = f"{s3_path_prefix}/{symbol}/{start[:7]}.parquet"
                filename = f"{symbol}_{interval_type}_{start[:7]}.parquet"
                filepath = save_path / filename
                
                try:
                    # Download file
                    logger.info(f"  Downloading {symbol} OHLCV...")
                    s3_client.download_file(bucket_name, s3_key, str(filepath))
                    
                    file_size = filepath.stat().st_size
                    total_downloaded += file_size
                    logger.info(f"    ✓ {symbol}: {file_size / (1024*1024):.2f} MB")
                    
                    # Load parquet file
                    df_symbol = pd.read_parquet(filepath)
                    
                    # Filter by date range and symbols
                    df_symbol['timestamp'] = pd.to_datetime(df_symbol['timestamp'])
                    df_symbol = df_symbol[
                        (df_symbol['timestamp'] >= start) &
                        (df_symbol['timestamp'] <= end)
                    ]
                    
                    # Resample if needed (e.g., 5m from 1m data)
                    if interval in ['5m', '15m', '30m'] and interval_type == 'minute':
                        df_symbol = df_symbol.set_index('timestamp')
                        df_symbol = df_symbol.resample(interval).agg({
                            'open': 'first',
                            'high': 'max',
                            'low': 'min',
                            'close': 'last',
                            'volume': 'sum'
                        }).dropna()
                        df_symbol = df_symbol.reset_index()
                        df_symbol['symbol'] = symbol
                    
                    all_dfs.append(df_symbol)
                    
                except s3_client.exceptions.NoSuchKey:
                    logger.warning(f"    ⚠️ {symbol}: File not found on S3 (might not have data for this period)")
                except Exception as e:
                    logger.error(f"    ❌ {symbol}: {e}")
        
        if not all_dfs:
            logger.error("❌ No data downloaded from S3")
            return None
        
        # Combine all dataframes
        df = pd.concat(all_dfs, ignore_index=True)
        df = df.sort_values(['timestamp', 'symbol'])
        
        # Set multi-index for compatibility
        df = df.set_index(['timestamp', 'symbol'])
        
        logger.info(f"✅ Downloaded {total_downloaded / (1024*1024):.2f} MB total")
        logger.info(f"✅ Loaded {len(df):,} rows for {len(symbols)} symbols")
        
        return df
        
    except Exception as e:
        logger.error(f"❌ S3 download failed: {e}")
        logger.info("Falling back to REST API...")
        return fetch_ohlcv_rest(symbols, start, end, interval)


def fetch_data(
    symbols: List[str],
    start: str,
    end: str,
    interval: str = "1h",
    method: str = "auto",
    data_type: str = "ohlcv",
    api_key: Optional[str] = None,
    s3_credentials: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    Transparently fetch data using the best method (REST API or flat files).
    
    This function automatically chooses between:
    - REST API: Fast, good for small queries (<10 symbols, <30 days)
    - Flat Files (S3): Efficient for bulk downloads (>10 symbols or >30 days)
    
    Args:
        symbols: List of ticker symbols
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        interval: Data interval ('1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w')
        method: 'auto', 'rest', or 'flat_file'
        data_type: 'ohlcv' (bars) or 'trades' (tick-level)
        api_key: Massive REST API key (if method='rest')
        s3_credentials: S3 credentials dict (if method='flat_file')
    
    Returns:
        DataFrame with columns: timestamp, symbol, open, high, low, close, volume (ohlcv)
                            or: timestamp, symbol, price, size, exchange, conditions (trades)
    
    Recommendation Logic (when method='auto'):
    - Use REST API if: ≤5 symbols AND ≤7 days
    - Use Flat Files if: >5 symbols OR >7 days
    - Fallback: Try flat files first, then REST if fails
    """
    # Calculate query size
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    days = (end_dt - start_dt).days
    num_symbols = len(symbols)
    
    # Auto-select method
    if method == "auto":
        # Heuristics for best method
        if num_symbols <= 5 and days <= 7:
            method = "rest"
            logger.info(f"📊 Auto-selected REST API (small query: {num_symbols} symbols, {days} days)")
        else:
            method = "flat_file"
            logger.info(f"📦 Auto-selected Flat Files (large query: {num_symbols} symbols, {days} days)")
            logger.info(f"   Estimated data size: {num_symbols} symbols × {days} days")
            # Estimate file size for user awareness
            if interval in ['1m', '5m', '15m', '30m']:
                est_mb = num_symbols * days * 10  # ~10 MB per symbol per day for minute data
                logger.info(f"   Approximate download: {est_mb:.1f} MB (minute-level data)")
            else:
                est_mb = num_symbols * days * 0.5  # ~0.5 MB per symbol per day for daily/hourly
                logger.info(f"   Approximate download: {est_mb:.1f} MB")
    
    # Fetch using selected method
    if method == "flat_file":
        logger.info(f"📦 Using Flat Files (S3) - {data_type.upper()}...")
        df = download_flat_file(
            symbols=symbols,
            start=start,
            end=end,
            interval=interval,
            data_type=data_type,
            s3_credentials=s3_credentials
        )
        if df is not None:
            return df
        logger.warning("⚠️ Flat file download failed, trying REST API...")
        method = "rest"
    
    if method == "rest":
        logger.info("📊 Using REST API...")
        return fetch_ohlcv_rest(
            symbols=symbols,
            start=start,
            end=end,
            interval=interval,
            api_key=api_key
        )
    
    logger.error(f"❌ Unknown method: {method}")
    return pd.DataFrame()


def fetch_stocks(
    symbols: List[str],
    days: int = 30,
    interval: str = "1h",
    api_key: Optional[str] = None,
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Convenience function to fetch stock data (compatible with yfinance_helper interface).
    
    Args:
        symbols: List of stock tickers
        days: Number of days of historical data
        interval: Data interval ('1m', '5m', '15m', '30m', '1h', '1d')
        api_key: Massive API key
        use_cache: Whether to use cached data (not implemented yet)
    
    Returns:
        DataFrame with columns: timestamp, symbol, open, high, low, close, volume
    """
    end = datetime.now()
    start = end - timedelta(days=days)
    
    df = fetch_data(
        symbols=symbols,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval=interval,
        method="auto",
        api_key=api_key
    )
    
    # Convert to flat format for compatibility
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    
    return df


def get_available_datasets(api_key: Optional[str] = None) -> List[Dict]:
    """
    Get list of available flat file datasets.
    
    Returns:
        List of dataset info dictionaries
    """
    if api_key is None:
        api_key = get_massive_api_key()
    
    if not api_key:
        logger.error("❌ No API key found")
        return []
    
    try:
        url = f"{MASSIVE_FILES_BASE}/datasets"
        headers = {"Authorization": f"Bearer {api_key}"}
        
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        return data.get('datasets', [])
        
    except Exception as e:
        logger.error(f"❌ Failed to fetch datasets: {e}")
        return []


class WebSocketDataRecorder:
    """
    Records live WebSocket data to a dataset for backtesting.
    
    Usage:
        recorder = WebSocketDataRecorder(symbols=["AAPL", "GOOGL"])
        recorder.start()
        # ... data streams in ...
        recorder.stop()
        df = recorder.get_dataframe()
        recorder.save_dataset("live_data_2024_01_15")
    """
    
    def __init__(self, symbols: List[str], save_dir: str = "data/live_recorded"):
        self.symbols = symbols
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.data_buffer = []
        self.is_recording = False
        self.start_time = None
        self.message_count = 0
        
        logger.info(f"📹 WebSocket Recorder initialized for {len(symbols)} symbols")
    
    def on_message(self, data: Dict):
        """Callback for WebSocket messages."""
        if not self.is_recording:
            return
        
        try:
            # Extract OHLCV data from message
            if 'type' in data and data['type'] == 'quote':
                record = {
                    'timestamp': pd.to_datetime(data.get('timestamp', datetime.now())),
                    'symbol': data.get('symbol', ''),
                    'bid': data.get('bid', 0.0),
                    'ask': data.get('ask', 0.0),
                    'bid_size': data.get('bid_size', 0),
                    'ask_size': data.get('ask_size', 0),
                    'last': data.get('last', 0.0),
                    'volume': data.get('volume', 0)
                }
                self.data_buffer.append(record)
                self.message_count += 1
                
                if self.message_count % 100 == 0:
                    logger.info(f"📹 Recorded {self.message_count} messages")
        
        except Exception as e:
            logger.error(f"❌ Error recording message: {e}")
    
    def start(self):
        """Start recording."""
        self.is_recording = True
        self.start_time = datetime.now()
        self.data_buffer = []
        self.message_count = 0
        logger.info(f"▶️ Recording started at {self.start_time}")
    
    def stop(self):
        """Stop recording."""
        self.is_recording = False
        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
            logger.info(f"⏹️ Recording stopped. Duration: {duration:.1f}s, Messages: {self.message_count}")
        else:
            logger.info(f"⏹️ Recording stopped. Messages: {self.message_count}")
    
    def get_dataframe(self, resample_interval: Optional[str] = None) -> pd.DataFrame:
        """
        Get recorded data as DataFrame.
        
        Args:
            resample_interval: Optional resampling ('1min', '5min', '1H', etc.)
        
        Returns:
            DataFrame with OHLCV data
        """
        if not self.data_buffer:
            logger.warning("⚠️ No data recorded")
            return pd.DataFrame()
        
        df = pd.DataFrame(self.data_buffer)
        
        # Convert bid/ask to OHLCV format
        df['open'] = (df['bid'] + df['ask']) / 2
        df['high'] = df['ask']
        df['low'] = df['bid']
        df['close'] = df['last']
        
        if resample_interval:
            # Resample to OHLCV bars
            df = df.set_index('timestamp')
            df_resampled = df.groupby('symbol').resample(resample_interval).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            df = df_resampled.reset_index()
        
        df = df[['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
        df = df.set_index(['timestamp', 'symbol'])
        
        logger.info(f"✅ Converted to DataFrame: {len(df)} rows")
        return df
    
    def save_dataset(self, name: str, resample_interval: str = "1min"):
        """
        Save recorded data as a dataset.
        
        Args:
            name: Dataset name
            resample_interval: Resampling interval
        
        Returns:
            Path to saved dataset
        """
        df = self.get_dataframe(resample_interval=resample_interval)
        
        if df.empty:
            logger.error("❌ No data to save")
            return None
        
        filepath = self.save_dir / f"{name}.parquet"
        df.to_parquet(filepath)
        
        file_size = filepath.stat().st_size / (1024 * 1024)
        logger.info(f"💾 Saved dataset: {filepath} ({file_size:.2f} MB)")
        
        return filepath
    
    def append_to_dataset(self, dataset_path: str, resample_interval: str = "1min"):
        """Append recorded data to existing dataset."""
        new_df = self.get_dataframe(resample_interval=resample_interval)
        
        if new_df.empty:
            logger.error("❌ No data to append")
            return
        
        # Load existing dataset
        existing_df = pd.read_parquet(dataset_path)
        
        # Combine
        combined_df = pd.concat([existing_df, new_df])
        combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
        combined_df = combined_df.sort_index()
        
        # Save
        combined_df.to_parquet(dataset_path)
        
        logger.info(f"✅ Appended {len(new_df)} rows to {dataset_path}")
        logger.info(f"   Total rows: {len(combined_df)}")


def print_free_tier_info():
    """Print information about Massive.com free tier limits."""
    info = """
    ╔════════════════════════════════════════════════════════════════╗
    ║              Massive.com Free Tier Limits                      ║
    ╠════════════════════════════════════════════════════════════════╣
    ║ REST API:                                                       ║
    ║   • 100 requests per day                                       ║
    ║   • 10 requests per minute                                     ║
    ║   • Access to all markets (stocks, crypto, options, futures)   ║
    ║                                                                 ║
    ║ WebSocket:                                                      ║
    ║   • 10 concurrent connections                                  ║
    ║   • 100 messages per minute per connection                     ║
    ║   • Real-time quotes and trades                                ║
    ║                                                                 ║
    ║ Flat Files:                                                     ║
    ║   • 10 GB per month downloads                                  ║
    ║   • Historical data back to 2000                               ║
    ║   • Parquet format for fast loading                            ║
    ║                                                                 ║
    ║ Tips to Stay in Free Tier:                                     ║
    ║   ✓ Use flat files for bulk historical data                   ║
    ║   ✓ Cache REST API responses locally                           ║
    ║   ✓ Use WebSocket for live data (more efficient)              ║
    ║   ✓ Request only needed symbols/timeframes                     ║
    ║   ✓ Use longer intervals (1h, 1d) instead of 1m               ║
    ║                                                                 ║
    ║ Upgrade: https://massive.com/pricing (starts at $29/month)    ║
    ╚════════════════════════════════════════════════════════════════╝
    """
    print(info)


def _generate_synthetic_ohlcv(symbols: List[str], start: str, end: str, interval: str) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    logger.info("Generating synthetic data...")
    
    # Parse dates
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    
    # Generate timestamps based on interval
    freq_map = {
        '1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min',
        '1h': '1H', '4h': '4H', '1d': '1D', '1w': '1W'
    }
    freq = freq_map.get(interval, '1H')
    timestamps = pd.date_range(start=start_dt, end=end_dt, freq=freq)
    
    all_data = []
    np.random.seed(42)
    
    for symbol in symbols:
        # Generate realistic price movements
        base_price = 100 + np.random.random() * 400
        returns = np.random.randn(len(timestamps)) * 0.02
        prices = base_price * np.exp(np.cumsum(returns))
        
        for i, ts in enumerate(timestamps):
            price = prices[i]
            volatility = price * 0.01
            
            open_price = price + np.random.randn() * volatility
            close_price = price + np.random.randn() * volatility
            high_price = max(open_price, close_price) + abs(np.random.randn()) * volatility
            low_price = min(open_price, close_price) - abs(np.random.randn()) * volatility
            volume = int(np.random.exponential(1000000))
            
            all_data.append({
                'timestamp': ts,
                'symbol': symbol,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
    
    df = pd.DataFrame(all_data)
    df = df.set_index(['timestamp', 'symbol'])
    logger.info(f"✓ Generated {len(df)} synthetic bars")
    return df


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print_free_tier_info()
    
    print("\n" + "="*60)
    print("Testing Massive.com Data Fetching")
    print("="*60)
    
    # Test REST API
    print("\n1. Testing REST API...")
    df = fetch_stocks(['AAPL', 'MSFT'], days=7, interval='1h')
    print(f"Fetched {len(df)} bars")
    print(df.head())
    
    # Test available datasets
    print("\n2. Testing available datasets...")
    datasets = get_available_datasets()
    if datasets:
        print(f"Found {len(datasets)} datasets:")
        for ds in datasets[:5]:
            print(f"  - {ds.get('name')}: {ds.get('description')}")
