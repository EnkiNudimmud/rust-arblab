"""Enhanced data fetching for intraday mean-reversion analysis.

Supports multiple data sources:
- CCXT (recommended for crypto - FREE, no API key needed!)
- Yahoo Finance (recommended for stocks - FREE, no API key needed!)
- Finnhub (stocks/forex - requires API key from api_keys.properties)
- Alpha Vantage (stocks/forex/crypto - requires API key, 25 calls/day)
- Massive (institutional data - requires API key, 100 calls/day + 10GB/month)
- Synthetic data (testing)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time

try:
    from python.data.fetchers.finnhub_helper import fetch_ohlcv as fh_fetch_ohlcv
    FH_AVAILABLE = True
except ImportError:
    fh_fetch_ohlcv = None
    FH_AVAILABLE = False
except Exception as e:
    fh_fetch_ohlcv = None
    FH_AVAILABLE = False
    import logging
    logging.getLogger(__name__).warning(f"Could not import Finnhub helper: {e}")

try:
    from python.data.fetchers.alpha_vantage_helper import (
        fetch_intraday as av_fetch_intraday,
        fetch_daily as av_fetch_daily,
        check_rate_limit,
        get_remaining_calls
    )
    AV_AVAILABLE = True
except ImportError:
    av_fetch_intraday = None
    av_fetch_daily = None
    check_rate_limit = None
    get_remaining_calls = None
    AV_AVAILABLE = False
except Exception as e:
    av_fetch_intraday = None
    av_fetch_daily = None
    check_rate_limit = None
    get_remaining_calls = None
    AV_AVAILABLE = False
    import logging
    logging.getLogger(__name__).warning(f"Could not import Alpha Vantage helper: {e}")

try:
    import yfinance as yf
    from python.data.fetchers.yfinance_helper import validate_date_range
    YF_AVAILABLE = True
except ImportError:
    yf = None
    validate_date_range = None
    YF_AVAILABLE = False

try:
    from python.data.fetchers.ccxt_helper import create_exchange, fetch_ohlcv_range
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
except Exception as e:
    CCXT_AVAILABLE = False
    import logging
    logging.getLogger(__name__).warning(f"Could not import CCXT helper: {e}")

try:
    from python.data.fetchers.massive_helper import fetch_ohlcv_rest as massive_fetch_ohlcv
    MASSIVE_AVAILABLE = True
except ImportError:
    massive_fetch_ohlcv = None
    MASSIVE_AVAILABLE = False
except Exception as e:
    massive_fetch_ohlcv = None
    MASSIVE_AVAILABLE = False
    import logging
    logging.getLogger(__name__).warning(f"Could not import Massive helper: {e}")


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
        # Check if symbols look like crypto pairs
        is_crypto = any('/' in s or 'USDT' in s or 'BTC' in s for s in symbols)
        
        if is_crypto and CCXT_AVAILABLE:
            source = "ccxt"
        elif YF_AVAILABLE:
            # Yahoo Finance is the default for stocks - it's free and reliable
            source = "yfinance"
        elif CCXT_AVAILABLE:
            # Try CCXT for stocks too (some exchanges have stock tokens)
            source = "ccxt"
        else:
            # Fallback to synthetic data for testing
            source = "synthetic"
    
    if source == "ccxt":
        return _fetch_ccxt(symbols, start, end, interval)
    elif source == "finnhub":
        return _fetch_finnhub(symbols, start, end, interval)
    elif source == "alpha_vantage":
        return _fetch_alpha_vantage(symbols, start, end, interval)
    elif source == "yfinance":
        return _fetch_yfinance(symbols, start, end, interval)
    elif source == "massive":
        return _fetch_massive(symbols, start, end, interval)
    else:
        return _generate_synthetic(symbols, start, end, interval)


def _fetch_ccxt(symbols: List[str], start: str, end: str, interval: str, exchange_id: str = 'binance') -> pd.DataFrame:
    """Fetch data from crypto exchanges using CCXT with parallel processing and progress tracking."""
    if not CCXT_AVAILABLE:
        raise ImportError(
            "❌ CCXT library not installed!\\n"
            "Install it with: pip install ccxt\\n"
            "CCXT provides FREE access to 100+ crypto exchanges with no API key required."
        )
    
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import streamlit as st
    
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    
    # Map interval to CCXT timeframe
    interval_map = {
        '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
        '1h': '1h', '2h': '2h', '4h': '4h', '1d': '1d'
    }
    timeframe = interval_map.get(interval, '1h')
    
    all_data = []
    errors = []
    
    # Create progress bar and status immediately
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text(f"🚀 Starting to fetch {len(symbols)} symbols from {exchange_id}...")
    
    # Track timing for ETA
    start_time = time.time()
    completion_times = []
    
    def fetch_single_symbol(symbol):
        """Fetch data for a single symbol"""
        symbol_start = time.time()
        try:
            exchange = create_exchange(exchange_id)
            # CCXT uses format like 'BTC/USDT', convert if needed
            ccxt_symbol = symbol if '/' in symbol else f"{symbol}/USDT"
            
            df = fetch_ohlcv_range(exchange, ccxt_symbol, timeframe, start_dt, end_dt)
            
            symbol_time = time.time() - symbol_start
            
            if not df.empty:
                df['symbol'] = symbol
                df = df.rename(columns={'timestamp': 'timestamp'})
                return df[['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']], None, symbol_time
            return None, f"No data returned for {symbol}", symbol_time
        except Exception as e:
            symbol_time = time.time() - symbol_start
            return None, f"{symbol}: {str(e)}", symbol_time
    
    # Use ThreadPoolExecutor for parallel fetching (6 concurrent requests for better speed)
    completed = 0
    with ThreadPoolExecutor(max_workers=6) as executor:
        # Submit all tasks immediately so progress bar shows right away
        futures = {executor.submit(fetch_single_symbol, symbol): symbol for symbol in symbols}
        
        # Show initial progress immediately after submitting
        status_text.text(f"⏳ Fetching 0/{len(symbols)} symbols from {exchange_id}... (0%) - Starting...")
        
        for future in as_completed(futures):
            result, error, symbol_time = future.result()
            completed += 1
            progress = completed / len(symbols)
            progress_bar.progress(progress)
            
            # Track completion times for better ETA
            completion_times.append(symbol_time)
            
            # Calculate ETA based on recent completion times
            if len(completion_times) >= 2:
                # Use average of recent completions for more accurate ETA
                recent_avg = sum(completion_times[-3:]) / len(completion_times[-3:])
                remaining = len(symbols) - completed
                # Account for parallel execution (divide by workers)
                eta_seconds = (remaining / 6) * recent_avg
                
                if eta_seconds < 60:
                    eta_str = f"~{max(1, int(eta_seconds))}s remaining"
                else:
                    eta_str = f"~{int(eta_seconds/60)}m {int(eta_seconds%60)}s remaining"
            else:
                eta_str = "Calculating..."
            
            status_text.text(
                f"⏳ Fetching {completed}/{len(symbols)} symbols from {exchange_id}... "
                f"({progress*100:.0f}%) - {eta_str}"
            )
            
            if result is not None:
                all_data.append(result)
            if error:
                errors.append(error)
    
    progress_bar.empty()
    
    # Show detailed error and success information
    if errors:
        error_summary = "\\n  • ".join(errors[:10])  # Show first 10 errors
        if len(errors) > 10:
            error_summary += f"\\n  • ... and {len(errors) - 10} more"
        
        # Show warning in Streamlit
        st.warning(f"⚠️  {len(errors)} symbol(s) failed to fetch:\\n{error_summary}")
    
    if not all_data:
        raise ValueError(
            f"❌ Failed to fetch data from CCXT ({exchange_id}) for symbols: {symbols}\\n"
            f"Possible reasons:\\n"
            f"  • Invalid symbol format (use 'BTC/USDT' or 'BTC')\\n"
            f"  • Exchange doesn't have these symbols\\n"
            f"  • Network issues\\n\\n"
            f"💡 Tip: Binance has the most symbols. Try: 'BTC/USDT', 'ETH/USDT', 'SOL/USDT'"
        )
    
    # Show success summary
    success_msg = f"✅ Successfully fetched {len(all_data)}/{len(symbols)} symbols"
    if len(all_data) < len(symbols):
        success_msg += f" ({len(symbols) - len(all_data)} failed)"
    status_text.success(success_msg)
    time.sleep(2)  # Show success message a bit longer
    status_text.empty()
    
    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.set_index(['timestamp', 'symbol']).sort_index()
    return combined


def _fetch_yfinance(symbols: List[str], start: str, end: str, interval: str) -> pd.DataFrame:
    """Fetch data from Yahoo Finance with enhanced error handling and retry logic."""
    if not YF_AVAILABLE:
        raise ImportError(
            "❌ yfinance not installed!\n"
            "Install it with: pip install yfinance\n"
            "Yahoo Finance provides FREE stock data with 1-minute intraday resolution."
        )
    
    import streamlit as st
    
    # Map interval names
    yf_interval_map = {
        "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
        "1h": "1h", "2h": "2h", "1d": "1d", "1wk": "1wk", "1mo": "1mo"
    }
    yf_interval = yf_interval_map.get(interval, "1h")
    
    # Validate date range for intraday data
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    days_range = (end_dt - start_dt).days
    
    # Check Yahoo Finance limitations and warn user
    if yf_interval in ["1m", "5m", "15m", "30m"] and days_range > 60:
        st.warning(
            f"⚠️ Yahoo Finance limits {yf_interval} data to max 60 days. "
            f"Requested: {days_range} days. Will fetch last 60 days only. "
            f"For longer history, use 1h or 1d interval."
        )
    elif yf_interval == "1h" and days_range > 730:
        st.warning(
            f"⚠️ Yahoo Finance limits 1h data to max ~730 days (2 years). "
            f"Requested: {days_range} days ({days_range/365:.1f} years). "
            f"Will fetch last 730 days only. For full history, use 1d interval."
        )
    
    all_data = []
    failed_symbols = []
    
    # Create progress bar and status immediately
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text(f"🚀 Starting to fetch {len(symbols)} symbols from Yahoo Finance...")
    
    # Track timing for ETA
    start_time = time.time()
    completion_times = []
    
    for i, symbol in enumerate(symbols, 1):
        symbol_start = time.time()
        
        # Update progress bar
        progress = (i - 1) / len(symbols)
        progress_bar.progress(progress)
        
        # Calculate ETA
        if completion_times:
            avg_time = sum(completion_times) / len(completion_times)
            remaining = len(symbols) - (i - 1)
            eta_seconds = remaining * avg_time
            
            if eta_seconds < 60:
                eta_str = f"~{max(1, int(eta_seconds))}s remaining"
            else:
                eta_str = f"~{int(eta_seconds/60)}m {int(eta_seconds%60)}s remaining"
        else:
            eta_str = "Calculating..."
        
        status_text.text(
            f"⏳ Fetching {i-1}/{len(symbols)} symbols from Yahoo Finance... "
            f"({progress*100:.0f}%) - {eta_str} - Current: {symbol}"
        )
        
        retry_count = 0
        max_retries = 3
        success = False
        
        while retry_count < max_retries and not success:
            try:
                ticker = yf.Ticker(symbol)  # type: ignore[union-attr]
                df = pd.DataFrame()
                
                # Use different methods based on interval and Yahoo's limitations
                if yf_interval in ["1m", "5m", "15m", "30m"]:
                    # Short-term intraday: use period-based approach (limited history)
                    if days_range <= 7:
                        df = ticker.history(period="7d", interval=yf_interval)
                    elif days_range <= 60:
                        df = ticker.history(period="60d", interval=yf_interval)
                    else:
                        # Yahoo Finance limitation: max 60 days for these intervals
                        df = ticker.history(period="60d", interval=yf_interval)
                elif yf_interval == "1h":
                    # 1h data: Yahoo allows max ~730 days (2 years)
                    if days_range <= 730:
                        df = ticker.history(start=start, end=end, interval=yf_interval)
                    else:
                        # Use max period and truncate to requested range
                        df = ticker.history(period="730d", interval=yf_interval)
                else:
                    # For daily and above, try period-based first (more reliable)
                    # Map days to appropriate period
                    if days_range <= 5:
                        df = ticker.history(period="5d", interval=yf_interval)
                    elif days_range <= 30:
                        df = ticker.history(period="1mo", interval=yf_interval)
                    elif days_range <= 90:
                        df = ticker.history(period="3mo", interval=yf_interval)
                    elif days_range <= 180:
                        df = ticker.history(period="6mo", interval=yf_interval)
                    elif days_range <= 365:
                        df = ticker.history(period="1y", interval=yf_interval)
                    elif days_range <= 730:
                        df = ticker.history(period="2y", interval=yf_interval)
                    elif days_range <= 1825:
                        df = ticker.history(period="5y", interval=yf_interval)
                    else:
                        df = ticker.history(period="max", interval=yf_interval)
                    
                    # If period-based failed or returned empty, try date range
                    if df.empty:
                        df = ticker.history(start=start, end=end, interval=yf_interval)
                
                if not df.empty:
                    # Filter to exact date range requested (handle timezone-aware timestamps)
                    try:
                        # Try to access tz attribute (exists for DatetimeIndex)
                        index_tz = getattr(df.index, 'tz', None)
                        if index_tz is not None:
                            # Make comparison timestamps timezone-aware
                            start_dt_tz = start_dt.tz_localize('UTC').tz_convert(index_tz)
                            end_dt_tz = end_dt.tz_localize('UTC').tz_convert(index_tz)
                            df = df[(df.index >= start_dt_tz) & (df.index <= end_dt_tz)]
                        else:
                            df = df[(df.index >= start_dt) & (df.index <= end_dt)]
                    except (AttributeError, TypeError):
                        df = df[(df.index >= start_dt) & (df.index <= end_dt)]
                    
                    if not df.empty:
                        df = df.reset_index()
                        df.columns = [c.lower() for c in df.columns]
                        df['symbol'] = symbol
                        
                        # Handle different index column names
                        if 'date' in df.columns:
                            df = df.rename(columns={'date': 'timestamp'})
                        elif 'datetime' in df.columns:
                            df = df.rename(columns={'datetime': 'timestamp'})
                        
                        # Ensure timestamp is datetime
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        
                        # Select and reorder columns
                        required_cols = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
                        df = df[required_cols]
                        
                        all_data.append(df)
                        success = True
                    else:
                        failed_symbols.append((symbol, "No data in date range"))
                        success = True  # Don't retry if date range is the issue
                else:
                    failed_symbols.append((symbol, "No data available from Yahoo Finance"))
                    success = True  # Don't retry if symbol doesn't exist
                
                # Track completion time
                symbol_time = time.time() - symbol_start
                completion_times.append(symbol_time)
                
                # Rate limiting: be nice to Yahoo's servers
                time.sleep(0.5)
                
            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    wait_time = retry_count * 2  # Exponential backoff
                    status_text.text(
                        f"⚠️ {symbol}: Retry {retry_count}/{max_retries} in {wait_time}s... "
                        f"({i-1}/{len(symbols)} completed)"
                    )
                    time.sleep(wait_time)
                else:
                    failed_symbols.append((symbol, str(e)))
                    # Track completion time even for failures
                    symbol_time = time.time() - symbol_start
                    completion_times.append(symbol_time)
    
    # Final progress update
    progress_bar.progress(1.0)
    
    # Show summary
    if failed_symbols:
        error_summary = "\n  • ".join([f"{sym}: {err}" for sym, err in failed_symbols[:10]])
        if len(failed_symbols) > 10:
            error_summary += f"\n  • ... and {len(failed_symbols) - 10} more"
        st.warning(f"⚠️ {len(failed_symbols)} symbol(s) failed:\n{error_summary}")
    
    success_msg = f"✅ Successfully fetched {len(all_data)}/{len(symbols)} symbols"
    if len(all_data) < len(symbols):
        success_msg += f" ({len(symbols) - len(all_data)} failed)"
    status_text.success(success_msg)
    time.sleep(2)
    status_text.empty()
    progress_bar.empty()
    
    # Provide detailed feedback
    if failed_symbols:
        print(f"\n⚠️  Failed to fetch {len(failed_symbols)}/{len(symbols)} symbols:")
        for sym, reason in failed_symbols:
            print(f"   • {sym}: {reason}")
    
    if not all_data:
        error_details = "\n".join([f"   • {sym}: {reason}" for sym, reason in failed_symbols[:5]])
        raise ValueError(
            f"❌ Failed to fetch data from Yahoo Finance for all symbols: {symbols}\n\n"
            f"Common issues:\n"
            f"  • Invalid symbol format:\n"
            f"    - Stocks: Use 'AAPL', 'MSFT', 'TSLA' (not 'AAPL.US')\n"
            f"    - Crypto: Use 'BTC-USD', 'ETH-USD' (with hyphen)\n"
            f"    - International: Add exchange suffix 'NESN.SW', 'SAP.DE'\n\n"
            f"  • Date range limitations:\n"
            f"    - 1m data: Last 7 days only\n"
            f"    - 5m/15m/30m data: Last 60 days only\n"
            f"    - 1h+ data: Multiple years available\n\n"
            f"  • Network or Yahoo Finance API issues\n\n"
            f"Failed symbols details:\n{error_details}\n\n"
            f"💡 Tips:\n"
            f"  • For crypto, use 'CCXT' data source (better coverage)\n"
            f"  • For longer history, use higher timeframes (1h, 1d)\n"
            f"  • Verify symbols at finance.yahoo.com"
        )
    
    print(f"\n✅ Successfully fetched {len(all_data)}/{len(symbols)} symbols")
    
    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.set_index(['timestamp', 'symbol']).sort_index()
    return combined


def _fetch_finnhub(symbols: List[str], start: str, end: str, interval: str) -> pd.DataFrame:
    """Fetch data from Finnhub."""
    if not FH_AVAILABLE or fh_fetch_ohlcv is None:
        raise ImportError(
            "Finnhub helper not available. This could be due to:\n"
            "1. Missing api_keys.properties file\n"
            "2. Invalid FINNHUB_API_KEY in api_keys.properties\n"
            "3. Missing python.connectors.finnhub module\n"
            "Please check your configuration and try 'Yahoo Finance' instead."
        )
    
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
        raise ValueError(
            f"❌ Failed to fetch data from Finnhub for symbols: {symbols}\n"
            f"Possible reasons:\n"
            f"  • Invalid API key in api_keys.properties\n"
            f"  • Symbols not available on Finnhub (use format 'BINANCE:BTCUSDT' for crypto)\n"
            f"  • Free tier API limits exceeded (60 calls/minute)\n"
            f"  • Date range not supported by your subscription tier\n\n"
            f"💡 Recommended alternative: Use 'CCXT' data source which supports:\n"
            f"  • Multiple exchanges (Binance, Kraken, Coinbase, etc.)\n"
            f"  • FREE historical data with second/minute intervals\n"
            f"  • No API key required for public data\n"
            f"  • Better coverage for crypto markets"
        )
    
    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.set_index(['timestamp', 'symbol']).sort_index()
    return combined


def _fetch_alpha_vantage(symbols: List[str], start: str, end: str, interval: str) -> pd.DataFrame:
    """Fetch data from Alpha Vantage with rate limiting."""
    if not AV_AVAILABLE or av_fetch_intraday is None:
        raise ImportError(
            "Alpha Vantage helper not available. This could be due to:\n"
            "1. Missing api_keys.properties file\n"
            "2. Invalid ALPHA_VANTAGE_API_KEY in api_keys.properties\n"
            "3. Missing requests library\n"
            "Please check your configuration and try 'Yahoo Finance' or 'CCXT' instead."
        )
    
    # Check rate limits before starting
    can_call, message = check_rate_limit()  # type: ignore
    if not can_call:
        raise RuntimeError(
            f"⚠️ Alpha Vantage rate limit exceeded: {message}\n\n"
            f"Free tier limitations:\n"
            f"  • 25 API calls per day\n"
            f"  • 5 API calls per minute\n"
            f"  • Limit resets at midnight UTC\n\n"
            f"💡 Alternatives:\n"
            f"  • Use Yahoo Finance (unlimited, no API key)\n"
            f"  • Use CCXT for crypto (unlimited, no API key)\n"
            f"  • Wait for rate limit reset\n"
            f"  • Upgrade to paid Alpha Vantage plan"
        )
    
    # Get remaining calls
    daily_remaining, minute_remaining = get_remaining_calls()  # type: ignore
    
    # Warn if fetching too many symbols
    if len(symbols) > daily_remaining:
        raise RuntimeError(
            f"⚠️ Too many symbols requested!\n"
            f"Requested: {len(symbols)} symbols\n"
            f"Available: {daily_remaining} API calls remaining today\n\n"
            f"💡 Solutions:\n"
            f"  • Reduce number of symbols to {daily_remaining} or less\n"
            f"  • Use saved datasets from previous fetches\n"
            f"  • Switch to Yahoo Finance or CCXT\n"
            f"  • Wait until tomorrow for limit reset"
        )
    
    if len(symbols) > minute_remaining:
        print(
            f"⏳ Note: Fetching {len(symbols)} symbols will take at least "
            f"{(len(symbols) - minute_remaining) * 12} seconds due to rate limiting (5 calls/minute max)"
        )
    
    # Map interval to Alpha Vantage format
    av_interval_map = {
        '1m': '1min',
        '5m': '5min',
        '15m': '15min',
        '30m': '30min',
        '1h': '60min',
        '60m': '60min',
    }
    av_interval = av_interval_map.get(interval, '5min')
    
    # Use daily endpoint for daily data
    use_daily = interval in ['1d', 'daily']
    
    all_data = []
    
    for i, symbol in enumerate(symbols):
        try:
            # Check rate limit before each call
            can_call, message = check_rate_limit()  # type: ignore
            if not can_call:
                print(f"⚠️ Rate limit reached after {i} symbols: {message}")
                break
            
            if use_daily:
                df = av_fetch_daily(symbol, outputsize='compact')  # type: ignore
            else:
                df = av_fetch_intraday(symbol, interval=av_interval)  # type: ignore
            
            if isinstance(df, pd.DataFrame) and not df.empty:
                # Alpha Vantage helper returns: timestamp, open, high, low, close, volume
                df['symbol'] = symbol
                all_data.append(df[['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']])
                print(f"✅ Fetched {len(df)} points for {symbol} ({i+1}/{len(symbols)})")
            else:
                print(f"⚠️ No data returned for {symbol}")
            
            # Rate limiting: wait 12 seconds between calls (5 calls/minute max)
            if i < len(symbols) - 1:
                print(f"⏳ Waiting 12 seconds (rate limit: 5 calls/minute)...")
                time.sleep(12)
                
        except RuntimeError as e:
            # Rate limit or other runtime errors
            if "Rate limit" in str(e):
                print(f"❌ Rate limit reached after {i} symbols")
                break
            else:
                print(f"Warning: Failed to fetch {symbol}: {e}")
        except Exception as e:
            print(f"Warning: Failed to fetch {symbol} from Alpha Vantage: {e}")
    
    if not all_data:
        raise ValueError(
            f"❌ Failed to fetch data from Alpha Vantage for symbols: {symbols}\n"
            f"Possible reasons:\n"
            f"  • Invalid API key in api_keys.properties\n"
            f"  • Symbols not available (use stock tickers like 'AAPL', 'MSFT')\n"
            f"  • Rate limits exceeded (25 calls/day, 5 calls/minute)\n"
            f"  • Invalid date range or interval\n\n"
            f"💡 Recommended alternatives:\n"
            f"  • Yahoo Finance: unlimited stock data, no API key\n"
            f"  • CCXT: unlimited crypto data, no API key\n"
            f"  • Reduce number of symbols and try again"
        )
    
    # Update session state with API call count
    try:
        import streamlit as st
        if 'av_calls_today' in st.session_state:
            st.session_state.av_calls_today += len(all_data)
    except:
        pass  # Not in Streamlit context
    
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


def _fetch_massive(symbols: List[str], start: str, end: str, interval: str) -> pd.DataFrame:
    """Fetch data from Massive.com with rate limiting and progress tracking."""
    if not MASSIVE_AVAILABLE or massive_fetch_ohlcv is None:
        raise ImportError(
            "❌ Massive helper not available!\n"
            "Make sure massive_helper.py is in the python/ directory.\n"
            "Install dependencies: pip install requests websockets"
        )
    
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info(f"📊 Fetching {len(symbols)} symbols from Massive.com...")
    logger.info("💡 Free tier: 100 req/day, 10 req/min - Use 'massive' for better rates than yfinance!")
    
    # Fetch data using Massive REST API
    df = massive_fetch_ohlcv(
        symbols=symbols,
        start=start,
        end=end,
        interval=interval
    )
    
    if df is None or df.empty:
        logger.warning("⚠️ No data returned from Massive, using synthetic fallback")
        return _generate_synthetic(symbols, start, end, interval)
    
    logger.info(f"✅ Massive.com: Fetched {len(df)} total bars")
    return df


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


# ==============================================================================
# COMPREHENSIVE DATA FETCHER CLASS
# ==============================================================================

class DataFetcher:
    """
    Comprehensive data fetcher with support for:
    - Multiple data sources (Yahoo, CCXT, Finnhub, Alpha Vantage, Websockets)
    - Major market indices (S&P 500, Dow Jones, NASDAQ, etc.)
    - Commodities and materials
    - Real-time websocket streaming
    - Full OHLCV data (including volume)
    - Automatic symbol discovery for indices
    """
    
    # Market Index Compositions
    INDICES = {
        "SP500": "S&P 500 stocks",
        "DOW30": "Dow Jones 30 stocks",
        "NASDAQ100": "NASDAQ 100 stocks",
        "RUSSELL2000": "Russell 2000 stocks"
    }
    
    # Commodity/Material Categories
    COMMODITIES = {
        "precious_metals": ["GC=F", "SI=F", "PL=F", "PA=F"],  # Gold, Silver, Platinum, Palladium
        "base_metals": ["HG=F", "ALI=F"],  # Copper, Aluminum
        "energy": ["CL=F", "NG=F", "BZ=F", "RB=F", "HO=F"],  # Crude, Nat Gas, Brent, RBOB, Heating Oil
        "agriculture": ["ZC=F", "ZW=F", "ZS=F", "KC=F", "SB=F", "CT=F"],  # Corn, Wheat, Soy, Coffee, Sugar, Cotton
    }
    
    def __init__(self, api_keys: Optional[Dict[str, str]] = None):
        """
        Initialize DataFetcher
        
        Args:
            api_keys: Dict with keys 'finnhub', 'alpha_vantage', 'polygon', etc.
        """
        self.api_keys = api_keys or {}
        self._ws_connections = {}
        self._ws_data_buffers = {}
        
    def get_index_symbols(self, index_name: str) -> List[str]:
        """
        Get all symbols for a major market index
        
        Args:
            index_name: 'SP500', 'DOW30', 'NASDAQ100', or 'RUSSELL2000'
            
        Returns:
            List of ticker symbols
        """
        if index_name == "SP500":
            return self._get_sp500_symbols()
        elif index_name == "DOW30":
            return self._get_dow30_symbols()
        elif index_name == "NASDAQ100":
            return self._get_nasdaq100_symbols()
        elif index_name == "RUSSELL2000":
            return self._get_russell2000_symbols()
        else:
            raise ValueError(f"Unknown index: {index_name}. Use: {list(self.INDICES.keys())}")
    
    def get_commodity_symbols(self, category: str = "all") -> List[str]:
        """
        Get commodity/material symbols
        
        Args:
            category: 'precious_metals', 'base_metals', 'energy', 'agriculture', or 'all'
            
        Returns:
            List of futures ticker symbols
        """
        if category == "all":
            all_symbols = []
            for symbols in self.COMMODITIES.values():
                all_symbols.extend(symbols)
            return all_symbols
        elif category in self.COMMODITIES:
            return self.COMMODITIES[category].copy()
        else:
            raise ValueError(f"Unknown category: {category}. Use: {list(self.COMMODITIES.keys())} or 'all'")
    
    def fetch_historical(
        self,
        symbols: List[str],
        start: str,
        end: str,
        interval: str = "1h",
        source: str = "auto",
        include_volume: bool = True
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data
        
        Args:
            symbols: List of symbols
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            interval: '1m', '5m', '15m', '30m', '1h', '1d'
            source: 'yfinance', 'ccxt', 'finnhub', 'auto'
            include_volume: Include volume column
            
        Returns:
            DataFrame with columns [open, high, low, close, volume]
        """
        return fetch_intraday_data(symbols, start, end, interval, source)
    
    def fetch_index_data(
        self,
        index_name: str,
        start: str,
        end: str,
        interval: str = "1h",
        max_symbols: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch data for entire market index
        
        Args:
            index_name: 'SP500', 'DOW30', 'NASDAQ100', 'RUSSELL2000'
            start: Start date
            end: End date
            interval: Data interval
            max_symbols: Limit number of symbols (None = all)
            
        Returns:
            DataFrame with OHLCV data for all index constituents
        """
        symbols = self.get_index_symbols(index_name)
        
        if max_symbols:
            symbols = symbols[:max_symbols]
        
        print(f"Fetching {index_name} data for {len(symbols)} symbols...")
        
        return self.fetch_historical(symbols, start, end, interval)
    
    def start_websocket_stream(
        self,
        symbols: List[str],
        exchange: str = "binance",
        interval: str = "1m",
        callback: Optional[callable] = None
    ):
        """
        Start real-time websocket data stream
        
        Args:
            symbols: List of symbols to stream
            exchange: Exchange name ('binance', 'coinbase', 'kraken', etc.)
            interval: Timeframe for candle aggregation
            callback: Function to call with new data: callback(symbol, ohlcv_dict)
        """
        if not CCXT_AVAILABLE:
            raise ImportError("CCXT required for websocket streaming. Install: pip install ccxt")
        
        # This would use ccxt.pro for websocket streaming
        # For now, provide polling-based alternative
        print(f"Starting websocket stream for {len(symbols)} symbols on {exchange}")
        print("Note: Full websocket implementation requires ccxt.pro")
        print("Using polling fallback...")
        
        self._start_polling_stream(symbols, exchange, interval, callback)
    
    def _start_polling_stream(
        self,
        symbols: List[str],
        exchange: str,
        interval: str,
        callback: Optional[callable]
    ):
        """Polling-based alternative to websockets"""
        import threading
        import time
        
        def poll_loop():
            ex = create_exchange(exchange)
            
            while self._ws_connections.get(exchange, False):
                for symbol in symbols:
                    try:
                        # Fetch latest candle
                        ohlcv = ex.fetch_ohlcv(symbol, interval, limit=1)
                        if ohlcv and callback:
                            latest = ohlcv[-1]
                            data = {
                                'timestamp': datetime.fromtimestamp(latest[0] / 1000),
                                'open': latest[1],
                                'high': latest[2],
                                'low': latest[3],
                                'close': latest[4],
                                'volume': latest[5]
                            }
                            callback(symbol, data)
                    except Exception as e:
                        print(f"Error polling {symbol}: {e}")
                
                time.sleep(5)  # Poll every 5 seconds
        
        self._ws_connections[exchange] = True
        thread = threading.Thread(target=poll_loop, daemon=True)
        thread.start()
        
        print(f"✓ Polling stream started for {exchange}")
    
    def stop_websocket_stream(self, exchange: str = "binance"):
        """Stop websocket stream for exchange"""
        self._ws_connections[exchange] = False
        print(f"✓ Stopped stream for {exchange}")
    
    def append_to_dataset(
        self,
        existing_data: pd.DataFrame,
        new_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Append new streaming data to existing dataset
        
        Args:
            existing_data: Current dataset
            new_data: New data to append
            
        Returns:
            Combined dataset with duplicates removed
        """
        combined = pd.concat([existing_data, new_data])
        
        # Remove duplicates (keep latest)
        combined = combined[~combined.index.duplicated(keep='last')]
        
        return combined.sort_index()
    
    # ========== INDEX SYMBOL LISTS ==========
    
    def _get_sp500_symbols(self) -> List[str]:
        """Get S&P 500 constituent symbols"""
        # Top 100 S&P 500 stocks by market cap (2024)
        return [
            # Tech (FAANG+)
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO", "ORCL", "ADBE",
            "CRM", "CSCO", "ACN", "AMD", "IBM", "INTC", "QCOM", "TXN", "NOW", "INTU",
            "AMAT", "MU", "ADI", "LRCX", "KLAC", "SNPS", "CDNS", "MRVL", "FTNT", "PANW",
            
            # Finance
            "JPM", "BAC", "WFC", "C", "GS", "MS", "BLK", "SCHW", "AXP", "USB",
            "PNC", "TFC", "COF", "BK", "STT", "DFS", "SYF", "AIG", "MET", "PRU",
            
            # Healthcare
            "UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE", "TMO", "ABT", "DHR", "BMY",
            "AMGN", "CVS", "ELV", "CI", "HUM", "GILD", "VRTX", "REGN", "ZTS", "ISRG",
            
            # Consumer
            "WMT", "HD", "PG", "KO", "PEP", "COST", "MCD", "NKE", "TGT", "SBUX",
            "LOW", "TJX", "DG", "ROST", "CMG", "YUM", "ORLY", "AZO", "ULTA", "DPZ",
            
            # Industrial
            "CAT", "BA", "HON", "UPS", "RTX", "LMT", "GE", "MMM", "DE", "UNP",
            "EMR", "ITW", "ETN", "PH", "CMI", "ROK", "PCAR", "CARR", "OTIS", "EMR",
            
            # Energy
            "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "HAL",
            
            # Materials
            "LIN", "APD", "ECL", "SHW", "DD", "NEM", "FCX", "NUE", "VMC", "MLM",
            
            # Utilities
            "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL", "ED", "PEG",
            
            # Real Estate
            "AMT", "PLD", "CCI", "EQIX", "PSA", "SPG", "WELL", "DLR", "O", "AVB",
            
            # Communication
            "NFLX", "DIS", "CMCSA", "VZ", "T", "TMUS", "CHTR", "EA", "TTWO", "NWSA",
            
            # Discretionary
            "TSLA", "HD", "NKE", "SBUX", "TGT", "LOW", "TJX", "BKNG", "MAR", "ABNB",
            
            # More key stocks
            "V", "MA", "PYPL", "SQ", "FIS", "FISV", "ADP", "PAYX", "ROP", "BR",
            "INFO", "IQV", "SPGI", "MCO", "CME", "ICE", "MSCI", "TRU", "VRSK", "EW",
            
            # Remaining top companies
            "BRK.B", "AVGO", "WMT", "LLY", "V", "MA", "UNH", "XOM", "JNJ", "ORCL",
            "HD", "PG", "COST", "ABBV", "MRK", "ASML", "KO", "PEP", "TMO", "CSCO",
            "ACN", "MCD", "ABT", "NFLX", "DHR", "CMCSA", "WFC", "ADBE", "PM", "DIS",
            "VZ", "CRM", "TXN", "NEE", "INTC", "UPS", "QCOM", "BMY", "RTX", "SPGI",
            "AMGN", "HON", "UNP", "LOW", "IBM", "COP", "BA", "AMAT", "GS", "DE"
        ]
    
    def _get_dow30_symbols(self) -> List[str]:
        """Get Dow Jones 30 constituent symbols"""
        return [
            "AAPL", "AMGN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS", "DOW",
            "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM",
            "MRK", "MSFT", "NKE", "PG", "TRV", "UNH", "V", "VZ", "WBA", "WMT"
        ]
    
    def _get_nasdaq100_symbols(self) -> List[str]:
        """Get NASDAQ 100 constituent symbols"""
        return [
            # Mega caps
            "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA", "AVGO", "COST",
            
            # Large tech
            "NFLX", "ADBE", "CSCO", "INTC", "CMCSA", "PEP", "AMD", "QCOM", "TXN", "INTU",
            "AMGN", "AMAT", "ISRG", "HON", "BKNG", "VRTX", "ADP", "SBUX", "GILD", "ADI",
            
            # Mid/Large tech
            "MU", "LRCX", "REGN", "MDLZ", "PYPL", "MELI", "KLAC", "SNPS", "CDNS", "ASML",
            "NXPI", "MNST", "CSX", "ABNB", "MRVL", "ORLY", "FTNT", "CHTR", "ADSK", "PCAR",
            
            # Growth/Mid caps  
            "PAYX", "AEP", "ROST", "ODFL", "CPRT", "PANW", "DXCM", "FAST", "EA", "KDP",
            "VRSK", "CTAS", "EXC", "CTSH", "LULU", "XEL", "TEAM", "IDXX", "ANSS", "KHC",
            
            # More constituents
            "GEHC", "MCHP", "CCEP", "TTWO", "ON", "ZS", "FANG", "BIIB", "DDOG", "CSGP",
            "CRWD", "WBD", "ILMN", "GFS", "MDB", "MRNA", "WDAY", "ALGN", "DASH", "ARM",
            
            # Additional
            "DLTR", "MAR", "CDW", "WBA", "ZM", "LCID", "RIVN", "HOOD", "COIN", "RBLX"
        ]
    
    def _get_russell2000_symbols(self) -> List[str]:
        """
        Get sample Russell 2000 symbols (subset - full list has 2000!)
        Returns top 50 by market cap
        """
        return [
            # Sample of largest Russell 2000 companies
            "FLEX", "RBC", "SSNC", "GTLS", "MANH", "CNK", "CELH", "RMBS", "EXLS", "GTLS",
            "SITM", "UFPI", "PEGA", "SMAR", "VIRT", "TGTX", "INSM", "NSP", "TMDX", "PEN",
            "DV", "CHWY", "PCVX", "RUN", "ENPH", "BMRN", "PLTR", "COIN", "HOOD", "RIVN",
            "LCID", "RBLX", "UPST", "AFRM", "OPEN", "ROOT", "GDRX", "CLOV", "SOFI", "MTTR",
            "NAVI", "UPST", "DKNG", "PENN", "LYV", "MSG", "MSGS", "CZR", "MGM", "WYNN"
        ]


# ==============================================================================
# WEBSOCKET DATA STREAMING (Enhanced)
# ==============================================================================

class WebsocketDataStreamer:
    """
    Real-time websocket data streaming and reconstruction
    
    Features:
    - Multi-exchange support (Binance, Coinbase, Kraken, etc.)
    - Tick-by-tick data capture
    - OHLCV reconstruction from ticks
    - Automatic reconnection
    - Buffer management
    """
    
    def __init__(self, exchange: str = "binance"):
        self.exchange = exchange
        self.active_streams = {}
        self.tick_buffers = {}
        self.candle_data = {}
        
    def start_tick_stream(
        self,
        symbol: str,
        on_tick: Optional[callable] = None,
        on_candle: Optional[callable] = None,
        candle_interval: str = "1m"
    ):
        """
        Start streaming individual ticks and reconstruct candles
        
        Args:
            symbol: Trading pair symbol
            on_tick: Callback for each tick: on_tick(price, volume, timestamp)
            on_candle: Callback for completed candles: on_candle(ohlcv_dict)
            candle_interval: Candle timeframe ('1m', '5m', '15m', '1h')
        """
        print(f"Starting tick stream for {symbol} on {self.exchange}")
        print(f"Candle reconstruction: {candle_interval}")
        
        # This requires ccxt.pro or exchange-specific websocket libraries
        # Implementation would use asyncio websocket connections
        
        raise NotImplementedError(
            "Full websocket implementation requires ccxt.pro or exchange SDK.\n"
            "Install: pip install ccxt[pro]\n"
            "Alternative: Use DataFetcher.start_websocket_stream() for polling-based approach"
        )
    
    def reconstruct_candles_from_ticks(
        self,
        ticks: List[Dict],
        interval: str = "1m"
    ) -> pd.DataFrame:
        """
        Reconstruct OHLCV candles from tick data
        
        Args:
            ticks: List of tick dicts with keys: timestamp, price, volume
            interval: Candle timeframe
            
        Returns:
            DataFrame with OHLCV data
        """
        df = pd.DataFrame(ticks)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        
        # Resample to candles
        candles = df.resample(interval).agg({
            'price': ['first', 'max', 'min', 'last'],
            'volume': 'sum'
        })
        
        candles.columns = ['open', 'high', 'low', 'close', 'volume']
        
        return candles.dropna()


# ==============================================================================
# UTILITY FUNCTION: MERGE STREAMING DATA
# ==============================================================================

def merge_streaming_data(
    existing_df: pd.DataFrame,
    new_ticks: List[Dict],
    interval: str = "1m"
) -> pd.DataFrame:
    """
    Merge new streaming tick data into existing dataset
    
    Args:
        existing_df: Existing OHLCV DataFrame
        new_ticks: New tick data from websocket
        interval: Candle interval
        
    Returns:
        Updated DataFrame with new candles appended
    """
    streamer = WebsocketDataStreamer()
    new_candles = streamer.reconstruct_candles_from_ticks(new_ticks, interval)
    
    combined = pd.concat([existing_df, new_candles])
    combined = combined[~combined.index.duplicated(keep='last')]
    
    return combined.sort_index()
