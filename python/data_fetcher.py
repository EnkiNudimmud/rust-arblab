"""Enhanced data fetching for intraday mean-reversion analysis.

Supports multiple data sources:
- CCXT (recommended for crypto - FREE, no API key needed!)
- Yahoo Finance (stocks and major crypto)
- Finnhub (requires API key)
- Synthetic data (testing)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time

try:
    from python.finnhub_helper import fetch_ohlcv as fh_fetch_ohlcv
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
    import yfinance as yf
    from python.yfinance_helper import validate_date_range
    YF_AVAILABLE = True
except ImportError:
    yf = None
    validate_date_range = None
    YF_AVAILABLE = False

try:
    from python.ccxt_helper import create_exchange, fetch_ohlcv_range
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
except Exception as e:
    CCXT_AVAILABLE = False
    import logging
    logging.getLogger(__name__).warning(f"Could not import CCXT helper: {e}")


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
            source = "yfinance"
        elif FH_AVAILABLE and fh_fetch_ohlcv is not None:
            source = "finnhub"
        else:
            source = "synthetic"
    
    if source == "ccxt":
        return _fetch_ccxt(symbols, start, end, interval)
    elif source == "finnhub":
        return _fetch_finnhub(symbols, start, end, interval)
    elif source == "yfinance":
        return _fetch_yfinance(symbols, start, end, interval)
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
