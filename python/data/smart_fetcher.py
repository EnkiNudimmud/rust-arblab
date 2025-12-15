
import pandas as pd
from typing import List, Dict, Union, Optional, Tuple
import logging
from datetime import datetime, timedelta

# Import fetchers
from python.data.fetchers.alpaca_helper import fetch_alpaca_batch
from python.data.fetchers.yfinance_helper import fetch_yahoo_batch
# Assuming ccxt logic is in data_fetcher or similar, we might need to import it or rely on fetch_intraday_data
from python.data.data_fetcher import fetch_intraday_data

logger = logging.getLogger(__name__)

class SmartDataFetcher:
    """
    Intelligently routes data requests to the best available source based on:
    - Asset Type (Crypto vs Stock)
    - Granularity (1s, 1m, 1h, 1d)
    - Data Source Limits (Free tier restrictions)
    - Available API keys
    """
    
    def __init__(self, progress_callback=None):
        self.progress_callback = progress_callback
        # Check available keys
        from python.api_keys import get_api_key
        self.has_alpaca = bool(get_api_key("ALPACA_API_KEY"))
        self.has_massive = bool(get_api_key("MASSIVE_API_KEY"))
        self.has_finnhub = bool(get_api_key("FINNHUB_API_KEY"))

    def update_progress(self, current, total, message):
        if self.progress_callback:
            self.progress_callback(current, total, message)
        logger.info(f"[{current}/{total}] {message}")

    def classify_symbol(self, symbol: str) -> str:
        """Determines if a symbol is Crypto or Stock."""
        symbol = symbol.upper().strip()
        if '/' in symbol or '-USD' in symbol or symbol in ['BTC', 'ETH', 'SOL', 'USDT', 'USDC']:
            return 'crypto'
        if symbol in ['BTCUSD', 'ETHUSD']: 
            return 'crypto'
        return 'stock'

    def fetch_auto(self, 
                   symbols: List[str], 
                   intervals: List[str], 
                   start_date: datetime, 
                   end_date: datetime) -> Dict[str, pd.DataFrame]:
        """
        Automatically fetch data for mixed symbols and multiple intervals.
        Returns a dict of {interval: dataframe}.
        """
        
        results = {}
        total_intervals = len(intervals)
        current_interval_idx = 0
        
        # Group symbols by type
        crypto_symbols = []
        stock_symbols = []
        
        for s in symbols:
            if self.classify_symbol(s) == 'crypto':
                crypto_symbols.append(s)
            else:
                stock_symbols.append(s)
                
        self.update_progress(0, 100, f"Classified {len(crypto_symbols)} crypto and {len(stock_symbols)} stock symbols")
        
        for interval in intervals:
            # Calculate progress percentage based on current interval
            interval_progress = int((current_interval_idx / max(total_intervals, 1)) * 100)
            
            interval_df_list = []
            
            # --- 1. Fetch Stocks ---
            if stock_symbols:
                self.update_progress(interval_progress, 100, f"Fetching stocks for {interval}...")
                
                # Decision Matrix for Stocks
                # 1s -> Alpaca (Free Tier: IEX feed fallback handled in helper)
                # 1m -> Alpaca (if available) > Yahoo (7 days limit)
                # 5m-1h -> Yahoo (60 days limit for <1h)
                # 1d -> Yahoo (Unlim)
                
                stock_data = pd.DataFrame()
                
                try:
                    if interval == '1s':
                        if self.has_alpaca:
                            # Alpaca handles batching internally, but let's be safe
                            # Alpaca helper handles "1Sec" mapping
                            alpaca_tf = "1Sec"
                            stock_data = fetch_alpaca_batch(
                                stock_symbols, 
                                start_date.strftime('%Y-%m-%d'), 
                                end_date.strftime('%Y-%m-%d'), 
                                timeframe=alpaca_tf
                            )
                        else:
                            logger.warning("1s interval requested for stocks but Alpaca key missing. Skipping.")
                            
                    elif interval == '1m':
                        # Prefer Alpaca for 1m if available (better history than Yahoo's 7 days)
                        if self.has_alpaca:
                             stock_data = fetch_alpaca_batch(
                                stock_symbols, 
                                start_date.strftime('%Y-%m-%d'), 
                                end_date.strftime('%Y-%m-%d'), 
                                timeframe="1Min"
                            )
                        else:
                            # Fallback to Yahoo (warn about 7 day limit?)
                            # fetch_intraday_data usually handles yahoo
                            duration_days = (end_date - start_date).days
                            if duration_days > 7:
                                logger.warning("Yahoo Finance 1m data limited to last 7 days. Request may be truncated.")
                            
                            for batch in self._chunk_list(stock_symbols, 10):
                                df = fetch_intraday_data(batch, start_date, end_date, interval, source='yfinance')
                                if not df.empty:
                                    interval_df_list.append(df)
                            
                    else:
                        # > 1m (5m, 15m, 1h, 1d) -> Yahoo is usually fine and free
                        # Batching done inside some helpers, but let's batch to be safe/granular with progress
                        batch_size = 20
                        batches = list(self._chunk_list(stock_symbols, batch_size))
                        for i, batch in enumerate(batches):
                            batch_progress = int(interval_progress + (i / max(len(batches), 1)) * 25)
                            self.update_progress(batch_progress, 100, f"Fetching batch {i+1}/{len(batches)} of stocks ({interval})...")
                            try:
                                df = fetch_intraday_data(batch, start_date, end_date, interval, source='yfinance')
                                if not df.empty:
                                    interval_df_list.append(df)
                            except Exception as e:
                                logger.error(f"Error fetching batch {batch}: {e}")
                            
                except Exception as e:
                    logger.error(f"Stock fetch failed: {e}")

                if not stock_data.empty:
                    interval_df_list.append(stock_data)
                    
            # --- 2. Fetch Crypto ---
            if crypto_symbols:
                self.update_progress(int(interval_progress + 25), 100, f"Fetching crypto for {interval}...")
                
                # CCXT is best for crypto
                # Batching: Fetch one by one usually for CCXT unless async
                total_crypto = len(crypto_symbols)
                for i, sym in enumerate(crypto_symbols):
                    crypto_pct = int((i / max(total_crypto, 1)) * 25)
                    if i % max(1, total_crypto // 5) == 0:  # Update every ~20% of symbols
                        self.update_progress(int(interval_progress + 25 + crypto_pct), 100, f"Fetching {sym} ({interval}) [{i+1}/{total_crypto}]")
                    
                    try:
                        # Fix symbol format for CCXT if needed (Yahoo uses BTC-USD, CCXT uses BTC/USDT)
                        ccxt_sym = sym.replace('-', '/')
                        if '/' not in ccxt_sym and not ccxt_sym.endswith('USD'):
                             ccxt_sym += '/USDT' # Default assumption
                        
                        df = fetch_intraday_data([ccxt_sym], start_date, end_date, interval, source='ccxt')
                        if not df.empty:
                            interval_df_list.append(df)
                    except Exception as e:
                         logger.error(f"Crypto fetch failed for {sym}: {e}")
            
            # Combine all for this interval
            if interval_df_list:
                combined_df = pd.concat(interval_df_list, ignore_index=False)
                # Cleanup duplicates if any
                if 'symbol' in combined_df.columns and 'timestamp' in combined_df.columns:
                     combined_df = combined_df.drop_duplicates(subset=['symbol', 'timestamp'])
                
                # Add interval column for multi-interval datasets
                combined_df['interval'] = interval
                results[interval] = combined_df
            
            current_interval_idx += 1

        self.update_progress(100, 100, "Auto-fetch complete!")
        return results

    def _chunk_list(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

def auto_fetch_wrapper(symbols, start, end, intervals, progress_bar=None):
    """Wrapper for Streamlit usage"""
    
    def on_progress(current, total, msg):
        if progress_bar:
            # Normalize to 0-1.0
            val = min(1.0, max(0.0, float(current) / float(total if total > 0 else 1)))
            try:
                progress_bar.progress(val, text=msg)
            except:
                pass
                 
    fetcher = SmartDataFetcher(progress_callback=on_progress)
    return fetcher.fetch_auto(symbols, intervals, start, end)
