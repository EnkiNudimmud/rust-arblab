import os
import pandas as pd
from alpaca_trade_api import REST
from alpaca_trade_api.rest import TimeFrame, TimeFrameUnit

# Load API keys using shared utility to support properties file
from python.api_keys import get_api_key

ALPACA_API_KEY = get_api_key("ALPACA_API_KEY")
ALPACA_API_SECRET = get_api_key("ALPACA_API_SECRET")
ALPACA_BASE_URL = get_api_key("ALPACA_BASE_URL") or "https://paper-api.alpaca.markets/v2"

# Helper to get Alpaca REST API

def get_alpaca_api():
    return REST(ALPACA_API_KEY, ALPACA_API_SECRET, ALPACA_BASE_URL)

def get_timeframe_object(tf_str):
    """Convert string timeframe to Alpaca TimeFrame object"""
    try:
        if tf_str == "1Sec":
             return TimeFrame(1, TimeFrameUnit.Second)
        elif tf_str == "1Min":
             return TimeFrame(1, TimeFrameUnit.Minute)
        elif tf_str == "5Min":
             return TimeFrame(5, TimeFrameUnit.Minute)
        elif tf_str == "15Min":
             return TimeFrame(15, TimeFrameUnit.Minute)
        elif tf_str == "30Min":
             return TimeFrame(30, TimeFrameUnit.Minute)
        elif tf_str == "1Hour":
             return TimeFrame(1, TimeFrameUnit.Hour)
        elif tf_str == "1Day":
             return TimeFrame(1, TimeFrameUnit.Day)
        return tf_str
    except Exception:
        return tf_str

# Fetch 1-second bars for a symbol
def fetch_alpaca_bars(symbol, start, end, timeframe="1Sec", limit=10000):
    api = get_alpaca_api()
    # Convert string timeframe to object if necessary
    tf_obj = get_timeframe_object(timeframe)
    
    try:
        # Try default feed (SIP) first
        bars = api.get_bars(symbol, tf_obj, start, end, limit=limit).df
    except Exception as e:
        # Fallback to IEX feed if SIP is restricted (common on free tier for recent data)
        if "subscription does not permit" in str(e):
            print(f"⚠️ SIP data restricted for {symbol}. Falling back to IEX feed (Free Tier compatible).")
            bars = api.get_bars(symbol, tf_obj, start, end, limit=limit, feed='iex').df
        else:
            raise e
            
    if bars.empty:
        return pd.DataFrame()

    bars.reset_index(inplace=True)
    bars["symbol"] = symbol
    return bars[["timestamp", "symbol", "open", "high", "low", "close", "volume"]]

# Batch fetch for multiple symbols
def fetch_alpaca_batch(symbols, start, end, timeframe="1Sec", limit=10000):
    dfs = []
    for symbol in symbols:
        try:
            df = fetch_alpaca_bars(symbol, start, end, timeframe, limit)
            dfs.append(df)
        except Exception as e:
            print(f"Alpaca fetch failed for {symbol}: {e}")
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()

# WebSocket streaming for live quotes/trades (example)
def stream_alpaca_quotes(symbols, on_quote):
    from alpaca_trade_api.stream import Stream
    stream = Stream(ALPACA_API_KEY, ALPACA_API_SECRET, base_url=ALPACA_BASE_URL)
    for symbol in symbols:
        stream.subscribe_quotes(on_quote, symbol)
    stream.run()
