import os
import pandas as pd
from alpaca_trade_api import REST

# Load API keys from environment or config
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

# Helper to get Alpaca REST API

def get_alpaca_api():
    return REST(ALPACA_API_KEY, ALPACA_API_SECRET, ALPACA_BASE_URL)

# Fetch 1-second bars for a symbol
def fetch_alpaca_bars(symbol, start, end, timeframe="1Sec", limit=10000):
    api = get_alpaca_api()
    bars = api.get_bars(symbol, timeframe, start, end, limit=limit).df
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
