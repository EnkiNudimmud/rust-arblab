# Massive.com Data Source Integration

Complete integration of Massive.com institutional-grade market data platform.

## Overview

Massive.com provides professional-grade market data with a **generous free tier**:
- **REST API**: 100 requests/day, 10 requests/minute
- **WebSocket**: 10 concurrent connections, 100 messages/minute  
- **Flat Files**: 10 GB/month bulk historical data downloads

## Features

### 1. REST API (Historical & Real-time Data)
- OHLCV data for stocks, options, futures, forex, crypto
- Configurable timeframes (1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w, 1M)
- Automatic rate limiting to stay within free tier
- Returns pandas DataFrame compatible with existing workflows

### 2. WebSocket Streaming (Live Trading)
- Real-time quotes and trades
- Supports 10 concurrent symbol streams
- Thread-safe callbacks for Streamlit integration
- Automatic reconnection on connection loss
- Rate limit monitoring (100 messages/minute)

### 3. Flat File Downloads (Bulk Historical)
- Download complete historical datasets
- 10 GB/month download quota (free tier)
- Direct to pandas DataFrame
- Efficient for backtesting large datasets

## Setup

### 1. Get API Key
1. Sign up at [massive.com](https://massive.com) (free)
2. Navigate to API Keys section
3. Generate new API key for REST API

### 2. Get S3 Credentials (For Flat Files)
1. In your Massive.com account dashboard
2. Click on **API Keys** → **Accessing Flat Files (S3)** tab
3. Copy your **Access Key ID** and **Secret Access Key**
4. Note: These are **separate** from the REST API key

### 3. Configure API Keys

Add your credentials to `api_keys.properties`:
```properties
# Massive.com REST API Key (Free tier: 100 requests/day, 10/minute)
MASSIVE_API_KEY=your_api_key_here

# Massive.com S3 Credentials for Flat File Downloads (separate from API key)
# Get from: https://massive.com/account/api-keys → "Accessing Flat Files (S3)" tab
MASSIVE_S3_ACCESS_KEY_ID=fe272bbe-1cd7-4977-b0a3-57a55fcd5271
MASSIVE_S3_SECRET_ACCESS_KEY=c_lbvj2jFpptvKLd36GYPA3aLJpJXuBp
MASSIVE_S3_ENDPOINT=https://files.massive.com
MASSIVE_S3_BUCKET=flatfiles
```

**Important:** 
- REST API key is for API calls (100/day limit)
- S3 credentials are for bulk file downloads (10 GB/month limit)
- They are **not interchangeable**

## Usage

### Data Loader (Streamlit UI)

1. Open **Data Loader** page
2. Select **"Massive (Institutional-grade - FREE 100 calls/day)"** as data source
3. **Choose fetch method:**
   - 🤖 **Auto (Recommended)**: Automatically picks best method
     - Uses REST API for small queries (≤5 symbols, ≤7 days)
     - Uses Flat Files for large queries (>5 symbols or >7 days)
   - 📡 **REST API**: Fast for small queries, counts against 100/day limit
   - 📦 **Flat Files (S3)**: Bulk downloads, 10 GB/month, requires S3 credentials
4. View free tier status with remaining quota
5. Enter symbols (stocks, crypto, forex, etc.)
6. Select date range and interval
7. Click **Fetch Data**

**Rate Limit Display:**
- ✅ Green: >50 REST calls remaining
- 🔶 Orange: 20-50 REST calls remaining  
- 🔴 Red: <20 REST calls remaining
- 🚫 Gray: Daily limit reached (resets at midnight UTC)

**Smart Method Selection (Auto Mode):**
| Query Size | Days | Symbols | Recommended Method | Reason |
|-----------|------|---------|-------------------|--------|
| Small | ≤7 | ≤5 | REST API | Fast, low quota usage |
| Medium | >7 | ≤5 | Flat Files | Saves REST calls |
| Large | Any | >5 | Flat Files | Efficient for bulk |
| Huge | >30 | >10 | Flat Files | S3 is designed for this |

**Tips for Free Tier:**
- Use **Auto mode** - it picks the best method for your query
- **REST API** is best for quick lookups (1-5 symbols, 1-7 days)
- **Flat Files (S3)** are best for:
  - Bulk downloads (>10 symbols)
  - Long historical periods (>30 days)
  - Building large backtesting datasets
- Use **Append mode** to incrementally build datasets
- Save datasets after fetching to avoid re-fetching
- Use **WebSocket streaming** for live trading (doesn't count against any quota)

### Live Trading (WebSocket Streaming)

1. Open **Live Trading** page
2. Select **"massive"** as Data Source
3. Choose **"Streaming (WebSocket)"** connection mode
4. Select symbols to stream
5. Click **Start Trading**

**Free Tier WebSocket Limits:**
- ✅ 10 concurrent connections (10 symbols streaming simultaneously)
- ✅ 100 messages/minute across all connections
- ✅ Automatic reconnection on disconnection

### Live Data Recording & Backtesting (NEW!)

Record live WebSocket data and test strategies on it in real-time:

#### Recording Live Data

1. **Start Live Trading** with Massive connector (WebSocket mode)
2. Navigate to **"🧪 Test on Live Data"** tab
3. Click **▶️ Start Recording**
4. Let it record for desired duration (1-60 minutes recommended)
5. Click **⏹️ Stop Recording**
6. Data is automatically saved as a dataset

**What Gets Recorded:**
- Real-time bid/ask quotes
- Trade prices and volumes
- Timestamp for each data point
- Automatically resampled to OHLCV bars (1min default)

#### Testing Strategies on Live Data

1. **Select a recorded dataset** from the dropdown
2. **Choose a strategy** to backtest
3. Click **🚀 Run Backtest**
4. View results:
   - 📈 Equity curve
   - 📊 Performance metrics (return, Sharpe, drawdown)
   - 📝 Trade log

**Why This Is Powerful:**
- ✅ Test strategies on **real market conditions** without risking capital
- ✅ See how strategies perform on **live data patterns** (gaps, spreads, volatility)
- ✅ Build a library of **real market scenarios** for robust testing
- ✅ Compare strategy performance across **different market conditions**
- ✅ **Validate backtests** - does your strategy work on live data too?

**Typical Workflow:**
```
1. Record live data during:
   - Market open (high volatility)
   - Mid-day (normal conditions)
   - News events (unusual volatility)
   - Different market regimes (trending, ranging, volatile)

2. Build a dataset library:
   - massive_live_morning_20250104.parquet
   - massive_live_afternoon_20250104.parquet  
   - massive_live_fomc_20250201.parquet
   - massive_live_earnings_season_20250315.parquet

3. Test strategies on each:
   - Mean reversion → Works best mid-day?
   - Momentum → Works best at open?
   - Market making → Struggles during news?

4. Optimize:
   - Adjust parameters for different regimes
   - Use HMM regime detection to switch strategies
   - Build adaptive strategies that detect market conditions
```

### Python API

#### Unified Data Fetching (Recommended)
```python
from python.massive_helper import fetch_data

# Smart fetch - automatically chooses REST API or Flat Files
df = fetch_data(
    symbols=["AAPL", "GOOGL", "MSFT"],
    start="2024-01-01",
    end="2024-01-31",
    interval="1h",
    method="auto"  # or "rest" or "flat_file"
)

# Auto mode logic:
# - Small queries (≤5 symbols, ≤7 days): Uses REST API
# - Large queries (>5 symbols or >7 days): Uses Flat Files
# - Falls back gracefully if one method fails

print(df.head())
# Output:
#            timestamp symbol    open    high     low   close     volume
# 0 2024-01-01 09:30:00   AAPL  180.50  182.30  180.10  181.90  5000000
# 1 2024-01-01 10:30:00   AAPL  181.90  183.50  181.50  182.80  4500000
```

#### REST API (Historical Data)
```python
from python.massive_helper import fetch_ohlcv_rest

# Explicit REST API call
df = fetch_ohlcv_rest(
    symbols=["AAPL", "GOOGL", "MSFT"],
    start="2024-01-01",
    end="2024-01-31",
    interval="1h",
    api_key="your_api_key"  # Optional - auto-loaded from api_keys.properties
)
```

#### Flat File Downloads (S3)
```python
from python.massive_helper import download_flat_file

# Download from S3 bucket (requires S3 credentials)
df = download_flat_file(
    symbols=["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"],
    start="2024-01-01",
    end="2024-12-31",
    interval="1d",
    # S3 credentials auto-loaded from api_keys.properties
)

# Efficient for large queries:
# - Doesn't count against REST API quota
# - Can handle hundreds of symbols
# - Perfect for backtesting datasets
```

#### WebSocket Streaming
```python
from python.massive_helper import stream_quotes_websocket
import asyncio

async def main():
    async for quote in stream_quotes_websocket(
        symbols=["AAPL", "GOOGL"],
        api_key="your_api_key"
    ):
        print(f"{quote['symbol']}: Bid={quote['bid']}, Ask={quote['ask']}")

asyncio.run(main())
```

#### WebSocket Data Recording
```python
from python.massive_helper import WebSocketDataRecorder

# Create recorder
recorder = WebSocketDataRecorder(
    symbols=["AAPL", "GOOGL", "MSFT"],
    save_dir="data/live_recorded"
)

# Start recording
recorder.start()

# ... let it record live data ...

# Stop and save
recorder.stop()
filepath = recorder.save_dataset("my_live_session_20250104", resample_interval="1min")

# Or get DataFrame directly
df = recorder.get_dataframe(resample_interval="1min")
print(f"Recorded {len(df):,} bars")

# Append to existing dataset
recorder.append_to_dataset("path/to/existing_dataset.parquet", resample_interval="1min")
```

#### Integration with Live Trading Connector
```python
from python.rust_bridge import get_connector

# Get Massive connector
connector = get_connector("massive")

# Enable recording
connector.enable_recording(symbols=["AAPL", "GOOGL"])

# Start streaming (data will be recorded automatically)
def handle_quote(symbol, orderbook):
    print(f"{symbol}: Bid={orderbook['bids'][0][0]}")

connector.start_stream(["AAPL", "GOOGL"], callback=handle_quote)

# ... stream for a while ...

# Stop and save recording
filepath = connector.disable_recording()
print(f"Saved recording to: {filepath}")

# Get recorded data as DataFrame
df = connector.get_recorded_dataframe(resample_interval="1min")
```

### Data Fetcher Integration

Massive is integrated into the unified `data_fetcher.py` interface:

```python
from python.data_fetcher import fetch_intraday_data

# Automatic source selection (Massive will be chosen if available)
df = fetch_intraday_data(
    symbols=["AAPL", "GOOGL"],
    start="2024-01-01",
    end="2024-01-31",
    interval="1h",
    source="auto"  # or explicitly: source="massive"
)
```

### Live Trading Connector

Massive is available as a WebSocket connector in live trading:

```python
from python.rust_bridge import get_connector

# Get Massive connector
connector = get_connector("massive")

# List available symbols
symbols = connector.list_symbols()
print(symbols)  # ['BTCUSDT', 'ETHUSDT', 'AAPL', 'GOOGL', ...]

# Start WebSocket stream
def handle_quote(symbol, orderbook):
    print(f"{symbol}: Bid={orderbook['bids'][0][0]}, Ask={orderbook['asks'][0][0]}")

connector.start_stream(["AAPL", "GOOGL"], callback=handle_quote)

# Get stats
stats = connector.get_stats()
print(f"Active streams: {stats['active_streams']}/{stats['max_streams']}")
print(f"Messages this minute: {stats['messages_this_minute']}/{stats['max_messages_per_minute']}")

# Stop streaming
connector.stop_stream(["AAPL", "GOOGL"])  # Stop specific symbols
# or
connector.stop_stream()  # Stop all
```

## Free Tier Management

### Rate Limiting Strategy

**REST API (100 calls/day, 10/minute):**
- 1 symbol = 1 API call
- 10 symbols = 10 API calls
- Fetching 10 symbols uses 10% of daily quota

**Recommendation:**
- Use **Append mode** in Data Loader to build datasets incrementally
- Fetch 1-5 days at a time, save dataset, then append more
- For bulk historical data, use **flat file downloads** instead

**WebSocket (10 connections, 100 messages/minute):**
- 1 symbol = 1 connection
- Can stream 10 symbols simultaneously
- 100 messages/minute shared across all symbols (~10 messages/symbol/minute)

**Recommendation:**
- Use WebSocket for live trading (real-time streaming)
- Use REST for historical backtesting data
- Use flat files for large historical datasets

### Monitoring Quota

**In Streamlit UI:**
- Data Loader shows real-time quota status (green/orange/red indicators)
- Live Trading displays WebSocket connection count and message rate

**In Python:**
```python
from python.massive_helper import print_free_tier_info

# Display free tier limits and tips
print_free_tier_info()
```

Output:
```
=== Massive.com Free Tier Limits ===
REST API: 100 requests/day, 10 requests/minute
WebSocket: 10 concurrent connections, 100 messages/minute  
Flat Files: 10 GB/month downloads

Tips:
- Use Append mode to incrementally build datasets
- Save datasets after fetching to avoid re-fetching
- Use flat file downloads for bulk historical data
- WebSocket is best for live trading (doesn't count against REST quota)
```

## Appending to Existing Datasets

Massive data can be appended to existing datasets:

```python
from python.data_fetcher import fetch_intraday_data
from python.data_persistence import load_dataset, save_dataset, stack_data

# Load existing dataset
existing_df = load_dataset("my_stocks_dataset")

# Fetch new data from Massive
new_df = fetch_intraday_data(
    symbols=["AAPL", "GOOGL"],
    start="2024-02-01",
    end="2024-02-28",
    interval="1h",
    source="massive"
)

# Append new data
combined_df = stack_data(existing_df, new_df, mode="append")

# Save combined dataset
save_dataset(combined_df, "my_stocks_dataset", append=True)
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Streamlit UI Layer                    │
├─────────────────────────────────────────────────────────┤
│  Data Loader (REST)          Live Trading (WebSocket)   │
│  - Rate limit display        - Real-time streaming      │
│  - Quota monitoring          - Multi-symbol support     │
│  - Append mode               - Connection management    │
└──────────────────┬──────────────────┬───────────────────┘
                   │                  │
┌──────────────────▼──────────────────▼───────────────────┐
│              Data Integration Layer                     │
├─────────────────────────────────────────────────────────┤
│  data_fetcher.py             rust_bridge.py             │
│  - Source selection          - Connector registry       │
│  - DataFrame conversion      - Live trading bridge      │
└──────────────────┬──────────────────┬───────────────────┘
                   │                  │
┌──────────────────▼──────────────────▼───────────────────┐
│                 Massive.com Layer                       │
├─────────────────────────────────────────────────────────┤
│  massive_helper.py          connectors/massive.py       │
│  - REST API                 - WebSocket streaming       │
│  - Flat file downloads      - Live trading connector    │
│  - Rate limiting            - Thread-safe callbacks     │
│  - DataFrame conversion     - Reconnection logic        │
└──────────────────┬──────────────────┬───────────────────┘
                   │                  │
┌──────────────────▼──────────────────▼───────────────────┐
│                   Massive.com API                       │
├─────────────────────────────────────────────────────────┤
│  REST API                   WebSocket API               │
│  - Historical data          - Real-time quotes          │
│  - OHLCV                    - Trades                    │
│  - Multiple timeframes      - Orderbook                 │
└─────────────────────────────────────────────────────────┘
```

## Files Modified/Created

### New Files
- `python/massive_helper.py` - REST API, WebSocket, flat file functions
- `python/connectors/massive.py` - Live trading WebSocket connector
- `docs/MASSIVE_INTEGRATION.md` - This documentation

### Modified Files
- `python/api_keys.py` - Added `get_massive_key()` function
- `api_keys.properties.example` - Added `MASSIVE_API_KEY` configuration
- `python/data_fetcher.py` - Added Massive as data source option
- `python/rust_bridge.py` - Registered Massive connector
- `app/pages/data_loader.py` - Added Massive UI option with rate limit display
- `app/pages/live_trading.py` - Added Massive WebSocket support

## Installation

Massive.com support requires additional Python packages:

```bash
# Install required packages
pip install boto3 websockets

# Or use the requirements file
pip install -r requirements-py313.txt
```

**Dependencies:**
- `boto3>=1.34.0` - For S3 flat file downloads
- `websockets>=12.0` - For WebSocket streaming
- `requests>=2.31.0` - For REST API calls
- `pandas>=2.2.0` - For data handling

## Troubleshooting

### "No API key found" Warning
**Cause:** `MASSIVE_API_KEY` not set in `api_keys.properties`  
**Solution:** Add your API key to `api_keys.properties`
```properties
MASSIVE_API_KEY=your_api_key_here
```

### "No S3 credentials found" Warning (Flat Files)
**Cause:** `MASSIVE_S3_ACCESS_KEY_ID` or `MASSIVE_S3_SECRET_ACCESS_KEY` not set  
**Solution:** Get credentials from Massive dashboard and add to `api_keys.properties`
```properties
# From "Accessing Flat Files (S3)" tab in your Massive account
MASSIVE_S3_ACCESS_KEY_ID=fe272bbe-1cd7-4977-b0a3-57a55fcd5271
MASSIVE_S3_SECRET_ACCESS_KEY=c_lbvj2jFpptvKLd36GYPA3aLJpJXuBp
MASSIVE_S3_ENDPOINT=https://files.massive.com
MASSIVE_S3_BUCKET=flatfiles
```
**Note:** When S3 credentials are missing, the system automatically falls back to REST API

### "boto3 not installed" Error
**Cause:** boto3 package not installed  
**Solution:** Install boto3
```bash
pip install boto3
```
System will automatically fall back to REST API if boto3 is not available

### "Daily limit reached" Error
**Cause:** Used 100 REST API calls today  
**Solutions:**
1. Wait until midnight UTC for quota reset
2. Use WebSocket streaming (doesn't count against REST quota)
3. Use flat file downloads for bulk data

### "Rate limit exceeded" Error  
**Cause:** Made >10 REST requests in 1 minute  
**Solution:** Wait 1 minute, then retry. Massive helper automatically throttles requests.

### WebSocket Connection Failed
**Causes:**
1. Invalid API key
2. Maximum connections reached (10 concurrent)
3. Network issues

**Solutions:**
1. Verify API key in `api_keys.properties`
2. Stop some connections: `connector.stop_stream(["AAPL"])`
3. Check network connectivity

### Synthetic Data Fallback
**Cause:** No API key configured (testing mode)  
**Behavior:** Massive helper generates synthetic OHLCV data for testing  
**Solution:** Add API key to use real market data

## Best Practices

### 1. Data Collection Strategy
```python
# ❌ Bad: Fetch all data at once (uses many API calls)
df = fetch_ohlcv_rest(symbols=["AAPL", "GOOGL", "MSFT", ...], start="2020-01-01", end="2024-01-01")

# ✅ Good: Fetch incrementally and save
for year in range(2020, 2025):
    df = fetch_ohlcv_rest(
        symbols=["AAPL", "GOOGL", "MSFT"],
        start=f"{year}-01-01",
        end=f"{year}-12-31",
        interval="1d"
    )
    save_dataset(df, f"stocks_{year}", append=True)
```

### 2. Live Trading
```python
# ✅ Use WebSocket for live trading (doesn't use REST quota)
connector = get_connector("massive")
connector.start_stream(symbols, callback=handle_quote)

# ❌ Don't poll REST API for live data (wastes quota)
while True:
    df = fetch_ohlcv_rest(symbols, ...)  # Bad!
    time.sleep(1)
```

### 3. Historical Analysis
```python
# ✅ Use flat file downloads for bulk historical data
df = download_flat_file(
    symbols=["AAPL", "GOOGL", "MSFT"],
    start="2020-01-01",
    end="2023-12-31",
    interval="1d"
)
# Uses file download quota (10 GB/month), not REST quota
```

## Support

- **Massive.com Docs**: https://massive.com/docs
- **API Status**: https://status.massive.com
- **Support**: support@massive.com

## Comparison with Other Sources

| Feature | Massive | Finnhub | Alpha Vantage | Yahoo Finance | CCXT |
|---------|---------|---------|---------------|---------------|------|
| **REST API Quota** | 100/day | 60/min | 25/day | Unlimited | Unlimited |
| **WebSocket** | 10 connections | 1 (free) | No | No | Yes |
| **Data Types** | All | All | All | Stocks/Crypto | Crypto |
| **Historical Depth** | Full | Limited (free) | Full | Limited | Exchange-dependent |
| **Real-time Delay** | None | 15min (free) | 15-20min | 15min | None |
| **API Key Required** | Yes | Yes | Yes | No | No (public) |

**When to use Massive:**
- ✅ Need institutional-grade data quality
- ✅ Want generous free tier (100 REST + 10 WebSocket)
- ✅ Trading stocks, options, futures, forex, crypto
- ✅ Need both historical and real-time data
- ✅ Running HFT strategies with WebSocket feeds

**When to use alternatives:**
- Use **CCXT** for crypto-only with no API key
- Use **Yahoo Finance** for simple stock data (no API key)
- Use **Finnhub** if already have premium subscription
- Use **Alpha Vantage** for quick stock quotes (25/day is enough)
