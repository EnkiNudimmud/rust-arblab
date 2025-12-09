# Massive.com S3 Flat Files & Live Data Recording Setup

## Overview

This document describes the enhanced Massive.com integration with:
1. **S3 Flat File Downloads** - Bulk historical data via S3-compatible API
2. **Transparent Data Fetching** - Automatic method selection (REST vs S3)
3. **Live Data Recording** - Capture WebSocket streams for backtesting
4. **Live Strategy Testing** - Backtest on recorded live data

## Features Implemented

### 1. S3 Flat File Support ✅

**What Changed:**
- Added `boto3` integration for S3-compatible downloads
- Separate S3 credentials (different from REST API key)
- Downloads from `https://files.massive.com` bucket

**Configuration:**
```properties
# api_keys.properties
MASSIVE_S3_ACCESS_KEY_ID=fe272bbe-1cd7-4977-b0a3-57a55fcd5271
MASSIVE_S3_SECRET_ACCESS_KEY=c_lbvj2jFpptvKLd36GYPA3aLJpJXuBp
MASSIVE_S3_ENDPOINT=https://files.massive.com
MASSIVE_S3_BUCKET=flatfiles
```

**Benefits:**
- ✅ Doesn't count against REST API quota (100 calls/day)
- ✅ Efficient for bulk downloads (10 GB/month)
- ✅ Perfect for large backtesting datasets
- ✅ Automatic fallback to REST if S3 fails

### 2. Transparent Data Fetching ✅

**New Function:** `fetch_data()` in `massive_helper.py`

**Auto Mode Logic:**
```python
if symbols ≤ 5 AND days ≤ 7:
    use REST API  # Fast for small queries
else:
    use Flat Files (S3)  # Efficient for bulk
```

**Usage:**
```python
from python.massive_helper import fetch_data

# Automatic method selection
df = fetch_data(
    symbols=["AAPL", "GOOGL"],
    start="2024-01-01",
    end="2024-12-31",
    interval="1d",
    method="auto"  # or "rest" or "flat_file"
)
```

**Streamlit UI:**
- Radio button to select method (Auto/REST/Flat File)
- Color-coded warnings for REST API quota
- Helpful tips for each method
- Automatic fallback if method fails

### 3. WebSocket Data Recording ✅

**New Class:** `WebSocketDataRecorder` in `massive_helper.py`

**Features:**
- Records live bid/ask quotes
- Automatic OHLCV resampling
- Save to Parquet datasets
- Append to existing datasets

**Usage:**
```python
from python.massive_helper import WebSocketDataRecorder

recorder = WebSocketDataRecorder(symbols=["AAPL", "GOOGL"])
recorder.start()
# ... record live data ...
recorder.stop()
filepath = recorder.save_dataset("live_session_20250104")
```

**Connector Integration:**
```python
from python.rust_bridge import get_connector

connector = get_connector("massive")
connector.enable_recording(symbols=["AAPL", "GOOGL"])
connector.start_stream(symbols, callback=handler)
# ... stream and record ...
filepath = connector.disable_recording()
```

### 4. Live Strategy Testing ✅

**New Tab:** "🧪 Test on Live Data" in Live Trading page

**Features:**
- Start/stop recording with UI controls
- View recording stats in real-time
- Select recorded datasets
- Run backtests on live data
- Compare strategy performance

**Workflow:**
1. Start WebSocket streaming (Massive connector)
2. Enable recording in "Test on Live Data" tab
3. Record for desired duration
4. Stop and save dataset
5. Select strategy to test
6. Run backtest
7. View equity curve, metrics, trades

**Why This Matters:**
- ✅ Test strategies on real market conditions
- ✅ Capture specific scenarios (volatility, gaps, news events)
- ✅ Build library of real market conditions
- ✅ Validate that backtests work on live data too

## Files Modified

### Core Functionality
1. **`python/api_keys.py`**
   - Added `get_massive_s3_credentials()` function
   
2. **`python/massive_helper.py`** (~200 lines added)
   - Added `boto3` import and error handling
   - Rewrote `download_flat_file()` to use S3 API
   - Added `fetch_data()` for transparent method selection
   - Added `WebSocketDataRecorder` class (140 lines)

3. **`python/connectors/massive.py`** (~80 lines added)
   - Added recording support to connector
   - `enable_recording()` / `disable_recording()` methods
   - `get_recorded_dataframe()` method
   - Auto-recording in `_trigger_callbacks()`

### UI Updates
4. **`app/pages/data_loader.py`** (~80 lines added)
   - Added method selection radio button
   - Updated rate limit warnings for each method
   - Call `fetch_data()` with selected method
   - Smart quota tracking (only counts REST calls)

5. **`app/pages/live_trading.py`** (~200 lines added)
   - Added new "🧪 Test on Live Data" tab
   - Recording controls (start/stop, duration)
   - Recording stats display
   - Dataset selection and backtest runner
   - Equity curve and metrics visualization

### Configuration
6. **`api_keys.properties.example`**
   - Added S3 credential fields with examples

7. **`requirements-py313.txt`**
   - Added `boto3>=1.34.0`
   - Added `websockets>=12.0`

### Documentation
8. **`docs/MASSIVE_INTEGRATION.md`** (extensively updated)
   - S3 credentials setup
   - Transparent fetching guide
   - Live recording tutorial
   - Strategy testing workflow
   - Installation requirements
   - Troubleshooting for S3 issues

## Testing Checklist

### S3 Flat Files
- [ ] Add S3 credentials to `api_keys.properties`
- [ ] Test `download_flat_file()` with valid credentials
- [ ] Test fallback to REST when S3 fails
- [ ] Verify downloaded file sizes and data quality

### Transparent Fetching
- [ ] Test auto mode with small query (≤5 symbols, ≤7 days)
- [ ] Test auto mode with large query (>5 symbols or >7 days)
- [ ] Test explicit REST mode
- [ ] Test explicit flat_file mode
- [ ] Verify REST quota is only counted for REST calls

### Live Recording
- [ ] Start Massive WebSocket streaming
- [ ] Enable recording
- [ ] Verify data is being captured
- [ ] Stop recording and check saved file
- [ ] Load recorded dataset in Data Loader

### Live Backtesting
- [ ] Record live data for 5-10 minutes
- [ ] Navigate to "Test on Live Data" tab
- [ ] Select recorded dataset
- [ ] Run backtest with a strategy
- [ ] Verify equity curve displays correctly
- [ ] Check trade log for accuracy

## Usage Examples

### Example 1: Small Query (Auto → REST)
```python
# 3 symbols, 5 days → Uses REST API
df = fetch_data(
    symbols=["AAPL", "GOOGL", "MSFT"],
    start="2024-12-01",
    end="2024-12-05",
    interval="1h",
    method="auto"
)
# Output: "📊 Auto-selected REST API (small query: 3 symbols, 5 days)"
```

### Example 2: Large Query (Auto → S3)
```python
# 20 symbols, 90 days → Uses Flat Files
df = fetch_data(
    symbols=["AAPL", "GOOGL", "MSFT", ...],  # 20 symbols
    start="2024-01-01",
    end="2024-03-31",
    interval="1d",
    method="auto"
)
# Output: "📦 Auto-selected Flat Files (large query: 20 symbols, 90 days)"
```

### Example 3: Live Recording Session
```python
from python.rust_bridge import get_connector

# Setup
connector = get_connector("massive")
connector.enable_recording(symbols=["AAPL", "GOOGL", "MSFT", "TSLA"])

# Stream for 10 minutes
def on_quote(symbol, orderbook):
    bid = orderbook['bids'][0][0]
    ask = orderbook['asks'][0][0]
    print(f"{symbol}: {bid:.2f} / {ask:.2f}")

connector.start_stream(["AAPL", "GOOGL", "MSFT", "TSLA"], callback=on_quote)

# ... wait 10 minutes ...

# Save recording
filepath = connector.disable_recording()
print(f"Saved to: {filepath}")
# Output: "Saved to: data/live_recorded/massive_live_20250104_143022.parquet"
```

## Performance Benefits

### REST API vs Flat Files
| Metric | REST API | Flat Files (S3) |
|--------|----------|-----------------|
| Best for | <5 symbols, <7 days | >10 symbols, >30 days |
| Speed | Fast (~1-2s) | Medium (~5-10s) |
| Quota | Counts against 100/day | 10 GB/month (separate) |
| Fallback | None | Falls back to REST |
| Use case | Quick lookups | Bulk backtesting |

### Live Recording Benefits
- **Real Market Conditions**: Test on actual spreads, gaps, volatility
- **Specific Scenarios**: Record during news, earnings, market open
- **Regime Testing**: Capture different market states (trending, ranging, volatile)
- **Validation**: Verify backtest results match live performance
- **Library Building**: Create collection of market scenarios for robust testing

## Next Steps

1. **Get S3 Credentials**
   - Log into Massive.com account
   - Go to API Keys → "Accessing Flat Files (S3)" tab
   - Copy Access Key ID and Secret Key
   - Add to `api_keys.properties`

2. **Test Flat Files**
   - Try downloading a large dataset (20+ symbols, 1+ year)
   - Verify data quality and completeness
   - Check file sizes against 10 GB quota

3. **Start Recording**
   - Enable WebSocket streaming with Massive
   - Record live data during different market conditions
   - Build a library of scenarios

4. **Test Strategies**
   - Run backtests on recorded live data
   - Compare with synthetic/historical backtests
   - Identify strategies that work on live data

5. **Optimize Workflow**
   - Use Auto mode for most queries
   - Switch to Flat Files for large backtesting datasets
   - Use WebSocket recording for strategy validation
   - Build adaptive strategies using HMM regime detection

## Support

- **Massive.com Docs**: https://docs.massive.com
- **S3 API Docs**: https://docs.massive.com/flat-files
- **Support**: support@massive.com
- **Issue Tracker**: File issues in rust-hft-arbitrage-lab repo
