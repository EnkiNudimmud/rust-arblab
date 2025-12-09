# Alpha Vantage Integration Guide

## Overview

Alpha Vantage has been successfully integrated into the HFT Arbitrage Lab as a new data source for stocks, forex, and cryptocurrency market data.

## Implementation Details

### Architecture

**Python-Only Approach**
- Alpha Vantage connector implemented purely in Python (no Rust bindings yet)
- Direct REST API integration using `requests` library
- Full rate limiting and free-tier protection built-in

### Components Created

#### 1. **Rust Connector** (`rust_core/connectors/alpha_vantage/`)
   - `src/lib.rs` - Complete REST API client implementation
   - Rate limiting constants: `FREE_TIER_DAILY_LIMIT = 25`, `FREE_TIER_PER_MINUTE_LIMIT = 5`
   - Endpoints supported:
     - `TIME_SERIES_INTRADAY` - 1min, 5min, 15min, 30min, 60min intervals
     - `TIME_SERIES_DAILY` - Daily prices
     - `GLOBAL_QUOTE` - Real-time quotes (15-20 min delay)
     - `FX_INTRADAY` - Forex intraday data
     - `DIGITAL_CURRENCY_DAILY` - Crypto daily prices
   - **Note:** Not yet exposed via PyO3 bindings (Python helper used directly)

#### 2. **Python Helper** (`python/alpha_vantage_helper.py`)
   - High-level API wrapper following Finnhub pattern
   - `RateLimiter` class for tracking API usage:
     - Daily call counter (resets at midnight)
     - Per-minute call tracking
     - Automatic rate limit enforcement
   - Functions:
     - `fetch_intraday()` - Intraday time series
     - `fetch_daily()` - Daily time series
     - `fetch_quote()` - Real-time quote
     - `fetch_forex_intraday()` - Forex data
     - `fetch_crypto_daily()` - Crypto data
     - `check_rate_limit()` - Check if call can be made
     - `get_remaining_calls()` - Get remaining daily/per-minute calls

#### 3. **API Keys System** (Updated)
   - `api_keys.properties.example` - Added `ALPHA_VANTAGE_API_KEY`
   - `python/api_keys.py` - Added `get_alpha_vantage_key()` function

#### 4. **Data Fetcher Integration** (`python/data_fetcher.py`)
   - Added `_fetch_alpha_vantage()` function
   - Automatic rate limit checking before fetching
   - Symbol count validation against daily limits
   - 12-second delays between requests (5 calls/minute max)
   - Session state tracking for call counting

#### 5. **Data Loader UI** (`app/pages/data_loader.py`)
   - Added "Alpha Vantage (API - FREE 25 calls/day)" to data source dropdown
   - **Free-Tier Status Display:**
     - Green (>15 calls remaining): ✅ Normal status
     - Yellow (5-15 calls): ⚠️ Warning status
     - Red (<5 calls): 🚨 Critical status
     - Blocked (0 calls): 🚫 Limit reached
   - **UI Features:**
     - Real-time call counter
     - Daily limit tracking (resets at midnight UTC)
     - Per-minute rate limit display
     - Tips for staying within limits
     - Recommendations to fetch 1-5 symbols at a time

## Free Tier Limitations

### API Rate Limits
- **Daily Limit:** 25 API calls per day
- **Per-Minute Limit:** 5 API calls per minute (12 seconds between calls)
- **Reset Time:** Midnight UTC

### Data Restrictions
- Real-time quotes have 15-20 minute delay
- Limited historical data on free tier
- No WebSocket streaming (REST only)

### UI Protections
The system prevents users from exceeding limits by:
1. Displaying remaining calls prominently
2. Validating symbol count before fetch
3. Enforcing 12-second delays between requests
4. Suggesting alternatives when limit reached

## Usage Examples

### Getting an API Key
1. Visit https://www.alphavantage.co/support/#api-key
2. Enter email to receive free API key
3. Add to `api_keys.properties`:
   ```
   ALPHA_VANTAGE_API_KEY=your_key_here
   ```

### Using in Data Loader (Streamlit UI)
1. Navigate to "Data Loader" page
2. Select "Alpha Vantage (API - FREE 25 calls/day)" as data source
3. Check rate limit status display
4. Enter 1-5 symbols (recommended for free tier)
5. Select date range and interval
6. Click "🔄 Fetch Data"
7. System automatically enforces rate limits

### Using Programmatically (Python)
```python
from python.alpha_vantage_helper import (
    fetch_intraday,
    fetch_daily,
    fetch_quote,
    check_rate_limit,
    get_remaining_calls
)

# Check rate limit before fetching
can_call, message = check_rate_limit()
if can_call:
    # Fetch intraday data
    df = fetch_intraday(symbol="AAPL", interval="5min")
    print(df.head())
    
    # Check remaining calls
    daily, minute = get_remaining_calls()
    print(f"Remaining: {daily} daily, {minute} per minute")
else:
    print(f"Rate limit exceeded: {message}")

# Fetch daily data
df_daily = fetch_daily(symbol="MSFT", outputsize="compact")

# Fetch real-time quote
quote = fetch_quote(symbol="GOOGL")
print(f"Current price: ${quote['price']}")

# Forex data
df_forex = fetch_forex_intraday(
    from_symbol="EUR",
    to_symbol="USD",
    interval="5min"
)

# Crypto data
df_crypto = fetch_crypto_daily(
    symbol="BTC",
    market="USD"
)
```

### Using in Data Fetcher
```python
from python.data_fetcher import fetch_intraday_data

# Fetch using Alpha Vantage
df = fetch_intraday_data(
    symbols=["AAPL", "MSFT", "GOOGL"],
    start="2024-12-01",
    end="2024-12-02",
    interval="5m",
    source="alpha_vantage"
)
```

## Best Practices

### Staying Within Free Tier Limits
1. **Fetch Wisely:**
   - Fetch 1-5 symbols at a time
   - Use saved datasets to avoid re-fetching
   - Prefer "compact" output (last 100 data points)

2. **Use Alternatives:**
   - Yahoo Finance for unlimited stock data
   - CCXT for unlimited crypto data
   - Alpha Vantage for specific needs only

3. **Monitor Usage:**
   - Check rate limit status before fetching
   - Track remaining calls in UI
   - Plan fetches to stay under daily limit

4. **Optimize Requests:**
   - Batch similar symbols together
   - Save fetched data for reuse
   - Use daily data when intraday not needed

### Upgrading to Paid Tier
For production use or higher volumes:
- Visit https://www.alphavantage.co/premium/
- Paid tiers offer:
  - 75-1200+ API calls per minute
  - Extended historical data
  - Premium data feeds
  - Technical support

## Technical Notes

### Why Python-Only (No Rust Bindings)?
- Alpha Vantage is REST-only (no WebSocket)
- Used for historical data fetching, not real-time streaming
- Python `requests` library is sufficient
- Avoids PyO3 async complexity for non-critical path
- Can add Rust bindings later if needed for performance

### Rate Limiter Implementation
- Thread-safe rate limiting using Python's `datetime`
- Tracks call timestamps in memory
- Automatic cleanup of old timestamps
- Daily counter resets at midnight

### Session State Integration
- Streamlit session state tracks daily calls
- Counter persists across page navigations
- Resets automatically on new day
- Used for UI status display

## Troubleshooting

### Common Issues

**"Rate limit exceeded" error:**
- **Solution:** Wait 12+ seconds between calls, or wait until next day for daily reset

**"Invalid API key" error:**
- **Solution:** Check `ALPHA_VANTAGE_API_KEY` in `api_keys.properties`

**"No data returned" error:**
- **Solution:** Verify symbol format (stocks: "AAPL", crypto: "BTC", forex: "EUR/USD")

**"Too many symbols requested" error:**
- **Solution:** Reduce symbol count to stay within daily limit

### Debugging
```python
# Enable logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check API key
from python.api_keys import get_alpha_vantage_key
print(f"API Key: {get_alpha_vantage_key()}")

# Test rate limiter
from python.alpha_vantage_helper import check_rate_limit, get_remaining_calls
can_call, msg = check_rate_limit()
print(f"Can call: {can_call}, Message: {msg}")
daily, minute = get_remaining_calls()
print(f"Remaining: {daily} daily, {minute} per minute")
```

## Future Enhancements

### Potential Improvements
1. **Rust Bindings:** Add PyO3 bindings for performance-critical paths
2. **Caching:** Implement request caching to reduce API calls
3. **Batch Fetching:** Optimize multi-symbol fetching
4. **Advanced Features:** Support for technical indicators, company fundamentals
5. **Premium Tier:** Add support for premium endpoints

### Contributing
To extend Alpha Vantage integration:
1. Add new endpoints in `rust_core/connectors/alpha_vantage/src/lib.rs`
2. Add Python wrappers in `python/alpha_vantage_helper.py`
3. Update UI in `app/pages/data_loader.py`
4. Update this documentation

## References

- **Alpha Vantage API Docs:** https://www.alphavantage.co/documentation/
- **Free API Key:** https://www.alphavantage.co/support/#api-key
- **Premium Plans:** https://www.alphavantage.co/premium/
- **Finnhub Helper (Reference):** `python/finnhub_helper.py`
- **CCXT Helper (Alternative):** `python/ccxt_helper.py`

## Summary

Alpha Vantage integration provides:
- ✅ Free-tier stock, forex, and crypto data
- ✅ Built-in rate limiting and protection
- ✅ User-friendly UI with status indicators
- ✅ Comprehensive error handling
- ✅ Docker deployment ready

The system is production-ready with all free-tier limitations properly handled and communicated to users.
