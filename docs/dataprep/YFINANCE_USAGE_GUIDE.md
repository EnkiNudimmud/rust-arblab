# Yahoo Finance Data Source Guide

Complete guide for using Yahoo Finance as a FREE stock market data source with optimized usage patterns.

## 🌟 Overview

**Yahoo Finance** provides FREE access to:
- ✅ **Stock market data** (NYSE, NASDAQ, international exchanges)
- ✅ **Cryptocurrency data** (major pairs like BTC-USD, ETH-USD)
- ✅ **Intraday data** down to 1-minute resolution
- ✅ **Historical data** going back years
- ✅ **No API key required** - completely free!

However, there are important limitations to be aware of.

---

## ⚡ Quick Start

### Basic Usage (Data Fetcher)

```python
from python.data_fetcher import fetch_intraday_data

# Fetch stock data
df = fetch_intraday_data(
    symbols=['AAPL', 'MSFT', 'TSLA'],
    start='2024-01-01',
    end='2024-01-31',
    interval='1h',
    source='yfinance'
)
```

### Advanced Usage (Helper Module with Caching)

```python
from python.yfinance_helper import fetch_stocks, fetch_crypto

# Fetch stocks with automatic caching
df = fetch_stocks(['AAPL', 'MSFT'], days=30, interval='1h')

# Fetch crypto (BTC, ETH automatically converted to BTC-USD, ETH-USD)
df = fetch_crypto(['BTC', 'ETH'], days=7, interval='1m')
```

---

## 📊 Data Limitations

### Intraday Data Limits

| Interval | Maximum History | Recommendation |
|----------|----------------|----------------|
| **1m** | **7 days** | Use for very short-term analysis only |
| **5m** | **60 days** | Good for intraday strategies |
| **15m** | **60 days** | Balanced intraday resolution |
| **30m** | **60 days** | Lower frequency intraday |
| **1h** | **Unlimited** | Best for longer backtests |
| **1d** | **Unlimited** | Daily timeframe, years of history |

### Important Notes

1. **1-minute data**: Only last 7 days available
   - Automatically enforced by Yahoo Finance
   - Older requests will return empty or truncated data
   - Use 5m or higher for longer periods

2. **5m/15m/30m data**: ~60 days maximum
   - Sufficient for most intraday strategies
   - Use 1h or daily for longer backtests

3. **Rate Limiting**: 
   - No official limits, but be respectful
   - Built-in 0.5s delay between requests
   - Automatic retry with exponential backoff

---

## 🎯 Symbol Format

### Stocks

```python
# US Stocks (simple ticker)
symbols = ['AAPL', 'MSFT', 'TSLA', 'GOOGL']

# International Stocks (add exchange suffix)
symbols = [
    'NESN.SW',   # Nestle (Swiss)
    'SAP.DE',    # SAP (German)
    '7203.T',    # Toyota (Tokyo)
    'SHOP.TO'    # Shopify (Toronto)
]
```

### Cryptocurrencies

```python
# Yahoo Finance format (with hyphen)
symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD']

# Or use the helper (auto-converts)
from python.yfinance_helper import fetch_crypto
df = fetch_crypto(['BTC', 'ETH', 'SOL'], days=7, interval='1h')
```

### Common Symbol Errors

❌ **Wrong**: `AAPL.US` (Yahoo doesn't use this format)
✅ **Right**: `AAPL`

❌ **Wrong**: `BTCUSDT` (no hyphen)
✅ **Right**: `BTC-USD`

---

## 🚀 Optimization Features

### 1. Smart Caching

The `yfinance_helper` module includes automatic caching to reduce redundant API calls:

```python
from python.yfinance_helper import fetch_stocks, clear_cache, get_cache_info

# First call - fetches from Yahoo Finance
df = fetch_stocks(['AAPL'], days=7, interval='1h')

# Second call - loads from cache (much faster!)
df = fetch_stocks(['AAPL'], days=7, interval='1h')

# Check cache
info = get_cache_info()
print(f"Cache size: {info['size_mb']} MB")

# Clear old cache (older than 24 hours)
clear_cache(older_than_hours=24)
```

**Cache Expiry Times**:
- 1m data: 5 minutes
- 5m data: 15 minutes
- 15m data: 30 minutes
- 30m data: 1 hour
- 1h data: 3 hours
- 1d data: 24 hours

### 2. Automatic Retry Logic

The enhanced fetcher includes:
- **3 retry attempts** for failed requests
- **Exponential backoff** (2s, 4s, 6s delays)
- **Detailed error messages** showing what went wrong

### 3. Date Range Validation

Automatic validation prevents requesting impossible date ranges:

```python
from python.yfinance_helper import validate_date_range

# Check if request is valid
is_valid, warning = validate_date_range(
    interval='1m',
    start='2024-01-01',
    end='2024-12-31'
)

if warning:
    print(warning)
# Output: "⚠️  Yahoo Finance: 1m data limited to 7 days..."
```

### 4. Progress Indicators

See real-time progress when fetching multiple symbols:

```
📊 Fetching 5 symbols from Yahoo Finance...
  [1/5] AAPL... ✅ 168 bars
  [2/5] MSFT... ✅ 168 bars
  [3/5] TSLA... ⚠️  No data in range
  [4/5] GOOGL... ✅ 168 bars
  [5/5] NVDA... ✅ 168 bars

✅ Successfully fetched 4/5 symbols
```

---

## 📝 Usage Examples

### Example 1: Stock Screening

```python
from python.yfinance_helper import fetch_stocks

# Get data for multiple tech stocks
tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'TSLA']
df = fetch_stocks(tech_stocks, days=30, interval='1h')

# Calculate returns
for symbol in tech_stocks:
    symbol_data = df[df['symbol'] == symbol]
    returns = (symbol_data['close'].iloc[-1] / symbol_data['close'].iloc[0] - 1) * 100
    print(f"{symbol}: {returns:.2f}%")
```

### Example 2: Intraday Mean Reversion Strategy

```python
from python.yfinance_helper import fetch_stocks
import pandas as pd

# Fetch high-frequency data
df = fetch_stocks(['SPY'], days=7, interval='1m', use_cache=True)

# Calculate mean reversion signals
df['sma_20'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(20).mean())
df['z_score'] = (df['close'] - df['sma_20']) / df.groupby('symbol')['close'].transform(lambda x: x.rolling(20).std())

# Entry signals
df['long_signal'] = df['z_score'] < -2
df['short_signal'] = df['z_score'] > 2

print(f"Long signals: {df['long_signal'].sum()}")
print(f"Short signals: {df['short_signal'].sum()}")
```

### Example 3: Multi-Asset Portfolio

```python
from python.yfinance_helper import fetch_stocks, fetch_crypto

# Fetch stocks
stocks_df = fetch_stocks(['SPY', 'TLT', 'GLD'], days=365, interval='1d')

# Fetch crypto
crypto_df = fetch_crypto(['BTC', 'ETH'], days=365, interval='1d')

# Combine
portfolio_df = pd.concat([stocks_df, crypto_df], ignore_index=True)

# Analyze correlations
pivot_df = portfolio_df.pivot(index='timestamp', columns='symbol', values='close')
correlations = pivot_df.corr()
print(correlations)
```

### Example 4: Batch Processing with Caching

```python
from python.yfinance_helper import fetch_stocks, get_cache_info

# Define stock universe
sp500_sample = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'WMT']

# First run - fetches from Yahoo Finance (slower)
print("First fetch...")
df1 = fetch_stocks(sp500_sample, days=30, interval='1h', use_cache=True)

# Check cache
cache_info = get_cache_info()
print(f"Cache files: {cache_info['files']}, Size: {cache_info['size_mb']} MB")

# Second run - loads from cache (much faster!)
print("\nSecond fetch (cached)...")
df2 = fetch_stocks(sp500_sample, days=30, interval='1h', use_cache=True)

# Verify data is identical
assert df1.equals(df2), "Cached data doesn't match!"
print("✅ Cache working perfectly!")
```

---

## 🔧 Troubleshooting

### Problem: "No data available"

**Causes**:
1. Invalid symbol format
2. Symbol doesn't exist on Yahoo Finance
3. Date range outside available data

**Solutions**:
```python
# Verify symbol at finance.yahoo.com
# For international stocks, check the correct suffix
symbols = ['SAP.DE']  # Not 'SAP' for German SAP

# Use correct crypto format
symbols = ['BTC-USD']  # Not 'BTCUSDT'

# Check date range for intraday data
# 1m data: last 7 days only
# 5m data: last 60 days only
```

### Problem: "Data may be incomplete"

**Cause**: Requesting more history than available for the interval

**Solution**:
```python
# Instead of:
df = fetch_stocks(['AAPL'], days=90, interval='1m')  # Will fail!

# Use:
df = fetch_stocks(['AAPL'], days=7, interval='1m')   # OK
# Or:
df = fetch_stocks(['AAPL'], days=90, interval='1h')  # OK
```

### Problem: Slow performance

**Solutions**:
1. **Enable caching**:
```python
df = fetch_stocks(['AAPL'], days=30, interval='1h', use_cache=True)
```

2. **Use higher timeframes** for backtesting:
```python
# Instead of 1m (7 days max), use 1h (unlimited)
df = fetch_stocks(['AAPL'], days=365, interval='1h')
```

3. **Clear old cache**:
```python
from python.yfinance_helper import clear_cache
clear_cache(older_than_hours=24)
```

### Problem: Rate limiting errors

**Solution**: The helper already includes rate limiting (0.3s delays) and retry logic. If you still hit limits:

```python
# Reduce batch size
symbols = ['AAPL', 'MSFT', 'GOOGL']  # Process in smaller batches
df1 = fetch_stocks(symbols[:2], days=7, interval='1m')
time.sleep(5)  # Additional delay between batches
df2 = fetch_stocks(symbols[2:], days=7, interval='1m')
```

---

## 🆚 Comparison with Other Data Sources

| Feature | Yahoo Finance | CCXT | Finnhub |
|---------|---------------|------|---------|
| **Cost** | FREE | FREE | Paid plans |
| **API Key** | Not required | Not required | Required |
| **Stocks** | ✅ Excellent | ❌ Not available | ✅ Good |
| **Crypto** | ⚠️ Limited | ✅ Excellent (100+ exchanges) | ⚠️ Limited |
| **Intraday** | ✅ 1m resolution | ✅ 1m resolution | ✅ 1m resolution |
| **History (1m)** | 7 days | Varies by exchange | Limited on free tier |
| **History (1h+)** | Years | Varies | Limited on free tier |
| **Reliability** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Rate Limits** | Soft limits | Per exchange | Strict limits |

### When to Use Yahoo Finance

✅ **Best for**:
- Stock market analysis
- Long-term backtesting (1h, 1d data)
- US and international equities
- No-cost solution for stocks

⚠️ **Not ideal for**:
- Cryptocurrency trading (use CCXT instead)
- Very long intraday backtests (7-60 day limits)
- Ultra-high-frequency trading

---

## 🔮 Best Practices

### 1. Choose the Right Interval

```python
# For strategy development (use higher timeframes)
df = fetch_stocks(['AAPL'], days=365, interval='1h')

# For live trading prep (use actual timeframe)
df = fetch_stocks(['AAPL'], days=7, interval='1m')
```

### 2. Always Use Caching

```python
# Cache dramatically speeds up repeated requests
df = fetch_stocks(symbols, days=30, interval='1h', use_cache=True)
```

### 3. Handle Missing Data

```python
from python.yfinance_helper import fetch_stocks

try:
    df = fetch_stocks(['AAPL', 'INVALID_SYMBOL'], days=7, interval='1h')
except ValueError as e:
    print(f"Error: {e}")
    # Handle error gracefully
```

### 4. Validate Before Large Requests

```python
from python.yfinance_helper import validate_date_range

# Check if request is reasonable
is_valid, warning = validate_date_range('1m', '2024-01-01', '2024-12-31')
if not is_valid:
    print(f"Warning: {warning}")
    # Adjust your request
```

### 5. Monitor Cache Size

```python
from python.yfinance_helper import get_cache_info, clear_cache

# Check cache periodically
info = get_cache_info()
if info['size_mb'] > 100:  # If cache > 100 MB
    clear_cache(older_than_hours=24)
```

---

## 📚 API Reference

### fetch_stocks()

Fetch stock market data with optimal settings.

**Parameters**:
- `symbols` (List[str]): Stock tickers (e.g., `['AAPL', 'MSFT']`)
- `days` (int): Number of days of historical data (default: 30)
- `interval` (str): Data interval - `'1m'`, `'5m'`, `'15m'`, `'30m'`, `'1h'`, `'1d'` (default: `'1h'`)
- `use_cache` (bool): Enable caching (default: `True`)

**Returns**: `pd.DataFrame` with columns: `timestamp`, `symbol`, `open`, `high`, `low`, `close`, `volume`

### fetch_crypto()

Fetch cryptocurrency data from Yahoo Finance.

**Parameters**:
- `coins` (List[str]): Coin symbols WITHOUT `-USD` suffix (e.g., `['BTC', 'ETH']`)
- `days` (int): Number of days of historical data (default: 7)
- `interval` (str): Data interval (default: `'1h'`)
- `use_cache` (bool): Enable caching (default: `True`)

**Returns**: `pd.DataFrame` with columns: `timestamp`, `symbol`, `open`, `high`, `low`, `close`, `volume`

**Note**: For better crypto coverage, use CCXT instead (see `python/ccxt_helper.py`)

### validate_date_range()

Validate if date range is supported for the given interval.

**Parameters**:
- `interval` (str): Data interval
- `start` (str): Start date
- `end` (str): End date

**Returns**: `Tuple[bool, Optional[str]]` - `(is_valid, warning_message)`

### clear_cache()

Clear cached data.

**Parameters**:
- `older_than_hours` (Optional[int]): Only clear cache older than this many hours. If `None`, clears all cache.

**Returns**: `int` - Number of cache files removed

### get_cache_info()

Get information about the current cache.

**Returns**: `Dict` with keys: `cache_dir`, `exists`, `files`, `size_mb`

---

## 🎓 Learning Resources

- **Yahoo Finance**: https://finance.yahoo.com
- **yfinance Documentation**: https://pypi.org/project/yfinance/
- **Symbol Lookup**: Search on finance.yahoo.com to find correct ticker formats

---

## 🆘 Support

Having issues? Check:

1. **Symbol format** - Verify on finance.yahoo.com
2. **Date range limits** - Respect intraday data limitations
3. **Error messages** - They provide detailed troubleshooting steps
4. **Cache** - Try clearing with `clear_cache()`

For crypto data, consider using CCXT instead: See `docs/CCXT_DATA_SOURCE_GUIDE.md`

---

**Last Updated**: January 2024
**Module**: `python/yfinance_helper.py`
**Related**: `python/data_fetcher.py`, `docs/CCXT_DATA_SOURCE_GUIDE.md`
