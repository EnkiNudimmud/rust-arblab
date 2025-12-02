# CCXT Data Source Guide

## Overview

**CCXT** (CryptoCurrency eXchange Trading library) is now the **recommended data source** for crypto market data in this project. It provides:

- ✅ **FREE access** to 100+ cryptocurrency exchanges
- ✅ **No API key required** for public market data
- ✅ **Second-level historical data** (1m, 5m, 15m, 30m, 1h, etc.)
- ✅ **Unified API** across all exchanges
- ✅ **Better coverage** than Finnhub for crypto markets

## Supported Exchanges

### Recommended Exchanges

| Exchange | Best For | Max Limit | Timeframes |
|----------|----------|-----------|------------|
| **Binance** | Most pairs, high liquidity | 1000 | 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M |
| **Kraken** | BTC pairs, regulated | 720 | 1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w, 15d |
| **Coinbase** | US trading, major pairs | 300 | 1m, 5m, 15m, 1h, 6h, 1d |
| **Bybit** | Perpetual futures | 200 | 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d, 1w, 1M |
| **OKX** | Wide variety of altcoins | 300 | 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d, 1w, 1M |

### All Supported Exchanges

CCXT supports 100+ exchanges including:
- Binance, Binance US, Binance Futures
- Coinbase, Coinbase Pro
- Kraken, Kraken Futures
- Bitfinex, Bitstamp
- Bybit, OKX, Huobi
- KuCoin, Gate.io
- Gemini, Bittrex
- And many more...

## Usage in the Application

### Via UI (Recommended)

1. Open the **Data Loader** page
2. Select **"CCXT - Crypto Exchanges (FREE! ⭐)"** as your data source
3. Choose an exchange (default: Binance)
4. Enter your symbols in CCXT format:
   - `BTC/USDT` for Bitcoin/Tether pair
   - `ETH/USDT` for Ethereum/Tether pair
   - `SOL/USDT` for Solana/Tether pair
5. Select your date range and interval
6. Click "Fetch Data"

### Programmatically

```python
from python.ccxt_helper import quick_fetch

# Fetch last 7 days of BTC/USDT 1-hour data from Binance
df = quick_fetch('BTC/USDT', 'binance', '1h', days_back=7)
print(df.head())
```

Or with more control:

```python
from python.ccxt_helper import create_exchange, fetch_ohlcv_range
from datetime import datetime, timedelta

# Create exchange instance
exchange = create_exchange('binance')

# Fetch data for specific date range
end_date = datetime.now()
start_date = end_date - timedelta(days=30)

df = fetch_ohlcv_range(
    exchange=exchange,
    symbol='ETH/USDT',
    timeframe='5m',
    start_date=start_date,
    end_date=end_date
)
```

## Symbol Format

CCXT uses a unified symbol format across all exchanges:

| Asset Type | Format | Example |
|------------|--------|---------|
| Spot | `BASE/QUOTE` | `BTC/USDT`, `ETH/BTC` |
| Futures | `BASE/QUOTE:SETTLE` | `BTC/USDT:USDT` |
| Perpetual | `BASE/QUOTE:SETTLE` | `ETH/USDT:USDT` |

**Common Pairs:**
- `BTC/USDT` - Bitcoin / Tether
- `ETH/USDT` - Ethereum / Tether
- `ETH/BTC` - Ethereum / Bitcoin
- `SOL/USDT` - Solana / Tether
- `DOGE/USDT` - Dogecoin / Tether
- `BNB/USDT` - Binance Coin / Tether

## Available Timeframes

Most exchanges support these timeframes:
- **Seconds**: Not commonly available
- **Minutes**: `1m`, `3m`, `5m`, `15m`, `30m`
- **Hours**: `1h`, `2h`, `4h`, `6h`, `8h`, `12h`
- **Days**: `1d`, `3d`
- **Weeks**: `1w`
- **Months**: `1M`

**Note**: Specific timeframes vary by exchange. Check the exchange info using:
```python
from python.ccxt_helper import get_available_exchanges
exchanges = get_available_exchanges()
print(exchanges['binance']['timeframes'])
```

## Rate Limits

Each exchange has rate limits to prevent abuse:

| Exchange | Rate Limit (ms) | Requests/Second |
|----------|-----------------|-----------------|
| Binance | 1200 | ~0.8 |
| Kraken | 3000 | ~0.3 |
| Coinbase | 1000 | ~1.0 |
| Bybit | 1000 | ~1.0 |
| OKX | 1000 | ~1.0 |

**Important**: The library automatically respects rate limits when you set `enableRateLimit=True` (enabled by default).

## Advantages over Finnhub

| Feature | CCXT | Finnhub |
|---------|------|---------|
| **API Key Required** | ❌ No | ✅ Yes |
| **Free Tier Limits** | None for public data | 60 calls/minute |
| **Crypto Coverage** | 100+ exchanges | Limited |
| **Historical Data** | ✅ Extensive | Limited on free tier |
| **Second-level Data** | ✅ Yes | Premium only |
| **Real-time** | ✅ Yes | Yes |

## Best Practices

1. **Choose the Right Exchange**:
   - Use Binance for maximum liquidity and most pairs
   - Use Kraken for regulatory compliance
   - Use Coinbase for US-based trading

2. **Respect Rate Limits**:
   - Always set `enableRateLimit=True`
   - Don't make parallel requests to the same exchange
   - Add delays between large batch requests

3. **Handle Errors Gracefully**:
   ```python
   try:
       df = fetch_ohlcv_data(exchange, 'BTC/USDT', '1h')
   except ValueError as e:
       print(f"Error: {e}")
       # Fall back to different exchange or symbol
   ```

4. **Cache Data**:
   - Historical data doesn't change
   - Cache fetched data locally
   - Reuse cached data for backtesting

5. **Use Appropriate Timeframes**:
   - Lower timeframes (1m, 5m) = more data = slower
   - Higher timeframes (1h, 1d) = less data = faster
   - Choose based on your strategy needs

## Troubleshooting

### "Symbol not found"
- Check if the symbol exists on that exchange
- Use correct format: `BTC/USDT` not `BTCUSDT`
- Try a different exchange

### "Network Error"
- Check your internet connection
- The exchange may be down for maintenance
- Try a different exchange

### "Rate Limit Exceeded"
- Slow down your requests
- Increase delays between fetches
- Use caching to reduce requests

### "No data returned"
- The symbol may not have data for that time period
- Check if the exchange supports that timeframe
- Try a shorter time range

## Examples

### Example 1: Fetch Bitcoin Data
```python
from python.ccxt_helper import quick_fetch

# Last 24 hours of 5-minute BTC/USDT data
df = quick_fetch('BTC/USDT', 'binance', '5m', days_back=1)
print(f"Fetched {len(df)} candles")
print(df.describe())
```

### Example 2: Multiple Symbols
```python
from python.ccxt_helper import fetch_multiple_symbols
from datetime import datetime, timedelta

symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
end = datetime.now()
start = end - timedelta(days=7)

df = fetch_multiple_symbols('binance', symbols, '1h', start, end)
print(df.head(20))
```

### Example 3: Compare Exchanges
```python
from python.ccxt_helper import create_exchange, fetch_ohlcv_data

exchanges = ['binance', 'kraken', 'coinbase']

for exchange_id in exchanges:
    exchange = create_exchange(exchange_id)
    df = fetch_ohlcv_data(exchange, 'BTC/USDT', '1h', limit=100)
    print(f"{exchange_id}: {len(df)} candles, latest price: ${df['close'].iloc[-1]:.2f}")
```

## Additional Resources

- [CCXT Documentation](https://docs.ccxt.com/)
- [CCXT GitHub](https://github.com/ccxt/ccxt)
- [Supported Exchanges](https://github.com/ccxt/ccxt/wiki/Exchange-Markets)
- [Exchange-specific Documentation](https://docs.ccxt.com/#/README?id=exchanges)

## Contributing

Found a bug or want to add support for more exchanges? Contributions are welcome!

1. Check if the exchange is supported by CCXT
2. Add exchange configuration to `RECOMMENDED_EXCHANGES` in `python/ccxt_helper.py`
3. Test the integration
4. Submit a pull request

---

**Last Updated**: November 2025
**CCXT Version**: 4.2.0+
