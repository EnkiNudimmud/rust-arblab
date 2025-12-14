# Alpaca Data Source Integration

## Features
- **1-second bars** for US stocks (NYSE, NASDAQ, AMEX)
- **Up to 5 years** of historical data
- **Live trading and streaming** via WebSocket
- **Free account required** (no credit card)
- **API keys** configured in `api_keys.properties`
- **Rate limit:** 200 requests/minute
- **Market hours only**

## How to Use
1. Go to **Data Loader**
2. Select **Alpaca (FREE! US Stocks, 1s bars)** as data source
3. Enter US stock symbols (e.g. `AAPL, MSFT, TSLA`)
4. Choose date range and frequency (1Sec recommended)
5. Click **Fetch Data**
6. Data will be auto-converted to MultiIndex format for analysis

## Limitations
- Only US stocks supported
- No ETFs, no crypto
- 1-second bars only available for market hours
- API rate limits apply
- Requires free Alpaca account and API keys

## Example Python Usage
```python
from python.data.fetchers.alpaca_helper import fetch_alpaca_batch
symbols = ["AAPL", "MSFT"]
start = "2025-01-01"
end = "2025-01-07"
df = fetch_alpaca_batch(symbols, start, end, timeframe="1Sec", limit=10000)
```

## Example Rust Usage
```rust
use rust_connector::alpaca::fetch_alpaca_bars;
// ...async context...
let bars = fetch_alpaca_bars(
    api_key, api_secret, base_url,
    "AAPL", "2025-01-01T09:30:00Z", "2025-01-01T16:00:00Z",
    "1Sec", 10000
).await?;
```

## Live Trading / Streaming
- Use `stream_alpaca_quotes` in Python
- Use `stream_alpaca_quotes` in Rust (WebSocket stub)

## Troubleshooting
- Ensure API keys are set in `api_keys.properties`
- Only US stocks are supported
- If you hit rate limits, reduce symbol count or date range
- For live trading, use Alpaca's paper trading endpoint

## References
- [Alpaca API Docs](https://alpaca.markets/docs/api-references/trading-api/)
- [Alpaca Data Docs](https://alpaca.markets/docs/api-references/market-data-api/)
