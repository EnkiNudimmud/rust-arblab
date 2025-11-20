# Multi-Strategy HFT Trading Platform

A comprehensive, unified Streamlit application for high-frequency trading strategy development, backtesting, and live execution.

## Features

### 📊 Data Loading
- **Multiple Data Sources**: Finnhub, Yahoo Finance, CSV upload, Mock/Synthetic data
- **Historical Data**: Load OHLCV data at various intervals (1m, 5m, 15m, 30m, 1h, 1d)
- **Data Preview**: Interactive charts, statistics, and data quality metrics
- **Export**: Download data in CSV or Parquet format
- **Caching**: Efficient data caching for fast re-access

### ⚡ Strategy Backtesting
- **Multiple Strategies**:
  - Mean Reversion (PCA, CARA, Sharpe optimization)
  - Pairs Trading
  - Triangular Arbitrage
  - Market Making
  - Statistical Arbitrage
- **Performance Metrics**: Sharpe ratio, max drawdown, total return, win rate
- **Visualizations**: Equity curves, portfolio weights, trade analysis
- **Transaction Costs**: Configurable transaction costs in basis points

### 🔴 Live Trading
- **Real-Time Data**: WebSocket and REST API support
- **Live Strategy Execution**: Run strategies on real-time market data
- **Risk Management**: Position limits, stop-loss, take-profit
- **Trade Logging**: Complete trade history with timestamps
- **Multiple Connectors**: Support for various exchanges and data providers

### 💼 Portfolio View
- **Holdings Management**: Track all open positions with P&L
- **Performance Tracking**: Portfolio value evolution over time
- **Allocation Analysis**: Visual breakdown of portfolio allocation
- **P&L Analytics**: Win/loss ratio, distribution, cumulative P&L
- **Manual Positions**: Add/remove positions manually

### 📈 Options & Futures
- **Options Chain**: Complete options chain with Greeks (Delta, Gamma, Theta, Vega, Rho)
- **Black-Scholes Pricing**: Accurate options pricing model
- **Futures Curves**: Term structure analysis with contango/backwardation detection
- **Payoff Diagrams**: Interactive payoff visualizations
- **Strategy Builder**: (Coming soon) Multi-leg options strategies

## Installation

### Prerequisites
```bash
# Python 3.7 or higher (3.8+ recommended for best performance)
python --version

# Verify version compatibility
python -c "import sys; assert sys.version_info >= (3, 7), 'Python 3.7+ required'"

# Install dependencies
pip install -r requirements.txt
```

### Required Packages
```
# Minimum versions (compatible with Python 3.7+)
streamlit>=1.22.0
pandas>=1.3.0
numpy>=1.21.0
plotly>=5.0.0
scipy>=1.7.0
yfinance>=0.1.70

# For Python 3.8+, you can use newer versions:
# streamlit>=1.28.0
# pandas>=2.0.0
# numpy>=1.24.0
```

### Optional (for enhanced features)
```
# Rust acceleration (10-100x speedup)
# Build from rust_connector/ directory
cd rust_connector
cargo build --release
```

## Configuration

### API Keys
Create `api_keys.properties` in the project root:

```properties
# Finnhub (recommended for real-time data)
finnhub.api_key=your_finnhub_api_key_here

# Additional connectors
# kraken.api_key=your_kraken_key
# kraken.api_secret=your_kraken_secret
```

See `api_keys.properties.example` for template.

### Data Sources
- **Finnhub**: Real-time and historical market data (API key required)
- **Yahoo Finance**: Free historical data (no API key needed)
- **CSV Upload**: Use your own data files
- **Mock**: Synthetic data for testing

## Usage

### Starting the App

```bash
# Run the multi-page app
streamlit run app/main_app.py

# Or use the development script
./dev.sh
```

The app will open in your browser at `http://localhost:8501`

### Quick Start Workflow

1. **Load Data** (📊 Data Loading page)
   - Select data source (Finnhub, Yahoo Finance, CSV, or Mock)
   - Choose symbols and date range
   - Click "Fetch Data" to load historical data
   - Preview charts and statistics

2. **Backtest Strategy** (⚡ Strategy Backtest page)
   - Select a trading strategy
   - Configure strategy parameters
   - Set initial capital and transaction costs
   - Click "Run Backtest" to see results
   - Analyze equity curves, trades, and performance metrics

3. **Live Trading** (🔴 Live Trading page)
   - Configure data source and symbols
   - Choose between Polling (REST) or Streaming (WebSocket)
   - Optionally enable strategy execution
   - Click "Start Live Feed" to begin receiving market data
   - Monitor real-time quotes, charts, and signals

4. **Monitor Portfolio** (💼 Portfolio View page)
   - View all open positions with current P&L
   - Track portfolio value evolution
   - Analyze allocation and performance
   - Add/remove positions manually if needed

5. **Derivatives** (📈 Options & Futures page)
   - Generate options chains with Greeks
   - Analyze futures term structure
   - View payoff diagrams
   - (Coming soon) Build multi-leg options strategies

## Architecture

### Project Structure
```
app/
├── main_app.py              # Main application entry point
├── pages/                   # Individual page modules
│   ├── __init__.py
│   ├── data_loader.py       # Data loading and preview
│   ├── strategy_backtest.py # Strategy backtesting
│   ├── live_trading.py      # Live trading execution
│   ├── portfolio_view.py    # Portfolio management
│   └── derivatives.py       # Options and futures
└── utils/                   # Shared utilities
    ├── __init__.py
    └── common.py            # Common functions

python/                      # Core Python modules
├── rust_bridge.py           # Rust integration
├── data_fetcher.py          # Data fetching utilities
├── meanrev.py               # Mean reversion strategies
└── strategies/              # Strategy implementations
    ├── definitions.py
    └── executor.py

rust_core/                   # Rust core (optional, for performance)
rust_connector/              # Rust bindings
```

### Session State Management
The app uses Streamlit's session state to maintain data across pages:
- `historical_data`: Loaded market data
- `portfolio`: Virtual portfolio state
- `backtest_results`: Strategy backtest results
- `live_data_buffer`: Live market data buffer
- `trade_log`: All executed trades

## Strategies

### Mean Reversion
Constructs mean-reverting portfolios using:
- **PCA**: Principal Component Analysis to find cointegrated portfolios
- **CARA**: Constant Absolute Risk Aversion utility optimization
- **Sharpe**: Maximum Sharpe ratio optimization

### Pairs Trading
Statistical arbitrage on correlated pairs using z-score of spread.

### Triangular Arbitrage
Exploit price discrepancies across three currency pairs.

### Market Making
Provide liquidity by quoting bid/ask with inventory management.

## Performance Metrics

The app calculates comprehensive performance metrics:
- **Returns**: Total return, annualized return
- **Risk**: Sharpe ratio, Sortino ratio, max drawdown, volatility
- **Trading**: Win rate, average win/loss, number of trades
- **Advanced**: Calmar ratio, VaR, CVaR

## Data Export

All data can be exported:
- **Historical Data**: CSV or Parquet format
- **Backtest Results**: Performance metrics and trades
- **Portfolio**: JSON format with positions and history
- **Trade Log**: Complete trade history

## WebSocket Support

The app supports WebSocket connections for real-time data:
- **Streaming Mode**: Real-time market data updates
- **Auto-reconnect**: Handles connection interruptions
- **Multiple Symbols**: Subscribe to multiple symbols simultaneously
- **Low Latency**: Sub-second update frequency

## Risk Management

Built-in risk management features:
- **Position Limits**: Maximum position size per symbol
- **Portfolio Limits**: Maximum total exposure
- **Stop Loss**: Automatic position exit on adverse moves
- **Transaction Costs**: Realistic cost modeling
- **Slippage**: (Coming soon) Slippage modeling for realistic backtests

## Customization

### Adding New Strategies
1. Create strategy definition in `python/strategies/definitions.py`
2. Implement execution logic in `python/strategies/executor.py`
3. Add to strategy selection in `app/pages/strategy_backtest.py`

### Adding New Data Sources
1. Implement connector in `python/connectors/`
2. Register in `python/rust_bridge.py`
3. Add to connector selection in pages

### Custom Visualizations
Use the plotting helpers in `app/utils/common.py` or create custom Plotly charts.

## Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Ensure all dependencies are installed
pip install -r requirements.txt

# Add project to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**2. API Key Issues**
- Verify `api_keys.properties` exists in project root
- Check API key format (no quotes, no extra spaces)
- Ensure API key is active and has necessary permissions

**3. Data Loading Fails**
- Check internet connection
- Verify symbols are correct
- Try different data source (Yahoo Finance as fallback)
- Check API rate limits

**4. Performance Issues**
- Enable Rust acceleration for 10-100x speedup
- Use data caching (`@st.cache_data`)
- Reduce data range or number of symbols
- Use lower frequency data (1h instead of 1m)

## Future Enhancements

- [ ] Multi-strategy portfolio optimization
- [ ] Options strategy builder with Greeks hedging
- [ ] Machine learning strategy integration
- [ ] Order execution simulation
- [ ] Historical strategy comparison
- [ ] Advanced risk analytics
- [ ] Custom indicator builder
- [ ] Alerts and notifications
- [ ] Mobile-responsive design
- [ ] Multi-user support

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

See LICENSE file in project root.

## Support

For issues, questions, or feature requests:
- Open an issue on GitHub
- Check existing documentation
- Review example notebooks in `examples/notebooks/`

## Acknowledgments

Built with:
- [Streamlit](https://streamlit.io/) - Interactive web apps
- [Plotly](https://plotly.com/) - Interactive visualizations
- [Pandas](https://pandas.pydata.org/) - Data manipulation
- [NumPy](https://numpy.org/) - Numerical computing
- [SciPy](https://scipy.org/) - Scientific computing
- [Rust](https://www.rust-lang.org/) - High-performance computing (optional)

---

**Version**: 1.0.0  
**Last Updated**: November 2025  
**Author**: Rust HFT Arbitrage Lab
