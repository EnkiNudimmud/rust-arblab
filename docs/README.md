# Documentation

## User Guides

### Getting Started
- [Environment Setup](ENVIRONMENT_SETUP.md) - Set up Python environment and dependencies
- [Quick Config](QUICK_CONFIG.md) - Fast configuration guide
- [Dashboard Quickstart](DASHBOARD_QUICKSTART.md) - Get started with the Streamlit dashboard
- [Quickstart App](QUICKSTART_APP.md) - Application quickstart guide

### Trading Strategies
- [Multi-Strategy Guide](MULTI_STRATEGY_GUIDE.md) - Overview of all trading strategies
- [Chiarella Signals Guide](CHIARELLA_SIGNALS_GUIDE.md) - Chiarella model signals and usage
- [Rough Heston Guide](ROUGH_HESTON_GUIDE.md) - Rough Heston affine volatility models

### Data & Connectivity
- [Kraken WebSocket Guide](KRAKEN_WEBSOCKET_GUIDE.md) - Real-time data via Kraken WebSocket
- [Finnhub Usage](FINNHUB_USAGE.md) - Using Finnhub market data API

### Reference
- [Navigation Guide](NAVIGATION_GUIDE.md) - Navigate the application
- [Quick Reference](QUICK_REFERENCE.md) - Command and feature reference
- [Python Version Guide](PYTHON_VERSION_GUIDE.md) - Python version compatibility
- [Testing Checklist](TESTING_CHECKLIST.md) - Testing guidelines

## Project Structure

```
rust-hft-arbitrage-lab/
├── app/                    # Streamlit application
│   ├── pages/             # Strategy dashboards
│   └── utils/             # UI utilities
├── python/                # Python implementation
│   ├── data_fetcher.py   # Market data fetching
│   ├── rough_heston.py   # Rough volatility models
│   └── ...               # Strategy implementations
├── rust_core/            # Rust core library
├── rust_python_bindings/ # PyO3 Python bindings
├── tests/                # Test suite
├── examples/             # Jupyter notebooks
└── docs/                 # Documentation (you are here)
```

## Contributing

Please see the main [README.md](../README.md) for contribution guidelines.
