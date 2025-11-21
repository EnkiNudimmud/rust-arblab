# 🎯 HFT Arbitrage Lab - Quick Navigation Guide

## 🚀 Starting the Application

```bash
cd /Users/melvinalvarez/Documents/Workspace/rust-hft-arbitrage-lab
streamlit run app/main_app.py
```

The app will open at: **http://localhost:8501**

## 📋 Available Pages & Features

### 1. 🏠 Main Dashboard (Home)
- View platform status (Rust engine, API keys, market data)
- Browse 12+ strategy cards organized by category
- Quick navigation to all features

### 2. 📊 Data Loading (`pages/data_loader.py`)
**What it does:**
- Load historical market data from multiple sources
- Preview and validate data before backtesting
- Export data in CSV or Parquet format

**Features:**
- ✅ Finnhub API integration
- ✅ Yahoo Finance fallback
- ✅ CSV file upload
- ✅ Mock/synthetic data generation
- ✅ Universe selection (Tech, Finance, Healthcare, etc.)
- ✅ Data quality checks

**How to access:**
- From main dashboard: Any strategy card with "Load Data" button
- Direct navigation in sidebar

### 3. ⚡ Strategy Backtest (`pages/strategy_backtest.py`)
**What it does:**
- Backtest trading strategies on historical data
- Visualize equity curves, drawdowns, and metrics
- Compare strategy performance

**Available Strategies:**
- Mean Reversion (Z-score based)
- Pairs Trading
- Triangular Arbitrage
- Market Making
- Statistical Arbitrage
- Mean Reversion variants (PCA, CARA, Sharpe)

**Key Features:**
- ✅ Configure parameters (window, thresholds, capital)
- ✅ Transaction cost modeling
- ✅ Performance metrics (Sharpe, max DD, win rate)
- ✅ Interactive charts with Plotly
- ✅ Trade log with entry/exit analysis

**How to access:**
- Main dashboard → "🚀 Launch Mean Reversion Lab"
- Main dashboard → "🚀 Launch Pairs Trading"

### 4. 📈 Derivatives (`pages/derivatives.py`)
**What it does:**
- Options chain analysis with Greeks
- Futures term structure
- Options strategy builder
- Payoff diagrams

**Strategy Types:**
- Straddle / Strangle
- Butterfly Spread
- Iron Condor / Iron Butterfly
- Vertical Spreads (Bull/Bear)
- Calendar Spreads
- Covered Call / Cash-Secured Put
- Ratio Spreads
- Single options (Calls/Puts)

**Features:**
- ✅ Black-Scholes pricing
- ✅ Greeks calculation (Delta, Gamma, Theta, Vega, Rho)
- ✅ Interactive payoff diagrams
- ✅ Break-even analysis
- ✅ Max profit/loss calculations
- ✅ Implied volatility surface

**How to access:**
- Main dashboard → "🚀 Launch Options Lab"
- Main dashboard → "🚀 Launch Vol Trading"

### 5. 🔴 Live Trading (`pages/live_trading.py`)
**What it does:**
- Real-time market data via WebSocket
- Live trading signals
- Order execution simulation
- P&L tracking

**Supported Exchanges:**
- Binance (crypto)
- Kraken (crypto)
- Coinbase (crypto)
- CoinGecko (price data)

**Features:**
- ✅ Multi-exchange WebSocket streams
- ✅ Real-time price updates
- ✅ Live signal generation
- ✅ Position tracking
- ✅ P&L monitoring
- ✅ Risk management alerts

**How to access:**
- Main dashboard → "🚀 Launch Live Trading"

### 6. 📊 Portfolio Monitor (`pages/portfolio_view.py`)
**What it does:**
- Track all open positions
- Monitor portfolio performance
- Risk metrics and exposure analysis
- Multi-asset portfolio management

**Features:**
- ✅ Real-time position values
- ✅ Profit/Loss tracking
- ✅ Equity curve visualization
- ✅ Drawdown analysis
- ✅ Sector exposure
- ✅ Risk metrics (Sharpe, volatility, max DD)
- ✅ Manual position entry
- ✅ Portfolio export (JSON)

**How to access:**
- Main dashboard → "🚀 Launch Portfolio"

## 🔧 Advanced Features

### Rust Analytics Module
High-performance computations for large datasets (100+ symbols):
```python
import hft_py

# Correlation matrix
corr = hft_py.analytics.compute_correlation_matrix(returns)

# Rolling z-scores
zscores = hft_py.analytics.compute_rolling_zscores(prices, window=20)

# Statistical metrics
mean = hft_py.analytics.compute_mean(data)
std = hft_py.analytics.compute_std(data)
skew = hft_py.analytics.compute_skewness(data)
```

### Universe Selection
Pre-defined symbol universes available:
- `sp500_top100` - Top 100 S&P 500 stocks
- `tech` - 30 technology stocks
- `finance` - 30 financial stocks
- `healthcare` - 30 healthcare stocks
- `energy` - 20 energy stocks
- `crypto_major` - 25 major cryptocurrencies
- `crypto_defi` - 15 DeFi tokens
- `etf_indices` - 10 major index ETFs
- `etf_sector` - 20 sector ETFs

```python
from python.universes import get_universe, get_available_universes

symbols = get_universe("tech")
all_universes = get_available_universes()
```

### Regime Detection
Adaptive strategy selection based on market conditions:
```python
from python.regime_detector import RegimeDetector, get_regime_metrics

detector = RegimeDetector(lookback_window=50)
regime = detector.detect_regime(returns)  # 'mean_reverting', 'trending', 'high_volatility'
metrics = get_regime_metrics(returns)
```

### Live Monitoring & Alerts
Real-time signal monitoring with multiple alert types:
```python
from python.signal_monitor import SignalMonitor

monitor = SignalMonitor(alert_file='alerts.jsonl', verbose=True)
monitor.check_signal_threshold("AAPL", z_score=2.5)
monitor.check_regime_change("AAPL", "trending", probabilities)
alerts = monitor.get_alerts(since=datetime.now() - timedelta(hours=1))
```

## 🐛 Troubleshooting

### Pages showing blank?
- ✅ **Fixed!** All pages now call their `render()` functions
- If issue persists, restart Streamlit: `pkill -f streamlit && streamlit run app/main_app.py`

### "No module named 'python'" error?
```bash
# Make sure you're in the project root
cd /Users/melvinalvarez/Documents/Workspace/rust-hft-arbitrage-lab
export PYTHONPATH=$PYTHONPATH:$(pwd)
streamlit run app/main_app.py
```

### API key errors?
```bash
# Check API keys are configured
cat api_keys.properties

# Should contain:
# FINNHUB_API_KEY=your_key_here
```

### Rust module not loading?
```bash
# Reinstall Rust module
cd rust_python_bindings
maturin develop --release
```

## 📊 Typical Workflow

1. **Load Data**
   - Navigate to Data Loading page
   - Select data source (Finnhub recommended)
   - Choose universe or custom symbols
   - Set date range and timeframe
   - Click "Load Data"

2. **Backtest Strategy**
   - Navigate to Strategy Backtest page
   - Select strategy type
   - Configure parameters
   - View equity curve and metrics
   - Analyze trade log

3. **Analyze Derivatives** (optional)
   - Navigate to Derivatives page
   - Select options strategy
   - Configure strikes and expiry
   - View payoff diagram and Greeks

4. **Monitor Live** (optional)
   - Navigate to Live Trading page
   - Start WebSocket streams
   - Monitor real-time signals
   - Track P&L

5. **Review Portfolio**
   - Navigate to Portfolio Monitor
   - View all positions
   - Check risk metrics
   - Export for records

## 🎨 UI Tips

- Use **wide layout** for better visibility (already configured)
- **Sidebar** always shows navigation
- **Status indicators** at top show system health
- **Strategy cards** are clickable - launch directly from main page
- Most charts are **interactive** (zoom, pan, hover for details)

## 📚 Documentation Files

- `README.md` - Project overview
- `RUST_REFACTORING_SUMMARY.md` - Rust module details
- `STREAMLIT_FIX.md` - Navigation fix details
- `ENHANCEMENT_SUMMARY.md` - New features summary
- `QUICK_REFERENCE.md` - This file

---

**Status**: ✅ All pages working, navigation fixed, fully functional platform
**Last Updated**: 2025-11-20
