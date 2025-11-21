# Multi-Strategy Trading Platform - Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 0: Check Python Version (Optional but Recommended)
```bash
# Check your Python version
python --version

# Minimum: Python 3.7 (works but not recommended)
# Recommended: Python 3.8+
# Best: Python 3.10 or 3.11

# If you have Python 3.7, you may need specific package versions
# See PYTHON_VERSION_GUIDE.md for details
```

### Step 1: Install Dependencies
```bash
# For Python 3.8+ (recommended)
pip install -r app/requirements.txt

# For Python 3.7 (minimum supported)
pip install "streamlit>=1.22.0,<1.28" "pandas>=1.3.0,<2.0" "numpy>=1.21.0,<1.24"
pip install plotly>=5.0.0 scipy>=1.7.0 yfinance>=0.1.70
```

### Step 2: Configure API Keys (Optional)
```bash
# Copy the example file
cp api_keys.properties.example api_keys.properties

# Edit with your API keys
# For Finnhub (free tier available): https://finnhub.io/
nano api_keys.properties
```

Add your API key:
```properties
finnhub.api_key=your_api_key_here
```

**Note:** API keys are optional. You can use Yahoo Finance (no key needed) or upload CSV files.

### Step 3: Run the App
```bash
# Using the run script
./run_app.sh

# Or directly with streamlit
streamlit run app/main_app.py
```

The app will open in your browser at **http://localhost:8501**

---

## 📚 First Time User Guide

### 1. Load Historical Data

**Navigate to:** 📊 Data Loading

1. Select data source:
   - **Yahoo Finance** (easiest, no API key needed)
   - **Finnhub** (best for real-time, requires API key)
   - **Upload CSV** (use your own data)
   - **Mock** (synthetic data for testing)

2. Enter symbols (e.g., AAPL, MSFT, GOOGL)

3. Choose date range and interval

4. Click **"Fetch Data"**

5. Preview your data with interactive charts

**Example:**
```
Data Source: Yahoo Finance
Symbols: AAPL, MSFT, GOOGL
Start Date: 2023-01-01
End Date: 2024-11-01
Interval: 1h
```

### 2. Backtest a Strategy

**Navigate to:** ⚡ Strategy Backtest

1. Your loaded data will be available automatically

2. Select a strategy:
   - **Mean Reversion (PCA)** - Good for multiple correlated stocks
   - **Mean Reversion (CARA)** - Risk-averse optimization
   - **Pairs Trading** - For two correlated assets
   - **Triangular Arbitrage** - For cryptocurrency triangles

3. Configure parameters:
   - Entry Z-Score: 2.0 (how far from mean to enter)
   - Exit Z-Score: 0.5 (when to exit)
   - Initial Capital: $100,000

4. Click **"Run Backtest"**

5. View results:
   - Equity curve
   - Portfolio weights
   - Trade log
   - Performance metrics (Sharpe, drawdown, etc.)

### 3. Try Live Data

**Navigate to:** 🔴 Live Trading

1. Select connector (Finnhub recommended)

2. Choose connection mode:
   - **Polling (REST)** - Fetch every few seconds
   - **Streaming (WebSocket)** - Real-time updates

3. Select symbols to monitor

4. Click **"Start Live Feed"**

5. Watch real-time:
   - Bid/ask quotes
   - Price charts
   - Spread analysis

**Note:** This is paper trading only - no real money is involved!

### 4. Monitor Your Portfolio

**Navigate to:** 💼 Portfolio View

View:
- All open positions with P&L
- Portfolio value over time
- Allocation breakdown
- Trade history

You can also add manual positions for testing.

### 5. Explore Derivatives

**Navigate to:** 📈 Options & Futures

Generate:
- **Options Chain**: Complete chain with Greeks
- **Futures Curve**: Term structure analysis
- **Payoff Diagrams**: Visualize option strategies

---

## 💡 Tips & Best Practices

### Data Loading
- Start with Yahoo Finance (no API key needed)
- Use 1h or 1d intervals for faster loading
- Limit to 3-5 symbols for initial testing
- Download data for offline use

### Backtesting
- Start with simple strategies (Mean Reversion PCA)
- Use at least 6 months of data
- Include transaction costs (10 bps is realistic)
- Check multiple symbols for diversification

### Live Trading
- Use WebSocket for real-time data
- Start with just 1-2 symbols
- Monitor for a few minutes to see data flow
- Enable strategy execution only after testing

### Portfolio Management
- Reset portfolio to start fresh testing
- Export portfolio for backup
- Track P&L over time

---

## 🔧 Common Tasks

### Change Data Source
```
1. Go to Data Loading page
2. Select different source from dropdown
3. Fetch data again
```

### Add More Symbols
```
1. Go to Data Loading page
2. Add symbols in text area (one per line or comma-separated)
3. Fetch data again
```

### Export Data
```
1. Load data on Data Loading page
2. Go to "Export" tab
3. Click "Download as CSV" or "Download as Parquet"
```

### Reset Everything
```
1. Go to Portfolio View
2. Click "Reset Portfolio" in sidebar
3. Or just refresh the browser (F5)
```

---

## 🐛 Troubleshooting

### Python Version Errors
- **Check your Python version**: `python --version`
- Minimum required: Python 3.7
- Recommended: Python 3.8+
- See `PYTHON_VERSION_GUIDE.md` for detailed compatibility info

### Data Won't Load
- **Check internet connection**
- Try Yahoo Finance instead of Finnhub
- Verify symbols are correct (use standard tickers)
- Try fewer symbols or shorter date range

### Backtest Fails
- Ensure data is loaded first (go to Data Loading page)
- Check that data has enough rows (need at least 30)
- Try different strategy or parameters
- Look at error message for specific issue

### Live Feed Not Updating
- Check API key in `api_keys.properties`
- Verify connector is supported
- Try Polling mode instead of WebSocket
- Refresh the page and start again

### Slow Performance
- Use fewer symbols
- Use longer intervals (1h instead of 1m)
- Reduce date range
- Build Rust acceleration for 10-100x speedup

---

## 🎯 Example Workflows

### Workflow 1: Simple Mean Reversion Backtest
```
1. Load Data: AAPL, MSFT, GOOGL (1h, last 6 months)
2. Backtest: Mean Reversion (PCA)
3. Parameters: Entry Z=2.0, Exit Z=0.5
4. Analyze: Look at Sharpe ratio and equity curve
5. Iterate: Try different parameters or symbols
```

### Workflow 2: Live Monitoring + Portfolio Tracking
```
1. Load Data: BTC/USD, ETH/USD (Yahoo or Finnhub)
2. Start Live Feed: WebSocket mode
3. Monitor: Watch real-time prices
4. Portfolio: Add manual positions to track
5. Track: See portfolio value evolve
```

### Workflow 3: Options Analysis
```
1. Navigate to: Options & Futures
2. Enter: AAPL, Spot=$150, 30 days to expiry
3. Generate: Options chain
4. Explore: Greeks (Delta, Gamma, Theta)
5. Visualize: Payoff diagrams
```

---

## 📖 Next Steps

1. **Read full documentation**: See `app/README.md`
2. **Explore strategies**: Check `python/strategies/`
3. **View examples**: Browse `examples/notebooks/`
4. **Customize**: Add your own strategies
5. **Contribute**: Submit pull requests!

---

## 🆘 Need Help?

- **Documentation**: Check `app/README.md` for detailed info
- **Examples**: See `examples/notebooks/` for Jupyter notebooks
- **Issues**: Open an issue on GitHub
- **API Keys**: See `QUICK_CONFIG.md` for setup

---

## 🎉 Have Fun Trading!

Remember: This is a **paper trading platform** for strategy development and testing. Always thoroughly test strategies before using real money!

**Happy Trading! 📈**
