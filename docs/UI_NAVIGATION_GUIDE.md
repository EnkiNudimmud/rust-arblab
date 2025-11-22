# Navigation Guide - Reorganized UI

## 📚 Overview

The HFT Arbitrage Lab UI has been reorganized into a clean, section-based navigation structure inspired by modern course platforms. The interface is divided into four main sections:

## 🗺️ Navigation Structure

### 🏠 Home
**Main landing page** (`HFT_Arbitrage_Lab.py`)
- System status dashboard
- Quick start guide
- Platform overview
- Navigation to all sections

---

### 📊 Data & Market

#### 💾 Data Loader
Load historical and real-time market data
- **Data Sources**: Finnhub, Binance, Kraken, Yahoo Finance
- **Preset Lists**: 
  - 11 Sector categories (Technology, Financials, Healthcare, etc.)
  - 4 Major indexes (S&P 500, Dow Jones, NASDAQ, Russell 2000)
  - 13 ETF categories (SPY, QQQ, sector ETFs, crypto ETFs)
  - 4 Crypto categories (Top 10, DeFi, Layer 1, Meme coins)
- **Features**: CSV/Parquet export, data visualization, statistics
- **Navigation**: Direct links to Mean Reversion Lab and Arbitrage Analysis

#### 💼 Portfolio View
Real-time portfolio tracking and management
- Current positions and P&L
- Performance metrics
- Risk analysis
- Transaction history

---

### 🔬 Research Labs

Interactive research environments for quantitative analysis.

#### 📉 Mean Reversion Lab
Statistical arbitrage with Z-score analysis
- **Features**:
  - Rolling Z-score calculation
  - Entry/exit threshold configuration
  - Cointegration testing for pairs
  - Spread analysis
  - Trading signal generation
- **Parameters**: Window size, thresholds, pair selection
- **Outputs**: Price charts, Z-score plots, statistics, recent signals

#### 📈 Rough Heston Lab
Stochastic volatility modeling with rough fractional processes
- **Features**:
  - Path simulation with fractional Brownian motion
  - Parameter calibration (Hurst exponent H < 0.5)
  - Volatility surface analysis
  - Monte Carlo option pricing
- **Parameters**: Hurst, mean reversion, vol-of-vol, correlation
- **Outputs**: Price and volatility paths, distribution statistics

#### 🌀 Chiarella Model Lab
Agent-based market dynamics
- **Features**:
  - Fundamentalist vs Chartist interaction
  - Regime detection (fundamentalist/chartist dominated)
  - Bifurcation analysis
  - Trading signals from agent dynamics
- **Parameters**: Agent strengths, switching rate, volatility
- **Outputs**: Price dynamics, agent fractions, regime classification

#### ✍️ Signature Methods Lab
Path signature analysis for time series
- **Features**:
  - Signature computation up to level N
  - Path classification and pattern recognition
  - Feature extraction from price paths
  - Model-free analysis
- **Use Cases**: Optimal execution, regime detection, signal generation
- **Theory**: Rough paths, iterated integrals, signature kernels

#### 📊 Portfolio Analytics Lab
Advanced portfolio optimization and risk metrics
- **Features**:
  - Mean-variance optimization (uses Rust CARA optimization)
  - Efficient frontier visualization
  - Sharpe ratio maximization
  - Risk metrics (VaR, CVaR, max drawdown)
  - Performance attribution
- **Methods**: MPT, CARA utility, risk-return tradeoff
- **Outputs**: Optimal weights, portfolio metrics, efficient frontier

---

### ⚡ Trading Strategies

Production-ready strategy implementations.

#### 📈 Strategy Backtest
Comprehensive backtesting framework
- Multiple strategy types
- Transaction cost modeling
- Risk-adjusted performance metrics
- Parameter optimization

#### 🔄 Arbitrage Analysis
Direct application of arbitrage strategies to loaded data
- **Strategies**:
  - Statistical arbitrage
  - Triangular arbitrage
  - Pairs trading
  - Cross-exchange arbitrage
- **Features**: Real-time signal generation, P&L tracking

#### 📊 Derivatives Strategies
Complex derivatives trading strategies
- Multi-leg option spreads
- Delta-hedging
- Volatility trading

#### 🎯 Options Strategies
Options-specific analysis and execution
- Greeks calculation
- Implied volatility analysis
- Strategy payoff diagrams

---

### 🔴 Live Trading

Real-world execution and monitoring.

#### ⚡ Live Trading
WebSocket-based live trading execution
- **Exchanges**: Binance, Kraken, Coinbase
- **Features**:
  - Real-time order book
  - Low-latency execution
  - Position management
  - Real-time P&L
- **Safety**: Paper trading mode, stop losses, risk limits

#### 🎲 Live Derivatives
Live derivatives trading
- Options execution
- Futures trading
- Real-time Greeks

#### 📐 Affine Models
Affine term structure models for derivatives
- Pricing and hedging
- Model calibration
- Risk analysis

---

## 🎨 UI Design Principles

### Consistent Styling
- **Gradient cards**: Feature highlights with purple gradient
- **Status cards**: System health and metrics
- **Metric boxes**: Clean, bordered metric display
- **Navigation buttons**: Always available at page top

### Color Scheme
- **Primary**: Purple gradient (#667eea → #764ba2)
- **Success**: Green (#10b981)
- **Warning**: Yellow/Orange
- **Error**: Red (#ef4444)
- **Info**: Light blue (#f0f7ff)

### Typography
- **Headers**: Large, gradient text
- **Metrics**: Bold, colored numbers
- **Body**: Clear, readable fonts
- **Code**: Monospace for technical content

---

## 🔗 Cross-Page Navigation

### Data Flow
1. **Data Loader** → Load market data → Navigate to:
   - Mean Reversion Lab (analyze Z-scores)
   - Arbitrage Analysis (apply strategies)
   - Portfolio Analytics (optimize allocation)

2. **Research Labs** → Develop insights → Navigate to:
   - Strategy Backtest (test hypotheses)
   - Arbitrage Analysis (implement strategies)

3. **Strategy Pages** → Validated strategies → Navigate to:
   - Live Trading (deploy live)
   - Portfolio View (track performance)

### Navigation Buttons
Every page includes:
- **🏠 Home**: Return to homepage
- **Context-specific buttons**: Relevant next steps

---

## 📱 Usage Workflow

### Typical Research Workflow

1. **Start**: Go to Home page
2. **Load Data**: Data Loader → Select symbols or presets → Fetch data
3. **Explore**: Navigate to relevant Research Lab
   - Mean Reversion Lab for statistical arbitrage
   - Rough Heston Lab for volatility analysis
   - Chiarella Lab for regime detection
4. **Optimize**: Portfolio Analytics Lab → Find optimal weights
5. **Backtest**: Strategy Backtest → Validate performance
6. **Deploy**: Live Trading → Execute with real data

### Quick Analysis Workflow

1. **Home** → Click "Go to Data Loader"
2. **Load preset** (e.g., "Technology" sector)
3. **Click "Mean Rev Lab"** button
4. **Analyze** Z-scores and signals
5. **Click "Strategy Backtest"** to validate
6. **Review results** and iterate

---

## 🎓 Learning Path

### Beginners
1. Start with **Data Loader** to understand data structure
2. Explore **Mean Reversion Lab** for basic statistical concepts
3. Try **Portfolio Analytics** for optimization basics
4. Review **Strategy Backtest** for performance evaluation

### Intermediate
1. **Rough Heston Lab**: Advanced volatility modeling
2. **Chiarella Lab**: Agent-based dynamics
3. **Arbitrage Analysis**: Multi-strategy implementation
4. **Live Trading**: Paper trading practice

### Advanced
1. **Signature Methods Lab**: Cutting-edge path analysis
2. **Derivatives Strategies**: Complex multi-leg trades
3. **Affine Models**: Advanced term structure modeling
4. **Live Trading**: Production deployment with risk management

---

## 🔧 Technical Notes

### st-pages Package
The navigation uses `st-pages` for section-based structure:
```python
from st_pages import Page, Section, show_pages

show_pages([
    Page("app/HFT_Arbitrage_Lab.py", "Home", "⚡"),
    Section("📊 Data & Market", "📊"),
    Page("app/pages/data_loader.py", "Data Loader", "💾", in_section=True),
    # ... more pages
])
```

### Session State Management
All pages share session state:
- `st.session_state.historical_data`: Loaded market data
- `st.session_state.portfolio`: Portfolio positions
- `st.session_state.symbols`: Selected symbols
- `st.session_state.derivatives_data`: Options/futures data

### Page Switching
Navigate programmatically:
```python
if st.button("Go to Mean Rev Lab"):
    st.switch_page("app/pages/lab_mean_reversion.py")
```

---

## 📚 References

- **DE Zoomcamp UI**: https://github.com/hamagistral/de-zoomcamp-ui
- **st-pages docs**: https://github.com/blackary/st_pages
- **Streamlit docs**: https://docs.streamlit.io

---

**Last Updated**: November 22, 2025  
**Version**: 2.0 (Reorganized UI)
