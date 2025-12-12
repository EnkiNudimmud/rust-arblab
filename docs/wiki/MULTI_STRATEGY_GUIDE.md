# Multi-Strategy Trading System Guide

## Overview

This guide covers the comprehensive multi-strategy trading system implemented in the `streamlit_all_strategies.py` dashboard. The system integrates multiple quantitative trading strategies with:

- **Mathematical foundations** and equations for each strategy
- **Rich interactive visualizations** using Plotly
- **Multi-strategy backtesting** with performance comparison
- **Real-world intraday data** support via Finnhub/synthetic data
- **Rust-accelerated** computation where available

---

## Strategies Implemented

### 1. Mean Reversion Strategies

**Theory:**
Mean-reverting portfolios exploit temporary price deviations from long-term equilibrium.

**Ornstein-Uhlenbeck Process:**
```
dS_t = θ(μ - S_t)dt + σdW_t
```

Where:
- `θ` = mean reversion speed
- `μ` = long-term mean
- `σ` = volatility

#### 1.1 PC1 Mean Reversion
Uses Principal Component Analysis to find the most dominant mean-reverting direction in the asset space.

**Implementation:**
1. Compute log returns matrix
2. Apply SVD/PCA to extract principal components
3. Use first principal component (PC1) as portfolio weights
4. Backtest with z-score entry/exit thresholds

#### 1.2 CARA Optimal Portfolio
Maximizes Constant Absolute Risk Aversion (CARA) utility function.

**Equation:**
```
U(W) = -exp(-γW)

Optimal weights: w* = (1/γ) Σ⁻¹ μ
```

Where:
- `γ` = risk aversion parameter (higher = more conservative)
- `Σ` = covariance matrix
- `μ` = expected returns vector

#### 1.3 Sharpe Optimal Portfolio
Maximizes risk-adjusted returns (Sharpe ratio).

**Equation:**
```
Sharpe = (E[Rp] - rf) / σp

Optimal weights: w* = Σ⁻¹(μ - rf·1) / (1ᵀ Σ⁻¹(μ - rf·1))
```

Where:
- `rf` = risk-free rate
- `σp` = portfolio volatility

**Parameters:**
- `Entry Z-score`: Enter position when |z| > threshold (default: 2.0)
- `Exit Z-score`: Exit position when |z| < threshold (default: 0.5)
- `CARA γ`: Risk aversion coefficient (default: 2.0)
- `Risk-free rate`: Annual risk-free rate (default: 0.02)
- `Transaction cost`: Per-trade cost in basis points (default: 10 bps)

---

### 2. Pairs Trading (Statistical Arbitrage)

**Theory:**
Exploit mean-reverting relationships between cointegrated asset pairs.

**OLS Hedge Ratio:**
```
y_t = β·x_t + c + ε_t
```

**Spread:**
```
s_t = y_t - β·x_t
```

**Z-Score:**
```
z_t = (s_t - μ_s) / σ_s
```

**Trading Signals:**
- **Long spread** (buy Y, sell X) when `z < -2`
- **Short spread** (sell Y, buy X) when `z > 2`
- **Exit** when `|z| < 0.5`

**Implementation:**
1. Select two correlated assets
2. Compute OLS regression to find hedge ratio β
3. Calculate spread and rolling z-score
4. Generate trading signals based on z-score thresholds
5. Compute PnL from spread changes weighted by position

**Parameters:**
- `Rolling window`: Lookback period for z-score (default: 50)
- `Entry Z`: Threshold to enter positions (default: 2.0)

**Visualizations:**
- Spread evolution over time
- Z-score with entry/exit thresholds
- Cumulative PnL
- Position changes

---

### 3. Triangular Arbitrage

**Theory:**
Exploit price inconsistencies in triangular currency/crypto relationships.

**Three-asset triangle:** A/B, B/C, C/A

**Forward path price:**
```
P_forward = P_AB × P_BC × P_CA
```

**Arbitrage condition:**
```
P_forward ≠ 1
```

**Profit:**
```
π = |1 - P_forward| - transaction_costs
```

**Implementation:**
1. Select three assets forming a triangle
2. Compute synthetic cross rates
3. Calculate forward path product
4. Detect arbitrage when deviation exceeds threshold
5. Compute theoretical PnL (scaled for visibility)

**Parameters:**
- `Arb threshold`: Minimum deviation to trigger arbitrage (default: 0.001 = 0.1%)

**Visualizations:**
- Arbitrage signal magnitude over time
- Threshold line
- Cumulative PnL from arbitrage captures
- Number of opportunities detected

**Note:** Real-world implementation requires:
- Ultra-low latency execution
- Precise order routing
- Accurate fee modeling

---

### 4. Market Making

**Theory:**
Provide liquidity by continuously quoting bid/ask prices, profiting from the spread.

**Quote Prices with Inventory Control:**
```
P_bid = P_mid - s/2 - γ·I
P_ask = P_mid + s/2 - γ·I
```

Where:
- `s` = spread
- `I` = current inventory
- `γ` = inventory aversion parameter

**Mark-to-Market PnL:**
```
PnL_t = Cash_t + I_t × P_mid,t
```

**Implementation:**
1. Quote bid/ask around mid price
2. Adjust quotes based on inventory (skew quotes to reduce risk)
3. Simulate random fills (buy on bid, sell on ask)
4. Track inventory and cash
5. Mark positions to market continuously

**Parameters:**
- `Spread`: Bid-ask spread in basis points (default: 20 bps)
- `Inventory aversion`: How much to adjust quotes per unit inventory (default: 0.1)

**Visualizations:**
- Mid price evolution
- Inventory over time (with zero line)
- Cumulative PnL

**Risk Management:**
- Inventory limits prevent runaway exposure
- Dynamic quote adjustment (wider spread when inventory large)
- Mark-to-market accounting captures unrealized gains/losses

---

## Usage Instructions

### 1. Launch the Dashboard

```bash
cd /path/to/rust-arblab
streamlit run app/streamlit_all_strategies.py
```

The app will open in your browser at `http://localhost:8501`.

---

### 2. Configure Data

**Sidebar → Data Configuration:**

1. **Market**: Choose `crypto` or `stocks`
2. **Symbols**: Select up to 30 symbols (start with 10 for speed)
3. **Interval**: Time resolution (`1min`, `5min`, `15min`, `1h`)
4. **Days of history**: Number of days to fetch (default: 7)

**Data Sources:**
- **Synthetic**: Fast, always available, good for testing
- **Finnhub**: Real market data (set `FINNHUB_API_KEY` env var)
- **Other**: Extend `data_fetcher.py` for additional sources

---

### 3. Select Strategies

**Sidebar → Strategy Selection:**

Check which strategies to run:
- ✅ Mean Reversion (PC1, CARA, Sharpe)
- ✅ Pairs Trading
- ✅ Triangular Arb
- ✅ Market Making

---

### 4. Configure Parameters

Each strategy has dedicated parameter controls in the sidebar:

**Mean Reversion:**
- Entry/Exit Z-scores
- CARA risk aversion (γ)
- Risk-free rate

**Pairs Trading:**
- Rolling window size
- Entry Z-score

**Triangular Arbitrage:**
- Arbitrage threshold

**Market Making:**
- Spread (bps)
- Inventory aversion

**Universal:**
- Transaction cost (applies to all)

---

### 5. Run Backtest

Click **"🚀 Run Backtest"** in the sidebar.

The app will:
1. Fetch data from the selected source
2. Compute all selected strategies
3. Generate visualizations
4. Display performance metrics

---

### 6. Analyze Results

#### Strategy-Specific Plots
Each strategy shows its own detailed visualizations:
- **Mean Reversion**: PnL curves for PC1/CARA/Sharpe
- **Pairs Trading**: Spread, Z-score, PnL
- **Triangular Arb**: Arbitrage signal, opportunities, PnL
- **Market Making**: Mid price, inventory, PnL

#### Multi-Strategy Comparison
When multiple strategies are active:
- **PnL Comparison Chart**: All strategies on one plot
- **Metrics Table**: Total PnL, Sharpe, Max Drawdown, Volatility
- **Weights Heatmap**: Portfolio allocation by strategy (mean reversion only)

---

## Performance Metrics

### Total PnL
Cumulative profit/loss over the backtest period.

```
Total PnL = Final_Portfolio_Value - Initial_Value
```

### Sharpe Ratio
Risk-adjusted returns, annualized.

```
Sharpe = (μ_returns / σ_returns) × √252
```

Higher is better (>1.0 is good, >2.0 is excellent).

### Max Drawdown
Largest peak-to-trough decline.

```
DD_t = Portfolio_t - max(Portfolio_0...t)
Max DD = min(DD_t)
```

Measures downside risk (smaller absolute value is better).

### Volatility
Standard deviation of returns, annualized.

```
σ_annual = σ_daily × √252
```

Lower volatility with same returns = better risk-adjusted performance.

---

## Implementation Details

### Rust Acceleration

The system uses Rust for computationally intensive operations when available:

**Rust Functions:**
- `compute_pca_rust()`: SVD-based PCA on large return matrices
- `estimate_ou_process_rust()`: MLE estimation of OU parameters
- `cara_optimal_weights_rust()`: Matrix inversion and optimization
- `sharpe_optimal_weights_rust()`: Sharpe maximization
- `backtest_with_costs_rust()`: Fast backtesting with transaction costs
- `optimal_thresholds_rust()`: Optimal entry/exit calculation

**Fallback:**
If Rust is not available, Python implementations (NumPy/SciPy) are used automatically.

**Performance Gains:**
- 10-100× faster for large matrices (100+ assets, 10k+ periods)
- Enables real-time strategy recalculation in the dashboard

---

### Data Pipeline

```
Finnhub/Synthetic
    ↓
fetch_intraday_data()
    ↓
get_close_prices()
    ↓
Forward-fill & clean
    ↓
Strategy compute()
    ↓
Backtesting & metrics
    ↓
Visualizations
```

---

## Best Practices

### 1. Start Small
- Begin with 5-10 symbols
- Use 1-7 days of data
- Test strategies one at a time

### 2. Validate with Synthetic Data
- Synthetic data is instant and deterministic
- Good for parameter tuning
- Switch to real data for final validation

### 3. Transaction Costs Matter
- Even 10 bps can drastically affect profitability
- High-frequency strategies are especially sensitive
- Always include realistic cost estimates

### 4. Watch Correlations
- Pairs trading requires strong correlation (>0.7)
- Triangular arb needs liquid, actively traded assets
- Mean reversion works best with co-integrated portfolios

### 5. Risk Management
- Monitor Max Drawdown closely
- Set stop-loss levels for live trading
- Diversify across multiple strategies
- Use position limits

---

## Extending the System

### Add New Strategies

1. Create a new strategy class:

```python
class MyNewStrategy:
    @staticmethod
    def theory() -> str:
        return r"""
        ### My Strategy Theory
        
        Equations here...
        """
    
    @staticmethod
    def compute(prices: pd.DataFrame, params: Dict) -> Dict:
        # Strategy logic
        return {'MyStrategy': {'pnl': pnl_series, ...}}
```

2. Add to sidebar strategy selection
3. Add to main computation logic
4. Create visualizations

### Add Data Sources

Extend `python/data_fetcher.py`:

```python
def fetch_from_my_source(symbols, start, end, interval):
    # Implementation
    return dataframe
```

Update `fetch_intraday_data()` to support new source.

### Enhance Visualizations

Use Plotly's rich API:
- `make_subplots()`: Multi-panel layouts
- `add_trace()`: Overlay multiple series
- `add_annotation()`: Mark important events
- `update_layout()`: Styling and formatting

---

## Troubleshooting

### "No module named 'plotly'"
```bash
pip install plotly
```

### "Rust connector not available"
Rebuild the Rust extension:
```bash
cd rust_connector
maturin develop --release
```

### "Data fetching failed"
- Check internet connection
- Verify API keys (for Finnhub)
- Try synthetic data source
- Check symbol validity

### "Strategy computation failed"
- Ensure enough data points (need >50 for most strategies)
- Check for NaN values in data
- Reduce number of symbols
- Check parameter ranges (no negative values, etc.)

### Performance Issues
- Reduce number of symbols
- Shorten time period
- Use hourly instead of minute data
- Ensure Rust extension is built (`--release` mode)

---

## Mathematical References

### Mean Reversion
- d'Aspremont, A. (2011). "Identifying small mean-reverting portfolios"
- [Mean Reversion Paper](https://www.di.ens.fr/~aspremon/PDF/MeanRevVec.pdf)

### Pairs Trading
- Vidyamurthy, G. (2004). "Pairs Trading: Quantitative Methods and Analysis"

### Triangular Arbitrage
- Fenn, D. J., et al. (2009). "Temporal evolution of financial-market correlations"

### Market Making
- Avellaneda, M., & Stoikov, S. (2008). "High-frequency trading in a limit order book"

---

## License

See LICENSE file in repository root.

---

## Support

- **GitHub Issues**: Report bugs or request features
- **Documentation**: See `README.md`, `IMPLEMENTATION_SUMMARY.md`
- **Examples**: Check `examples/notebooks/` for Jupyter tutorials

---

**Happy Trading! 🚀📈**
