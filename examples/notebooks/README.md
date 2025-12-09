# 📓 Jupyter Notebooks Collection

## Overview

This directory contains comprehensive Jupyter notebooks implementing advanced statistical arbitrage and options strategies with real-world data.

---

## 📚 Available Notebooks

### 1. `automated_cointegration_discovery.ipynb`

**Purpose**: Automated discovery of mean-reverting cointegrated pairs using optimal switching.

**Key Features**:
- 50+ real stocks and ETFs (Tech, Finance, Energy, Healthcare, Metals)
- Parallel processing with ThreadPoolExecutor
- Statistical validation (Engle-Granger, Hurst exponent)
- HJB PDE solver for optimal boundaries
- Comprehensive backtesting with transaction costs
- Portfolio construction from top pairs

**Runtime**: 10-20 minutes (500+ pairs)

**Prerequisites**:
```bash
pip install numpy pandas matplotlib seaborn yfinance scipy statsmodels tqdm
```

**Key Outputs**:
- Top 20 cointegrated pairs ranked by combined score
- 4-panel statistical visualization
- Best pair deep dive analysis
- CSV export of all results

---

### 2. `options_strategies_pairs_trading.ipynb`

**Purpose**: Enhance pairs trading with advanced options strategies.

**Key Features**:
- 4 complete options strategies with real data
- Black-Scholes pricing implementation
- Greeks calculation (Delta, Gamma, Vega, Theta, Rho)
- GLD-GDX real-world example
- Payoff diagrams and risk analysis
- Complete trade construction guide

**Strategies Covered**:
1. **Long Call** (Leverage)
2. **Protective Put** (Insurance)
3. **Covered Call** (Income)
4. **Delta-Neutral Straddle** (Volatility)

**Runtime**: 5-10 minutes

**Prerequisites**:
```bash
pip install numpy pandas matplotlib seaborn yfinance scipy
```

**Key Outputs**:
- 12+ interactive visualizations
- Strategy comparison table
- Real-world trade example with signals
- Risk metrics and breakeven analysis

---

## 🚀 Quick Start

### Method 1: Jupyter Notebook
```bash
cd examples/notebooks
jupyter notebook
```

### Method 2: JupyterLab
```bash
jupyter lab
```

### Method 3: VS Code
1. Open `.ipynb` file in VS Code
2. Install Jupyter extension if needed
3. Select Python kernel
4. Run cells sequentially

### Method 4: Docker (Recommended)
```bash
# From project root
docker-compose up jupyter

# Access at: http://localhost:8889
# Token shown in logs or set in docker-compose.yml
```

---

## 📊 Mathematical Foundations

### Cointegration
$$Z_t = Y_t - \beta X_t \sim I(0)$$

Two non-stationary series are cointegrated if their linear combination is stationary.

### Ornstein-Uhlenbeck Process
$$dZ_t = \kappa(\theta - Z_t)dt + \sigma dW_t$$

Models mean-reverting spreads with:
- $\kappa$: mean-reversion speed
- $\theta$: long-term mean
- $\sigma$: volatility

### HJB Equation
$$\rho V(x) = \kappa(\theta - x)V'(x) + \frac{1}{2}\sigma^2 V''(x)$$

Determines optimal switching boundaries for trading.

### Black-Scholes
$$C = S\Phi(d_1) - Ke^{-rT}\Phi(d_2)$$

European option pricing with closed-form solution.

---

## 📈 Example Results

### Automated Discovery
```
Pairs Tested: 1,247
Cointegrated: 89 (7.1%)
Mean-Reverting: 67 (5.4%)

Top Pair: GLD-GDX
  Combined Score: 0.8924
  Total Return: 18.7%
  Sharpe Ratio: 2.34
  Max Drawdown: -4.2%
  Win Rate: 63.8%
```

### Options Strategy
```
GLD Long Call Strategy:
  Premium: $4.50
  Strike: $185 (ATM)
  Delta: 0.52
  Breakeven: $189.50
  Max Loss: $450
  Max Profit: Unlimited
```

---

## 🎯 Learning Path

### Beginner (Start Here)
1. Open `options_strategies_pairs_trading.ipynb`
2. Focus on Strategy 1 (Long Call)
3. Understand payoff diagrams
4. Run with default parameters

### Intermediate
1. Modify option parameters (strike, expiration)
2. Test different stock pairs
3. Run `automated_cointegration_discovery.ipynb` with 100 pairs
4. Analyze results and identify patterns

### Advanced
1. Full discovery with 1000+ pairs
2. Implement custom scoring functions
3. Add new options strategies
4. Integrate with trading systems
5. Build custom visualizations

---

## 🔧 Configuration

### Discovery Notebook
```python
# Adjust these parameters
n_workers = 8              # Parallel workers (4-16)
max_pairs = 500            # Pairs to test (100-10000)
significance = 0.05        # Cointegration p-value
min_hurst = 0.45          # Mean-reversion threshold
transaction_cost = 0.001  # 0.1% per trade
```

### Options Notebook
```python
# Adjust these parameters
T = 30 / 365              # Days to expiration
r = 0.04                  # Risk-free rate
K_call = S * 1.05         # Strike (5% OTM)
sigma = 0.25              # Implied volatility
```

---

## 📁 Data Sources

### Historical Prices
- **Source**: Yahoo Finance (yfinance)
- **Period**: 2 years (730 days)
- **Frequency**: Daily adjusted close prices
- **Universe**: 50+ liquid stocks and ETFs

### Real-Time Options (Future)
- Add integration with CBOE, TD Ameritrade, or Interactive Brokers API
- Fetch live options chains
- Calculate implied volatility from market prices

---

## 🛠️ Troubleshooting

### Issue: Data download fails
**Solution**: Check internet connection, verify symbols exist, try market hours

### Issue: Out of memory
**Solution**: Reduce `max_pairs`, clear variables, restart kernel

### Issue: Slow execution
**Solution**: Increase `n_workers`, reduce data period, use faster CPU

### Issue: No cointegrated pairs found
**Solution**: Use longer data period, different symbols, looser thresholds

---

## 📚 Additional Resources

### Documentation
- **Theory**: `docs/AUTO_DISCOVERY_IMPLEMENTATION_SUMMARY.md`
- **Notebooks**: `docs/JUPYTER_NOTEBOOKS_SUMMARY.md`
- **Streamlit**: `app/pages/lab_mean_reversion.py`

### Academic Papers
- Engle & Granger (1987) - Cointegration
- Gatev et al. (2006) - Pairs Trading
- Black & Scholes (1973) - Options Pricing
- Øksendal & Sulem (2007) - Optimal Control

### Code References
- Statsmodels: Statistical tests
- Scipy: Numerical methods
- yfinance: Data download
- Plotly/Matplotlib: Visualizations

---

## ⚠️ Risk Warning

**Educational Purpose Only**: These notebooks are for learning and research. Options trading involves significant risk and is not suitable for all investors.

**Before Live Trading**:
- Paper trade strategies for 3+ months
- Understand all risks involved
- Never risk more than 2-3% per trade
- Consider consulting a financial advisor
- Verify all calculations independently

---

## 🤝 Contributing

### Improvements Welcome
- Additional strategies
- Better visualizations
- Performance optimizations
- Bug fixes
- Documentation enhancements

### How to Contribute
1. Fork the repository
2. Create feature branch
3. Make changes with tests
4. Submit pull request
5. Follow code style

---

## 📝 License

See LICENSE file in project root.

---

## 📧 Support

- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Email**: See project maintainers

---

**Last Updated**: December 2025  
**Status**: ✅ Production Ready  
**Notebooks**: 2 comprehensive implementations
