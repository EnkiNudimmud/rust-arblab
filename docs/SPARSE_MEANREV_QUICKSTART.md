# 🚀 Sparse Mean-Reversion Quick Start Guide

## 5-Minute Setup and Demo

### Prerequisites
- Python 3.8+ with conda/venv
- Rust toolchain (optional, for building from source)

### Option 1: Use Pre-Built Wheel (Fastest)
```bash
cd /Users/melvinalvarez/Documents/Workspace/rust-hft-arbitrage-lab
pip install target/wheels/rust_connector-0.1.0-*.whl
```

### Option 2: Build from Source
```bash
cd rust_connector
pip install maturin
maturin build --release
pip install target/wheels/rust_connector-0.1.0-*.whl
```

## Quick Demo

### 1. Test Rust Functions
```bash
python -c "from python.sparse_meanrev import RUST_AVAILABLE; print(f'Rust: {RUST_AVAILABLE}')"
# Output: Rust: True ⚡
```

### 2. Run Built-in Demo
```bash
cd /Users/melvinalvarez/Documents/Workspace/rust-hft-arbitrage-lab
python python/sparse_meanrev.py
```

Expected output:
```
Sparse Mean-Reverting Portfolios Demo
============================================================

1. Sparse PCA
------------------------------------------------------------
Sparse PCA Results:
  Components: 3
  Total Variance Explained: X.XX%
  Average Sparsity: XX.XX%
  Average Iterations: XXX.X

2. Hurst Exponent
------------------------------------------------------------
[... detailed results ...]

✓ Demo complete!
Rust acceleration: ENABLED ⚡
```

### 3. Launch Streamlit Dashboard
```bash
cd app
streamlit run HFT_Arbitrage_Lab.py
```

Then:
1. Navigate to **"Mean Reversion Lab"** page
2. Load data via **"Data Loader"** (need 5+ assets, 100+ periods)
3. Go to **"Sparse Mean-Reversion"** tab
4. Select algorithm: **Sparse PCA**
5. Set λ = 0.2, Components = 3
6. Click **"🚀 Run Sparse Analysis"**

## Python API Quick Reference

### Import
```python
from python.sparse_meanrev import (
    sparse_pca,
    box_tao_decomposition,
    hurst_exponent,
    sparse_cointegration,
    generate_sparse_meanrev_signals
)
```

### Example 1: Sparse PCA
```python
import pandas as pd
import numpy as np

# Generate or load data
returns = pd.DataFrame(np.random.randn(1000, 20))

# Extract sparse components
result = sparse_pca(returns, n_components=3, lambda_=0.2)

print(result.summary())
# Output:
# Sparse PCA Results:
#   Components: 3
#   Total Variance Explained: 45.23%
#   Average Sparsity: 25.00%
#   Average Iterations: 157.3

# Get portfolio weights
portfolio = result.get_portfolio(0)  # First component
print(f"Active assets: {(portfolio.abs() > 0.001).sum()}")
```

### Example 2: Test Mean-Reversion
```python
# Create portfolio value series
portfolio_value = (1 + (returns * portfolio).sum(axis=1)).cumprod()

# Test for mean-reversion
hurst_result = hurst_exponent(portfolio_value)

print(hurst_result.summary())
# Output:
# Hurst Exponent Analysis:
#   H = 0.3842 ± 0.0284
#   95% CI: [0.3558, 0.4126]
#   Interpretation: Mean-reverting (anti-persistent)
#   Mean-Reverting: True

if hurst_result.is_mean_reverting:
    print("✅ Portfolio is mean-reverting - suitable for trading!")
```

### Example 3: Generate Trading Signals
```python
# Load price data
prices = pd.read_csv('prices.csv', index_col=0)

# Generate signals
signals = generate_sparse_meanrev_signals(
    prices,
    method='sparse_pca',
    lambda_=0.15,
    lookback=252
)

# Check latest signal
latest = signals.iloc[-1]
print(f"Signal: {latest['signal']}")  # -1, 0, or 1
print(f"Z-Score: {latest['z_score']:.2f}")
print(f"Hurst: {latest['hurst']:.4f}")
print(f"Mean-Reverting: {latest['is_mean_reverting']}")

# Trading logic
if latest['is_mean_reverting']:
    if latest['signal'] == 1:
        print("🟢 BUY")
    elif latest['signal'] == -1:
        print("🔴 SELL")
    else:
        print("⚪ HOLD")
```

## Algorithm Selection Guide

### Sparse PCA
**Use when:** You want to find sparse portfolios that capture maximum variance
**Best for:** Diversified sparse portfolios, factor analysis
**Parameters:** 
- λ = 0.1-0.3 (moderate sparsity)
- n_components = 2-5

### Box & Tao Decomposition
**Use when:** You want to separate common factors from idiosyncratic behavior
**Best for:** Finding hidden mean-reversion in noisy data
**Parameters:**
- λ = 0.05-0.15 (sparse component sparsity)
- μ = 0.01 (ADMM parameter)

### Sparse Cointegration
**Use when:** You know a target asset and want a sparse hedging portfolio
**Best for:** Pairs/multi-asset cointegration strategies
**Parameters:**
- λ_l1 = 0.1-0.2 (sparsity)
- λ_l2 = 0.01 (regularization)

## Parameter Tuning Tips

### Sparsity (λ)
- **Low (0.01-0.1)**: More assets, denser portfolios
- **Medium (0.1-0.3)**: Balanced (5-10 assets typically)
- **High (0.3-1.0)**: Very sparse (1-3 assets)

### Lookback Period
- **Short (50-100)**: Fast adaptation, more noise
- **Medium (100-252)**: Balanced
- **Long (252-500)**: Stable but slow to adapt

### Entry Thresholds (Z-score)
- **Conservative**: |z| > 2.5 (fewer trades, higher confidence)
- **Moderate**: |z| > 2.0 (recommended)
- **Aggressive**: |z| > 1.5 (more trades, lower confidence)

## Common Issues

### Issue: "Rust functions not available"
**Solution:**
```bash
cd rust_connector
maturin build --release
pip install --force-reinstall target/wheels/rust_connector-0.1.0-*.whl
```

### Issue: "NaN in Hurst exponent"
**Cause:** Portfolio has zero or constant variance
**Solution:** Check that portfolio returns are non-trivial

### Issue: "Insufficient data points"
**Cause:** Need at least 100 periods for reliable analysis
**Solution:** Load more historical data

### Issue: "All weights are zero"
**Cause:** λ too high (over-penalizing)
**Solution:** Reduce λ parameter

## Performance Tips

### For Large Universes (100+ assets)
1. Use Rust implementations (10-100x faster)
2. Reduce lookback period
3. Pre-filter assets by liquidity/correlation

### For Real-Time Trading
1. Pre-compute sparse portfolios daily
2. Update only z-scores intraday
3. Monitor Hurst exponent weekly

### For Backtesting
1. Use expanding window for robustness
2. Include transaction costs
3. Test multiple λ values

## Next Steps

1. **Explore Notebook**: Open `examples/notebooks/sparse_meanrev_portfolios.ipynb`
2. **Try Dashboard**: Use Streamlit interface for interactive analysis
3. **Customize**: Modify parameters for your specific use case
4. **Backtest**: Integrate with your backtesting framework
5. **Deploy**: Connect to live data feed and execute signals

## Support

- **Documentation**: See `docs/SPARSE_MEANREV_SUMMARY.md` for full details
- **API Reference**: Docstrings in `python/sparse_meanrev.py`
- **Math Theory**: LaTeX equations in Jupyter notebook
- **Examples**: Demo script in `python/sparse_meanrev.py`

## Quick Checklist

- [ ] Rust extension installed and working
- [ ] Demo script runs successfully
- [ ] Streamlit dashboard accessible
- [ ] Data loaded (5+ assets, 100+ periods)
- [ ] Sparse analysis produces results
- [ ] Hurst exponent computed correctly
- [ ] Trading signals generated

Once all items checked, you're ready to deploy! 🚀
