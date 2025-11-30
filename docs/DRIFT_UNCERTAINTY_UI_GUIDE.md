# Drift Uncertainty Optimizer - Streamlit UI Guide

## Overview

A dedicated Streamlit page for **robust portfolio optimization** under drift uncertainty, based on the Bismuth-Guéant-Pu paper.

## Location

`app/pages/drift_uncertainty_optimizer.py`

## Features

### 🛡️ Tab 1: Robust Portfolio Choice
- **Input**: Select assets, initial wealth
- **Parameters**: Risk aversion (γ), drift uncertainty (δ)
- **Output**:
  - Optimal portfolio weights
  - Expected vs worst-case returns
  - Portfolio variance and utility
  - Sensitivity analysis across uncertainty levels
  - Position sizes in dollars and shares

**Use Case**: Find optimal portfolio allocation that performs well even if expected returns are misestimated

### 📉 Tab 2: Optimal Liquidation
- **Input**: Position size, time horizon, market impact parameter
- **Parameters**: Risk aversion, drift uncertainty
- **Output**:
  - Trading schedule (remaining position over time)
  - Trading velocity (shares per day)
  - Expected vs worst-case liquidation costs
  - Summary statistics (peak rate, cost per share)

**Use Case**: Liquidate a large position over time while minimizing costs and managing uncertainty

### 🔄 Tab 3: Portfolio Transition
- **Input**: Current weights, target weights, assets
- **Parameters**: Transaction costs, time horizon, drift uncertainty
- **Output**:
  - Weight trajectory (how weights evolve over time)
  - Trading velocities by asset
  - Transition costs (expected and worst-case)
  - Summary table comparing current, target, and final weights

**Use Case**: Rebalance from one portfolio allocation to another optimally

### ⚠️ Tab 4: Risk Analysis
- **Input**: Assets, weights, confidence level
- **Parameters**: Drift uncertainty, time horizon
- **Output**:
  - VaR (Value at Risk) at specified confidence level
  - CVaR (Expected Shortfall)
  - Comparison across multiple confidence levels
  - Portfolio composition details

**Use Case**: Assess portfolio risk accounting for uncertain expected returns

## UI Components

### Sidebar Parameters
- **Risk Aversion (γ)**: 0.5 - 10.0 (default: 2.0)
- **Drift Uncertainty (δ)**: 0.0 - 0.15 (default: 0.02)
- **Time Horizon**: 1 - 30 days (default: 10)
- **Transaction Cost**: 0 - 100 bps (default: 10)
- **Time Steps**: 20 - 200 (default: 100)

### Data Requirements
- Requires market data loaded in session state (`st.session_state.historical_data`)
- Compatible with both dict and DataFrame formats
- Extracts close prices and computes log returns
- Calculates mean returns and covariance matrix

### Error Handling
- Graceful degradation when module not available
- Shows helpful error messages with build instructions
- Links to documentation and Jupyter notebook
- Detailed error traces in expandable sections

## Module Availability Detection

The page automatically detects if the Rust bindings are available:

```python
try:
    import hft_py.portfolio_drift as pdrift
    DRIFT_AVAILABLE = True
except ImportError:
    DRIFT_AVAILABLE = False
```

### When Module NOT Available
- Shows error banner with build instructions
- Displays "About" section explaining drift uncertainty
- Links to documentation and Jupyter notebook
- Stops execution gracefully

### When Module IS Available
- Shows success banner
- Renders all 4 tabs with full functionality
- Enables all optimization features

## Visualization Features

### Interactive Charts (Plotly)
1. **Portfolio Weights**: Bar chart with percentages
2. **Sensitivity Analysis**: Line chart showing expected vs worst-case returns
3. **Liquidation Schedule**: Dual subplot (position + velocity)
4. **Transition Trajectory**: Multi-line chart with weight evolution
5. **Trading Velocities**: Stacked area chart
6. **Risk Metrics**: Grouped bar chart (VaR vs CVaR)

### Data Tables
- Formatted with proper precision (2-4 decimal places)
- Highlighted columns for key metrics
- Summary statistics in metric cards
- Expandable error details

## Integration with Navigation

The page uses the standard navigation system:

```python
from utils.ui_components import render_sidebar_navigation, apply_custom_css

render_sidebar_navigation(current_page="Drift Uncertainty Optimizer")
apply_custom_css()
```

**Note**: You may need to update the navigation configuration to include this page.

## Usage Workflow

### Basic Flow
1. **Load Data**: Use Data Loader page to load market data
2. **Navigate**: Go to Drift Uncertainty Optimizer page
3. **Configure**: Set parameters in sidebar
4. **Select Tab**: Choose optimization type
5. **Input Assets**: Select assets and specify constraints
6. **Optimize**: Click optimization button
7. **Analyze**: Review results and visualizations

### Example: Robust Portfolio
1. Load AAPL, MSFT, GOOGL, AMZN, NVDA
2. Set risk aversion = 2.0, uncertainty = 0.02
3. Select assets for portfolio
4. Click "Optimize Robust Portfolio"
5. Review:
   - Optimal weights
   - Expected return: 8.5%
   - Worst-case return: 7.2%
   - Sensitivity across uncertainty levels

### Example: Liquidation
1. Position: 10,000 shares
2. Time horizon: 10 days
3. Market impact: 0.01
4. Click "Compute Liquidation Strategy"
5. Review trading schedule and costs

## Mathematical Details

### Robust Portfolio Choice
```
max_w min_{μ ∈ [μ̂-δ, μ̂+δ]} E[U(w^T μ - (γ/2)w^T Σ w)]
```

### CARA Utility
```
U(W) = -exp(-γW)
```

### Liquidation Cost
```
Cost = Σ_t (drift × position_t + impact × rate_t²)
```

### Transaction Cost
```
Cost = tc × Σ_t |Δw_t|
```

## Performance Notes

- **Sensitivity Analysis**: Runs 5-6 optimizations (progress bar shown)
- **Risk Analysis**: Computes VaR/CVaR for 5 confidence levels
- **Time Steps**: Higher values = more accurate but slower
- Typical optimization time: 0.5-2 seconds per call

## Error Messages

### Common Errors
1. **"Failed to compute statistics"**: Data quality issue, check for NaN/inf values
2. **"Weights must sum to 1.0"**: Input validation error in transition tab
3. **"Need N weights, got M"**: Mismatch in risk analysis input

### Debug Mode
- All exceptions caught and displayed
- Traceback available in expandable section
- Helps identify Rust-side errors

## Future Enhancements

Potential additions:
- [ ] Save/load portfolio configurations
- [ ] Export results to CSV/Excel
- [ ] Compare multiple scenarios side-by-side
- [ ] Historical backtest of strategies
- [ ] Monte Carlo simulation visualization
- [ ] Portfolio construction wizard

## Testing Checklist

Before deploying:
- [ ] Module imports correctly
- [ ] All tabs render without errors
- [ ] Data loading works from session state
- [ ] Optimizations run successfully
- [ ] Charts display properly
- [ ] Error handling works (try invalid inputs)
- [ ] Navigation links work
- [ ] Documentation links accessible
- [ ] Mobile responsive (check on small screens)

## Dependencies

### Python
- streamlit
- pandas
- numpy
- plotly
- hft_py.portfolio_drift (Rust bindings)

### Rust
- rust_core::portfolio_drift_uncertainty
- PyO3 bindings in rust_python_bindings

## References

- **Paper**: Bismuth, A., Guéant, O., & Pu, J. (2017). Portfolio choice, portfolio liquidation, and portfolio transition under drift uncertainty. *SIAM Journal on Financial Mathematics*.
- **Implementation**: `docs/DRIFT_UNCERTAINTY_IMPLEMENTATION.md`
- **Jupyter Notebook**: `examples/notebooks/portfolio_drift_uncertainty.ipynb`
- **Rust Source**: `rust_core/src/portfolio_drift_uncertainty.rs`
- **Python Bindings**: `rust_python_bindings/src/portfolio_drift_bindings.rs`
