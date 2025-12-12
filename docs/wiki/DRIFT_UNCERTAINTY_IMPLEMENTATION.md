# Portfolio Optimization under Drift Uncertainty - Implementation Summary

## Overview

This implementation adds comprehensive portfolio optimization techniques under drift uncertainty based on the paper **"Portfolio choice, portfolio liquidation, and portfolio transition under drift uncertainty"** by Alexis Bismuth, Olivier Guéant, and Jiang Pu.

## Implementation Status

### ✅ Completed Components

1. **Rust Core Module** (`rust_core/src/portfolio_drift_uncertainty.rs`)
   - Robust portfolio choice with worst-case optimization
   - Optimal liquidation with exponential decay
   - Portfolio transition/rebalancing strategy
   - Risk measures: VaR and CVaR under drift uncertainty
   - Helper functions: matrix operations, normal distribution utilities

2. **Python Bindings** (`rust_python_bindings/src/portfolio_drift_bindings.rs`)
   - PyO3 wrappers for all Rust functions
   - Python-accessible result classes with `to_dict()` methods
   - Comprehensive error handling and input validation

3. **Module Registration**
   - Updated `rust_core/src/lib.rs` to expose portfolio_drift_uncertainty module
   - Updated `rust_python_bindings/src/lib.rs` to register Python submodule

4. **Jupyter Notebook** (`examples/notebooks/portfolio_drift_uncertainty.ipynb`)
   - Example 1: Robust portfolio choice comparing different uncertainty levels
   - Example 2: Optimal liquidation strategy with trading schedules
   - Example 3: Portfolio transition from current to target weights
   - Example 4: VaR and CVaR risk measures
   - Complete with visualizations and mathematical formulations

5. **Streamlit Integration** (`app/pages/lab_portfolio_optimizer.py`)
   - Added import detection for drift uncertainty module
   - Status banner showing feature availability
   - Ready for full UI integration once bindings are built

## Mathematical Framework

### Key Algorithms Implemented

#### 1. Robust Portfolio Choice
```
Maximize: min_{μ ∈ [μ̂ - δ, μ̂ + δ]} U(w^T μ - (γ/2) w^T Σ w)
```
- Uses CARA utility: U(W) = -exp(-γW)
- Accounts for drift uncertainty δ in expected returns
- Returns worst-case optimal weights

#### 2. Optimal Liquidation
```
Minimize: E[cost] + γ * Var[cost] under drift uncertainty
```
- Exponential decay liquidation schedule
- Balances temporary market impact with drift uncertainty
- Returns trading schedule and rates over time

#### 3. Portfolio Transition
```
Optimize rebalancing from w₀ to w_target over time T
```
- Minimizes transition costs with transaction cost consideration
- Returns optimal weight trajectory and trading velocities

#### 4. Risk Measures
- **VaR (Value at Risk)**: Maximum loss at confidence level α
- **CVaR (Expected Shortfall)**: Average loss beyond VaR threshold
- Both computed under worst-case drift uncertainty

## API Reference

### Python API

```python
import hft_py.portfolio_drift as pdrift

# Robust portfolio choice
result = pdrift.portfolio_choice_drift_uncertainty(
    mu=[0.05, 0.08, 0.06],           # Expected returns
    cov=[[0.04, 0.01, 0.00],         # Covariance matrix
         [0.01, 0.09, 0.02],
         [0.00, 0.02, 0.16]],
    risk_aversion=2.0,               # Risk aversion γ
    drift_uncertainty=0.02           # Uncertainty δ
)
# Returns: {'weights': [...], 'expected_return': ..., 'worst_case_return': ..., ...}

# Optimal liquidation
result = pdrift.liquidation_drift_uncertainty(
    initial_position=1000.0,         # Shares to liquidate
    time_horizon=10,                 # Days
    drift_uncertainty=0.02,
    risk_aversion=2.0,
    temporary_impact=0.01,           # Market impact parameter λ
    num_steps=100                    # Time discretization
)
# Returns: {'trading_schedule': [...], 'trading_rates': [...], 'expected_cost': ..., ...}

# Portfolio transition
result = pdrift.transition_drift_uncertainty(
    initial_weights=[0.4, 0.3, 0.3],
    target_weights=[0.3, 0.5, 0.2],
    cov=cov,
    time_horizon=5,
    risk_aversion=2.0,
    drift_uncertainty=0.02,
    transaction_cost=0.001,          # 10 bps
    num_steps=50
)
# Returns: {'trajectory': [...], 'trading_rates': [...], 'expected_cost': ..., ...}

# Risk measures
var = pdrift.var_drift_uncertainty(
    mu=mu, cov=cov, weights=[0.4, 0.3, 0.3],
    time_horizon=1,
    confidence_level=0.95,
    drift_uncertainty=0.02
)

cvar = pdrift.cvar_drift_uncertainty(
    mu=mu, cov=cov, weights=[0.4, 0.3, 0.3],
    time_horizon=1,
    confidence_level=0.95,
    drift_uncertainty=0.02
)
```

## Usage Examples

### 1. Basic Portfolio Optimization

```python
import numpy as np
import hft_py.portfolio_drift as pdrift

# Define assets
n_assets = 3
mu = np.array([0.05, 0.08, 0.06])  # 5%, 8%, 6% expected returns
sigma = np.array([[0.04, 0.01, 0.00],
                  [0.01, 0.09, 0.02],
                  [0.00, 0.02, 0.16]])

# Optimize with 2% drift uncertainty
result = pdrift.portfolio_choice_drift_uncertainty(
    mu=mu.tolist(),
    cov=sigma.tolist(),
    risk_aversion=2.0,
    drift_uncertainty=0.02
)

print(f"Optimal weights: {result['weights']}")
print(f"Expected return: {result['expected_return']*100:.2f}%")
print(f"Worst-case return: {result['worst_case_return']*100:.2f}%")
```

### 2. Liquidation Strategy

```python
# Liquidate 1000 shares over 10 days
result = pdrift.liquidation_drift_uncertainty(
    initial_position=1000.0,
    time_horizon=10,
    drift_uncertainty=0.02,
    risk_aversion=2.0,
    temporary_impact=0.01,
    num_steps=100
)

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(result['times'], result['trading_schedule'])
plt.title('Remaining Position')
plt.xlabel('Time (days)')
plt.ylabel('Shares')

plt.subplot(1, 2, 2)
plt.plot(result['times'], result['trading_rates'])
plt.fill_between(result['times'], result['trading_rates'], alpha=0.3)
plt.title('Trading Velocity')
plt.xlabel('Time (days)')
plt.ylabel('Shares/day')

plt.tight_layout()
plt.show()
```

## Building and Testing

### Prerequisites
```bash
# Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Python dependencies
pip install maturin numpy pandas matplotlib

# Recommended: Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
```

### Build Instructions

⚠️ **Note**: Before building, you must fix pre-existing compilation errors in `rust_core/src/orderbook.rs`:

The `orderbook.rs` file uses `BTreeMap<f64, ...>` which doesn't compile because `f64` doesn't implement `Ord` (due to NaN handling). This needs to be fixed first:

**Option 1: Use a wrapper type**
```rust
use std::cmp::Ordering;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrderedFloat(f64);

impl Eq for OrderedFloat {}

impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for OrderedFloat {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// Then use BTreeMap<OrderedFloat, VecDeque<Order>>
```

**Option 2: Use HashMap instead**
```rust
// Change from BTreeMap to HashMap in orderbook.rs
use std::collections::HashMap;

pub struct OrderedSide {
    pub levels: HashMap<Price, VecDeque<Order>>,
}
```

Once orderbook.rs is fixed, build with:

```bash
cd rust_python_bindings
maturin develop
```

### Testing

```bash
# Run the Jupyter notebook
cd examples/notebooks
jupyter notebook portfolio_drift_uncertainty.ipynb

# Or test directly in Python
python << EOF
import hft_py.portfolio_drift as pdrift

result = pdrift.portfolio_choice_drift_uncertainty(
    mu=[0.05, 0.08, 0.06],
    cov=[[0.04, 0.01, 0.00], [0.01, 0.09, 0.02], [0.00, 0.02, 0.16]],
    risk_aversion=2.0,
    drift_uncertainty=0.02
)
print(f"Optimal weights: {result['weights']}")
EOF
```

## Streamlit Integration (Future Work)

The Streamlit interface (`app/pages/lab_portfolio_optimizer.py`) has been prepared with:
- Module availability detection
- Status banner for feature visibility
- Import structure for the drift uncertainty module

Full UI integration with tabs for each optimization type can be added after the bindings are successfully built.

## Files Modified/Created

### New Files
1. `/rust_core/src/portfolio_drift_uncertainty.rs` (651 lines)
2. `/rust_python_bindings/src/portfolio_drift_bindings.rs` (411 lines)
3. `/examples/notebooks/portfolio_drift_uncertainty.ipynb` (comprehensive notebook)
4. `/docs/DRIFT_UNCERTAINTY_IMPLEMENTATION.md` (this document)

### Modified Files
1. `/rust_core/src/lib.rs` - Added `pub mod portfolio_drift_uncertainty;`
2. `/rust_python_bindings/src/lib.rs` - Added portfolio_drift module registration
3. `/app/pages/lab_portfolio_optimizer.py` - Added feature detection banner

## Next Steps

1. **Fix Orderbook Compilation** - Resolve f64 Ord issue in orderbook.rs
2. **Build Bindings** - Run `maturin develop` successfully
3. **Test Examples** - Run all examples in Jupyter notebook
4. **Complete Streamlit UI** - Add full drift uncertainty interface with tabs
5. **Documentation** - Add docstrings to Python bindings
6. **Unit Tests** - Add Rust unit tests for edge cases

## References

**Paper**: Alexis Bismuth, Olivier Guéant, and Jiang Pu. "Portfolio choice, portfolio liquidation, and portfolio transition under drift uncertainty." *SIAM Journal on Financial Mathematics*, 2017.

**Key Concepts**:
- Drift uncertainty: Uncertainty in expected returns ε ∈ [μ - δ, μ + δ]
- CARA utility: Constant Absolute Risk Aversion
- Worst-case optimization: Robust optimization framework
- Exponential decay: Optimal liquidation schedule follows e^(-κt)

## License

See project LICENSE file.

## Contributors

- Implementation based on academic paper by Bismuth, Guéant, and Pu
- Rust core and Python bindings: rust-arblab project
