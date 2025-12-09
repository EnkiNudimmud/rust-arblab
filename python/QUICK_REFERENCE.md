"""
Quick Reference: Refactored Python Structure
=============================================

## Import Cheat Sheet

### ✅ Strategies
```python
from python.strategies.adaptive_strategies import AdaptiveMeanReversion
from python.strategies.meanrev import calculate_half_life
from python.strategies.sparse_meanrev import box_tao_decomposition
```

### ✅ Optimization
```python
from python.optimization.advanced_optimization import HMMRegimeDetector, MCMCOptimizer
from python.optimization.signature_methods import SignatureMethods
```

### ✅ Data
```python
from python.data.data_fetcher import fetch_intraday_data
from python.data.fetchers.alpha_vantage_helper import fetch_intraday
from python.data.fetchers.ccxt_helper import create_exchange
```

### ✅ Models
```python
from python.models.rough_heston import simulate_rough_heston
from python.models.regime_detector import RegimeDetector
```

### ✅ Utils
```python
from python.utils.data_persistence import save_dataset, load_dataset
from python.utils.retry_utils import retry_with_backoff
from python.utils.signal_monitor import SignalMonitor
```

### ✅ Type Fixes (NEW!)
```python
from python.type_fixes import safe_mean, safe_std, as_numpy, ArrayLike
```

### ✅ Factories (NEW!)
```python
from python.factories import StrategyFactory, DataFetcherFactory
```

## Common Tasks

### Fix numpy/pandas type errors
```python
from python.type_fixes import safe_mean, safe_std

# Instead of:
mean = np.mean(df['close'].values)  # ❌ ExtensionArray error

# Use:
mean = safe_mean(df['close'])  # ✅ Works with any array-like
```

### Create strategies dynamically
```python
from python.factories import StrategyFactory

strategy = StrategyFactory.create('adaptive_meanrev', 
                                  n_regimes=3,
                                  lookback_period=100)
```

### Inherit from base classes
```python
from python.core.base import BaseStrategy

class MyStrategy(BaseStrategy):
    def generate_signal(self, data):
        # Your implementation
        pass
```

## Directory Structure
```
python/
├── core/           - Base classes, type utils, errors
├── data/           - Data fetchers and persistence
│   └── fetchers/   - API-specific implementations
├── strategies/     - Trading strategy implementations
├── models/         - Statistical/ML models
├── optimization/   - Optimization algorithms
├── utils/          - Helper utilities
├── type_fixes.py   - Quick type conversions
└── factories.py    - Factory pattern implementations
```

## Error Status
- ✅ **App files**: 0 errors
- ✅ **Strategy files**: 0 errors  
- ✅ **Optimization files**: 0 errors
- ✅ **Data files**: Minor pandas type hint warnings (non-blocking)
- ⚠️ **Pylance cache**: Shows 1 error for non-existent file (ignore)

## Files Moved (Quick Lookup)
| Old Location | New Location |
|--------------|-------------|
| `python/adaptive_strategies.py` | `python/strategies/adaptive_strategies.py` |
| `python/meanrev.py` | `python/strategies/meanrev.py` |
| `python/sparse_meanrev.py` | `python/strategies/sparse_meanrev.py` |
| `python/advanced_optimization.py` | `python/optimization/advanced_optimization.py` |
| `python/signature_methods.py` | `python/optimization/signature_methods.py` |
| `python/rough_heston.py` | `python/models/rough_heston.py` |
| `python/regime_detector.py` | `python/models/regime_detector.py` |
| `python/data_persistence.py` | `python/utils/data_persistence.py` |
| `python/retry_utils.py` | `python/utils/retry_utils.py` |
| `python/signal_monitor.py` | `python/utils/signal_monitor.py` |
| `python/alpha_vantage_helper.py` | `python/data/fetchers/alpha_vantage_helper.py` |
| `python/finnhub_helper.py` | `python/data/fetchers/finnhub_helper.py` |
| `python/coingecko_helper.py` | `python/data/fetchers/coingecko_helper.py` |
| `python/yfinance_helper.py` | `python/data/fetchers/yfinance_helper.py` |
| `python/ccxt_helper.py` | `python/data/fetchers/ccxt_helper.py` |
| `python/massive_helper.py` | `python/data/fetchers/massive_helper.py` |
| `python/data_fetcher.py` | `python/data/data_fetcher.py` |
"""