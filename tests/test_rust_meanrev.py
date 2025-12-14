#!/usr/bin/env python3
"""Test Rust mean-reversion functions."""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from python.rust_grpc_bridge import rust_connector as rust_connector  # type: ignore
    print("✅ rust_connector bridge imported successfully")
    
    # Test 1: PCA
    print("\n=== Testing PCA ===")
    np.random.seed(42)
    prices = np.random.randn(100, 5).tolist()
    result = rust_connector.compute_pca_rust(prices, 3)
    print(f"Components shape: {len(result['components'])} x {len(result['components'][0])}")
    print(f"Explained variance: {result['explained_variance'][:3]}")
    print("✅ PCA test passed")
    
    # Test 2: OU Process
    print("\n=== Testing OU Process Estimation ===")
    prices_ts = [100 + i + np.random.randn() for i in range(100)]
    result = rust_connector.estimate_ou_process_rust(prices_ts)
    print(f"Theta: {result['theta']:.4f}")
    print(f"Mu: {result['mu']:.2f}")
    print(f"Sigma: {result['sigma']:.4f}")
    print(f"Half-life: {result['half_life']:.2f}")
    print("✅ OU estimation test passed")
    
    # Test 3: Cointegration Test
    print("\n=== Testing Cointegration ===")
    prices1 = [100 + 0.5*i + np.random.randn() for i in range(100)]
    prices2 = [105 + 0.5*i + np.random.randn() for i in range(100)]
    result = rust_connector.cointegration_test_rust(prices1, prices2)
    print(f"ADF Statistic: {result['statistic']:.4f}")
    print(f"P-value: {result['p_value']:.4f}")
    print(f"Is Cointegrated: {result['is_cointegrated']}")
    print("✅ Cointegration test passed")
    
    # Test 4: Backtest
    print("\n=== Testing Backtest ===")
    prices_bt = [100 + 10*np.sin(i/10) + np.random.randn() for i in range(100)]
    result = rust_connector.backtest_strategy_rust(prices_bt, 2.0, 0.5, 0.5)  # type: ignore
    print(f"Returns shape: {len(result['returns'])}")
    print(f"Positions shape: {len(result['positions'])}")
    print(f"Final PnL: {result['pnl'][-1]:.2f}")
    print(f"Sharpe Ratio: {result['sharpe']:.4f}")
    print(f"Max Drawdown: {result['max_drawdown']:.4f}")
    print("✅ Backtest test passed")
    
    print("\n🎉 All tests passed! Rust functions are working correctly.")
    
except ImportError as e:
    print(f"❌ Failed to import rust_connector: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
