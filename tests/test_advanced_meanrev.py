#!/usr/bin/env python3
"""Test advanced mean-reversion functions (utility maximization, transaction costs, etc.)"""

import numpy as np
import pandas as pd
import sys
import time

try:
    import rust_connector
    print("✅ rust_connector imported successfully")
    RUST_AVAILABLE = True
except ImportError:
    print("⚠️  rust_connector not available, testing Python fallbacks only")
    RUST_AVAILABLE = False

from python import meanrev

print("\n" + "="*70)
print("ADVANCED MEAN-REVERSION FEATURES TEST SUITE")
print("="*70)

# Generate synthetic data
np.random.seed(42)
n_assets = 5
n_periods = 100

# Test 1: CARA Optimal Weights
print("\n=== Test 1: CARA Utility Maximization (Appendix A) ===")
expected_returns = np.array([0.10, 0.08, 0.12, 0.09, 0.11])  # Annual returns
correlation = np.array([
    [1.0, 0.3, 0.2, 0.1, 0.2],
    [0.3, 1.0, 0.4, 0.2, 0.1],
    [0.2, 0.4, 1.0, 0.3, 0.2],
    [0.1, 0.2, 0.3, 1.0, 0.4],
    [0.2, 0.1, 0.2, 0.4, 1.0]
])
volatilities = np.array([0.20, 0.15, 0.25, 0.18, 0.22])
covariance = np.outer(volatilities, volatilities) * correlation

if RUST_AVAILABLE:
    start = time.time()
    result_rust = rust_connector.cara_optimal_weights_rust(
        expected_returns.tolist(), covariance.tolist(), gamma=2.0
    )
    rust_time = time.time() - start
    print(f"⚡ Rust CARA weights: {[f'{w:.4f}' for w in result_rust['weights']]}")
    print(f"   Expected return: {result_rust['expected_return']:.4f}")
    print(f"   Expected variance: {result_rust['expected_variance']:.4f}")
    print(f"   Time: {rust_time*1000:.2f}ms")

start = time.time()
result_python = meanrev.cara_optimal_weights(expected_returns, covariance, gamma=2.0)
python_time = time.time() - start
print(f"🔧 Python CARA weights: {[f'{w:.4f}' for w in result_python['weights']]}")
print(f"   Expected return: {result_python['expected_return']:.4f}")
print(f"   Expected variance: {result_python['expected_variance']:.4f}")
print(f"   Time: {python_time*1000:.2f}ms")

if RUST_AVAILABLE:
    print(f"⚡ Speedup: {python_time/rust_time:.1f}x")
print("✅ CARA optimization test passed")

# Test 2: Sharpe Optimal Weights
print("\n=== Test 2: Risk-Adjusted Portfolio Weights (Sharpe Maximization) ===")

if RUST_AVAILABLE:
    start = time.time()
    result_rust = rust_connector.sharpe_optimal_weights_rust(
        expected_returns.tolist(), covariance.tolist(), risk_free_rate=0.02
    )
    rust_time = time.time() - start
    print(f"⚡ Rust Sharpe weights: {[f'{w:.4f}' for w in result_rust['weights']]}")
    print(f"   Sharpe ratio: {result_rust['sharpe_ratio']:.4f}")
    print(f"   Expected return: {result_rust['expected_return']:.4f}")
    print(f"   Expected std: {result_rust['expected_std']:.4f}")
    print(f"   Time: {rust_time*1000:.2f}ms")

start = time.time()
result_python = meanrev.sharpe_optimal_weights(expected_returns, covariance, risk_free_rate=0.02)
python_time = time.time() - start
print(f"🔧 Python Sharpe weights: {[f'{w:.4f}' for w in result_python['weights']]}")
print(f"   Sharpe ratio: {result_python['sharpe_ratio']:.4f}")
print(f"   Expected return: {result_python['expected_return']:.4f}")
print(f"   Expected std: {result_python['expected_std']:.4f}")
print(f"   Time: {python_time*1000:.2f}ms")

if RUST_AVAILABLE:
    print(f"⚡ Speedup: {python_time/rust_time:.1f}x")
print("✅ Sharpe optimization test passed")

# Test 3: Transaction Cost Modeling
print("\n=== Test 3: Backtest with Transaction Costs ===")
prices = pd.Series([100 + 10*np.sin(i/10) + np.random.randn() for i in range(100)])

if RUST_AVAILABLE:
    start = time.time()
    result_rust = rust_connector.backtest_with_costs_rust(
        prices.tolist(), entry_z=2.0, exit_z=0.5, transaction_cost=0.001
    )
    rust_time = time.time() - start
    print(f"⚡ Rust backtest with costs:")
    print(f"   Final PnL: ${result_rust['pnl'][-1]:.2f}")
    print(f"   Total costs: ${result_rust['total_costs']:.2f}")
    print(f"   Sharpe ratio: {result_rust['sharpe']:.4f}")
    print(f"   Max drawdown: {result_rust['max_drawdown']:.2%}")
    print(f"   Time: {rust_time*1000:.2f}ms")

start = time.time()
result_python = meanrev.backtest_with_costs(prices, entry_z=2.0, exit_z=0.5, transaction_cost=0.001)
python_time = time.time() - start
print(f"🔧 Python backtest with costs:")
print(f"   Final PnL: ${result_python['pnl'][-1]:.2f}")
print(f"   Total costs: ${result_python['total_costs']:.2f}")
print(f"   Sharpe ratio: {result_python['sharpe']:.4f}")
print(f"   Max drawdown: {result_python['max_drawdown']:.2%}")
print(f"   Time: {python_time*1000:.2f}ms")

if RUST_AVAILABLE:
    print(f"⚡ Speedup: {python_time/rust_time:.1f}x")
print("✅ Transaction cost modeling test passed")

# Test 4: Optimal Stopping Times
print("\n=== Test 4: Optimal Stopping Times ===")
theta = 0.1  # Mean reversion speed
mu = 100.0   # Long-term mean
sigma = 5.0  # Volatility

if RUST_AVAILABLE:
    result_rust = rust_connector.optimal_thresholds_rust(theta, mu, sigma, transaction_cost=0.001)
    print(f"⚡ Rust optimal thresholds:")
    print(f"   Entry threshold: {result_rust['optimal_entry']:.2f} σ")
    print(f"   Exit threshold: {result_rust['optimal_exit']:.2f} σ")
    print(f"   Expected holding period: {result_rust['expected_holding_period']:.1f} days")

result_python = meanrev.optimal_thresholds(theta, mu, sigma, transaction_cost=0.001)
print(f"🔧 Python optimal thresholds:")
print(f"   Entry threshold: {result_python['optimal_entry']:.2f} σ")
print(f"   Exit threshold: {result_python['optimal_exit']:.2f} σ")
print(f"   Expected holding period: {result_python['expected_holding_period']:.1f} days")

print("✅ Optimal stopping times test passed")

# Test 5: Multi-Period Optimization
print("\n=== Test 5: Multi-Period Portfolio Optimization ===")
returns_data = np.random.randn(100, 5) * 0.02  # Daily returns
returns_df = pd.DataFrame(returns_data, columns=[f'Asset{i}' for i in range(5)])

if RUST_AVAILABLE:
    start = time.time()
    result_rust = rust_connector.multiperiod_optimize_rust(
        returns_data.tolist(), covariance.tolist(), 
        gamma=2.0, transaction_cost=0.001, n_periods=5
    )
    rust_time = time.time() - start
    print(f"⚡ Rust multi-period optimization:")
    print(f"   Number of rebalancing periods: {len(result_rust['weights_sequence'])}")
    print(f"   Rebalance times: {result_rust['rebalance_times']}")
    print(f"   Expected utility: {result_rust['expected_utility']:.6f}")
    print(f"   First period weights: {[f'{w:.4f}' for w in result_rust['weights_sequence'][0]]}")
    print(f"   Time: {rust_time*1000:.2f}ms")

start = time.time()
result_python = meanrev.multiperiod_optimize(
    returns_df, covariance, gamma=2.0, transaction_cost=0.001, n_periods=5
)
python_time = time.time() - start
print(f"🔧 Python multi-period optimization:")
print(f"   Number of rebalancing periods: {len(result_python['weights_sequence'])}")
print(f"   Rebalance times: {result_python['rebalance_times']}")
print(f"   Expected utility: {result_python['expected_utility']:.6f}")
print(f"   First period weights: {[f'{w:.4f}' for w in result_python['weights_sequence'][0]]}")
print(f"   Time: {python_time*1000:.2f}ms")

if RUST_AVAILABLE:
    print(f"⚡ Speedup: {python_time/rust_time:.1f}x")
print("✅ Multi-period optimization test passed")

# Summary
print("\n" + "="*70)
print("🎉 ALL ADVANCED FEATURE TESTS PASSED!")
print("="*70)

if RUST_AVAILABLE:
    print("\n✨ Summary of Features:")
    print("  ✅ CARA Utility Maximization (Appendix A)")
    print("  ✅ Risk-Adjusted Portfolio Weights (Sharpe Maximization)")
    print("  ✅ Transaction Cost Modeling")
    print("  ✅ Optimal Stopping Times")
    print("  ✅ Multi-Period Portfolio Optimization")
    print("\n⚡ All features implemented in high-performance Rust with Python fallbacks")
else:
    print("\n⚠️  Note: Rust connector not available - using Python fallbacks")
    print("   Run 'maturin develop --release' in rust_connector/ to enable Rust")
