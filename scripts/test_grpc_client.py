#!/usr/bin/env python3
"""
Quick test script for gRPC pair discovery client.
Run this after starting the gRPC server to verify connectivity.

Usage:
    python scripts/test_grpc_client.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from python.pair_discovery_client import PairDiscoveryClient

def test_connection():
    """Test basic gRPC connection."""
    print("Testing gRPC connection...")
    try:
        client = PairDiscoveryClient()
        print(f"✅ Connected to {client.host}:{client.port}")
        return client
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        sys.exit(1)

def test_ou_estimation(client):
    """Test OU parameter estimation."""
    print("\nTesting OU parameter estimation...")
    series = np.random.randn(100)
    
    try:
        result = client.estimate_ou_params(series)
        print(f"✅ OU Parameters:")
        print(f"   Theta: {result['theta']:.6f}")
        print(f"   Mu:    {result['mu']:.6f}")
        print(f"   Sigma: {result['sigma']:.6f}")
        print(f"   Half-Life: {result['half_life']:.2f}")
    except Exception as e:
        print(f"❌ OU estimation failed: {e}")

def test_cointegration(client):
    """Test cointegration testing."""
    print("\nTesting cointegration...")
    
    # Generate cointegrated pair
    x = np.cumsum(np.random.randn(200)) + 100
    y = 1.5 * x + np.random.randn(200) * 0.5
    
    try:
        result = client.test_cointegration(x, y)
        print(f"✅ Cointegration Test:")
        print(f"   Is Cointegrated: {result['is_cointegrated']}")
        print(f"   P-Value:         {result['p_value']:.6f}")
        print(f"   Hedge Ratio:     {result['hedge_ratio']:.4f}")
    except Exception as e:
        print(f"❌ Cointegration test failed: {e}")

def test_hurst(client):
    """Test Hurst exponent calculation."""
    print("\nTesting Hurst exponent...")
    series = np.random.randn(300)
    
    try:
        result = client.calculate_hurst(series)
        print(f"✅ Hurst Exponent:")
        print(f"   H:               {result['hurst_exponent']:.4f}")
        print(f"   Mean Reverting:  {result['is_mean_reverting']}")
        print(f"   95% CI:          ({result['confidence_interval'][0]:.4f}, {result['confidence_interval'][1]:.4f})")
    except Exception as e:
        print(f"❌ Hurst calculation failed: {e}")

def test_pair_full(client):
    """Test full pair analysis."""
    print("\nTesting full pair analysis...")
    
    # Generate cointegrated pair
    x = np.cumsum(np.random.randn(500)) + 100
    y = 1.2 * x + np.random.randn(500) * 2
    
    try:
        result = client.test_pair(x, y, pair_name="TEST-PAIR")
        print(f"✅ Pair Test Results:")
        print(f"   Pair Name:        {result['pair_name']}")
        print(f"   Is Cointegrated:  {result['is_cointegrated']}")
        print(f"   P-Value:          {result['p_value']:.6f}")
        print(f"   Hedge Ratio:      {result['hedge_ratio']:.4f}")
        print(f"   Hurst Exponent:   {result['hurst_exponent']:.4f}")
        print(f"   Half-Life:        {result['half_life']:.2f}")
        print(f"   Entry Threshold:  {result['entry_threshold']:.4f}")
        print(f"   Exit Threshold:   {result['exit_threshold']:.4f}")
    except Exception as e:
        print(f"❌ Pair test failed: {e}")

def test_hjb_solver(client):
    """Test HJB PDE solver."""
    print("\nTesting HJB solver...")
    
    try:
        result = client.solve_hjb(
            theta=0.5,
            mu=0.0,
            sigma=1.0,
            x_min=-2.0,
            x_max=2.0,
            n_points=51,
            dt=0.01,
            max_iterations=1000
        )
        print(f"✅ HJB Solution:")
        print(f"   Converged:        {result['converged']}")
        print(f"   Iterations:       {result['iterations']}")
        print(f"   Entry Threshold:  {result['policy']['entry_threshold']:.4f}")
        print(f"   Exit Threshold:   {result['policy']['exit_threshold']:.4f}")
        print(f"   Value Function:   {len(result['value_function'])} points")
    except Exception as e:
        print(f"❌ HJB solver failed: {e}")

def test_backtest(client):
    """Test strategy backtesting."""
    print("\nTesting strategy backtest...")
    
    spread = np.random.randn(1000) * 2
    
    try:
        result = client.backtest_strategy(
            spread=spread,
            entry_threshold=2.0,
            exit_threshold=0.5,
            transaction_cost=0.001,
            initial_capital=100000.0
        )
        print(f"✅ Backtest Results:")
        print(f"   Total Return:     {result['total_return']:.2f}%")
        print(f"   Sharpe Ratio:     {result['sharpe_ratio']:.4f}")
        print(f"   Max Drawdown:     {result['max_drawdown']:.2f}%")
        print(f"   Number of Trades: {result['num_trades']}")
        print(f"   Win Rate:         {result['win_rate']:.2f}%")
    except Exception as e:
        print(f"❌ Backtest failed: {e}")

def main():
    print("=" * 60)
    print("gRPC Pair Discovery Client Test Suite")
    print("=" * 60)
    
    # Test connection
    client = test_connection()
    
    # Run all tests
    test_ou_estimation(client)
    test_cointegration(client)
    test_hurst(client)
    test_pair_full(client)
    test_hjb_solver(client)
    test_backtest(client)
    
    # Close connection
    client.close()
    
    print("\n" + "=" * 60)
    print("✅ All tests completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
