"""
Advanced Algorithm Test Suite for gRPC Server

Tests all real algorithm implementations including:
- Mean reversion with OU process
- Portfolio optimization (Sharpe, min-variance, risk parity)
- HMM regime detection
- Sparse portfolios
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import time
from python.grpc_client import TradingGrpcClient

def test_mean_reversion_advanced():
    """Test mean reversion with realistic price data"""
    print("\n📊 Testing Advanced Mean Reversion...")
    
    with TradingGrpcClient() as client:
        # Generate mean-reverting price series
        np.random.seed(42)
        n = 100
        prices = [100.0]
        for i in range(n-1):
            # OU process simulation
            mean_reversion_speed = 0.1
            long_term_mean = 100.0
            volatility = 0.5
            
            drift = mean_reversion_speed * (long_term_mean - prices[-1])
            shock = volatility * np.random.randn()
            prices.append(prices[-1] + drift + shock)
        
        start = time.time()
        result = client.calculate_mean_reversion(
            prices=prices,
            lookback=20,
            threshold=1.5
        )
        latency = (time.time() - start) * 1000
        
        print(f"  ✓ Signal: {result['signal']:.2f}")
        print(f"  ✓ Z-score: {result['zscore']:.2f}")
        print(f"  ✓ Mean: {result['metrics']['mean']:.2f}")
        print(f"  ✓ Std: {result['metrics']['std']:.2f}")
        print(f"  ✓ Entry signal: {result['entry_signal']}")
        print(f"  ⚡ Latency: {latency:.2f}ms")
        
        assert abs(result['zscore']) < 5.0, "Z-score should be reasonable"
        print("  ✅ Mean reversion test passed")

def test_portfolio_optimization_methods():
    """Test different portfolio optimization methods"""
    print("\n💼 Testing Portfolio Optimization Methods...")
    
    with TradingGrpcClient() as client:
        # Generate synthetic price data for 5 assets
        np.random.seed(42)
        n_assets = 5
        n_periods = 100
        
        prices = []
        for i in range(n_assets):
            asset_prices = [100.0]
            for _ in range(n_periods - 1):
                ret = np.random.normal(0.0005, 0.02)
                asset_prices.append(asset_prices[-1] * (1 + ret))
            prices.append({
                'symbol': f'ASSET_{i}',
                'prices': asset_prices
            })
        
        methods = ['max_sharpe', 'min_variance', 'risk_parity', 'equal_weight']
        
        for method in methods:
            start = time.time()
            result = client.optimize_portfolio(
                prices=prices,
                method=method,
                parameters={'risk_free_rate': 0.02}
            )
            latency = (time.time() - start) * 1000
            
            weights = np.array(result['weights'])
            print(f"\n  {method}:")
            print(f"    Weights: {weights}")
            print(f"    Sum: {weights.sum():.4f}")
            print(f"    Expected Return: {result['expected_return']:.4f}")
            print(f"    Volatility: {result['volatility']:.4f}")
            print(f"    Sharpe Ratio: {result['sharpe_ratio']:.4f}")
            print(f"    ⚡ Latency: {latency:.2f}ms")
            
            assert abs(weights.sum() - 1.0) < 0.01, f"Weights should sum to 1 for {method}"
            assert all(weights >= -0.01), f"No short positions for {method}"  # Allow small numerical errors
        
        print("  ✅ All optimization methods passed")

def test_hmm_regime_detection():
    """Test HMM for regime detection"""
    print("\n🔄 Testing HMM Regime Detection...")
    
    with TradingGrpcClient() as client:
        # Generate returns with regime changes
        np.random.seed(42)
        
        # Low volatility regime (50 samples)
        regime1 = np.random.normal(0.001, 0.01, 50)
        
        # High volatility regime (50 samples)
        regime2 = np.random.normal(-0.002, 0.03, 50)
        
        observations = np.concatenate([regime1, regime2]).tolist()
        
        start = time.time()
        result = client.run_hmm(
            observations=observations,
            n_states=2,
            max_iterations=50,
            tolerance=1e-4
        )
        latency = (time.time() - start) * 1000
        
        print(f"  ✓ Converged: {result['converged']}")
        print(f"  ✓ Log-likelihood: {result['log_likelihood']:.4f}")
        print(f"  ✓ State probabilities: {result['state_probabilities']}")
        print(f"  ✓ Emission means: {result['emission_means']}")
        print(f"  ✓ Emission stds: {result['emission_stds']}")
        print(f"  ⚡ Latency: {latency:.2f}ms")
        
        assert len(result['state_probabilities']) == 2
        assert result['converged'], "HMM should converge"
        print("  ✅ HMM regime detection passed")

def test_sparse_portfolio():
    """Test sparse mean-reverting portfolio discovery"""
    print("\n🎯 Testing Sparse Portfolio Discovery...")
    
    with TradingGrpcClient() as client:
        # Generate correlated price series (some mean-reverting)
        np.random.seed(42)
        n_assets = 10
        n_periods = 100
        
        prices = []
        
        # Create 3 mean-reverting pairs and 4 random walks
        for i in range(n_assets):
            asset_prices = [100.0]
            
            if i < 6:  # Mean-reverting pairs
                pair_idx = i // 2
                for t in range(n_periods - 1):
                    if i % 2 == 0:
                        # Lead asset
                        shock = np.random.normal(0, 1)
                        asset_prices.append(asset_prices[-1] + shock)
                    else:
                        # Follow asset (mean-reverting to lead)
                        lead_price = prices[i-1]['prices'][t+1]
                        mean_rev = 0.1 * (lead_price - asset_prices[-1])
                        shock = np.random.normal(0, 0.5)
                        asset_prices.append(asset_prices[-1] + mean_rev + shock)
            else:
                # Random walk
                for _ in range(n_periods - 1):
                    ret = np.random.normal(0.0001, 0.02)
                    asset_prices.append(asset_prices[-1] * (1 + ret))
            
            prices.append({
                'symbol': f'ASSET_{i}',
                'prices': asset_prices
            })
        
        # Test different sparsity levels
        lambdas = [0.1, 0.5, 1.0]
        
        for lambda_val in lambdas:
            start = time.time()
            result = client.calculate_sparse_portfolio(
                prices=prices,
                method='lasso',
                lambda_param=lambda_val,
                alpha=0.5
            )
            latency = (time.time() - start) * 1000
            
            weights = np.array(result['weights'])
            n_selected = result['n_assets_selected']
            
            print(f"\n  λ={lambda_val}:")
            print(f"    Assets selected: {n_selected}/{n_assets}")
            print(f"    Non-zero weights: {np.sum(np.abs(weights) > 1e-6)}")
            print(f"    Weight range: [{weights.min():.4f}, {weights.max():.4f}]")
            print(f"    Objective value: {result['objective_value']:.4f}")
            print(f"    ⚡ Latency: {latency:.2f}ms")
            
            assert n_selected <= n_assets, "Selected assets should be <= total"
            assert n_selected > 0, "Should select at least some assets"
        
        print("  ✅ Sparse portfolio discovery passed")

def performance_comparison():
    """Compare performance of different algorithms"""
    print("\n⚡ Performance Comparison...")
    
    with TradingGrpcClient() as client:
        np.random.seed(42)
        
        # Test data
        prices_simple = np.random.randn(100).cumsum() + 100
        prices_multi = [{'symbol': f'A{i}', 'prices': (np.random.randn(100).cumsum() + 100).tolist()} 
                       for i in range(5)]
        observations = np.random.randn(100).tolist()
        
        tests = [
            ('Mean Reversion', lambda: client.calculate_mean_reversion(prices_simple.tolist(), threshold=1.5, lookback=20)),
            ('Portfolio (Sharpe)', lambda: client.optimize_portfolio(prices_multi, 'max_sharpe', {})),
            ('Portfolio (Min Var)', lambda: client.optimize_portfolio(prices_multi, 'min_variance', {})),
            ('HMM (2 states)', lambda: client.run_hmm(observations, n_states=2, max_iterations=20, tolerance=1e-4)),
            ('Sparse Portfolio', lambda: client.calculate_sparse_portfolio(prices_multi, 'lasso', lambda_param=0.5, alpha=0.5)),
        ]
        
        print("\n  Algorithm             | Avg Latency | Throughput")
        print("  " + "-" * 55)
        
        for name, func in tests:
            latencies = []
            n_runs = 100
            
            start = time.time()
            for _ in range(n_runs):
                run_start = time.time()
                func()
                latencies.append(time.time() - run_start)
            total_time = time.time() - start
            
            avg_latency = np.mean(latencies) * 1000
            throughput = n_runs / total_time
            
            print(f"  {name:20s} | {avg_latency:8.2f}ms | {throughput:8.0f} ops/s")
        
        print("  ✅ Performance comparison complete")

def main():
    print("=" * 70)
    print("🧪 Advanced Algorithm Test Suite")
    print("=" * 70)
    
    try:
        test_mean_reversion_advanced()
        test_portfolio_optimization_methods()
        test_hmm_regime_detection()
        test_sparse_portfolio()
        performance_comparison()
        
        print("\n" + "=" * 70)
        print("🎉 All advanced tests passed!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
