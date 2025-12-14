"""
Test gRPC connectivity and performance.

Run this after starting the gRPC server to verify everything works.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import time
from python.grpc_client import TradingGrpcClient, GrpcConfig

def test_connectivity():
    """Test basic connectivity."""
    print("🔗 Testing connectivity...")
    try:
        client = TradingGrpcClient()
        client.connect()
        print("✓ Connected to gRPC server")
        client.close()
        return True
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

def test_mean_reversion():
    """Test mean reversion calculation."""
    print("\n📊 Testing mean reversion...")
    client = TradingGrpcClient()
    client.connect()
    
    # Generate test data
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    
    # Time the operation
    start = time.perf_counter()
    result = client.calculate_mean_reversion(prices, threshold=2.0, lookback=20)
    elapsed = (time.perf_counter() - start) * 1000  # ms
    
    print(f"✓ Mean reversion: signal={result['signal']:.2f}, zscore={result['zscore']:.2f}")
    print(f"  Latency: {elapsed:.2f}ms")
    
    client.close()
    return True

def test_portfolio_optimization():
    """Test portfolio optimization."""
    print("\n💼 Testing portfolio optimization...")
    client = TradingGrpcClient()
    client.connect()
    
    # Generate test data
    np.random.seed(42)
    prices = {
        'AAPL': 100 + np.cumsum(np.random.randn(100) * 0.5),
        'GOOGL': 200 + np.cumsum(np.random.randn(100) * 0.7),
        'MSFT': 150 + np.cumsum(np.random.randn(100) * 0.6),
    }
    
    start = time.perf_counter()
    result = client.optimize_portfolio(prices, method="markowitz")
    elapsed = (time.perf_counter() - start) * 1000
    
    print(f"✓ Portfolio optimized: {len(result['weights'])} assets")
    print(f"  Weights: {result['weights']}")
    print(f"  Latency: {elapsed:.2f}ms")
    
    client.close()
    return True

def test_streaming():
    """Test market data streaming."""
    print("\n📡 Testing market data streaming...")
    client = TradingGrpcClient()
    client.connect()
    
    symbols = ['BTC/USD', 'ETH/USD']
    count = 0
    max_messages = 5
    
    print(f"  Streaming {max_messages} messages...")
    start = time.perf_counter()
    
    for update in client.stream_market_data(symbols, exchange='binance', interval_ms=100):
        count += 1
        print(f"  [{count}] {update['symbol']}: bid={update['bid']:.2f}, ask={update['ask']:.2f}")
        if count >= max_messages:
            break
    
    elapsed = (time.perf_counter() - start) * 1000
    throughput = count / (elapsed / 1000)
    
    print(f"✓ Received {count} messages in {elapsed:.0f}ms ({throughput:.0f} msg/s)")
    
    client.close()
    return True

def benchmark_performance():
    """Benchmark gRPC performance."""
    print("\n⚡ Performance Benchmark")
    print("=" * 50)
    
    client = TradingGrpcClient()
    client.connect()
    
    # Test data
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    
    # Warmup
    for _ in range(10):
        client.calculate_mean_reversion(prices)
    
    # Benchmark
    n_calls = 1000
    start = time.perf_counter()
    
    for _ in range(n_calls):
        client.calculate_mean_reversion(prices)
    
    elapsed = time.perf_counter() - start
    avg_latency = (elapsed / n_calls) * 1000  # ms
    throughput = n_calls / elapsed
    
    print(f"Calls:      {n_calls}")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Avg latency: {avg_latency:.3f}ms")
    print(f"Throughput: {throughput:.0f} calls/s")
    
    client.close()

def main():
    """Run all tests."""
    print("🧪 gRPC Integration Tests")
    print("=" * 50)
    
    tests = [
        ("Connectivity", test_connectivity),
        ("Mean Reversion", test_mean_reversion),
        ("Portfolio Optimization", test_portfolio_optimization),
        ("Streaming", test_streaming),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            results.append((name, False))
    
    # Benchmark if all tests pass
    if all(r[1] for r in results):
        try:
            benchmark_performance()
        except Exception as e:
            print(f"❌ Benchmark failed: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary")
    print("=" * 50)
    for name, success in results:
        status = "✓ PASS" if success else "❌ FAIL"
        print(f"{status}: {name}")
    
    passed = sum(1 for _, s in results if s)
    total = len(results)
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! gRPC is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the error messages above.")

if __name__ == "__main__":
    main()
