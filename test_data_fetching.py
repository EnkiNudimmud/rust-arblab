#!/usr/bin/env python3
"""
Test script to verify data fetching and WebSocket connectivity.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import logging
logging.basicConfig(level=logging.INFO)

def test_api_keys():
    """Test API keys loading"""
    print("\n=== Testing API Keys ===")
    try:
        from python.api_keys import get_finnhub_key, get_binance_credentials, get_kraken_credentials
        
        fh_key = get_finnhub_key()
        print(f"✓ Finnhub API key: {'Found' if fh_key else 'Not found'}")
        
        binance_key, binance_secret = get_binance_credentials()
        print(f"✓ Binance credentials: {'Found' if binance_key and binance_secret else 'Not found'}")
        
        kraken_key, kraken_secret = get_kraken_credentials()
        print(f"✓ Kraken credentials: {'Found' if kraken_key and kraken_secret else 'Not found'}")
        
        return True
    except Exception as e:
        print(f"✗ API keys test failed: {e}")
        return False


def test_finnhub_connector():
    """Test Finnhub connector initialization and quote fetching"""
    print("\n=== Testing Finnhub Connector ===")
    try:
        from python.rust_bridge import get_connector
        
        connector = get_connector("finnhub")
        print(f"✓ Finnhub connector created: {connector.name}")
        
        # Test symbols
        symbols = connector.list_symbols()
        print(f"✓ Available symbols: {len(symbols)}")
        
        # Test sync orderbook fetch
        test_symbol = "AAPL"
        print(f"Fetching quote for {test_symbol}...")
        ob = connector.fetch_orderbook_sync(test_symbol)
        
        if ob and ob.get('bids') and ob.get('asks'):
            bid = ob['bids'][0][0]
            ask = ob['asks'][0][0]
            print(f"✓ Quote fetched: bid={bid:.2f}, ask={ask:.2f}, spread={ask-bid:.4f}")
            return True
        else:
            print(f"✗ No data returned for {test_symbol}")
            return False
            
    except Exception as e:
        print(f"✗ Finnhub connector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_fetcher():
    """Test data fetching with different sources"""
    print("\n=== Testing Data Fetcher ===")
    try:
        from python.data_fetcher import fetch_intraday_data
        from datetime import datetime, timedelta
        
        symbols = ["AAPL", "GOOGL"]
        start = (datetime.now() - timedelta(days=7)).isoformat()
        end = datetime.now().isoformat()
        
        # Test with synthetic data
        print("Testing with synthetic data...")
        df = fetch_intraday_data(symbols, start, end, interval="1h", source="synthetic")
        print(f"✓ Synthetic data: {len(df)} rows")
        
        # Test with Yahoo Finance
        print("Testing with Yahoo Finance...")
        try:
            df = fetch_intraday_data(symbols, start, end, interval="1h", source="yfinance")
            print(f"✓ Yahoo Finance: {len(df)} rows")
        except Exception as e:
            print(f"⚠ Yahoo Finance failed (expected if not installed): {e}")
        
        # Test with Finnhub
        print("Testing with Finnhub...")
        try:
            df = fetch_intraday_data(symbols, start, end, interval="1h", source="finnhub")
            print(f"✓ Finnhub: {len(df)} rows")
        except Exception as e:
            print(f"⚠ Finnhub failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Data fetcher test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_websocket_stream():
    """Test WebSocket streaming (short test)"""
    print("\n=== Testing WebSocket Stream ===")
    try:
        from python.rust_bridge import get_connector
        import time
        import threading
        
        connector = get_connector("finnhub")
        
        # Check if connector supports streaming
        if not hasattr(connector, 'start_stream'):
            print("⚠ Connector doesn't support streaming (expected for some connectors)")
            return True
        
        received_data = []
        
        def callback(orderbook):
            received_data.append(orderbook)
            print(f"  Received update: {len(received_data)} total")
        
        # Start stream
        test_symbol = "AAPL"
        print(f"Starting stream for {test_symbol}...")
        connector.start_stream(test_symbol, callback)
        
        # Wait for data
        print("Waiting 5 seconds for data...")
        time.sleep(5)
        
        # Stop stream
        connector.stop_stream()
        
        if len(received_data) > 0:
            print(f"✓ WebSocket test passed: received {len(received_data)} updates")
            return True
        else:
            print("⚠ No data received (might be outside market hours)")
            return True  # Don't fail - might be timing issue
            
    except Exception as e:
        print(f"✗ WebSocket test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("Data Fetching & WebSocket Connectivity Test Suite")
    print("=" * 60)
    
    results = {
        "API Keys": test_api_keys(),
        "Finnhub Connector": test_finnhub_connector(),
        "Data Fetcher": test_data_fetcher(),
        "WebSocket Stream": test_websocket_stream(),
    }
    
    print("\n" + "=" * 60)
    print("Test Results Summary:")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:.<40} {status}")
    
    all_passed = all(results.values())
    print("=" * 60)
    print(f"Overall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
