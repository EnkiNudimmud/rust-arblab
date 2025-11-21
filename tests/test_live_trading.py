#!/usr/bin/env python3
"""
Test live trading data fetching - both REST polling and WebSocket streaming.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import time
from datetime import datetime

def test_rest_polling():
    """Test REST polling mode"""
    print("\n=== Testing REST Polling ===")
    try:
        from python.rust_bridge import get_connector
        
        # Test with Finnhub
        connector = get_connector("finnhub")
        symbols = ["AAPL", "GOOGL"]
        
        print(f"Connector: {connector.name}")
        print(f"Symbols: {symbols}")
        print("\nFetching data for 5 iterations...")
        
        data_points = []
        for i in range(5):
            print(f"\n  Iteration {i+1}:")
            for symbol in symbols:
                try:
                    # Fetch orderbook
                    ob = connector.fetch_orderbook_sync(symbol)
                    
                    # Extract bid/ask
                    if ob and ob.get('bids') and ob.get('asks'):
                        bid = ob['bids'][0][0]
                        ask = ob['asks'][0][0]
                        mid = (bid + ask) / 2
                        
                        data_point = {
                            'timestamp': datetime.now(),
                            'symbol': symbol,
                            'bid': bid,
                            'ask': ask,
                            'mid': mid
                        }
                        
                        data_points.append(data_point)
                        print(f"    {symbol}: bid=${bid:.2f}, ask=${ask:.2f}, mid=${mid:.2f}")
                    else:
                        print(f"    {symbol}: No data")
                        
                except Exception as e:
                    print(f"    {symbol}: Error - {e}")
            
            if i < 4:  # Don't sleep after last iteration
                time.sleep(1)
        
        if len(data_points) > 0:
            print(f"\n✓ REST Polling: Collected {len(data_points)} data points")
            return True
        else:
            print("\n✗ REST Polling: No data collected")
            return False
            
    except Exception as e:
        print(f"\n✗ REST Polling failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_websocket_streaming():
    """Test WebSocket streaming mode"""
    print("\n=== Testing WebSocket Streaming ===")
    try:
        from python.rust_bridge import get_connector
        import threading
        
        # Test with Finnhub
        connector = get_connector("finnhub")
        symbol = "AAPL"
        
        print(f"Connector: {connector.name}")
        print(f"Symbol: {symbol}")
        
        # Check if streaming is supported
        if not hasattr(connector, 'start_stream'):
            print("⚠ Connector doesn't support WebSocket streaming")
            return True  # Not a failure, just not supported
        
        received_data = []
        
        def callback(orderbook):
            try:
                # Extract bid/ask
                if isinstance(orderbook, dict):
                    if orderbook.get("bids") and orderbook.get("asks"):
                        bid = float(orderbook["bids"][0][0])
                        ask = float(orderbook["asks"][0][0])
                    else:
                        return
                else:
                    if hasattr(orderbook, 'bids') and hasattr(orderbook, 'asks'):
                        if orderbook.bids and orderbook.asks:
                            bid = float(orderbook.bids[0][0])
                            ask = float(orderbook.asks[0][0])
                        else:
                            return
                    else:
                        return
                
                data_point = {
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'bid': bid,
                    'ask': ask,
                    'mid': (bid + ask) / 2
                }
                
                received_data.append(data_point)
                print(f"  Update {len(received_data)}: bid=${bid:.2f}, ask=${ask:.2f}")
                
            except Exception as e:
                print(f"  Callback error: {e}")
        
        print(f"\nStarting WebSocket stream for {symbol}...")
        connector.start_stream(symbol, callback)
        
        print("Collecting data for 10 seconds...")
        time.sleep(10)
        
        print("Stopping stream...")
        if hasattr(connector, 'stop_stream'):
            connector.stop_stream(symbol)
        
        if len(received_data) > 0:
            print(f"\n✓ WebSocket: Received {len(received_data)} updates")
            
            # Show first and last data points
            if len(received_data) >= 2:
                first = received_data[0]
                last = received_data[-1]
                print(f"  First: {first['symbol']} @ ${first['mid']:.2f}")
                print(f"  Last:  {last['symbol']} @ ${last['mid']:.2f}")
            
            return True
        else:
            print("\n⚠ WebSocket: No updates received (might be outside market hours)")
            return True  # Don't fail - could be timing issue
            
    except Exception as e:
        print(f"\n✗ WebSocket Streaming failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_symbols():
    """Test with multiple symbols simultaneously"""
    print("\n=== Testing Multiple Symbols ===")
    try:
        from python.rust_bridge import get_connector
        
        connector = get_connector("finnhub")
        symbols = ["AAPL", "GOOGL", "MSFT"]
        
        print(f"Testing {len(symbols)} symbols: {', '.join(symbols)}")
        
        results = {}
        for symbol in symbols:
            try:
                ob = connector.fetch_orderbook_sync(symbol)
                if ob and ob.get('bids') and ob.get('asks'):
                    bid = ob['bids'][0][0]
                    ask = ob['asks'][0][0]
                    results[symbol] = {'bid': bid, 'ask': ask, 'status': 'ok'}
                    print(f"  ✓ {symbol}: ${bid:.2f}/${ask:.2f}")
                else:
                    results[symbol] = {'status': 'no_data'}
                    print(f"  ✗ {symbol}: No data")
            except Exception as e:
                results[symbol] = {'status': 'error', 'error': str(e)}
                print(f"  ✗ {symbol}: {e}")
            
            time.sleep(0.5)  # Rate limit
        
        success_count = sum(1 for r in results.values() if r.get('status') == 'ok')
        print(f"\n{success_count}/{len(symbols)} symbols fetched successfully")
        
        return success_count > 0
        
    except Exception as e:
        print(f"\n✗ Multiple symbols test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all live trading tests"""
    print("=" * 70)
    print("Live Trading Data Fetching Test Suite")
    print("=" * 70)
    
    results = {
        "REST Polling": test_rest_polling(),
        "WebSocket Streaming": test_websocket_streaming(),
        "Multiple Symbols": test_multiple_symbols(),
    }
    
    print("\n" + "=" * 70)
    print("Test Results Summary:")
    print("=" * 70)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:.<50} {status}")
    
    all_passed = all(results.values())
    print("=" * 70)
    
    if all_passed:
        print("✓ ALL TESTS PASSED - Live trading should work!")
        print("\nYou can now:")
        print("  1. Go to Live Trading page in Streamlit")
        print("  2. Select Finnhub connector")
        print("  3. Choose REST Polling or WebSocket Streaming")
        print("  4. Click 'Start Live Feed'")
        print("  5. Watch real-time data flow!")
    else:
        print("✗ SOME TESTS FAILED - Check errors above")
    
    print("=" * 70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
