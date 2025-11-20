#!/usr/bin/env python3
"""
Quick validation script for WebSocket live trading functionality
Tests that WebSocket connections work and data flows correctly
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_websocket_connector():
    """Test that connector supports WebSocket streaming"""
    print("=" * 60)
    print("Testing WebSocket Connector Support")
    print("=" * 60)
    
    from python.rust_bridge import get_connector
    
    # Test Binance connector
    print("\n1. Testing Binance connector...")
    try:
        connector = get_connector("binance")
        
        # Check for required methods
        has_start_stream = hasattr(connector, 'start_stream')
        has_latest_snapshot = hasattr(connector, 'latest_snapshot')
        
        print(f"   ✓ Connector loaded")
        print(f"   ✓ start_stream available: {has_start_stream}")
        print(f"   ✓ latest_snapshot available: {has_latest_snapshot}")
        
        if has_start_stream and has_latest_snapshot:
            print("   ✅ Binance supports WebSocket streaming")
            return True
        else:
            print("   ❌ Binance missing WebSocket methods")
            return False
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_websocket_data_flow():
    """Test that WebSocket data flows correctly"""
    print("\n" + "=" * 60)
    print("Testing WebSocket Data Flow")
    print("=" * 60)
    
    from python.rust_bridge import get_connector
    import time
    
    print("\n2. Testing data reception...")
    try:
        connector = get_connector("binance")
        
        # Track updates
        updates = []
        
        def callback(orderbook):
            updates.append(orderbook)
            if len(updates) <= 3:
                print(f"   ✓ Update #{len(updates)} received")
        
        # Start stream
        print("   Starting WebSocket stream for BTCUSDT...")
        connector.start_stream("BTCUSDT", callback)
        
        # Wait for data
        print("   Waiting 5 seconds for data...")
        for i in range(5):
            time.sleep(1)
            snapshot = connector.latest_snapshot()
            if snapshot:
                print(f"   ✓ Snapshot available at {i+1}s")
                if len(updates) > 0:
                    break
        
        if len(updates) > 0:
            print(f"   ✅ Received {len(updates)} updates via WebSocket")
            return True
        else:
            snapshot = connector.latest_snapshot()
            if snapshot and hasattr(snapshot, 'bids') and snapshot.bids:
                print(f"   ✅ Snapshot cached (bid: {snapshot.bids[0][0]})")
                return True
            else:
                print("   ❌ No updates received and no snapshot")
                return False
                
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_extraction():
    """Test data extraction from orderbook"""
    print("\n" + "=" * 60)
    print("Testing Data Extraction")
    print("=" * 60)
    
    from python.rust_bridge import get_connector
    import time
    
    print("\n3. Testing bid/ask extraction...")
    try:
        connector = get_connector("binance")
        
        # Start stream
        connector.start_stream("BTCUSDT", lambda ob: None)
        time.sleep(2)
        
        # Get snapshot
        snapshot = connector.latest_snapshot()
        
        if snapshot:
            # Test extraction logic
            if hasattr(snapshot, 'bids') and hasattr(snapshot, 'asks'):
                if snapshot.bids and snapshot.asks:
                    bid = float(snapshot.bids[0][0])
                    ask = float(snapshot.asks[0][0])
                    mid = (bid + ask) / 2
                    spread = ask - bid
                    
                    print(f"   ✓ Bid: ${bid:,.2f}")
                    print(f"   ✓ Ask: ${ask:,.2f}")
                    print(f"   ✓ Mid: ${mid:,.2f}")
                    print(f"   ✓ Spread: ${spread:.2f}")
                    print("   ✅ Data extraction successful")
                    return True
        
        print("   ❌ No data available for extraction")
        return False
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all validation tests"""
    print("\n🔍 WebSocket Live Trading Validation")
    print("=" * 60)
    
    results = []
    
    # Test 1: Connector support
    results.append(("Connector Support", test_websocket_connector()))
    
    # Test 2: Data flow
    results.append(("Data Flow", test_websocket_data_flow()))
    
    # Test 3: Data extraction
    results.append(("Data Extraction", test_data_extraction()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Validation Summary")
    print("=" * 60)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
    
    all_passed = all(r[1] for r in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED - WebSocket functionality working!")
        print("\nYou can now:")
        print("  1. Run the app: ./run_app.sh")
        print("  2. Navigate to Live Trading page")
        print("  3. Select 'Streaming (WebSocket)' mode")
        print("  4. Click 'Start Live Feed'")
        print("  5. Watch real-time data stream in!")
    else:
        print("⚠️ SOME TESTS FAILED - Check errors above")
        print("\nYou can still use REST polling mode:")
        print("  1. Run the app: ./run_app.sh")
        print("  2. Navigate to Live Trading page")
        print("  3. Select 'Polling (REST)' mode instead")
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
