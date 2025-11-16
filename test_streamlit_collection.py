#!/usr/bin/env python3
"""
Test script to verify WebSocket collection works
"""
import sys
from pathlib import Path
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from python.rust_bridge import get_connector

print("=" * 60)
print("Testing WebSocket Data Collection")
print("=" * 60)

# Test 1: Get connector
print("\n1. Getting Binance connector...")
connector = get_connector("binance")
print(f"   ✓ Got connector: {connector}")

# Test 2: Check methods
print("\n2. Checking available methods...")
has_fetch = hasattr(connector, "fetch_orderbook_sync")
has_stream = hasattr(connector, "start_stream")
has_snapshot = hasattr(connector, "latest_snapshot")
print(f"   fetch_orderbook_sync: {'✓' if has_fetch else '✗'}")
print(f"   start_stream: {'✓' if has_stream else '✗'}")
print(f"   latest_snapshot: {'✓' if has_snapshot else '✗'}")

if not (has_fetch and has_stream and has_snapshot):
    print("\n   ✗ Missing required methods!")
    sys.exit(1)

# Test 3: Start WebSocket
print("\n3. Starting WebSocket stream...")
symbol = "BTCUSDT"
updates_received = []

def callback(ob):
    updates_received.append(ob)
    print(f"   📥 WebSocket update received (total: {len(updates_received)})")

try:
    connector.start_stream(symbol, callback)
    print(f"   ✓ WebSocket started for {symbol}")
except Exception as e:
    print(f"   ✗ Failed to start WebSocket: {e}")
    sys.exit(1)

# Test 4: Wait for data
print("\n4. Waiting for WebSocket data (10 seconds)...")
for i in range(10):
    time.sleep(1)
    snapshot = connector.latest_snapshot()
    if snapshot:
        print(f"   [{i+1}s] ✓ Snapshot available")
        # Try to parse it
        try:
            if hasattr(snapshot, 'bids') and hasattr(snapshot, 'asks'):
                if snapshot.bids and snapshot.asks:
                    bid = float(snapshot.bids[0][0])
                    ask = float(snapshot.asks[0][0])
                    print(f"          Bid: {bid:.2f}, Ask: {ask:.2f}, Spread: {ask-bid:.2f}")
        except Exception as e:
            print(f"          (Could not parse: {e})")
    else:
        print(f"   [{i+1}s] ⚠ No snapshot yet...")

# Test 5: Summary
print("\n5. Test Summary:")
print(f"   WebSocket updates received: {len(updates_received)}")
final_snapshot = connector.latest_snapshot()
if final_snapshot:
    print("   ✓ Final snapshot available")
else:
    print("   ✗ No final snapshot")

if len(updates_received) > 0 or final_snapshot:
    print("\n✅ WebSocket collection is WORKING!")
    print("\nStreamlit should now be able to collect data continuously.")
    print("Run: streamlit run app/streamlit_app.py")
    print("Then select 'Streaming (WebSocket)' mode")
else:
    print("\n✗ WebSocket collection NOT working")
    print("Check your internet connection and Binance API availability")

print("=" * 60)
