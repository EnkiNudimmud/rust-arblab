#!/usr/bin/env python3
"""
Quick test script for CCXT integration
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from python.ccxt_helper import quick_fetch, get_available_exchanges
import pandas as pd

def test_ccxt():
    """Test CCXT integration"""
    print("=" * 60)
    print("Testing CCXT Integration")
    print("=" * 60)
    
    # Test 1: Show available exchanges
    print("\n✅ Available Exchanges:")
    exchanges = get_available_exchanges()
    for exchange_id, info in exchanges.items():
        print(f"  • {info['name']}: {info['description']}")
    
    # Test 2: Fetch sample data
    print("\n✅ Fetching BTC/USDT data from Binance (last 24 hours, 1h interval)...")
    try:
        df = quick_fetch('BTC/USDT', 'binance', '1h', days_back=1)
        print(f"  • Successfully fetched {len(df)} candles")
        print(f"  • Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"  • Latest BTC price: ${df['close'].iloc[-1]:,.2f}")
        print(f"  • 24h change: {((df['close'].iloc[-1] / df['open'].iloc[0] - 1) * 100):.2f}%")
        print("\n  Sample data:")
        print(df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].tail(5).to_string())
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def test_data_fetcher():
    """Test data_fetcher integration"""
    print("\n" + "=" * 60)
    print("Testing data_fetcher.py integration")
    print("=" * 60)
    
    from python.data_fetcher import fetch_intraday_data
    from datetime import datetime, timedelta
    
    print("\n✅ Fetching ETH/USDT via data_fetcher...")
    try:
        end = datetime.now()
        start = end - timedelta(days=2)
        
        df = fetch_intraday_data(
            symbols=['ETH/USDT'],
            start=start.isoformat(),
            end=end.isoformat(),
            interval='1h',
            source='ccxt'
        )
        
        print(f"  • Successfully fetched {len(df)} records")
        print(f"  • Symbols: {df.index.get_level_values('symbol').unique().tolist()}")
        
        # Reset index to display
        df_display = df.reset_index()
        print("\n  Sample data:")
        print(df_display[['timestamp', 'symbol', 'open', 'high', 'low', 'close']].tail(5).to_string())
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n🚀 Starting CCXT Integration Tests\n")
    
    test1 = test_ccxt()
    test2 = test_data_fetcher()
    
    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)
    print(f"CCXT Helper:      {'✅ PASSED' if test1 else '❌ FAILED'}")
    print(f"Data Fetcher:     {'✅ PASSED' if test2 else '❌ FAILED'}")
    print("=" * 60)
    
    if test1 and test2:
        print("\n🎉 All tests passed! CCXT integration is working correctly.")
        print("\n💡 Next steps:")
        print("  1. Open the Streamlit app: ./run_app.sh")
        print("  2. Go to 'Data Loader' page")
        print("  3. Select 'CCXT - Crypto Exchanges (FREE! ⭐)'")
        print("  4. Try fetching BTC/USDT, ETH/USDT, or SOL/USDT")
    else:
        print("\n⚠️  Some tests failed. Please check the error messages above.")
        sys.exit(1)
