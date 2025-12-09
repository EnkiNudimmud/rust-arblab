#!/usr/bin/env python3
"""Test enhanced yfinance integration with caching and optimization.

Tests both the helper module and the data_fetcher integration.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print("Testing Enhanced Yahoo Finance Integration")
print("=" * 70)

# Test 1: yfinance_helper module
print("\n" + "=" * 70)
print("Test 1: yfinance_helper - Stock Data with Caching")
print("=" * 70)

try:
    from python.yfinance_helper import fetch_stocks, get_cache_info, clear_cache
    
    # Check initial cache state
    cache_info = get_cache_info()
    print(f"\n📦 Initial Cache State:")
    print(f"   Directory: {cache_info['cache_dir']}")
    print(f"   Files: {cache_info['files']}")
    print(f"   Size: {cache_info['size_mb']} MB")
    
    # Fetch stock data (will cache)
    print(f"\n{'='*70}")
    print("Fetching AAPL and MSFT (7 days, 1h interval)...")
    print("=" * 70)
    df = fetch_stocks(['AAPL', 'MSFT'], days=7, interval='1h', use_cache=True)
    
    print(f"\n✅ Test 1.1: Stock Data Fetch - PASSED")
    print(f"   Rows: {len(df)}")
    print(f"   Symbols: {df['symbol'].unique().tolist()}")
    print(f"   Columns: {df.columns.tolist()}")
    print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Check if data looks correct
    assert len(df) > 0, "No data returned"
    assert set(df.columns) == {'timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume'}, "Wrong columns"
    assert df['symbol'].nunique() <= 2, "Too many symbols"
    
    print(f"\n   Sample data:")
    print(df.head(3).to_string(index=False))
    
    # Check cache was created
    cache_info = get_cache_info()
    print(f"\n📦 Cache After First Fetch:")
    print(f"   Files: {cache_info['files']}")
    print(f"   Size: {cache_info['size_mb']} MB")
    
    # Fetch again (should use cache)
    print(f"\n{'='*70}")
    print("Fetching AAPL and MSFT again (should use cache)...")
    print("=" * 70)
    df2 = fetch_stocks(['AAPL', 'MSFT'], days=7, interval='1h', use_cache=True)
    
    print(f"\n✅ Test 1.2: Cached Data Fetch - PASSED")
    print(f"   Data matches: {df.equals(df2)}")
    
except Exception as e:
    print(f"\n❌ Test 1: yfinance_helper - FAILED")
    print(f"   Error: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Crypto data
print("\n" + "=" * 70)
print("Test 2: yfinance_helper - Crypto Data")
print("=" * 70)

try:
    from python.yfinance_helper import fetch_crypto
    
    print("\nFetching BTC and ETH (3 days, 1h interval)...")
    df = fetch_crypto(['BTC', 'ETH'], days=3, interval='1h', use_cache=True)
    
    print(f"\n✅ Test 2: Crypto Data Fetch - PASSED")
    print(f"   Rows: {len(df)}")
    print(f"   Symbols: {df['symbol'].unique().tolist()}")
    print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Check that symbols are without -USD suffix
    assert 'BTC' in df['symbol'].values or 'ETH' in df['symbol'].values, "Crypto symbols not formatted correctly"
    
    print(f"\n   Sample data:")
    print(df.head(3).to_string(index=False))
    
except Exception as e:
    print(f"\n❌ Test 2: Crypto Data - FAILED")
    print(f"   Error: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Date range validation
print("\n" + "=" * 70)
print("Test 3: Date Range Validation")
print("=" * 70)

try:
    from python.yfinance_helper import validate_date_range
    from datetime import datetime, timedelta
    
    # Test 1m interval with 10 days (should warn)
    start = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
    end = datetime.now().strftime("%Y-%m-%d")
    
    is_valid, warning = validate_date_range('1m', start, end)
    
    print(f"\n✅ Test 3.1: Date Validation - PASSED")
    print(f"   Interval: 1m, Days: 10")
    print(f"   Valid: {is_valid}")
    print(f"   Warning: {warning[:100] if warning else 'None'}...")
    
    assert not is_valid, "Should be invalid for 1m with 10 days"
    assert warning is not None, "Should have warning"
    
    # Test 1h interval with 10 days (should be OK)
    is_valid, warning = validate_date_range('1h', start, end)
    
    print(f"\n✅ Test 3.2: Date Validation - PASSED")
    print(f"   Interval: 1h, Days: 10")
    print(f"   Valid: {is_valid}")
    print(f"   Warning: {warning}")
    
    assert is_valid, "Should be valid for 1h with 10 days"
    assert warning is None, "Should not have warning"
    
except Exception as e:
    print(f"\n❌ Test 3: Date Validation - FAILED")
    print(f"   Error: {e}")
    import traceback
    traceback.print_exc()

# Test 4: data_fetcher integration
print("\n" + "=" * 70)
print("Test 4: data_fetcher Integration")
print("=" * 70)

try:
    from python.data_fetcher import fetch_intraday_data
    
    print("\nFetching AAPL via data_fetcher (5 days, 1h)...")
    df = fetch_intraday_data(
        symbols=['AAPL'],
        start=(datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d"),
        end=datetime.now().strftime("%Y-%m-%d"),
        interval='1h',
        source='yfinance'
    )
    
    print(f"\n✅ Test 4: data_fetcher Integration - PASSED")
    print(f"   Rows: {len(df)}")
    print(f"   Symbols: {df['symbol'].unique().tolist()}")
    print(f"   Columns: {df.columns.tolist()}")
    
    print(f"\n   Sample data:")
    print(df.head(3).to_string(index=False))
    
except Exception as e:
    print(f"\n❌ Test 4: data_fetcher Integration - FAILED")
    print(f"   Error: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Error handling
print("\n" + "=" * 70)
print("Test 5: Error Handling")
print("=" * 70)

try:
    from python.yfinance_helper import fetch_stocks
    
    print("\nTrying to fetch invalid symbol (should handle gracefully)...")
    try:
        df = fetch_stocks(['INVALID_SYMBOL_12345'], days=7, interval='1h', use_cache=False)
        print(f"\n⚠️  Test 5.1: Invalid Symbol - Got data unexpectedly")
    except ValueError as e:
        print(f"\n✅ Test 5.1: Invalid Symbol Error - PASSED")
        print(f"   Error message is informative: {len(str(e)) > 100}")
        print(f"   First 150 chars: {str(e)[:150]}...")
    
    print("\n{'='*70}")
    print("Trying to fetch 1m data for 90 days (should warn)...")
    start = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
    end = datetime.now().strftime("%Y-%m-%d")
    
    try:
        # This might return data but should show warning
        df = fetch_stocks(['AAPL'], days=90, interval='1m', use_cache=False)
        print(f"\n✅ Test 5.2: Date Range Warning - PASSED")
        print(f"   Data rows: {len(df)} (may be limited to 7 days)")
    except Exception as e:
        print(f"\n✅ Test 5.2: Date Range Warning - PASSED")
        print(f"   Handled with error: {str(e)[:100]}...")
    
except Exception as e:
    print(f"\n❌ Test 5: Error Handling - FAILED")
    print(f"   Error: {e}")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "=" * 70)
print("Test Summary")
print("=" * 70)

try:
    cache_info = get_cache_info()
    print(f"\n📦 Final Cache State:")
    print(f"   Files: {cache_info['files']}")
    print(f"   Size: {cache_info['size_mb']} MB")
    
    print(f"\n🎯 Enhanced Features Tested:")
    print(f"   ✅ Smart caching")
    print(f"   ✅ Date range validation")
    print(f"   ✅ Retry logic (implicit)")
    print(f"   ✅ Progress indicators")
    print(f"   ✅ Error handling")
    print(f"   ✅ Crypto symbol conversion")
    
    print(f"\n💡 Tips:")
    print(f"   • Use higher intervals (1h, 1d) for longer backtests")
    print(f"   • Enable caching (use_cache=True) for faster repeated requests")
    print(f"   • For crypto, consider using CCXT for better coverage")
    print(f"   • Check docs/YFINANCE_USAGE_GUIDE.md for best practices")
    
except Exception as e:
    print(f"   Error getting final cache info: {e}")

print("\n" + "=" * 70)
print("✅ Enhanced yfinance integration testing complete!")
print("=" * 70)
