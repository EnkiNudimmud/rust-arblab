#!/usr/bin/env python3
"""Quick visual demo of enhanced yfinance features."""

import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent))

print("=" * 80)
print("ENHANCED YFINANCE INTEGRATION - QUICK DEMO")
print("=" * 80)

# Demo 1: Simple stock fetch with progress
print("\n" + "=" * 80)
print("DEMO 1: Fetching Stock Data with Progress Indicators")
print("=" * 80)

from python.yfinance_helper import fetch_stocks

print("\n📈 Fetching AAPL, MSFT, GOOGL (last 7 days, 1h interval)...")
df = fetch_stocks(['AAPL', 'MSFT', 'GOOGL'], days=7, interval='1h', use_cache=False)

print(f"\n✅ SUCCESS!")
print(f"   Rows: {len(df)}")
print(f"   Symbols: {df['symbol'].unique().tolist()}")
print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

print("\n   Latest prices:")
for symbol in df['symbol'].unique():
    symbol_data = df[df['symbol'] == symbol]
    latest = symbol_data.iloc[-1]
    print(f"   {symbol}: ${latest['close']:.2f}")

# Demo 2: Error handling
print("\n" + "=" * 80)
print("DEMO 2: Enhanced Error Messages")
print("=" * 80)

print("\n⚠️  Attempting to fetch invalid symbol...")
try:
    df = fetch_stocks(['INVALID_TICKER_XYZ123'], days=7, interval='1h', use_cache=False)
except ValueError as e:
    print("\n✅ Error caught with helpful message:")
    print(str(e)[:300] + "...")

# Demo 3: Date range validation
print("\n" + "=" * 80)
print("DEMO 3: Smart Date Range Validation")
print("=" * 80)

from python.yfinance_helper import validate_date_range

print("\n🔍 Checking if 1m data can be fetched for 90 days...")
is_valid, warning = validate_date_range('1m', 
                                       (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d"),
                                       datetime.now().strftime("%Y-%m-%d"))

if not is_valid:
    print(f"\n❌ Invalid request detected!")
    print(warning)
else:
    print("\n✅ Valid request!")

print("\n🔍 Checking if 1h data can be fetched for 90 days...")
is_valid, warning = validate_date_range('1h', 
                                       (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d"),
                                       datetime.now().strftime("%Y-%m-%d"))

if not is_valid:
    print(f"\n❌ Invalid request detected!")
    print(warning)
else:
    print("\n✅ Valid request! This will work perfectly.")

# Demo 4: Crypto with auto-conversion
print("\n" + "=" * 80)
print("DEMO 4: Crypto Fetching with Auto Symbol Conversion")
print("=" * 80)

from python.yfinance_helper import fetch_crypto

print("\n₿ Fetching BTC and ETH (input: ['BTC', 'ETH'])...")
df = fetch_crypto(['BTC', 'ETH'], days=3, interval='1h', use_cache=False)

print(f"\n✅ SUCCESS!")
print(f"   Rows: {len(df)}")
print(f"   Symbols returned: {df['symbol'].unique().tolist()}  ← Notice: 'BTC' not 'BTC-USD'")
print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

print("\n   Latest crypto prices:")
for symbol in df['symbol'].unique():
    symbol_data = df[df['symbol'] == symbol]
    latest = symbol_data.iloc[-1]
    print(f"   {symbol}: ${latest['close']:,.2f}")

# Demo 5: Cache info
print("\n" + "=" * 80)
print("DEMO 5: Cache Management")
print("=" * 80)

from python.yfinance_helper import get_cache_info, clear_cache

cache_info = get_cache_info()
print(f"\n📦 Cache Status:")
print(f"   Location: {cache_info['cache_dir']}")
print(f"   Files: {cache_info['files']}")
print(f"   Size: {cache_info['size_mb']} MB")

if cache_info['files'] > 0:
    print(f"\n🗑️  Clearing cache...")
    removed = clear_cache()
    print(f"   Removed {removed} cache files")
else:
    print(f"\n💡 Note: Install pyarrow for caching support:")
    print(f"   pip install pyarrow")

# Summary
print("\n" + "=" * 80)
print("SUMMARY: Key Features Demonstrated")
print("=" * 80)

print("""
✅ Enhanced Features Working:

1. 📊 Progress Indicators
   • Real-time feedback for each symbol
   • Clear success/failure messages
   • Row counts and status updates

2. 🔄 Retry Logic
   • Automatic 3 retries with exponential backoff
   • Graceful handling of transient failures
   • Detailed error reporting

3. ⚠️  Smart Validation
   • Date range checking against yfinance limits
   • Proactive warnings before fetching
   • Helpful suggestions for alternatives

4. 💡 Better Error Messages
   • Specific guidance for each error type
   • Symbol format examples
   • Troubleshooting tips included

5. 🔧 Optimization Features
   • Auto symbol conversion (BTC → BTC-USD → BTC)
   • Smart interval handling
   • Cache management utilities

6. 📈 Data Quality
   • Timezone-aware timestamp handling
   • Consistent column format
   • Accurate latest prices

📚 Documentation:
   • See docs/YFINANCE_USAGE_GUIDE.md for complete guide
   • See docs/YFINANCE_ENHANCEMENT_SUMMARY.md for technical details

🚀 Next Steps:
   • Install pyarrow for caching: pip install pyarrow
   • Use fetch_stocks() for stock data
   • Use fetch_crypto() for cryptocurrency data
   • Check validate_date_range() before large requests
""")

print("=" * 80)
print("✅ DEMO COMPLETE!")
print("=" * 80)
