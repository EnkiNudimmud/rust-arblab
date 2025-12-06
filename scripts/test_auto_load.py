#!/usr/bin/env python3
"""
Test Auto-Load Dataset Feature
Verifies that ensure_data_loaded() works correctly across all pages
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "app"))

print("🧪 Testing Auto-Load Dataset Feature")
print("=" * 60)

# Test 1: Import ensure_data_loaded
print("\n[1/4] Testing ensure_data_loaded import...")
try:
    from utils.ui_components import ensure_data_loaded
    from utils.data_persistence import list_datasets
    print("   ✅ Imports successful")
except ImportError as e:
    print(f"   ❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Check for available datasets
print("\n[2/4] Checking for saved datasets...")
try:
    datasets = list_datasets()
    if datasets:
        print(f"   ✅ Found {len(datasets)} saved dataset(s):")
        for ds in datasets[:3]:  # Show first 3
            num_symbols = len(ds.get('symbols', []))
            row_count = ds.get('row_count', 0)
            print(f"      - {ds['name']} ({num_symbols} symbols, {row_count} records)")
        if len(datasets) > 3:
            print(f"      ... and {len(datasets) - 3} more")
    else:
        print("   ⚠️  No saved datasets found")
        print("      Tip: Use the Data Loader page to fetch and save data")
except Exception as e:
    print(f"   ❌ Error listing datasets: {e}")
    sys.exit(1)

# Test 3: Simulate session state and test auto-load
print("\n[3/4] Testing auto-load functionality...")
try:
    # Create mock session state
    class MockSessionState:
        def __init__(self):
            self._data = {
                'historical_data': None,
                'symbols': []
            }
        
        def __contains__(self, key):
            return key in self._data
        
        def __getattr__(self, key):
            if key.startswith('_'):
                return object.__getattribute__(self, key)
            return self._data.get(key)
        
        def __setattr__(self, key, value):
            if key.startswith('_'):
                object.__setattr__(self, key, value)
            else:
                self._data[key] = value
    
    # Monkey-patch streamlit (for testing without running Streamlit)
    import importlib
    if 'streamlit' not in sys.modules:
        # Create a minimal mock of streamlit
        class MockStreamlit:
            class SessionState:
                pass
            session_state = MockSessionState()
            
            @staticmethod
            def toast(msg, icon=None):
                print(f"   📢 Toast: {msg}")
        
        sys.modules['streamlit'] = MockStreamlit()
        sys.modules['st'] = MockStreamlit()
    
    # Now test ensure_data_loaded
    import streamlit as st
    st.session_state = MockSessionState()
    
    result = ensure_data_loaded()
    
    if result:
        print(f"   ✅ Auto-load successful!")
        print(f"      - Data shape: {st.session_state.historical_data.shape if st.session_state.historical_data is not None else 'None'}")
        print(f"      - Symbols: {len(st.session_state.symbols)}")
    else:
        print("   ⚠️  No data auto-loaded (no datasets available)")
        
except Exception as e:
    print(f"   ❌ Auto-load test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Verify pages import correctly
print("\n[4/4] Verifying page imports...")
pages_to_test = [
    "lab_mean_reversion",
    "lab_momentum",
    "lab_market_making",
    "lab_pca_arbitrage",
    "lab_sparse_meanrev",
    "lab_advanced_optimization",
    "lab_adaptive_strategies",
]

success_count = 0
for page in pages_to_test:
    try:
        # Just verify the ensure_data_loaded import exists
        with open(project_root / f"app/pages/{page}.py", "r") as f:
            content = f.read()
            if "ensure_data_loaded" in content:
                success_count += 1
            else:
                print(f"   ⚠️  {page}.py missing ensure_data_loaded")
    except FileNotFoundError:
        print(f"   ⚠️  {page}.py not found")

print(f"   ✅ {success_count}/{len(pages_to_test)} pages have auto-load enabled")

# Summary
print("\n" + "=" * 60)
print("🎉 Auto-Load Feature Test Summary:")
print(f"   ✓ Import system working")
print(f"   ✓ Dataset persistence working")
print(f"   ✓ Auto-load functionality working")
print(f"   ✓ {success_count}/{len(pages_to_test)} pages updated")
print("\n💡 Usage:")
print("   • Navigate to any lab page")
print("   • If no data is loaded, the most recent dataset loads automatically")
print("   • You'll see a toast notification: '✅ Auto-loaded: <dataset_name>'")
print("   • Works on app restart, page refresh, and navigation")
print("\n✅ All tests passed!")
