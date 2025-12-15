#!/usr/bin/env python3
"""
Test hft_py.regime_portfolio availability
"""
import sys

print("=" * 80)
print("Testing hft_py.regime_portfolio")
print("=" * 80)
print()

print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print()

try:
    import hft_py
    print("✓ hft_py imported successfully")
    print(f"  Location: {hft_py.__file__}")
    print()
    
    print("Available modules in hft_py:")
    for attr in dir(hft_py):
        if not attr.startswith('_'):
            print(f"  - {attr}")
    print()
    
    # Test regime_portfolio specifically
    if hasattr(hft_py, 'regime_portfolio'):
        print("✓ hft_py.regime_portfolio is available!")
        print(f"  Type: {type(hft_py.regime_portfolio)}")
        print()
        
        # Try to access submodules
        print("regime_portfolio contents:")
        for attr in dir(hft_py.regime_portfolio):
            if not attr.startswith('_'):
                print(f"  - {attr}")
    else:
        print("✗ hft_py.regime_portfolio is NOT available")
        print()
        print("This might mean:")
        print("  1. The Rust bindings need to be recompiled")
        print("  2. The Jupyter kernel needs to be restarted")
        print("  3. There's a Python environment mismatch")
        
except ImportError as e:
    print(f"✗ Failed to import hft_py: {e}")
    print()
    print("The Rust bindings are not installed in this Python environment.")
    print("Run: cd rust_python_bindings && maturin develop --release")

print()
print("=" * 80)
