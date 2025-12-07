#!/usr/bin/env python3
"""
Test script to verify all Streamlit pages can be imported without errors.
"""

import sys
from pathlib import Path
import importlib.util

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "app"))

def test_import_module(module_path: Path) -> tuple[bool, str]:
    """Test if a Python module can be imported without errors."""
    try:
        spec = importlib.util.spec_from_file_location("test_module", module_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return True, "OK"
    except Exception as e:
        return False, str(e)
    return False, "Unknown error"

def main():
    """Test all Streamlit pages."""
    pages_dir = project_root / "app" / "pages"
    pages = sorted(pages_dir.glob("*.py"))
    
    print("=" * 80)
    print("Testing Streamlit Pages Import")
    print("=" * 80)
    
    results = []
    for page in pages:
        if page.name.startswith("__"):
            continue
        
        print(f"\nTesting {page.name}...", end=" ")
        success, message = test_import_module(page)
        
        if success:
            print("✓ OK")
            results.append((page.name, True, ""))
        else:
            print(f"✗ FAILED")
            print(f"  Error: {message[:100]}")
            results.append((page.name, False, message))
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    passed = sum(1 for _, success, _ in results if success)
    failed = len(results) - passed
    
    print(f"\nTotal: {len(results)} pages")
    print(f"✓ Passed: {passed}")
    print(f"✗ Failed: {failed}")
    
    if failed > 0:
        print("\nFailed pages:")
        for name, success, error in results:
            if not success:
                print(f"  - {name}")
                if error:
                    print(f"    {error[:200]}")
        sys.exit(1)
    else:
        print("\n✓ All pages can be imported successfully!")
        sys.exit(0)

if __name__ == "__main__":
    main()
