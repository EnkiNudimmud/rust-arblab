#!/usr/bin/env python3
"""
Notebook Validator - Check all notebooks for common issues
"""
import json
import sys
from pathlib import Path
import re

def check_notebook(notebook_path):
    """Check a single notebook for issues"""
    issues = []
    
    try:
        with open(notebook_path) as f:
            nb = json.load(f)
    except Exception as e:
        return [f"ERROR: Cannot load notebook: {e}"]
    
    # Check each cell
    for i, cell in enumerate(nb.get('cells', [])):
        if cell['cell_type'] != 'code':
            continue
            
        source = ''.join(cell.get('source', []))
        
        # Check for common issues
        if 'import hft_py' in source or 'from hft_py' in source:
            issues.append(f"Cell {i}: Uses hft_py (Rust bindings required)")
        
        # Check for scipy version access (known issue)
        if 'scipy.__version__' in source:
            issues.append(f"Cell {i}: WARNING - Uses scipy.__version__ (may cause AttributeError)")
        
        # Check for potential data path issues
        if '../data/' in source or './data/' in source:
            issues.append(f"Cell {i}: Uses relative data paths")
    
    return issues

def main():
    notebooks_dir = Path('/Users/melvinalvarez/Documents/Enki/Workspace/rust-arblab/examples/notebooks')
    
    print("=" * 80)
    print("NOTEBOOK VALIDATION REPORT")
    print("=" * 80)
    print()
    
    all_notebooks = sorted(notebooks_dir.glob('*.ipynb'))
    
    for nb_path in all_notebooks:
        if '.ipynb_checkpoints' in str(nb_path):
            continue
            
        print(f"\n📓 {nb_path.name}")
        print("-" * 80)
        
        issues = check_notebook(nb_path)
        
        if not issues:
            print("  ✅ No obvious issues found")
        else:
            for issue in issues:
                print(f"  ⚠️  {issue}")
    
    print("\n" + "=" * 80)
    print(f"Total notebooks checked: {len(all_notebooks)}")
    print("=" * 80)

if __name__ == '__main__':
    main()
