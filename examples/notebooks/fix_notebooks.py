#!/usr/bin/env python3
"""
Fix common notebook issues
"""
import json
import sys
from pathlib import Path

def fix_scipy_version_issue(notebook_path):
    """Remove scipy.__version__ access that causes AttributeError"""
    with open(notebook_path) as f:
        nb = json.load(f)
    
    fixed = False
    for cell in nb.get('cells', []):
        if cell['cell_type'] != 'code':
            continue
        
        source = ''.join(cell.get('source', []))
        
        # Check if this cell has the scipy version print
        if 'scipy.__version__' in source:
            # Remove or comment out the scipy version line
            lines = source.split('\n')
            new_lines = []
            for line in lines:
                if 'scipy.__version__' in line and 'print' in line.lower():
                    # Comment it out instead of removing
                    new_lines.append('# ' + line + '  # Commented out - scipy may not have __version__')
                    fixed = True
                else:
                    new_lines.append(line)
            
            cell['source'] = [l + '\n' for l in new_lines[:-1]] + [new_lines[-1]] if new_lines else []
    
    if fixed:
        with open(notebook_path, 'w') as f:
            json.dump(nb, f, indent=1)
        return True
    return False

def add_hft_py_fallback(notebook_path):
    """Add try-except fallback for hft_py imports"""
    with open(notebook_path) as f:
        nb = json.load(f)
    
    fixed = False
    for cell in nb.get('cells', []):
        if cell['cell_type'] != 'code':
            continue
        
        source = ''.join(cell.get('source', []))
        
        # Check if this cell imports hft_py without try-except
        if 'import hft_py' in source and 'try:' not in source:
            lines = source.split('\n')
            
            # Find the hft_py import line
            for i, line in enumerate(lines):
                if 'import hft_py' in line and not line.strip().startswith('#'):
                    # Check if there's already error handling
                    if i > 0 and 'try' in lines[i-1]:
                        continue
                    
                    # Add try-except wrapper
                    indent = len(line) - len(line.lstrip())
                    indent_str = ' ' * indent
                    
                    new_lines = lines[:i] + [
                        indent_str + 'try:',
                        indent_str + '    ' + line.strip(),
                        indent_str + '    print("✓ hft_py (Rust bindings) loaded successfully")',
                        indent_str + 'except ImportError as e:',
                        indent_str + '    print(f"⚠️  Could not import hft_py: {e}")',
                        indent_str + '    print("Note: Rust bindings not available. Some features may be limited.")',
                    ] + lines[i+1:]
                    
                    cell['source'] = [l + '\n' for l in new_lines[:-1]] + [new_lines[-1]] if new_lines else []
                    fixed = True
                    break
    
    if fixed:
        with open(notebook_path, 'w') as f:
            json.dump(nb, f, indent=1)
        return True
    return False

def main():
    notebooks_dir = Path('/Users/melvinalvarez/Documents/Enki/Workspace/rust-arblab/examples/notebooks')
    
    print("=" * 80)
    print("FIXING NOTEBOOKS")
    print("=" * 80)
    print()
    
    # Fix scipy version issue
    superspace_nb = notebooks_dir / 'superspace_anomaly_detection.ipynb'
    if superspace_nb.exists():
        print(f"📓 Checking {superspace_nb.name}...")
        if fix_scipy_version_issue(superspace_nb):
            print("  ✅ Fixed scipy.__version__ issue")
        else:
            print("  ℹ️  No scipy.__version__ issue found")
    
    # Add fallbacks for hft_py notebooks
    hft_py_notebooks = [
        'regime_switching_jump_diffusion.ipynb',
        'portfolio_drift_uncertainty.ipynb',
        'delta_hedging_analysis.ipynb'
    ]
    
    for nb_name in hft_py_notebooks:
        nb_path = notebooks_dir / nb_name
        if nb_path.exists():
            print(f"\n📓 Checking {nb_name}...")
            if add_hft_py_fallback(nb_path):
                print("  ✅ Added hft_py import fallback")
            else:
                print("  ℹ️  hft_py import already has error handling or not found")
    
    print("\n" + "=" * 80)
    print("FIXES COMPLETE")
    print("=" * 80)

if __name__ == '__main__':
    main()
