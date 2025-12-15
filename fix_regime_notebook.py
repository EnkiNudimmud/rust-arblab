#!/usr/bin/env python3
"""
Fix regime_switching_jump_diffusion.ipynb to use correct API
"""
import json
from pathlib import Path

notebook_path = Path('/Users/melvinalvarez/Documents/Enki/Workspace/rust-arblab/examples/notebooks/regime_switching_jump_diffusion.ipynb')

with open(notebook_path) as f:
    nb = json.load(f)

# Find and fix cells that use hft_py.regime_portfolio
for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    
    source = ''.join(cell.get('source', []))
    
    # Check if this cell uses regime_portfolio incorrectly
    if 'hft_py.regime_portfolio.' in source and 'from hft_py.regime_portfolio import regime_portfolio' not in source:
        print("Found cell with incorrect API usage:")
        print(source[:200])
        print()
        
        # Fix the imports
        lines = source.split('\n')
        new_lines = []
        
        # Check if we need to add the import
        has_import = any('from hft_py.regime_portfolio import regime_portfolio' in line for line in lines)
        
        if not has_import:
            # Find where to insert the import (after other imports)
            import_idx = 0
            for i, line in enumerate(lines):
                if line.strip().startswith('import ') or line.strip().startswith('from '):
                    import_idx = i + 1
            
            # Insert the correct import
            lines.insert(import_idx, 'from hft_py.regime_portfolio import regime_portfolio as rp')
            lines.insert(import_idx + 1, '')
        
        # Replace hft_py.regime_portfolio. with rp.
        for i, line in enumerate(lines):
            if 'hft_py.regime_portfolio.' in line:
                lines[i] = line.replace('hft_py.regime_portfolio.', 'rp.')
        
        cell['source'] = [l + '\n' for l in lines[:-1]] + ([lines[-1]] if lines else [])
        
        print("Fixed to:")
        print(''.join(cell['source'])[:200])
        print()

# Save the fixed notebook
with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=1)

print(f"✅ Fixed {notebook_path.name}")
