import json
import glob
import os
import re

notebooks_dir = '/Users/melvinalvarez/Documents/Enki/Workspace/rust-arblab/examples/notebooks'
notebooks = glob.glob(os.path.join(notebooks_dir, '*.ipynb'))

print(f"Scanning {len(notebooks)} notebooks...")

for nb_path in notebooks:
    with open(nb_path, 'r', encoding='utf-8') as f:
        try:
            nb = json.load(f)
        except:
            print(f"Failed to load {nb_path}")
            continue
            
    found_imports = []
    for cell in nb.get('cells', []):
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            if 'hft_py' in source or 'optimizr' in source:
                # Extract the specific line
                lines = source.split('\n')
                for line in lines:
                    if 'hft_py' in line or 'optimizr' in line:
                        found_imports.append(line.strip())

    if found_imports:
        print(f"\nNotebook: {os.path.basename(nb_path)}")
        for line in found_imports:
            print(f"  - {line}")
