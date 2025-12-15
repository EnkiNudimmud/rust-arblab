import json
import sys

notebook_path = '/Users/melvinalvarez/Documents/Enki/Workspace/rust-arblab/examples/notebooks/superspace_anomaly_detection.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        print(f"\n--- Cell {i} (ID: {cell.get('id')}) ---")
        print(source[:500])
