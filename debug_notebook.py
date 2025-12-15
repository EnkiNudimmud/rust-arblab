import json

notebook_path = '/Users/melvinalvarez/Documents/Enki/Workspace/rust-arblab/examples/notebooks/superspace_anomaly_detection.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        print(f"--- Cell {i} ---")
        if "SciPy version" in source:
            print("FOUND KEYWORD 'SciPy version'")
            print(repr(source))
