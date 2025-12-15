import json
import os

notebook_path = '/Users/melvinalvarez/Documents/Enki/Workspace/rust-arblab/examples/notebooks/superspace_anomaly_detection.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the cell and modify it
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        # Join temporarily to check content
        full_source = "".join(source)
        if "SciPy version: {stats.__version__}" in full_source:
            print("Found target cell.")
            
            # Modify imports
            new_source = []
            for line in source:
                if line.strip() == "from scipy import stats":
                    new_source.append(line)
                    new_source.append("import scipy\n")
                elif "SciPy version: {stats.__version__}" in line:
                    new_source.append(line.replace("{stats.__version__}", "{scipy.__version__}"))
                else:
                    new_source.append(line)
            
            cell['source'] = new_source
            print("Cell modified.")
            break

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook saved.")
