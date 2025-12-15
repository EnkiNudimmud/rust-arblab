import json
import os

notebook_path = '/Users/melvinalvarez/Documents/Enki/Workspace/rust-arblab/examples/notebooks/superspace_anomaly_detection.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

found_import = False
mod_made = False

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        full_source = "".join(source)
        
        # Check if this is the imports cell
        if "from scipy import stats" in full_source or "import numpy" in full_source:
            
            new_source = []
            has_scipy_import = False
            
            for line in source:
                if "import scipy" in line and "from" not in line:
                    has_scipy_import = True
                new_source.append(line)
            
            # If explicit import scipy is missing, add it
            if not has_scipy_import:
                # Insert after 'from scipy import stats' or at top
                final_source = []
                for line in new_source:
                    final_source.append(line)
                    if "from scipy import stats" in line:
                        final_source.append("import scipy\n")
                        print("Added 'import scipy'")
                        mod_made = True
                
                # If we didn't find the insertion point but it's the right cell, append
                if not mod_made and "import numpy" in full_source:
                     # Fallback
                     pass 
                
                if mod_made:
                    cell['source'] = final_source

            # Fix version check if needed
            fixed_source = []
            for line in cell['source']:
                if "{stats.__version__}" in line:
                    fixed_source.append(line.replace("{stats.__version__}", "{scipy.__version__}"))
                    print("Fixed stats.__version__")
                    mod_made = True
                else:
                    fixed_source.append(line)
            cell['source'] = fixed_source
            
            if mod_made:
                break

if mod_made:
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Notebook updated.")
else:
    print("No changes needed or target not found.")
