import json

notebook_path = '/Users/melvinalvarez/Documents/Enki/Workspace/rust-arblab/examples/notebooks/superspace_anomaly_detection.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

target_id = "8ab58eb2"
found = False

new_source = [
    "# Core numerical and scientific libraries\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import scipy\n",
    "from scipy import stats\n",
    "from scipy.optimize import minimize\n",
    "import statsmodels.api as sm\n",
    "from statsmodels.tsa.stattools import coint, adfuller\n",
    "\n",
    "# Machine learning and data preprocessing\n",
    "from sklearn.decomposition import PCA\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "\n",
    "# Symbolic mathematics (install if needed)\n",
    "try:\n",
    "    import sympy as sp\n",
    "except ImportError:\n",
    "    import sys\n",
    "    import subprocess\n",
    "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'sympy'])\n",
    "    import sympy as sp\n",
    "\n",
    "# Visualization\n",
    "import matplotlib.pyplot as plt\n",
    "from mpl_toolkits.mplot3d import Axes3D\n",
    "import seaborn as sns\n",
    "\n",
    "# Plotly for interactive plots\n",
    "try:\n",
    "    import plotly.graph_objects as go\n",
    "    import plotly.express as px\n",
    "    from plotly.subplots import make_subplots\n",
    "except ImportError:\n",
    "    import sys\n",
    "    import subprocess\n",
    "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'plotly'])\n",
    "    import plotly.graph_objects as go\n",
    "    import plotly.express as px\n",
    "    from plotly.subplots import make_subplots\n",
    "\n",
    "# Configuration\n",
    "plt.style.use('seaborn-v0_8-darkgrid')\n",
    "sns.set_palette(\"husl\")\n",
    "np.random.seed(42)\n",
    "\n",
    "print(\"✓ All imports successful!\")\n",
    "print(f\"NumPy version: {np.__version__}\")\n",
    "print(f\"Pandas version: {pd.__version__}\")\n",
    "print(f\"SciPy version: {scipy.__version__}\")\n",
    "print(f\"SymPy version: {sp.__version__}\")"
]

for cell in nb['cells']:
    if cell.get('id') == target_id:
        cell['source'] = new_source
        found = True
        break

if found:
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Notebook updated successfully.")
else:
    print(f"Cell with ID {target_id} not found.")
