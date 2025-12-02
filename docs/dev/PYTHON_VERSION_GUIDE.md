# Python Version Compatibility Guide

## Supported Python Versions

The Multi-Strategy HFT Trading Platform is designed to be compatible across multiple Python versions:

### ✅ Fully Supported
- **Python 3.8+** - Recommended, all features work optimally
- **Python 3.9+** - Excellent performance, latest pandas/numpy
- **Python 3.10+** - Best performance, modern Python features
- **Python 3.11+** - Fastest performance, significant speed improvements

### ⚠️ Minimum Supported (with limitations)
- **Python 3.7** - Minimum version, some dependency versions may be older

### ❌ Not Supported
- **Python 3.6 and below** - Incompatible with Streamlit
- **Python 2.x** - Not supported

---

## Version-Specific Information

### Python 3.7
**Status:** Minimum supported version

**Dependencies:**
```
streamlit>=1.22.0
pandas>=1.3.0,<2.0.0
numpy>=1.21.0,<1.24.0
plotly>=5.0.0
scipy>=1.7.0
```

**Limitations:**
- Older versions of pandas (< 2.0)
- Older versions of numpy (< 1.24)
- May have slightly slower performance

**Notes:**
- Python 3.7 reached end-of-life in June 2023
- Security updates no longer provided
- Upgrade to 3.8+ recommended

### Python 3.8
**Status:** ✅ Recommended minimum

**Dependencies:**
```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.17.0
scipy>=1.11.0
```

**Benefits:**
- All modern package versions supported
- Better performance than 3.7
- Position-only parameters support
- Assignment expressions (walrus operator) available

**Notes:**
- Python 3.8 reaches end-of-life in October 2024
- Stable and well-tested
- Good balance of compatibility and features

### Python 3.9
**Status:** ✅ Fully recommended

**Benefits:**
- Dictionary merge operators (`|`, `|=`)
- Type hint generics in standard collections
- Improved performance
- Better error messages

**Best for:**
- Production deployments
- Long-term stability
- Active security support until October 2025

### Python 3.10
**Status:** ✅ Excellent choice

**Benefits:**
- Structural pattern matching
- Better error messages with location
- Union types with `|`
- Performance improvements (10-60% faster)

**Best for:**
- New projects
- Latest features
- Maximum performance

### Python 3.11+
**Status:** ✅ Best performance

**Benefits:**
- Up to 60% faster than 3.10
- Better error messages
- Enhanced type hints
- Improved debugging

**Best for:**
- Maximum performance
- Latest Python features
- Future-proof projects

---

## Installation by Python Version

### For Python 3.7
```bash
# Install with version constraints
pip install "streamlit>=1.22.0,<1.28" "pandas>=1.3.0,<2.0" "numpy>=1.21.0,<1.24"
pip install plotly>=5.0.0 scipy>=1.7.0 yfinance>=0.1.70
```

### For Python 3.8+
```bash
# Install latest compatible versions
pip install -r app/requirements.txt
```

### For Python 3.9+
```bash
# Use latest versions for best performance
pip install --upgrade streamlit pandas numpy plotly scipy yfinance
```

---

## Checking Your Python Version

### Command Line
```bash
# Check version
python --version
# or
python3 --version

# Check detailed info
python -c "import sys; print(f'Python {sys.version}')"
```

### In Python
```python
import sys
print(f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
```

### In the App
The app automatically checks Python version on startup and will:
- **Error** if version < 3.7
- **Warn** if version < 3.8 (but continue)
- **Display** version in terminal when using `run_app.sh`

---

## Compatibility Features

### Automatic Version Detection
The app includes a compatibility module (`app/utils/compat.py`) that:
- Checks Python version on startup
- Provides appropriate warnings
- Uses compatible imports
- Ensures graceful degradation

### Type Hints
- Uses `typing` module for Python 3.7-3.9
- Compatible with built-in generics in Python 3.10+
- No Python 3.8+ exclusive features used

### No Advanced Features Used
The codebase avoids:
- Walrus operator (`:=`) - Python 3.8+
- Positional-only parameters (`/`) - Python 3.8+
- Union types (`|`) - Python 3.10+
- Pattern matching - Python 3.10+
- f-string `=` specifier - Python 3.8+

---

## Dependency Compatibility

### Core Dependencies

| Package | Python 3.7 | Python 3.8+ | Python 3.9+ |
|---------|-----------|-------------|-------------|
| streamlit | 1.22+ | 1.28+ | Latest |
| pandas | 1.3.x | 2.0+ | Latest |
| numpy | 1.21.x | 1.24+ | Latest |
| plotly | 5.0+ | 5.17+ | Latest |
| scipy | 1.7+ | 1.11+ | Latest |
| yfinance | 0.1.70+ | 0.2+ | Latest |

### Optional Dependencies

| Package | Minimum Version | Notes |
|---------|----------------|-------|
| typing-extensions | 3.7.4+ | Only for Python 3.7 |
| numba | 0.54+ | Performance boost |
| ccxt | 4.0+ | Crypto exchanges |

---

## Performance by Version

Approximate relative performance (Python 3.7 = baseline):

| Version | Performance | Notes |
|---------|------------|-------|
| 3.7 | 100% | Baseline |
| 3.8 | 110% | ~10% faster |
| 3.9 | 115% | Incremental improvements |
| 3.10 | 125% | Significant speedup |
| 3.11 | 160% | Major performance gains |

**Note:** Performance varies by workload. Numerical operations benefit most from newer versions.

---

## Troubleshooting Version Issues

### Issue: "Python 3.7+ required"
```bash
# Check your version
python --version

# If too old, upgrade Python
# macOS with Homebrew:
brew install python@3.11

# Linux (Ubuntu/Debian):
sudo apt update
sudo apt install python3.11

# Create virtual environment with specific version
python3.11 -m venv venv
source venv/bin/activate
```

### Issue: Package Installation Fails
```bash
# For Python 3.7, use compatible versions
pip install "pandas<2.0" "numpy<1.24"

# Or create a new virtual environment
python3 -m venv venv_37
source venv_37/bin/activate
pip install -r app/requirements.txt
```

### Issue: Import Errors
```python
# If you see typing-related errors
pip install typing-extensions

# If you see pandas/numpy errors
pip install --upgrade pandas numpy
```

---

## Recommendations

### For New Installations
**Use Python 3.10 or 3.11** for:
- Best performance
- Latest features
- Long-term support
- Future compatibility

### For Existing Systems
**Python 3.8+** is sufficient for:
- All features work
- Good performance
- Stable dependencies

### For Legacy Systems
**Python 3.7** can work but:
- Upgrade soon (EOL)
- Some packages are outdated
- Performance is slower
- Security risks

---

## Migration Guide

### From Python 3.7 to 3.8+

1. **Backup your environment**
   ```bash
   pip freeze > requirements_37.txt
   ```

2. **Install Python 3.8+**
   ```bash
   # Download from python.org or use package manager
   ```

3. **Create new virtual environment**
   ```bash
   python3.8 -m venv venv_38
   source venv_38/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r app/requirements.txt
   ```

5. **Test the app**
   ```bash
   ./run_app.sh
   ```

---

## Version Checking in Code

The app includes automatic version checking:

```python
# app/utils/compat.py
from app.utils.compat import check_python_version, PYTHON_VERSION

# Check on startup
check_python_version()

# Access version info
print(f"Running Python {PYTHON_VERSION.major}.{PYTHON_VERSION.minor}")
```

---

## Future Compatibility

### Python 3.12+ (Future)
When released, the app will be compatible with:
- Even better performance
- New language features
- Modern type system

### Maintenance Promise
- Minimum version will track Streamlit's requirements
- Support for Python 3.7 may be dropped in future
- Regular updates for new Python releases

---

## Quick Reference

| Task | Command |
|------|---------|
| Check version | `python --version` |
| Minimum required | Python 3.7 |
| Recommended | Python 3.8+ |
| Best performance | Python 3.11+ |
| Install for 3.7 | `pip install "pandas<2.0" "numpy<1.24"` |
| Install for 3.8+ | `pip install -r app/requirements.txt` |
| Verify compatibility | `python -c "import app.utils.compat"` |

---

## Support

For version-related issues:
1. Check this guide first
2. Verify your Python version
3. Check dependency versions
4. Create virtual environment
5. Report issues on GitHub

---

**Last Updated:** November 19, 2025  
**Maintained for:** Python 3.7 - 3.11+  
**Recommended:** Python 3.10 or 3.11
