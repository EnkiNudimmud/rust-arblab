# Environment Setup Quick Reference

This guide provides all commands needed to set up, build, and rebuild the project in different environments.

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Docker Setup](#docker-setup)
3. [Local Setup by OS](#local-setup-by-os)
4. [Python Environment Options](#python-environment-options)
5. [Build Commands](#build-commands)
6. [Rebuild/Reset Commands](#rebuildreset-commands)
7. [Configuration Files](#configuration-files)
8. [Verification & Testing](#verification--testing)

---

## System Requirements

| Component | Version | Notes |
|-----------|---------|-------|
| Python | 3.11+ | 3.11 recommended |
| Rust | 1.70+ | Latest stable via rustup |
| Docker | 20.10+ | Optional, for containerized builds |
| Shell | zsh/bash | macOS/Linux default |
| OS | macOS/Linux/Windows | Docker recommended for Windows |

---

## Docker Setup

### Initial Build & Run
```bash
# From project root
cd /path/to/rust-hft-arbitrage-lab

# Build and start all services
docker compose build
docker compose up

# Access Streamlit UI at: http://localhost:8501
```

### Docker Environment Details
- **Base Image**: `python:3.11-slim`
- **Rust Toolchain**: Latest stable (installed during build)
- **System Packages**: build-essential, patchelf, libssl-dev, libopenblas-dev, liblapack-dev
- **Python Packages**: Installed from `docker/requirements.txt`
- **Rust Connector**: Auto-built with maturin during image creation

### Docker Commands Reference
```bash
# Start services (foreground)
docker compose up

# Start services (background)
docker compose up -d

# View logs
docker compose logs -f

# Stop services
docker compose down

# Rebuild and restart
docker compose up --build

# Clean rebuild (no cache)
docker compose build --no-cache
docker compose up

# Remove all containers and volumes
docker compose down --volumes --remove-orphans
```

---

## Local Setup by OS

### macOS

#### Prerequisites
```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python 3.11
brew install python@3.11

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

#### Python Environment (Conda - Recommended for macOS)
```bash
# Install Miniconda (if not installed)
brew install miniconda

# Create environment
conda create -n rhftlab python=3.11 -y
conda activate rhftlab

# Install dependencies
pip install --upgrade pip setuptools wheel maturin
pip install -r docker/requirements.txt

# Build Rust connector
maturin develop --manifest-path rust_connector/Cargo.toml --release
```

#### Python Environment (venv - Alternative)
```bash
# Create venv
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip setuptools wheel maturin
pip install -r docker/requirements.txt

# Build Rust connector
maturin develop --manifest-path rust_connector/Cargo.toml --release
```

### Linux (Ubuntu/Debian)

#### Prerequisites
```bash
# Update package list
sudo apt-get update

# Install system dependencies
sudo apt-get install -y \
    build-essential \
    curl \
    git \
    pkg-config \
    libssl-dev \
    python3.11 \
    python3.11-dev \
    python3-pip \
    python3.11-venv

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

#### Python Environment (venv)
```bash
# Create venv
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip setuptools wheel maturin
pip install -r docker/requirements.txt

# Build Rust connector
maturin develop --manifest-path rust_connector/Cargo.toml --release
```

### Linux (RHEL/CentOS/Fedora)

#### Prerequisites
```bash
# Install system dependencies
sudo dnf install -y \
    gcc \
    gcc-c++ \
    make \
    curl \
    git \
    pkg-config \
    openssl-devel \
    python3.11 \
    python3.11-devel

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

### Windows

#### Prerequisites
1. **Install Python 3.11**: Download from [python.org](https://www.python.org/downloads/)
2. **Install Rust**: Download from [rustup.rs](https://rustup.rs/)
3. **Install Visual Studio Build Tools**: Required for Rust compilation
   - Download from [visualstudio.microsoft.com](https://visualstudio.microsoft.com/downloads/)
   - Select "Desktop development with C++"

#### Python Environment (PowerShell)
```powershell
# Create venv
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install dependencies
pip install --upgrade pip setuptools wheel maturin
pip install -r docker/requirements.txt

# Build Rust connector
maturin develop --manifest-path rust_connector/Cargo.toml --release
```

**Note**: Docker is **strongly recommended** for Windows to avoid build tool complications.

---

## Python Environment Options

### Option 1: Conda (Recommended for macOS)
```bash
# Create
conda create -n rhftlab python=3.11 -y

# Activate
conda activate rhftlab

# Deactivate
conda deactivate

# Remove
conda env remove -n rhftlab
```

**Pros**: Best compatibility on macOS, handles system dependencies well
**Cons**: Requires Anaconda/Miniconda installation

### Option 2: venv (Lightweight)
```bash
# Create
python3.11 -m venv .venv

# Activate (macOS/Linux)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\Activate.ps1

# Deactivate
deactivate

# Remove
rm -rf .venv  # macOS/Linux
rmdir /s .venv  # Windows
```

**Pros**: Built into Python, lightweight, no extra tools
**Cons**: May have issues on macOS with native extensions

### Option 3: Docker (Most Consistent)
```bash
# No local Python environment needed
# Everything runs in container
docker compose up --build
```

**Pros**: Identical environment everywhere, no local setup
**Cons**: Requires Docker, slower iteration cycle

---

## Build Commands

### Full Build Sequence (Local)
```bash
# 1. Activate Python environment
source .venv/bin/activate  # or: conda activate rhftlab

# 2. Ensure build tools are installed
pip install --upgrade pip setuptools wheel maturin

# 3. Install Python dependencies
pip install -r docker/requirements.txt

# 4. Build and install Rust connector
maturin develop --manifest-path rust_connector/Cargo.toml --release

# 5. Verify installation
python -c "import rust_connector; print('✓ Rust connector installed')"
```

### Build Options

#### Development Build (Faster, with Debug Symbols)
```bash
maturin develop --manifest-path rust_connector/Cargo.toml
```

#### Release Build (Optimized, Production)
```bash
maturin develop --manifest-path rust_connector/Cargo.toml --release
```

#### Build Wheel (for Distribution)
```bash
maturin build --manifest-path rust_connector/Cargo.toml --release
# Output: target/wheels/rust_connector-*.whl
```

---

## Rebuild/Reset Commands

### Quick Rebuild (Rust Connector Only)
```bash
# Incremental rebuild (fast)
maturin develop --manifest-path rust_connector/Cargo.toml --release
```

### Clean Rebuild (Rust)
```bash
# Remove all Rust build artifacts
cargo clean

# Rebuild from scratch
maturin develop --manifest-path rust_connector/Cargo.toml --release
```

### Reset Python Environment (venv)
```bash
# 1. Deactivate and remove
deactivate
rm -rf .venv

# 2. Recreate
python3.11 -m venv .venv
source .venv/bin/activate

# 3. Reinstall everything
pip install --upgrade pip setuptools wheel maturin
pip install -r docker/requirements.txt
maturin develop --manifest-path rust_connector/Cargo.toml --release
```

### Reset Python Environment (Conda)
```bash
# 1. Deactivate and remove
conda deactivate
conda env remove -n rhftlab

# 2. Recreate
conda create -n rhftlab python=3.11 -y
conda activate rhftlab

# 3. Reinstall everything
pip install --upgrade pip setuptools wheel maturin
pip install -r docker/requirements.txt
maturin develop --manifest-path rust_connector/Cargo.toml --release
```

### Complete Docker Rebuild
```bash
# Remove everything
docker compose down --volumes --remove-orphans

# Clean build (no cache)
docker compose build --no-cache

# Start fresh
docker compose up
```

### Nuclear Option (Clean Everything)
```bash
# Remove Python environment
rm -rf .venv
conda env remove -n rhftlab

# Remove all Rust build artifacts
cargo clean
rm -rf target/
rm -rf rust_connector/target/
rm -rf rust_core/target/
rm -rf rust_python_bindings/target/

# Remove Python cache
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete

# Remove Docker artifacts
docker compose down --volumes --remove-orphans
docker system prune -af

# Now rebuild from scratch
```

---

## Configuration Files

### api_keys.properties
Location: Project root

```properties
# Exchange API Keys (for authenticated trading)
binance.api_key=your_binance_api_key_here
binance.api_secret=your_binance_api_secret_here

coinbase.api_key=your_coinbase_api_key_here
coinbase.api_secret=your_coinbase_api_secret_here

kraken.api_key=your_kraken_api_key_here
kraken.api_secret=your_kraken_api_secret_here

# Market Data API Keys
finnhub.api_key=your_finnhub_api_key_here
```

**Setup**:
```bash
# Copy example file
cp api_keys.properties.example api_keys.properties

# Edit with your keys
nano api_keys.properties  # or your preferred editor
```

**Security**:
- ✅ Included in `.gitignore`
- ✅ Never commit real keys
- ✅ Use read-only API keys when possible
- ✅ For Docker: mount as volume (already configured in docker-compose.yml)

### Environment Variables
Optional alternative to api_keys.properties:

```bash
# Export in shell (temporary)
export FINNHUB_API_KEY=your_key_here

# Add to shell profile (persistent)
echo 'export FINNHUB_API_KEY=your_key_here' >> ~/.zshrc
source ~/.zshrc

# Docker Compose (.env file)
echo 'FINNHUB_API_KEY=your_key_here' > .env
```

---

## Verification & Testing

### Environment Verification
```bash
# Check Python version
python --version
# Expected: Python 3.11.x

# Check Rust version
rustc --version
# Expected: rustc 1.7x.x or newer

# Check maturin
maturin --version
# Expected: maturin x.x.x

# Check if in virtual environment
which python
# Expected: path should include .venv or anaconda3/envs/rhftlab
```

### Installation Verification
```bash
# Test Rust connector import
python -c "import rust_connector; print('✓ rust_connector')"

# List available connectors
python -c "from python.rust_bridge import list_connectors; print(list_connectors())"
# Expected: ['binance', 'coinbase', 'kraken', 'finnhub', ...]

# Get connector and test
python -c "
from python.rust_bridge import get_connector
c = get_connector('binance')
print('✓ Binance connector:', c.list_symbols()[:3])
"
```

### Functional Testing
```bash
# Test WebSocket streaming
python test_websocket.py
# Expected: Should show updates from Binance and Kraken

# Test API keys loading
python test_api_keys.py
# Expected: Should display loaded API keys (masked)

# Run Streamlit app
streamlit run app/streamlit_app.py
# Expected: Browser opens at http://localhost:8501

# Test notebook
jupyter notebook examples/notebooks/triangular_arbitrage.ipynb
# Expected: Jupyter opens, notebook runs without errors
```

### Health Check Commands
```bash
# Quick health check
python << 'EOF'
import sys
print(f"✓ Python {sys.version}")

try:
    import rust_connector
    print("✓ rust_connector installed")
except ImportError:
    print("✗ rust_connector NOT installed")
    sys.exit(1)

from python.rust_bridge import list_connectors
connectors = list_connectors()
print(f"✓ {len(connectors)} connectors available: {connectors}")

print("\n✅ Environment healthy!")
EOF
```

---

## Common Issues & Solutions

### Issue: `maturin: command not found`
```bash
pip install --upgrade maturin
```

### Issue: `ModuleNotFoundError: No module named 'rust_connector'`
```bash
# Rebuild and install
maturin develop --manifest-path rust_connector/Cargo.toml --release
```

### Issue: Python/Rust ABI mismatch (macOS)
```bash
# Solution 1: Use conda environment
conda create -n rhftlab python=3.11 -y
conda activate rhftlab
pip install -r docker/requirements.txt
maturin develop --manifest-path rust_connector/Cargo.toml --release

# Solution 2: Use Docker
docker compose up --build
```

### Issue: Docker build fails with "patchelf not found"
```bash
# Fixed in latest Dockerfile
# Pull latest code and rebuild
git pull
docker compose build --no-cache
```

### Issue: WebSocket not receiving updates
```bash
# Verify latest code
git pull

# Rebuild Rust connector
cargo clean
maturin develop --manifest-path rust_connector/Cargo.toml --release

# Test
python test_websocket.py
```

### Issue: Streamlit not auto-refreshing
```bash
# Ensure latest app code
git pull

# Restart Streamlit
# Press Ctrl+C and run:
streamlit run app/streamlit_app.py
```

---

## Quick Command Cheatsheet

### Setup (First Time)
```bash
# Docker
docker compose up --build

# Local (macOS with conda)
conda create -n rhftlab python=3.11 -y && conda activate rhftlab
pip install -r docker/requirements.txt maturin
maturin develop --manifest-path rust_connector/Cargo.toml --release

# Local (Linux with venv)
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r docker/requirements.txt maturin
maturin develop --manifest-path rust_connector/Cargo.toml --release
```

### Daily Development
```bash
# Activate environment
source .venv/bin/activate  # or: conda activate rhftlab

# Run app
streamlit run app/streamlit_app.py

# Rebuild after Rust changes
maturin develop --manifest-path rust_connector/Cargo.toml --release

# Run tests
python test_websocket.py
```

### Troubleshooting
```bash
# Clean rebuild
cargo clean && maturin develop --manifest-path rust_connector/Cargo.toml --release

# Reset environment
rm -rf .venv && python3.11 -m venv .venv && source .venv/bin/activate

# Docker clean rebuild
docker compose down --volumes && docker compose build --no-cache && docker compose up
```

---

## Environment Variables Reference

```bash
# Python
export PYTHON_VERSION=3.11

# Rust optimization
export RUSTFLAGS="-C target-cpu=native"

# API Keys (alternative to api_keys.properties)
export FINNHUB_API_KEY=your_key
export BINANCE_API_KEY=your_key
export BINANCE_API_SECRET=your_secret

# Docker
export COMPOSE_BAKE=true  # Performance tuning

# Jupyter
export JUPYTER_PORT=8888
```

---

## Support

For more detailed information, see:
- **README.md**: Full project documentation
- **QUICK_CONFIG.md**: API keys configuration
- **KRAKEN_WEBSOCKET_GUIDE.md**: Kraken-specific setup
- **FINNHUB_USAGE.md**: Market data configuration

For issues:
- Check the [Troubleshooting section](#common-issues--solutions)
- Review GitHub issues
- Run verification commands above
