# Quick Command Reference Card

**One-page reference for common build/run commands**

---

## 🐳 Docker (Simplest)

```bash
# Build and start
docker compose up --build

# Stop
docker compose down

# Clean rebuild
docker compose down --volumes --remove-orphans
docker compose build --no-cache
docker compose up

# Access: http://localhost:8501
```

---

## 💻 Local Development

### First Time Setup

**macOS (conda recommended)**
```bash
conda create -n rhftlab python=3.11 -y
conda activate rhftlab
pip install -r docker/requirements.txt maturin
maturin develop --manifest-path rust_connector/Cargo.toml --release
```

**macOS/Linux (venv)**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r docker/requirements.txt maturin
maturin develop --manifest-path rust_connector/Cargo.toml --release
```

**Or use the automated script**
```bash
./setup_env.sh
```

### Daily Usage

```bash
# Activate environment
source .venv/bin/activate        # venv
conda activate rhftlab            # conda

# Run Streamlit
streamlit run app/streamlit_app.py

# Run Jupyter
jupyter notebook examples/notebooks/

# Test WebSocket
python test_websocket.py
```

---

## 🔧 Rebuild Commands

### Rust Connector Only (Fast)
```bash
maturin develop --manifest-path rust_connector/Cargo.toml --release
```

### Clean Rust Build
```bash
cargo clean
maturin develop --manifest-path rust_connector/Cargo.toml --release
```

### Reset Python Environment
```bash
# venv
deactivate && rm -rf .venv
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r docker/requirements.txt maturin

# conda
conda deactivate && conda env remove -n rhftlab
conda create -n rhftlab python=3.11 -y && conda activate rhftlab
pip install -r docker/requirements.txt maturin
```

### Nuclear Option (Everything)
```bash
# Local
rm -rf .venv target/ rust_connector/target/
cargo clean
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# Docker
docker compose down --volumes --remove-orphans
docker system prune -af
```

---

## ⚙️ Configuration

### API Keys
```bash
# Create config file
cp api_keys.properties.example api_keys.properties

# Edit with your keys
nano api_keys.properties
```

### Environment Variables (Alternative)
```bash
export FINNHUB_API_KEY=your_key_here
export BINANCE_API_KEY=your_key_here
export BINANCE_API_SECRET=your_secret_here
```

---

## ✅ Verification

```bash
# Check versions
python --version          # Should be 3.11+
rustc --version          # Should be 1.70+
maturin --version        # Should be installed

# Check installation
python -c "import rust_connector; print('✓ OK')"

# List connectors
python -c "from python.rust_bridge import list_connectors; print(list_connectors())"

# Test WebSocket
python test_websocket.py
```

---

## 🆘 Troubleshooting

**Problem**: `maturin: command not found`
```bash
pip install --upgrade maturin
```

**Problem**: `ModuleNotFoundError: No module named 'rust_connector'`
```bash
maturin develop --manifest-path rust_connector/Cargo.toml --release
```

**Problem**: Python/Rust mismatch on macOS
```bash
# Use conda instead
conda create -n rhftlab python=3.11 -y
conda activate rhftlab
# ... or use Docker
```

**Problem**: Docker build fails
```bash
docker compose build --no-cache
```

---

## 📂 Key Files

| File | Purpose |
|------|---------|
| `api_keys.properties` | API credentials (git-ignored) |
| `docker/requirements.txt` | Python dependencies |
| `rust_connector/Cargo.toml` | Rust dependencies |
| `setup_env.sh` | Interactive setup script |
| `ENVIRONMENT_SETUP.md` | Complete setup guide |

---

## 🎯 Quick Start (Choose One)

**Fastest - Docker**
```bash
make docker-up
# OR: docker compose up --build
# Open: http://localhost:8501
```

**Using Makefile (Easiest Local)**
```bash
make setup          # Complete setup
make run            # Start Streamlit
```

**Automated Script**
```bash
./setup_env.sh      # Interactive wizard
```

**Manual Local**
```bash
# 1. Create environment
python3.11 -m venv .venv && source .venv/bin/activate

# 2. Install deps
pip install -r docker/requirements.txt maturin

# 3. Build Rust
maturin develop --manifest-path rust_connector/Cargo.toml --release

# 4. Run
streamlit run app/streamlit_app.py
```

## 🛠️ Using Makefile

```bash
# See all commands
make help

# Complete setup (local)
make setup

# Run app
make run

# Run tests
make test

# Rebuild Rust
make rebuild

# Docker
make docker-up
make docker-down

# Clean
make clean          # Build artifacts only
make clean-all      # Everything including venv
```

---

## 📝 Environment Info

**System Requirements**
- Python: 3.11+
- Rust: 1.70+
- Docker: 20.10+ (optional)

**Shells Supported**
- zsh (macOS default)
- bash (Linux default)
- PowerShell (Windows - Docker recommended)

**Package Managers**
- macOS: Homebrew
- Ubuntu/Debian: apt
- RHEL/CentOS: dnf/yum
- Windows: Manual (or Docker)

---

For complete documentation, see: **[ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md)**
