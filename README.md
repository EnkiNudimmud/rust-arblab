# rust-hft-arbitrage-lab

> **🚀 Quick Start**: Run `./setup_env.sh` for guided setup, or see [QUICK_REFERENCE.md](QUICK_REFERENCE.md) for command cheatsheet

A modular, Docker-ready research lab for high-frequency trading (HFT) and arbitrage research.
This project combines Rust (for low-latency connectors and numeric kernels) and Python (for research, backtesting and visualization) to provide an end-to-end environment:

- Rust <-> Python bridge (PyO3 / maturin)
- Research notebooks (Triangular Arbitrage, Market Making, Pair Trading)
- Reusable Python backtester (PnL, Sharpe, drawdown)
- Streamlit UI for interactive backtesting, simulation and connector-driven live/paper runs
- Docker recipes to build & run everything reproducibly

Quick summary
- Ready-to-run notebooks with fallback Python implementations (so you can run them even without the Rust extension).
- Rust connector skeleton (WebSocket example + orderbook parsing + compute primitives).
- Streamlit app for visually running and inspecting strategies.
- Dockerfile that builds Rust and Python inside a container for consistent builds across hosts.

---

## 🚀 Quick Environment Setup

### System Requirements
- **Python**: 3.11+ (3.11 recommended)
- **Rust**: Latest stable (install via [rustup](https://rustup.rs/))
- **Shell**: bash/zsh (macOS/Linux) or PowerShell (Windows)
- **Docker** (optional): Latest Docker Desktop or Docker Engine

### Choose Your Environment

#### Option A: Docker (Recommended - Consistent Across All Systems)
```bash
# 1. Build and start all services
docker compose build
docker compose up

# 2. Access Streamlit UI
# Open: http://localhost:8501

# 3. Stop services
docker compose down
```

**Docker Environment Details:**
- Base Image: `python:3.11-slim`
- Rust: Latest stable toolchain
- System Dependencies: Pre-installed (build-essential, patchelf, libssl-dev, etc.)
- Python Dependencies: Auto-installed from `docker/requirements.txt`
- Rust Connector: Auto-built with maturin during image build

#### Option B: Local Development (macOS/Linux)

**Step 1: Install System Dependencies**
```bash
# macOS (using Homebrew)
brew install rust python@3.11

# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y build-essential curl git pkg-config \
    libssl-dev python3.11 python3.11-dev python3-pip

# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

**Step 2: Setup Python Environment**
```bash
# Using venv (lightweight)
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# OR using conda (recommended for macOS)
conda create -n rhftlab python=3.11 -y
conda activate rhftlab
```

**Step 3: Install Python Dependencies**
```bash
pip install --upgrade pip setuptools wheel maturin
pip install -r docker/requirements.txt
```

**Step 4: Build Rust Connector**
```bash
# Build and install Rust extension into current Python environment
maturin develop --manifest-path rust_connector/Cargo.toml --release
```

**Step 5: Run Application**
```bash
# Start Streamlit app
streamlit run app/streamlit_app.py

# OR start Jupyter for notebooks
jupyter notebook examples/notebooks/
```

### Configuration Files

#### API Keys Setup
Create `api_keys.properties` in project root:
```properties
# Exchange API Keys (optional - for live trading)
binance.api_key=your_binance_key_here
binance.api_secret=your_binance_secret_here

coinbase.api_key=your_coinbase_key_here
coinbase.api_secret=your_coinbase_secret_here

kraken.api_key=your_kraken_key_here
kraken.api_secret=your_kraken_secret_here

# Market Data (optional - for real-time data)
finnhub.api_key=your_finnhub_key_here
```

**Note**: See `api_keys.properties.example` for template. Never commit real keys to git!

### Rebuild Commands

#### Full Clean Rebuild (Docker)
```bash
# Remove old containers and rebuild from scratch
docker compose down --volumes --remove-orphans
docker compose build --no-cache
docker compose up
```

#### Rebuild Rust Connector (Local)
```bash
# Quick rebuild (incremental)
maturin develop --manifest-path rust_connector/Cargo.toml --release

# Clean rebuild
cargo clean
maturin develop --manifest-path rust_connector/Cargo.toml --release
```

#### Reset Python Environment (Local)
```bash
# Deactivate and remove venv
deactivate
rm -rf .venv

# Recreate from scratch
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel maturin
pip install -r docker/requirements.txt
maturin develop --manifest-path rust_connector/Cargo.toml --release
```

### Environment Variables Reference

```bash
# Optional: Finnhub API key for real market data
export FINNHUB_API_KEY=your_key_here

# Optional: Docker compose performance tuning
export COMPOSE_BAKE=true

# Python version (for venv creation)
export PYTHON_VERSION=3.11

# Rust compilation flags (optional optimization)
export RUSTFLAGS="-C target-cpu=native"
```

### Shell-Specific Commands

**Zsh/Bash (macOS/Linux)**
```bash
source .venv/bin/activate           # Activate venv
source $HOME/.cargo/env             # Load Rust environment
conda activate rhftlab              # Activate conda env
```

**PowerShell (Windows)**
```powershell
.venv\Scripts\Activate.ps1          # Activate venv
conda activate rhftlab              # Activate conda env
```

### Verification Commands

```bash
# Check Python version
python --version  # Should show 3.11.x

# Check Rust version
rustc --version   # Should show 1.70.0 or newer

# Check if Rust connector is installed
python -c "import rust_connector; print(rust_connector.__version__)"

# List available connectors
python -c "from python.rust_bridge import list_connectors; print(list_connectors())"

# Test WebSocket streaming
python test_websocket.py
```

### Troubleshooting Quick Fixes

**Problem**: `maturin: command not found`
```bash
pip install --upgrade maturin
```

**Problem**: `ModuleNotFoundError: No module named 'rust_connector'`
```bash
maturin develop --manifest-path rust_connector/Cargo.toml --release
```

**Problem**: Docker build fails with "patchelf not found"
- Fixed in latest Dockerfile (patchelf added to dependencies)
- Run: `docker compose build --no-cache`

**Problem**: Python/Rust ABI mismatch on macOS
- Use conda environment instead of system Python
- OR use Docker for consistent Linux build environment

**Problem**: WebSocket not receiving updates
- Ensure you're using latest code (fixed in recent commits)
- Check that connector is properly initialized: `python test_websocket.py`

---

---

## 📚 Documentation Quick Links

### Setup & Configuration
- **[SETUP_PATHS.md](SETUP_PATHS.md)** - 🎯 Visual decision tree (start here!)
- **[ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md)** - Complete environment setup guide
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - One-page command cheatsheet
- **[QUICK_CONFIG.md](QUICK_CONFIG.md)** - API keys configuration

### Feature-Specific Guides
- **[KRAKEN_WEBSOCKET_GUIDE.md](KRAKEN_WEBSOCKET_GUIDE.md)** - Kraken connector details
- **[FINNHUB_USAGE.md](FINNHUB_USAGE.md)** - Market data configuration

### Quick Actions
- **Interactive Setup**: Run `./setup_env.sh` for guided wizard
- **Quick Commands**: Run `make help` for all available commands

---

Table of contents
1. Features
2. Architecture overview
3. Quickstart (Docker — recommended)
4. Local dev (non-Docker)
5. Build & install Rust connector (maturin)
6. Using notebooks
7. Streamlit app (UI & examples)
8. Docker tips & troubleshooting
9. Development & testing
10. Contributing & license
11. References

---

## 1. Features

- **Real Market Data** via Finnhub (free tier: 60 API calls/min):
  - ✅ Crypto (Binance, Coinbase), Stocks, Forex
  - ✅ Real-time quotes and WebSocket streaming
  - ✅ Automatic fallback to synthetic data
  - 📖 See `FINNHUB_USAGE.md` for details
  
- **⚡ Mean-Reversion Portfolio Discovery** (NEW - based on d'Aspremont 2011):
  - ✅ High-performance Rust implementation (10-100x speedup)
  - ✅ PCA-based portfolio construction
  - ✅ Ornstein-Uhlenbeck parameter estimation
  - ✅ CARA utility maximization (Appendix A)
  - ✅ Risk-adjusted portfolio weights (Sharpe optimization)
  - ✅ Transaction cost modeling
  - ✅ Optimal stopping times
  - ✅ Multi-period portfolio optimization
  - ✅ Interactive Streamlit UI (`streamlit_meanrev.py`)
  - 📖 See `ADVANCED_MEANREV_FEATURES.md` for complete guide
  
- Rust connector exported as a Python extension using PyO3 (fast orderbook parsing, mean-reversion compute, triangular arbitrage).
- Python bridge (`python/rust_bridge.py`) that exposes the Rust functions to notebooks / Streamlit.
- Backtesting core (`python/backtest/core.py`) with:
  - Trade execution simulation
  - Mark-to-market equity curve recording
  - Metrics: returns, Sharpe (annualized), max drawdown
  - Plotly visualizations
  
- **12 Jupyter Notebooks** with real Finnhub data integration:
  - `mean_rev_vec.ipynb` - **NEW**: Mean-reverting portfolios with PCA
  - `triangular_arbitrage.ipynb` - Crypto cross-rate arbitrage
  - `stat_arb_pairs.ipynb` - Statistical arbitrage pairs trading
  - `market_making.ipynb` - Market making strategy
  - `market_making_imbalance.ipynb` - Imbalance-based MM
  - `hawkes_modeling.ipynb` - Hawkes process modeling
  - `pair_trading_optimal.ipynb` - Optimal pairs trading
  - `portfolio_hedging.ipynb` - Portfolio hedging strategies
  - `price_discovery.ipynb` - Price discovery analysis
  - `signature_optimal_stopping.ipynb` - Signature methods
  - `triangular_signature_optimal_stopping.ipynb` - Combined approach
  - `ws_orderbook_client_demo.ipynb` - WebSocket demo
  Each notebook contains equations, inline commentary, fallback implementations and visualization cells.
  
- **Three Streamlit Apps**:
  - `streamlit_app.py` - Market data explorer with live connectors
  - `streamlit_strategies.py` - Strategy backtesting with real/synthetic data
  - `streamlit_meanrev.py` - **NEW**: Interactive mean-reversion portfolio discovery
  - Attempts to start the Rust WebSocket connector if built
  - Displays equity, trades, orderbook snapshots and metrics
  
- Docker-based reproducible build (Rust + Python built inside container using maturin).

---

## 1.1 Real Market Data (Optional)

Get real market data from Finnhub (free tier: 60 API calls/min):

```bash
# Get free API key at https://finnhub.io/register
export FINNHUB_API_KEY=your_key_here

# Run notebooks or apps - they'll use real data automatically
streamlit run app/streamlit_strategies.py
```

**Features:**
- ✅ Real-time crypto, stocks, forex data
- ✅ Automatic fallback to synthetic data if no key
- ✅ Simple environment variable configuration

See `QUICK_CONFIG.md` for setup details.

---

## 2. Architecture overview

- rust_connector/ (Rust crate)
  - PyO3 module exposing:
    - OrderBook type
    - parse_orderbook(json)
    - compute_triangular_opportunity(...)
    - blocking_start_ws(url) — example to start WS in background
- python/
  - rust_bridge.py — thin shim to call the Rust module from Python
  - backtest/core.py — Backtest class + metrics
- app/
  - streamlit_app.py — interactive UI for backtesting and live simulation
  - joke_app.py — small demo (random joke generator)
- notebooks/ — example strategy notebooks (self-contained)
- Dockerfile, docker-compose.yml — build and run everything inside a container

Design notes:
- The Rust crate is built and installed into the Python environment using maturin. Rust crates (like pyo3) must be declared in Cargo.toml and NOT in pip requirements.
- The Streamlit app uses the Python bridge to call Rust when available, otherwise uses Python fallbacks included in the notebooks and app.

---

## 3. Quickstart (Docker — recommended)

Why Docker? It isolates host differences (macOS Python linking, architecture mismatches) and builds the Rust extension in a Linux environment where maturin + Python dev libs are available.

1) Build & run (from repository root)
```bash
# optionally improve performance on some setups
export COMPOSE_BAKE=true

# build and start (this will build Rust, install Python deps and start Streamlit)
docker-compose up --build
```

2) Open the UI:
- Streamlit interface: http://localhost:8501

3) Stop
```bash
docker-compose down
```

Notes:
- The image builds the Rust toolchain and runs `python -m maturin develop --manifest-path rust_connector/Cargo.toml --release` inside the container. That installs the compiled Python extension into the container's Python environment.
- If the build fails in Docker, copy the first ~100 error lines and check the Troubleshooting section below.

---

## 4. Local development (without Docker)

Prerequisites:
- Rust toolchain (rustup)
- Python 3.11+ (same interpreter you will use to run Streamlit)
- pip and virtualenv

Steps:
```bash
@# create & activate venv
python -m venv .venv
source .venv/bin/activate

# install maturin and dependencies
pip install --upgrade pip setuptools wheel maturin
pip install -r docker/requirements.txt    # note: docker/requirements.txt does NOT include Rust crates

# build & install the rust extension into the current Python env
python -m maturin develop --manifest-path rust_connector/Cargo.toml --release

# run streamlit
streamlit run app/streamlit_app.py
```

Important: run `maturin` using the same Python interpreter you will use to import the extension (the same venv).

---

## 5. Build & install Rust connector (maturin)

The Rust crate is at `rust_connector/`. Example Cargo.toml includes PyO3 and tokio-tungstenite.

Build & install (dev workflow):
```bash
# from repo root (ensure your venv/py env is active)
python -m pip install --upgrade maturin
python -m maturin develop --manifest-path rust_connector/Cargo.toml --release
```

Alternative: build a wheel and install it and/or publish it:
```bash
python -m maturin build --manifest-path rust_connector/Cargo.toml --release
pip install target/wheels/rust_connector-*.whl
```

Notes:
- Do NOT put `pyo3` in Python requirements.txt — `pyo3` is a Rust dependency (Cargo.toml).
- If you encounter macOS linker errors (missing _Py symbols), ensure the Python architecture matches the Rust build or use Docker to build in Linux.

Example Python usage (after building):
```python
from python.rust_bridge import parse_orderbook, compute_triangular_opportunity, start_ws

# parsing
ob = parse_orderbook('{"bids":[[100.0,1]],"asks":[[100.1,1]]}')

# compute (placeholders exist in skeleton)
profit, route = compute_triangular_opportunity(ob, ob, ob)

# start background websocket client (example)
start_ws("wss://example.exchange/ws")
```

---

## 6. Using the notebooks

Notebooks are self-contained and have fallback implementations:
- Run them in Jupyter or VSCode notebooks.
- They attempt to import `python.rust_bridge` but fall back to pure-Python logic so results are reproducible if Rust extension is missing.
- Each notebook includes:
  - Strategy explanation & equations
  - Synthetic dataset generator
  - Strategy function (detect & signal generation)
  - Backtest run and visualizations (Plotly)
  - Notes on production improvements (fees, depth, slippage, order management)

Quick run:
```bash
pip install -r docker/requirements.txt
jupyter notebook notebooks/Triangular.ipynb
```

---

## 7. Streamlit app (UI & examples)

File: `app/streamlit_app.py`

Key features:
- Choose mode: Backtest | Live (simulate) | Live (connector)
- Choose strategy: Triangular | Market Making | Pair Trading
- Data source: Synthetic | Upload CSV
- For Backtest: sliders & params in sidebar, Equity chart, Metrics, Trades table
- For Live (simulate): synthetic orderbook feed with orderbook snapshot and mid-price chart
- For Live (connector): attempts to call `rust_connector` to start a WS client

Example commands:
```bash
# run locally
streamlit run app/streamlit_app.py

# in Docker the container's CMD runs streamlit automatically
```

Security note: never hardcode API keys. Use Streamlit secrets (`.streamlit/secrets.toml`) or environment variables when connecting to real exchanges.

---

## 8. Docker tips & troubleshooting

Common build problem you reported:
- Error: `Could not find a version that satisfies the requirement pyo3`
  - Root cause: `pyo3` is a Rust crate, not a Python package; pip cannot install it.
  - Fix: remove `pyo3` from any `requirements.txt` and let `maturin` build the Rust crate.

Build troubleshooting checklist:
1. If pip fails saying it can't find pyo3 -> remove it from requirements.
2. If macOS linker fails with missing `_Py*` symbols:
   - Likely Python <-> Rust arch mismatch. Ensure you run maturin with the Python interpreter you will use for import.
   - Use Docker to build on Linux if host linking is problematic.
3. If Docker build is slow:
   - Consider enabling delegated builds: `export COMPOSE_BAKE=true` (informational prompt from compose).
   - Use Docker build cache or multi-stage builds to reduce final image size.
4. Use `--manifest-path rust_connector/Cargo.toml` with maturin to avoid workspace detection issues.

Example Docker rebuild (after fixing requirements):
```bash
docker-compose down --remove-orphans
export COMPOSE_BAKE=true     # optional
docker-compose up --build
```



---

## 9. Development & testing

- Add unit tests for:
  - Rust functions (use maturin + pytest to import & test behavior)
  - Backtester invariants (PnL commutativity, monotonicity in some setups)
  - Strategy deterministic behavior on synthetic snapshots
- Suggested CI:
  - CI job that builds the Rust wheel using maturin and runs Python tests in a Linux runner
  - Use multi-stage builds: builder stage (has rust toolchain) produces wheel, runtime stage installs the wheel and python deps

Example test command:
```bash
python -m pytest tests/
```

---

## 10. Contributing & Code Style

- Please open issues for bugs / feature requests.
- PRs: fork -> branch -> PR; include tests where relevant.
- Rust style: `cargo fmt` and `cargo clippy`.
- Python style: follow Black (optional), include unit tests for new features.

Suggested files to add (if not present):
- LICENSE (MIT or Apache-2.0)
- CONTRIBUTING.md
- .github/workflows/ci.yml (build + test + lint)

---

## 11. References & useful links

- PyO3: https://pyo3.rs
- Maturin: https://github.com/PyO3/maturin
- tokio-tungstenite: async websocket client for tokio (used in examples)
- Streamlit: https://streamlit.io
- Plotly: https://plotly.com/python

---

## 12. Minimal example snippets

Build rust and install into interpreter:
```bash
python -m pip install --upgrade maturin
python -m maturin develop --manifest-path rust_connector/Cargo.toml --release
```

Run Streamlit:
```bash
streamlit run app/streamlit_app.py
```

Python usage example:
```python
from python.rust_bridge import parse_orderbook, compute_triangular_opportunity

ob_json = '{"bids":[[100.0,1]],"asks":[[100.1,1]]}'
ob = parse_orderbook(ob_json)
profit, route = compute_triangular_opportunity(ob, ob, ob)
print("profit", profit, "route", route)
```

## 13. Quickstart
### 13.a Local mode (recommended)
1. Prepare Python environment (conda or venv). Example with conda (recommended on macOS):
   conda create -n rhftlab python=3.11 -y
   conda activate rhftlab

2. Install Python build tooling and maturin:
   python -m pip install --upgrade pip setuptools wheel maturin

3. Install Python deps for UIs:
   python -m pip install -r docker/requirements.txt

4. Build & install Rust extension (manifest-path avoids workspace ambiguity):
   python -m maturin develop --manifest-path rust_connector/Cargo.toml --release

   Note: ensure you use the same Python interpreter to build and run (the one activated in your shell). If you get linker/_Py symbols errors, use the same interpreter for maturin (see README section "Python/Rust ABI").

5. Run the Streamlit app:
   streamlit run app/streamlit_app.py

### 13.b Docker (reproducible dev image)

Docker build & run:
```bash
export COMPOSE_BAKE=true      # optional
docker-compose up --build
# open http://localhost:8501
```

1. Build and run:
   export COMPOSE_BAKE=true
   docker-compose up --build

2. Open: http://localhost:8501

Design and usage notes
- Streaming in Rust:
  - Connectors spawn background tasks in Tokio (pyo3-asyncio enables safe spawning).
  - Each connector maintains an in-memory snapshot (Arc<Mutex<Option<OrderBook>>>).
  - On each incoming update the Rust task updates the snapshot and calls the Python callback under the GIL with a new OrderBook pyobject.
  - This pattern is robust for UIs (Streamlit) and notebooks: Python receives a consistent snapshot and can process/update state safely.

- CEX vs DEX arbitrage:
  - CEX: we use WS or REST top-of-book snapshots (Binance/coinbase examples).
  - DEX: we read pair reserves via JSON‑RPC (ethers provider), compute implied price, and estimate swap impact with Uniswap formula (example helper provided).
  - compute_dex_cex_arbitrage implemented in Rust for speed; python bridge prefers Rust but falls back to Python implementation.

- Symbols and discovery:
  - For CEX, the Streamlit UI uses connector.list_symbols() (initial defaults provided). You can extend the Rust connector to fetch dynamic symbol lists via REST endpoints and return them to Python.
  - For Uniswap, discover pair addresses using The Graph or pre-populated configuration.

Automated market data collection (Streamlit)
- The Streamlit app includes an "Auto-collect" option: it repeatedly fetches snapshots and stores top-of-book points in session state, with adjustable interval.
- collected data can be visualized, inspected, and used to compute quick DEX↔CEX arbitrage opportunities.
- For heavier collection and long-term storage, replace in-memory session collection with a lightweight local database (SQLite) or time-series store (InfluxDB/Prometheus) and persist snapshots to disk.

Building notes & dependency tips
- ethers 2.x: the crate is split across subcrates (ethers-core, ethers-providers, ethers-contract). Cargo.toml in this repo uses explicit subcrates to import contract/provider types.
- If cargo/maturin complains about features or versions:
  - ensure Cargo.toml dependencies match available crate features (we use 2.0.14 in the example).
  - run `python -m maturin develop --manifest-path rust_connector/Cargo.toml --release` from the activated Python environment.
- macOS: to avoid PyO3 linker issues, prefer using the Docker image for builds if you experience ABI mismatch errors.
- If receiving warnings about workspace resolver, you may add `workspace.resolver = "2"` to root Cargo.toml if your workspace crates use edition 2021 — this is informational.

Extending for production
- Maintain full orderbook state in Rust per symbol and apply incremental diffs (Binance depthUpdate or Coinbase l2update) instead of reconstructing from snapshot messages.
- Add reconnection/backoff, rate-limits, metrics, and health checks to connectors.
- Implement task registry (start_stream returns a task id; stop_stream(task_id) cancels it).
- Add signing and authenticated endpoints for exchanges when you need order placement (keep keys out of repo — use secrets/vault).
- For on‑chain execution of arbitrage, integrate a safe relay/MEV path (Flashbots, bundle signing, or flash swap onchain), and simulate gas costs & slippage.

Examples
- Python interactive usage:
```py
import python.rust_bridge as bridge
bridge.list_connectors()
c = bridge.get_connector("binance")
c.list_symbols()
ob = c.fetch_orderbook_sync("BTCUSDT")
print(ob.top())
def cb(ob):
    print("update", ob.top())
c.start_stream("BTCUSDT", cb)
```

- Uniswap reserves:
```py
from python.rust_bridge import get_connector
# or call rust_connector.uniswap_get_reserves directly if extension installed
```
