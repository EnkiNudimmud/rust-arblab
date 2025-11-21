# Rust HFT Arbitrage Lab

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Rust](https://img.shields.io/badge/rust-stable-orange.svg)](https://www.rust-lang.org/)

> **A high-performance research platform for quantitative trading strategies combining Rust and Python**

A modular, production-ready framework for high-frequency trading (HFT) and arbitrage research. This project combines Rust's performance for low-latency computation with Python's flexibility for research, backtesting, and visualization.

## 🎯 Key Features

### Trading Strategies
- **Mean Reversion**: CARA utility optimization, multi-period strategies, Sharpe maximization
- **Pairs Trading**: Cointegration-based statistical arbitrage
- **Triangular Arbitrage**: Cross-exchange opportunity detection
- **Market Making**: Spread capture and inventory management
- **Rough Heston**: Advanced volatility modeling with affine structures
- **Chiarella Model**: Economic-based price dynamics

### Technology Stack
- 🦀 **Rust Core**: High-performance numerical computation (10-100× speedup)
- 🐍 **Python**: Research, backtesting, and strategy development
- 🔗 **PyO3 Bindings**: Seamless Rust-Python integration
- 📊 **Streamlit Dashboard**: Interactive multi-strategy visualization
- 📈 **Plotly Charts**: Professional-grade financial visualizations
- 🐳 **Docker**: Reproducible deployment across platforms

### Production Features
- ✅ Real-time WebSocket market data (Kraken, Finnhub)
- ✅ Transaction cost modeling and slippage
- ✅ Comprehensive risk metrics (Sharpe, Sortino, Max Drawdown, VaR)
- ✅ Multi-strategy portfolio construction
- ✅ Jupyter notebooks for research and education
- ✅ Extensive test suite with CI/CD ready structure

## 🚀 Quick Start

### Prerequisites
- **Python**: 3.11+ 
- **Rust**: Latest stable ([Install rustup](https://rustup.rs/))
- **Docker** (optional): For containerized deployment

### Installation

#### Option 1: Docker (Recommended)
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/rust-hft-arbitrage-lab.git
cd rust-hft-arbitrage-lab

# Build and start services
docker compose up --build

# Access dashboard at http://localhost:8501
```

#### Option 2: Local Development

**1. Setup Environment**
```bash
# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip setuptools wheel maturin
pip install -r docker/requirements.txt
```

**2. Build Rust Components**
```bash
# Build Rust-Python bindings
maturin develop --manifest-path rust_connector/Cargo.toml --release
```

**3. Configure API Keys**
```bash
# Copy example config
cp api_keys.properties.example api_keys.properties

# Edit with your API keys (optional - falls back to synthetic data)
nano api_keys.properties
```

**4. Run Application**
```bash
# Quick start (recommended)
./run_app.sh

# OR manually start Streamlit dashboard
streamlit run app/streamlit_all_strategies.py

# OR start Jupyter for notebooks
jupyter notebook examples/notebooks/
```

### 🔄 Development Workflow

#### Quick Restart Scripts

| Script | Purpose | Use Case |
|--------|---------|----------|
| `./restart_all.sh` | **Everything** (Rust + Streamlit + Jupyter) | After any code changes |
| `./restart_all.sh --quick` | Quick incremental Rust + all services | Fast iteration on Rust code |
| `./restart_all.sh --skip-rust` | Python services only | After Python-only changes |
| `./restart_rust.sh` | Full Rust rebuild with verification | Major Rust changes |
| `./quick_rust_build.sh` | Fast incremental Rust build | Minor Rust tweaks |
| `./clean_restart_streamlit.sh` | Streamlit only with cache clear | UI/Python changes |

#### Usage Examples

```bash
# Scenario 1: Modified Rust strategy implementation
./restart_all.sh                    # Rebuilds Rust + restarts all services

# Scenario 2: Quick fix in Rust code
./restart_all.sh --quick            # Incremental build (30s-2min) + restart

# Scenario 3: Modified Python strategy or Streamlit UI
./restart_all.sh --skip-rust        # No Rust rebuild, just restart services
# OR
./clean_restart_streamlit.sh        # Even faster, Streamlit only

# Scenario 4: Added new Rust connector
./restart_rust.sh                   # Full Rust rebuild with verification
# Then manually restart Streamlit when ready

# Scenario 5: Jupyter notebook work only
# No restart needed! Just refresh kernel if using rust_connector
```

#### What You Get

When running `./restart_all.sh`, the script:

1. **🐍 Activates virtual environment** - Ensures correct Python/dependencies
2. **🦀 Rebuilds Rust engine** - Compiles optimized performance code
   - Full build: 5-10 minutes first time, ~2-3 minutes subsequent
   - Quick build: 30 seconds - 2 minutes (incremental)
3. **🛑 Stops running services** - Clean shutdown of Streamlit/Jupyter
4. **🧹 Clears all caches** - Fresh state (Streamlit cache, Python `__pycache__`)
5. **🚀 Starts services** - Launches Streamlit at http://localhost:8501
6. **✅ Verifies status** - Shows what's running and performance gains

**Performance gains after Rust rebuild:**
- PCA computation: **10-100× faster**
- Matrix operations: **5-50× faster**
- Portfolio backtesting: **20-200× faster**
- WebSocket processing: **2-10× faster**

#### Interactive Options

```bash
# Show help and all options
./restart_all.sh --help

# Combine flags
./restart_all.sh --quick --skip-jupyter    # Quick Rust + Streamlit only
```

## 📊 Dashboard

The Streamlit dashboard provides an interactive interface for:

- **Multi-Strategy Analysis**: Compare Mean Reversion, Pairs Trading, Triangular Arbitrage, and Market Making
- **Rough Heston Models**: Advanced volatility modeling with leverage swaps and ATM skew
- **Real-Time Backtesting**: Instant parameter updates and performance visualization
- **Risk Metrics**: Sharpe ratio, max drawdown, VaR, and more
- **Mathematical Theory**: LaTeX equations and strategy explanations

**Access**: http://localhost:8501 after running the application

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_rust_analytics.py

# Run with coverage
pytest tests/ --cov=python --cov-report=html
```

## 📚 Documentation

Comprehensive documentation is available in the [`docs/`](docs/) directory:

- [Environment Setup](docs/ENVIRONMENT_SETUP.md)
- [Multi-Strategy Guide](docs/MULTI_STRATEGY_GUIDE.md)
- [Rough Heston Guide](docs/ROUGH_HESTON_GUIDE.md)
- [Kraken WebSocket Guide](docs/KRAKEN_WEBSOCKET_GUIDE.md)
- [Quick Reference](docs/QUICK_REFERENCE.md)

See [`docs/README.md`](docs/README.md) for the complete documentation index.

## 🏗️ Project Structure

```
rust-hft-arbitrage-lab/
├── app/                      # Streamlit dashboard
│   ├── pages/               # Strategy pages
│   │   ├── affine_models.py
│   │   └── ...
│   ├── utils/               # UI utilities
│   └── streamlit_all_strategies.py
├── python/                   # Python implementations
│   ├── rough_heston.py      # Rough Heston models
│   ├── data_fetcher.py      # Market data
│   ├── meanrev.py           # Mean reversion
│   └── ...
├── rust_core/               # Rust core library
│   └── src/
│       ├── rough_heston.rs
│       └── ...
├── rust_python_bindings/    # PyO3 bindings
├── rust_connector/          # Exchange connectors
├── tests/                   # Test suite
├── examples/                # Jupyter notebooks
├── docs/                    # Documentation
└── docker/                  # Docker configuration
```

## ⚙️ Configuration

### API Keys (Optional)

Create `api_keys.properties` for real market data:

```properties
# Market Data
finnhub.api_key=your_key_here

# Exchange APIs (for live trading)
kraken.api_key=your_key_here
kraken.api_secret=your_secret_here
```

See `api_keys.properties.example` for the complete template.

**Note**: The system falls back to synthetic data if API keys are not provided.

## 🧰 Development Commands

```bash
# Rebuild Rust components
maturin develop --manifest-path rust_connector/Cargo.toml --release

# Clean build
cargo clean && maturin develop --release

# Run tests
pytest tests/

# Format code
black python/
cargo fmt

# Lint
pylint python/
cargo clippy

```

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Code Style

- **Python**: Follow PEP 8, use `black` for formatting
- **Rust**: Follow Rust style guidelines, use `cargo fmt`
- **Tests**: Add tests for new features
- **Documentation**: Update docs for API changes

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
