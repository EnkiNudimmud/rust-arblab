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
- **Limit Order Book (LOB)**: Real-time orderbook analytics and visualization

### Technology Stack
- 🦀 **Rust Core**: High-performance numerical computation (10-100× speedup)
- 🐍 **Python**: Research, backtesting, and strategy development
- 🔗 **PyO3 Bindings**: Seamless Rust-Python integration
- 📊 **Streamlit Dashboard**: Interactive multi-strategy visualization
- 📈 **Plotly Charts**: Professional-grade financial visualizations
- 🐳 **Docker**: Reproducible deployment across platforms

### Production Features
- ✅ Real-time WebSocket market data (Kraken, Finnhub)
- ✅ **Limit Order Book (LOB) Analytics**: Rust-powered orderbook processing (10-100× faster)
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
streamlit run app/HFT_Arbitrage_Lab.py

# OR start Jupyter for notebooks
jupyter notebook examples/notebooks/
```

### 🔄 Development Workflow

#### Quick Restart Scripts

| Script | Purpose | Use Case |
|--------|---------|----------|
| `./scripts/restart_all.sh` | **Everything** (Rust + Streamlit + Jupyter) | After any code changes |
| `./scripts/restart_all.sh --quick` | Quick incremental Rust + all services | Fast iteration on Rust code |
| `./scripts/restart_all.sh --skip-rust` | Python services only | After Python-only changes |
| `./scripts/restart_rust.sh` | Full Rust rebuild with verification | Major Rust changes |
| `./scripts/quick_rust_build.sh` | Fast incremental Rust build | Minor Rust tweaks |
| `./scripts/clean_restart_streamlit.sh` | Streamlit only with cache clear | UI/Python changes |

#### Usage Examples

```bash
# Scenario 1: Modified Rust strategy implementation
./scripts/restart_all.sh                    # Rebuilds Rust + restarts all services

# Scenario 2: Quick fix in Rust code
./scripts/restart_all.sh --quick            # Incremental build (30s-2min) + restart

# Scenario 3: Modified Python strategy or Streamlit UI
./scripts/restart_all.sh --skip-rust        # No Rust rebuild, just restart services
# OR
./scripts/clean_restart_streamlit.sh        # Even faster, Streamlit only

# Scenario 4: Added new Rust connector
./scripts/restart_rust.sh                   # Full Rust rebuild with verification
# Then manually restart Streamlit when ready

# Scenario 5: Jupyter notebook work only
# No restart needed! Just refresh kernel if using rust_connector
```

#### What You Get

When running `./scripts/restart_all.sh`, the script:

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
./scripts/restart_all.sh --help

# Combine flags
./scripts/restart_all.sh --quick --skip-jupyter    # Quick Rust + Streamlit only
```

## 📊 Dashboard

The Streamlit dashboard provides an interactive interface for:

- **Multi-Strategy Analysis**: Compare Mean Reversion, Pairs Trading, Triangular Arbitrage, and Market Making
- **Rough Heston Models**: Advanced volatility modeling with leverage swaps and ATM skew
- **Real-Time Backtesting**: Instant parameter updates and performance visualization
- **Risk Metrics**: Sharpe ratio, max drawdown, VaR, and more
- **Mathematical Theory**: LaTeX equations and strategy explanations
- **Limit Order Book**: Live orderbook visualization with depth charts and analytics

**Access**: http://localhost:8501 after running the application

## 📖 Limit Order Book (LOB) Feature

### Overview

The LOB feature provides **high-performance orderbook recording and analytics** powered by Rust. Inspired by [pfei-sa/binance-LOB](https://github.com/pfei-sa/binance-LOB), this implementation captures multi-level orderbook data from Binance and other exchanges with **10-100× performance improvement** over pure Python.

### Architecture

```
┌─────────────────────┐
│  Exchange API/WS    │  (Binance, Kraken, etc.)
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Rust LOB Engine    │  ← Core Processing (rust_connector/src/lob.rs)
│  • Snapshot capture │     • Zero-copy data structures
│  • Diff updates     │     • O(log n) BTreeMap updates
│  • Analytics (20+)  │     • 20+ metrics calculation
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Python Wrapper     │  ← LOB Recorder (python/lob_recorder.py)
│  • Recording        │     • Persistence to disk
│  • Export to CSV    │     • DataFrame conversion
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Streamlit UI       │  ← Visualization (app/pages/live_trading.py)
│  • Depth charts     │     • 4 interactive tabs
│  • Heatmaps         │     • Real-time updates
│  • Time series      │     • Export functionality
└─────────────────────┘
```

### Features

#### 🎯 Core Capabilities
- **Multi-level Orderbook Capture**: Up to 100 price levels (configurable)
- **Snapshot & Differential Updates**: Full snapshots + efficient diff streams
- **Real-time Analytics**: 20+ metrics calculated in Rust
- **Persistent Storage**: JSONL format for historical analysis
- **CSV/JSON Export**: Easy integration with analysis tools

#### 📊 Analytics Metrics

| Category | Metrics |
|----------|---------|
| **Spread** | Absolute spread, spread in bps, mid-price |
| **Depth** | Volume at 0.1%, 0.5%, 1.0% from best bid/ask |
| **Imbalance** | Volume imbalance, price-weighted imbalance, depth imbalance |
| **Liquidity** | Effective spread, market impact for $10k order |
| **Book Shape** | Number of levels, total volumes |

### Usage

#### 1. Start LOB Recording

Navigate to **Live Trading** page in the Streamlit dashboard:

```python
# In the dashboard:
1. Select exchange (Binance/Kraken)
2. Choose trading pair (BTC/USD, ETH/USD, etc.)
3. Click "Start Live Trading"
4. Navigate to "📖 Limit Order Book" tab
```

#### 2. Programmatic Usage

```python
from python.lob_recorder import LOBRecorder, parse_binance_orderbook_py
from rust_connector import calculate_lob_analytics

# Initialize recorder
lob = LOBRecorder(
    symbols=['BTCUSDT'],
    max_levels=20,           # Number of price levels
    snapshot_interval=60,    # Snapshot every 60 seconds
    storage_path='./data/lob'
)

# Parse orderbook from exchange API
orderbook_data = {
    'lastUpdateId': 123456,
    'bids': [['50000.00', '1.5'], ['49999.50', '2.0']],
    'asks': [['50001.00', '1.2'], ['50002.50', '2.5']]
}
snapshot = parse_binance_orderbook_py(orderbook_data, 'BTCUSDT', 'binance')

# Record snapshot
lob.record_snapshot('BTCUSDT', snapshot)

# Calculate analytics (Rust implementation)
analytics = calculate_lob_analytics(snapshot)

print(f"Spread: {analytics.spread_bps:.2f} bps")
print(f"Mid Price: ${analytics.mid_price:.2f}")
print(f"Volume Imbalance: {analytics.volume_imbalance:.4f}")
print(f"Bid Depth (0.1%): ${analytics.bid_depth_1:.2f}")

# Export to DataFrame for analysis
df = lob.export_to_csv('BTCUSDT')
df.to_csv('orderbook_analytics.csv')
```

#### 3. Real-time WebSocket Integration

```python
import websocket
import json
from python.lob_recorder import LOBRecorder, parse_binance_diff_depth

lob = LOBRecorder(symbols=['BTCUSDT'])

def on_message(ws, message):
    data = json.loads(message)
    
    if data.get('e') == 'depthUpdate':
        # Parse differential update
        update = parse_binance_diff_depth(data, 'BTCUSDT')
        
        # Apply update (Rust implementation)
        lob.record_update('BTCUSDT', update)
        
        # Get latest analytics
        current_book = lob.get_current_book('BTCUSDT')
        if current_book:
            analytics = calculate_lob_analytics(current_book)
            print(f"Spread: {analytics.spread_bps:.2f} bps")

# Connect to Binance WebSocket
ws = websocket.WebSocketApp(
    "wss://stream.binance.com:9443/ws/btcusdt@depth@100ms",
    on_message=on_message
)
ws.run_forever()
```

### Performance Benchmarks

| Operation | Python | Rust | Speedup |
|-----------|--------|------|---------|
| Snapshot parsing | 1.2ms | 0.05ms | **24×** |
| Analytics calculation | 3.5ms | 0.08ms | **43×** |
| Diff update application | 0.8ms | 0.02ms | **40×** |
| 1000 updates/sec | ❌ | ✅ | **Feasible** |

### Visualization

The Live Trading dashboard provides 4 interactive tabs:

1. **📊 Orderbook Levels**: Real-time bid/ask table with depth visualization
2. **📈 Analytics Time Series**: Spread, imbalance, and market impact over time
3. **🔥 Orderbook Heatmap**: Price level intensity visualization
4. **💾 Export Data**: Download orderbook snapshots and analytics

### Data Storage

LOB data is stored in JSONL format for efficient append operations:

```
data/lob/
├── BTCUSDT_20241130_snapshots.jsonl    # Full snapshots
└── BTCUSDT_20241130_analytics.csv      # Computed analytics
```

Each line in the JSONL file is a complete orderbook snapshot:
```json
{
  "timestamp": "2024-11-30T12:34:56.789Z",
  "symbol": "BTCUSDT",
  "last_update_id": 123456,
  "exchange": "binance",
  "bids": [[50000.0, 1.5], [49999.5, 2.0]],
  "asks": [[50001.0, 1.2], [50002.5, 2.5]]
}
```

### Configuration

Configure LOB behavior in your code:

```python
lob = LOBRecorder(
    symbols=['BTCUSDT', 'ETHUSDT'],      # Multiple symbols
    max_levels=50,                        # Capture 50 price levels
    snapshot_interval=30,                 # Snapshot every 30 seconds
    storage_path='./data/lob',           # Storage directory
    enable_persistence=True               # Auto-save to disk
)
```

### Advanced: Rust Implementation Details

The Rust implementation uses efficient data structures:

```rust
// Core data structures (rust_connector/src/lob.rs)
#[pyclass]
pub struct OrderBookSnapshot {
    timestamp: String,              // ISO 8601 timestamp
    symbol: String,
    last_update_id: u64,
    exchange: String,
    bids: Vec<(f64, f64)>,         // (price, quantity) tuples
    asks: Vec<(f64, f64)>,
}

// Analytics with 20+ metrics
#[pyclass]
pub struct LOBAnalytics {
    // Spread metrics
    spread_abs: f64,
    spread_bps: f64,
    mid_price: f64,
    
    // Depth at 0.1%, 0.5%, 1.0%
    bid_depth_1: f64,
    ask_depth_1: f64,
    // ... more metrics
}

// Efficient update application with BTreeMap
pub fn apply_orderbook_update(
    snapshot: &OrderBookSnapshot,
    update: &OrderBookUpdate,
    max_levels: usize,
) -> PyResult<OrderBookSnapshot>
```

**Key optimizations:**
- **BTreeMap** for O(log n) price level updates
- **Zero-copy** where possible with references
- **SIMD-friendly** data layout for vectorization
- **Cache-efficient** contiguous memory for price levels

### Troubleshooting

**Issue: "Rust LOB module not available"**
```bash
# Rebuild Rust connector
cd rust_connector
maturin develop --release
```

**Issue: WebSocket connection fails**
```python
# Check exchange connectivity
import ccxt
exchange = ccxt.binance()
exchange.fetch_order_book('BTC/USDT', limit=20)
```

**Issue: High memory usage**
```python
# Reduce max_levels or snapshot_interval
lob = LOBRecorder(max_levels=10, snapshot_interval=120)
```

### Related Documentation

- [RUST_LOB_IMPLEMENTATION.md](docs/RUST_LOB_IMPLEMENTATION.md) - Technical details
- [Kraken WebSocket Guide](docs/KRAKEN_WEBSOCKET_GUIDE.md) - Exchange integration
- [binance-LOB Project](https://github.com/pfei-sa/binance-LOB) - Inspiration

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
