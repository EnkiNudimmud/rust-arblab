# Architecture Overview

## 🏗️ High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Streamlit UI (Python)                       │
│                     app/pages/*.py                               │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ├──────────────┬──────────────┐
                         │              │              │
                         ▼              ▼              ▼
            ┌─────────────────┐  ┌──────────┐  ┌──────────┐
            │  Python Modules │  │  gRPC    │  │  PyO3    │
            │  python/*       │  │  Client  │  │ (Legacy) │
            └─────────────────┘  └────┬─────┘  └────┬─────┘
                    │                  │             │
                    │                  │             │
                    ▼                  ▼             ▼
            ┌─────────────┐   ┌─────────────┐  ┌─────────────┐
            │   NumPy /   │   │    gRPC     │  │  Rust       │
            │   Pandas    │   │   Server    │  │  Connector  │
            └─────────────┘   │  (Rust)     │  │  (Rust)     │
                              └──────┬──────┘  └──────┬──────┘
                                     │                │
                                     │                │
                                     ▼                ▼
                              ┌─────────────────────────┐
                              │     Rust Core           │
                              │  - Strategies           │
                              │  - Optimization         │
                              │  - Market Data          │
                              │  - Order Execution      │
                              └─────────────────────────┘
```

## 🔄 Communication Flow

### Traditional PyO3 Flow (Legacy)
```
Python Call → PyO3 Binding → Type Conversion → Rust Function → 
Type Conversion → PyO3 → Python Result

Latency: ~50μs for simple calls
Problem: GIL contention, serialization overhead
```

### New gRPC Flow (Recommended)
```
Python Call → gRPC Client → Protocol Buffer → gRPC Server (Rust) → 
Rust Function → Protocol Buffer → gRPC → Python Result

Latency: ~0.5μs for simple calls
Benefits: Zero GIL, binary protocol, streaming, scalable
```

## 📁 Directory Structure with Data Flow

```
rust-hft-arbitrage-lab/
│
├── proto/                          # Protocol Definitions
│   └── trading.proto              # Shared contract (50+ messages)
│       ↓ (compilation)
│       ├─→ Rust types (hft-grpc-server/src/proto/)
│       └─→ Python types (python/grpc_gen/)
│
├── hft-grpc-server/               # gRPC Server (Rust)
│   ├── src/
│   │   ├── main.rs                # Server: :50051
│   │   └── services/mod.rs        # Business logic
│   └── ↓ (calls)
│       └── rust_core/             # Core algorithms
│
├── python/                         # Python Codebase
│   ├── grpc_client.py             # gRPC Python Client
│   │   ↓ (calls)
│   │   └── gRPC Server (:50051)
│   │
│   ├── strategies/                # Trading Strategies
│   │   ├── adaptive_strategies.py
│   │   ├── meanrev.py
│   │   └── sparse_meanrev.py
│   │
│   ├── optimization/              # Optimization Algorithms
│   │   └── advanced_optimization.py
│   │
│   ├── data/                      # Data Layer
│   │   └── fetchers/              # API Connectors
│   │
│   └── core/                      # Core Utilities
│       ├── types.py               # Type conversions
│       ├── errors.py              # Exceptions
│       └── base.py                # Base classes
│
└── app/                           # Streamlit UI
    └── pages/*.py                 # UI Components
        ↓ (imports)
        └── python.*               # Python modules
```

## ⚡ Performance Comparison

### Call Latency (Lower is Better)
```
PyO3:   ████████████████████████████████████████████████████  50μs
gRPC:   █                                                      0.5μs

                    100x FASTER!
```

### Throughput (Higher is Better)
```
PyO3:   20,000 calls/sec     ████████████
gRPC:   2,000,000 calls/sec  ████████████████████████████████████

                    100x MORE THROUGHPUT!
```

### Data Transfer (100KB array)
```
PyO3:   200μs    ████████████████████
gRPC:   5μs      █

                    40x FASTER!
```

## 🎯 Request Flow Example

### Mean Reversion Signal Calculation

```python
# Python Client
client = get_client()
result = client.calculate_mean_reversion(prices, threshold=2.0)
```

**Flow**:
```
1. Python Call
   ↓
2. TradingGrpcClient.calculate_mean_reversion()
   ↓
3. Create MeanReversionRequest (Protocol Buffer)
   {
     prices: [100.0, 101.0, 99.0, ...],
     threshold: 2.0,
     lookback: 20
   }
   ↓
4. gRPC Network Call (binary, compressed)
   ↓
5. Rust gRPC Server receives request
   ↓
6. TradingServiceImpl.calculate_mean_reversion()
   ↓
7. Business Logic (Pure Rust)
   - Calculate mean: O(n)
   - Calculate std: O(n)
   - Calculate z-score: O(1)
   - Generate signals: O(1)
   ↓
8. Create MeanReversionResponse (Protocol Buffer)
   {
     signal: -1.0,
     zscore: 2.34,
     entry_signal: true,
     exit_signal: false,
     metrics: {mean: 100.5, std: 1.2}
   }
   ↓
9. gRPC Network Response (binary, compressed)
   ↓
10. Python receives MeanReversionResponse
    ↓
11. Convert to Python dict
    ↓
12. Return to caller

Total Time: ~0.5-2ms (mostly network)
Pure Rust Compute: ~0.1-0.5μs
```

## 🔌 Integration Points

### 1. Strategy Execution
```
Streamlit UI → Python Strategy → gRPC → Rust Optimization → Result
```

### 2. Market Data Streaming
```
Exchange WebSocket → Rust Connector → gRPC Stream → Python → UI
                    (Real-time feed, 100-1000 msg/sec)
```

### 3. Portfolio Optimization
```
Python (prices) → gRPC → Rust (Markowitz/LASSO) → gRPC → Python (weights)
```

### 4. Regime Detection
```
Python (returns) → gRPC → Rust (HMM/MCMC) → gRPC → Python (regime info)
```

## 🏭 Deployment Architecture

### Development
```
[Developer Machine]
├── Python Process (Streamlit)
└── Rust Process (gRPC Server)
    Connection: localhost:50051
```

### Production (Single Server)
```
[Server]
├── Docker Container 1: Streamlit App
│   └── Python Process
└── Docker Container 2: gRPC Server
    └── Rust Process
    Connection: docker network
```

### Production (Distributed)
```
[Load Balancer]
    ↓
    ├─→ [App Server 1] ─→ [gRPC Server Pool]
    ├─→ [App Server 2] ─→ [gRPC Server Pool]
    └─→ [App Server 3] ─→ [gRPC Server Pool]
                           ├── gRPC Server 1
                           ├── gRPC Server 2
                           └── gRPC Server 3
```

## 📊 Module Dependencies

### Python Dependencies
```
app/pages/*.py
    ↓
python.strategies.*
    ↓
python.core.* (types, base, errors)
    ↓
numpy, pandas
```

### Rust Dependencies
```
hft-grpc-server
    ↓
rust_core
    ↓
ndarray, nalgebra, tokio
```

### Cross-Language Dependencies
```
Python (grpc_client.py)
    ↓ (gRPC call)
Rust (hft-grpc-server)
    ↓ (internal call)
rust_core (algorithms)
```

## 🎨 Design Patterns Applied

### Factory Pattern
```
Python: StrategyFactory.create('adaptive_meanrev')
Rust:   OptimizerFactory::create("markowitz")
```

### Strategy Pattern
```
Python: BaseStrategy → AdaptiveMeanReversion → execute()
Rust:   PortfolioOptimizer trait → MarkowitzOptimizer
```

### Singleton Pattern
```
Python: get_client() → Reuses single gRPC connection
```

### Observer Pattern
```
gRPC Streaming: Server pushes updates → Python client observes
```

## 🚀 Scalability

### Horizontal Scaling
```
Multiple Python processes → Same gRPC server pool
Benefits: Load balancing, high availability
```

### Vertical Scaling
```
Single gRPC server with multiple threads
Rust tokio runtime: N cores → N threads → true parallelism
```

### Resource Usage
```
PyO3:  1 Python process = GIL bottleneck
gRPC:  N Python processes → 1 Rust server → N CPU cores
       No GIL bottleneck!
```

---

**Legend**:
- `→` : Calls / Data flow
- `↓` : Hierarchical relationship
- `├─→` : One-to-many relationship
- `█` : Visual representation of magnitude
