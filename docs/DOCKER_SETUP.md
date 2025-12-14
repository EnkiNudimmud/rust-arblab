# Docker Configuration for HFT Arbitrage Lab

## Architecture

The Docker setup includes three main services:

### 1. **gRPC Server** (`hft-grpc-server`)
- **Purpose**: High-performance pair discovery backend in Rust
- **Port**: 50051
- **Features**:
  - Parallel pair testing with Tokio + Rayon
  - Optimal control via HJB PDE solver
  - OU parameter estimation
  - Cointegration testing (Engle-Granger)
  - Hurst exponent calculation
  - Strategy backtesting

### 2. **Streamlit Lab** (`lab`)
- **Purpose**: Main web interface
- **Port**: 8501
- **Features**:
  - Real-time market data visualization
  - Strategy configuration
  - Portfolio management
  - Connects to gRPC backend

### 3. **Jupyter Notebook** (`jupyter`)
- **Purpose**: Interactive data analysis and research
- **Port**: 8889
- **Features**:
  - Full access to `optimizr` library (Rust optimization algorithms)
  - gRPC client for pair discovery
  - Example notebooks included

## Quick Start

### Build and Run

```bash
# Using Makefile (recommended)
make docker-build
make docker-up

# Or manually with Docker Compose
DOCKER_BUILDKIT=1 docker compose build --parallel
docker compose up
```

### Run in Background

```bash
make docker-up-d
```

### View Logs

```bash
make docker-logs
```

### Stop Services

```bash
make docker-down
```

## Access Services

Once running, access:

- **Streamlit**: http://localhost:8501
- **Jupyter**: http://localhost:8889
- **gRPC Server**: localhost:50051 (internal, used by Python clients)

## Environment Variables

Configure in `docker-compose.yml`:

```yaml
environment:
  - GRPC_HOST=grpc-server      # gRPC server hostname
  - GRPC_PORT=50051            # gRPC server port
  - RUST_LOG=info              # Rust logging level
  - PYTHONPATH=/app            # Python module path
```

## Build Process

### Stage 1: Rust Builder
- Builds `hft-grpc-server` with gRPC support
- Compiles protobuf definitions
- Produces optimized release binary

### Stage 2: Python Builder
- Builds `optimizr` Python wheel (generic optimization library)
- Builds `rust_connector` wheel (HFT-specific bindings)
- Uses Maturin for PyO3 compilation

### Stage 3: Runtime
- Lightweight Python 3.11 image
- Installs all Python dependencies
- Installs compiled Rust wheels
- Generates gRPC Python stubs
- Copies pre-built binaries

## Volumes

```yaml
volumes:
  hft_data:         # Persistent data storage
  cargo-cache:      # Rust build cache
  target-cache:     # Compiled Rust artifacts
```

## Using the gRPC Client

### In Jupyter Notebooks

```python
from python.pair_discovery_client import PairDiscoveryClient

# Connect to gRPC server
client = PairDiscoveryClient()  # Uses GRPC_HOST/GRPC_PORT env vars

# Test a single pair
result = client.test_pair(prices_x, prices_y, pair_name="BTC-ETH")
print(f"Cointegrated: {result['is_cointegrated']}")
print(f"P-value: {result['p_value']}")

# Discover pairs in streaming mode
for pair_result in client.discover_pairs(price_matrix, pair_names):
    print(f"Found: {pair_result['pair_name']}")
```

### In Streamlit App

```python
import os
from python.pair_discovery_client import PairDiscoveryClient

# Connect using environment variables
grpc_host = os.getenv('GRPC_HOST', 'localhost')
grpc_port = int(os.getenv('GRPC_PORT', '50051'))
client = PairDiscoveryClient(host=grpc_host, port=grpc_port)
```

## Example Notebooks

See `examples/notebooks/grpc_pair_discovery_demo.ipynb` for comprehensive examples:

1. Testing single pairs
2. Streaming pair discovery
3. OU parameter estimation
4. HJB PDE solving
5. Strategy backtesting

## Rebuilding

### Full Rebuild

```bash
make docker-clean  # Remove all Docker artifacts
make docker-build  # Rebuild from scratch
```

### Rebuild Specific Service

```bash
docker compose build grpc-server  # Rebuild gRPC server
docker compose build jupyter      # Rebuild Jupyter
```

## Troubleshooting

### gRPC Connection Issues

Check if the server is running:
```bash
docker ps | grep hft-grpc-server
docker logs hft-grpc-server
```

Test connectivity:
```bash
docker exec -it hft-jupyter nc -zv grpc-server 50051
```

### Import Errors

Verify packages are installed:
```bash
docker exec -it hft-jupyter python -c "import optimizr; print('optimizr:', optimizr.__version__)"
docker exec -it hft-jupyter python -c "from python import pair_discovery_pb2; print('protobuf: OK')"
```

### Build Failures

Check build logs:
```bash
docker compose build --no-cache --progress=plain
```

## Performance Notes

- **gRPC**: Unary calls have ~1ms overhead, streaming is more efficient for batch operations
- **Parallel Processing**: The server uses all available CPU cores via Rayon
- **Memory**: Price matrices are transmitted via protobuf, consider batching for large datasets
- **Caching**: Cargo and target directories are cached between builds

## Development

### Generate Protobuf Stubs

```bash
make proto
# Or manually:
bash scripts/generate_proto.sh
```

### Local Testing (Without Docker)

```bash
# Terminal 1: Start gRPC server
cd hft-grpc-server
cargo run --release

# Terminal 2: Test client
cd ..
python -c "from python.pair_discovery_client import PairDiscoveryClient; print(PairDiscoveryClient().test_pair([1,2,3], [2,4,6]))"
```

## Next Steps

1. Review `examples/notebooks/grpc_pair_discovery_demo.ipynb`
2. Implement custom strategies using the gRPC client
3. Extend the protobuf schema for additional methods
4. Add authentication/TLS for production deployments
