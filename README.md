
# rust-hft-arbitrage-lab

Unified repository with **dual Docker setup** (toggle `tick` on/off), mock exchanges (FastAPI REST+WS), Rust core with Python bindings, example notebooks, and GitHub Actions.

## Quickstart (macOS 14 Intel i9)
- **Stable (no C++ builds)**:
  ```bash
  ./dev.sh up
  # or: make up
  ```
- **With `tick` (compiles C++ ext)**:
  ```bash
  TICK=1 ./dev.sh up
  # or: make TICK=1 up
  ```

Jupyter → http://localhost:8888  
Mock API → http://localhost:8000/health

## Structure
- `Dockerfile.no-tick` • Python 3.11, no C++ compiles
- `Dockerfile.tick`    • Python 3.10 multi-stage, builds wheel for `tick==0.6.0.0`
- `docker-compose.yml` + overrides `docker-compose.no-tick.yml` / `docker-compose.tick.yml`
- `mock_apis/` • FastAPI server + WS streaming, with mock data
- `rust_core/` • Rust library stubs (orderbook, matching engine, strategies)
- `rust_python_bindings/` • PyO3 module `hft_py`
- `python_client/` • backtesting/execution scaffolding
- `examples/notebooks/` • executable notebooks (imbalance MM, pairs, triangular, hawkes, price discovery, hedging, signature OS)
- `.github/workflows/` • CI + Release

