# rust-hft-arbitrage-lab-enriched

A modular research / prototype repository for high-frequency-trading (HFT) market data connectors, a lightweight aggregator, signature-based optimal stopping algorithms, and Python bindings / notebooks for experimentation. This project is structured as a Cargo workspace with Rust crates and PyO3 bindings for Python interoperability.

This README explains the repository layout, how to build and run components, how to supply API keys for exchange connectors, Docker examples (tick / no-tick modes), and pointers to the notebooks and Python bindings.

---

## Table of contents

- Project overview
- Repository layout
- Requirements
- Quickstart (build & run)
- Connectors and API key setup
- Signature optimal stopping module
- Python bindings and notebooks
- Docker (tick / no-tick compose)
- Development workflow
- Tests and CI guidance
- License and contributions

---

## Project overview

This repository contains:

- Exchange connectors (Binance, Kraken, Coinbase, CoinGecko) implemented as individual Rust crates.
- A central `aggregator` crate that consumes connector streams and exposes a unified tick event API.
- A `signature_optimal_stopping` crate implementing truncated-signature features and a simple ridge-based continuation value approximator.
- PyO3-based Python bindings (`signature_optimal_stopping_py` and `rust_python_bindings`) exposing Rust logic to Python/Jupyter.
- Example Jupyter notebooks under `examples/notebooks/` demonstrating triangular arbitrage and the signature optimal stopping workflow.
- Docker Compose examples to run the stack locally with or without simulated tick traffic.

---

## Repository layout

Top-level layout (important files and directories):

- Cargo.toml                # workspace manifest
- Makefile                 # common build / docker helpers
- README.md                # this file
- examples/
  - notebooks/             # ready-to-paste ipynb examples
- rust_core/
  - connectors/
    - common/
    - binance/
    - kraken/
    - coinbase/
    - coingecko/
  - aggregator/
  - signature_optimal_stopping/
  - signature_optimal_stopping_py/
- rust_python_bindings/    # PyO3 "hft_py" extension crate
- docker/
  - docker-compose.tick.yml
  - docker-compose.notick.yml

---

## Requirements

- Rust toolchain (rustup recommended) — stable channel
- Cargo
- Optional but recommended:
  - maturin (for building/installing PyO3 wheels): pip install maturin
  - Docker and Docker Compose v2 (for example compose files)
  - Python 3.8+ with virtualenv / venv for notebooks
  - OpenBLAS (only if `ndarray-linalg` with openblas-static is used)

---

## Quickstart

1. Clone and enter repository:
