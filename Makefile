# Top-level Makefile for rust-hft-arbitrage-lab-enriched
# Usage: make <target>
# Common targets: build, clean, test, fmt, clippy, docs, docker-up, docker-down

REPO_ROOT := $(shell pwd)
CARGO := cargo
MATURIN := maturin

.PHONY: all build clean fmt clippy test release docs \
        py-build py-install wheels \
        docker-up docker-down docker-logs \
        notebooks

all: build

# Rust workspace build
build:
    @echo "==> cargo build --workspace"
    $(CARGO) build --workspace

# Release build for production binaries / wheels
release:
    @echo "==> cargo build --workspace --release"
    $(CARGO) build --workspace --release

clean:
    @echo "==> cargo clean"
    $(CARGO) clean
    @rm -rf target/wheels || true

fmt:
    @echo "==> cargo fmt --all"
    $(CARGO) fmt --all

clippy:
    @echo "==> cargo clippy --all -- -D warnings"
    $(CARGO) clippy --all -- -D warnings

test:
    @echo "==> cargo test --workspace"
    $(CARGO) test --workspace

docs:
    @echo "==> cargo doc --workspace"
    $(CARGO) doc --workspace --no-deps

# Python extension build (PyO3 crates)
py-build:
    @echo "==> Build Python wheel using maturin for signature_optimal_stopping_py"
    @test -x $(MATURIN) || (echo "maturin not found in PATH; install with pip install maturin" && exit 1)
    cd rust_core/signature_optimal_stopping_py && $(MATURIN) build --release

py-install:
    @echo "==> Install Python wheel locally (requires maturin build output)"
    cd rust_core/signature_optimal_stopping_py && $(MATURIN) develop --release

wheels:
    @echo "==> Build all PyO3 wheels (signature_optimal_stopping_py and rust_python_bindings if configured)"
    cd rust_core/signature_optimal_stopping_py && $(MATURIN) build --release
    # add other PyO3 crates building if needed

# Notebooks helper
notebooks:
    @echo "==> List notebooks"
    @ls -1 examples/notebooks || true

# Docker integration helpers
# Two sample docker-compose files are included in docker/ directory:
# - docker/docker-compose.tick.yml         (with market tick generation & connectors)
# - docker/docker-compose.notick.yml       (no tick generation, connectors inactive)
docker-up:
    @echo "==> docker compose up -d (tick)"
    @docker compose -f docker/docker-compose.tick.yml up -d

docker-down:
    @echo "==> docker compose down"
    @docker compose -f docker/docker-compose.tick.yml down || true
    @docker compose -f docker/docker-compose.notick.yml down || true

docker-logs:
    @echo "==> docker-compose logs -f"
    @docker compose -f docker/docker-compose.tick.yml logs -f

# Convenience: show workspace members
members:
    @echo "Workspace members:"
    @grep -A20 'members' Cargo.toml || true

# Default help
help:
    @printf "Makefile targets:\n"
    @printf "  make build        Build all Rust crates in the workspace\n"
    @printf "  make release      Release build\n"
    @printf "  make clean        Clean target directories\n"
    @printf "  make fmt          Run rustfmt on workspace\n"
    @printf "  make clippy       Run clippy with deny warnings\n"
    @printf "  make test         Run tests\n"
    @printf "  make docs         Generate docs\n"
    @printf "  make py-build     Build Python wheel(s) with maturin\n"
    @printf "  make py-install   Install Python extension in editable mode\n"
    @printf "  make docker-up    Bring up example docker-compose (tick)\n"
    @printf "  make docker-down  Tear down docker-compose\n"
    @printf "  make docker-logs  Follow docker logs\n"
    @printf "  make notebooks    List notebooks\n"
