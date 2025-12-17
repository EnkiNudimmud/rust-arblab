# Makefile for rust-hft-arbitrage-lab
# Quick commands for building and running the project

# Detect Python (prefer python3.11, fallback to python3)
PYTHON := $(shell command -v python3.11 2>/dev/null || command -v python3 2>/dev/null)
VENV = .venv
PY = $(VENV)/bin/python
PIP = $(VENV)/bin/pip
MATURIN = $(VENV)/bin/maturin

# Detect if running in a virtual environment
ifdef VIRTUAL_ENV
	PY = python
	PIP = pip
	MATURIN = maturin
endif

.PHONY: help setup venv install build run test clean docker-build docker-up docker-down rebuild verify

help:
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  rust-hft-arbitrage-lab - Quick Commands"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	@echo "🚀 Quick Start:"
	@echo "  make setup          Complete local setup (venv + deps + build)"
	@echo "  make docker-up      Run everything in Docker"
	@echo ""
	@echo "🔨 Build Commands:"
	@echo "  make venv           Create Python virtual environment"
	@echo "  make install        Install Python dependencies"
	@echo "  make build          Build Rust connector"
	@echo "  make rebuild        Clean and rebuild Rust connector"
	@echo ""
	@echo "▶️  Run Commands:"
	@echo "  make run            Start Streamlit app"
	@echo "  make test           Run WebSocket tests"
	@echo "  make jupyter        Start Jupyter notebook server"
	@echo ""
	@echo "🐳 Docker Commands:"
	@echo "  make docker-build   Build Docker image"
	@echo "  make docker-up      Start Docker services"
	@echo "  make docker-down    Stop Docker services"
	@echo "  make docker-clean   Remove all Docker artifacts"
	@echo ""
	@echo "🧹 Maintenance:"
	@echo "  make clean          Remove build artifacts"
	@echo "  make clean-all      Remove everything (venv, builds, cache)"
	@echo "  make verify         Check installation"
	@echo ""
	@echo "📚 Documentation:"
	@echo "  ./setup_env.sh          Interactive setup wizard"
	@echo "  QUICK_REFERENCE.md      Command cheatsheet"
	@echo "  ENVIRONMENT_SETUP.md    Complete setup guide"
	@echo ""

# Complete setup (local)
setup: venv install build verify
	@echo "✅ Setup complete! Run 'make run' to start Streamlit"

# Create virtual environment
venv:
ifndef VIRTUAL_ENV
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv $(VENV)
	@echo "✓ Virtual environment created at $(VENV)"
	@echo "  Activate with: source $(VENV)/bin/activate"
else
	@echo "Already in a virtual environment: $(VIRTUAL_ENV)"
endif

# Install Python dependencies
install:
	@echo "Installing Python dependencies..."
	$(PIP) install --upgrade pip setuptools wheel maturin
	$(PIP) install -r docker/requirements.txt
	@echo "✓ Python dependencies installed"

# Build Rust connector (DEPRECATED - use gRPC instead via docker-up)
build:
	@echo "⚠️  Building legacy PyO3 rust_connector (DEPRECATED)"
	@echo "   For production, use gRPC backend: make docker-up"
	$(MATURIN) develop --manifest-path rust_connector/Cargo.toml --release
	@echo "✓ Legacy Rust connector built (fallback only)"

# Rebuild Rust (clean first)
rebuild:
	@echo "Cleaning Rust build artifacts..."
	cargo clean
	@echo "Rebuilding Rust connector..."
	$(MATURIN) develop --manifest-path rust_connector/Cargo.toml --release
	@echo "✓ Rust connector rebuilt"

# Run Streamlit app
run:
	@echo "Starting Streamlit app..."
	@echo "Access at: http://localhost:8501"
	$(PY) -m streamlit run app/HFT_Arbitrage_Lab.py

# Backwards-compatible alias for older docs and scripts
.PHONY: run-app
run-app: run
	@echo "Alias: run-app -> run"

# Run Streamlit with auth disabled (standalone/demo mode)
.PHONY: run-standalone
run-standalone:
	@echo "Starting Streamlit app (standalone; ENABLE_AUTH=false)"
	ENABLE_AUTH=false $(MAKE) run

# Run Streamlit in background (convenience)
.PHONY: run-background
run-background:
	@echo "Starting Streamlit app in background (logs: streamlit.log)"
	nohup $(PY) -m streamlit run app/HFT_Arbitrage_Lab.py> streamlit.log 2>&1 &

# Run gRPC server (for development)
.PHONY: run-server
run-server:
	@echo "Starting gRPC server on localhost:50051..."
	@echo "Press Ctrl+C to stop"
	cd rust_grpc_service && cargo run --release 2>&1 || echo "Note: rust_grpc_service may not be configured. Using docker-up for full stack."

# Run smoke test client
.PHONY: smoke-test-client
smoke-test-client:
	@echo "Running gRPC smoke test..."
	$(PY) scripts/grpc_smoke_test.py
test:
	@echo "Running WebSocket tests..."
	$(PY) test_websocket.py

# Run Jupyter
jupyter:
	@echo "Starting Jupyter notebook server..."
	$(PY) -m jupyter notebook examples/notebooks/

# Verify installation
verify:
	@echo "Verifying installation..."
	@$(PY) --version
	@$(PY) -c "from python.rust_grpc_bridge import compute_pca_rust; print('✓ gRPC bridge available')" || echo "⚠ gRPC bridge not available (will use numpy/pandas fallbacks)"
	@$(PY) -c "from python.rust_bridge import list_connectors; print('✓ Connectors:', ', '.join(list_connectors()))"

# Generate protobuf code
proto:
	@echo "Generating Python gRPC stubs..."
	bash scripts/generate_proto.sh

# Docker build with optimizr
docker-build:
	@echo "Building Docker image with gRPC server and optimizr..."
	DOCKER_BUILDKIT=1 docker compose build --parallel

# Docker up
docker-up:
	@echo "Starting Docker services..."
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  Access services at:"
	@echo "    🌐 Streamlit:     http://localhost:8501"
	@echo "    📓 Jupyter:       http://localhost:8889"
	@echo "    ⚡ gRPC Server:   localhost:50051"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	docker compose up

# Docker up in detached mode
docker-up-d:
	@echo "Starting Docker services in detached mode..."
	docker compose up -d
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  Services running in background"
	@echo "    🌐 Streamlit:     http://localhost:8501"
	@echo "    📓 Jupyter:       http://localhost:8889"
	@echo "    ⚡ gRPC Server:   localhost:50051"
	@echo "  View logs: make docker-logs"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Docker down
docker-down:
	@echo "Stopping Docker services..."
	docker compose down

# Docker logs
docker-logs:
	@echo "Tailing Docker logs (Ctrl+C to exit)..."
	docker compose logs -f

# Docker clean rebuild
docker-clean:
	@echo "Cleaning Docker artifacts..."
	docker compose down --volumes --remove-orphans
	docker system prune -f
	@echo "Rebuilding Docker image..."
	docker compose build --no-cache

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build dist target *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "✓ Build artifacts cleaned"

# Clean everything including venv
clean-all: clean
	@echo "Removing virtual environment..."
	rm -rf $(VENV)
	@echo "Cleaning Rust targets..."
	cargo clean
	rm -rf rust_connector/target rust_core/target rust_python_bindings/target
	@echo "✓ Everything cleaned"