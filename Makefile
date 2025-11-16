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

# Build Rust connector
build:
	@echo "Building Rust connector..."
	$(MATURIN) develop --manifest-path rust_connector/Cargo.toml --release
	@echo "✓ Rust connector built"

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
	$(PY) -m streamlit run app/streamlit_app.py

# Run tests
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
	@$(PY) -c "import rust_connector; print('✓ rust_connector installed')" || echo "✗ rust_connector NOT installed"
	@$(PY) -c "from python.rust_bridge import list_connectors; print('✓ Connectors:', ', '.join(list_connectors()))"

# Docker build
docker-build:
	@echo "Building Docker image..."
	docker compose build

# Docker up
docker-up:
	@echo "Starting Docker services..."
	@echo "Access Streamlit at: http://localhost:8501"
	docker compose up

# Docker down
docker-down:
	@echo "Stopping Docker services..."
	docker compose down

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