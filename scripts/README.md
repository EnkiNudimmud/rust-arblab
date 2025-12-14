# Scripts Directory

This directory contains helper scripts for running and managing the HFT Arbitrage Lab.

## Quick Start Scripts

### Running the Application

| Script | Description | Usage |
|--------|-------------|-------|
| `run_app.sh` | Run Streamlit app (auto-detects environment) | `./scripts/run_app.sh` |
| `run_standalone.sh` | Run without authentication | `./scripts/run_standalone.sh` |
| `run_with_auth.sh` | Run with authentication (requires secrets.toml) | `./scripts/run_with_auth.sh` |
| `clean_restart_streamlit.sh` | Kill existing Streamlit and start fresh | `./scripts/clean_restart_streamlit.sh` |

### Docker Commands

| Script | Description | Usage |
|--------|-------------|-------|
| `dev.sh` | Docker compose wrapper | `MODE=standalone ./scripts/dev.sh up` |
| | Standalone mode (no auth) | `MODE=standalone ./scripts/dev.sh up` |
| | Production mode (with auth) | `MODE=prod ./scripts/dev.sh up` |
| | Stop containers | `./scripts/dev.sh down` |
| | Build containers | `./scripts/dev.sh build` |
| | View logs | `./scripts/dev.sh logs` |

### Rust Build Scripts

| Script | Description | Usage |
|--------|-------------|-------|
| `quick_rust_build.sh` | Quick incremental Rust build | `./scripts/quick_rust_build.sh` |
| `restart_rust.sh` | Clean rebuild Rust connector | `./scripts/restart_rust.sh` |

### Development Scripts

| Script | Description | Usage |
|--------|-------------|-------|
| `restart_all.sh` | Restart entire development environment | `./scripts/restart_all.sh` |
| `setup_env.sh` | Environment setup helper | `./scripts/setup_env.sh` |
| `activate.sh` | Activate Python virtual environment | `source ./scripts/activate.sh` |
| `start_jupyter.sh` | Start Jupyter notebook server | `./scripts/start_jupyter.sh` |

## Common Workflows

### First Time Setup
```bash
# 1. Setup environment
./scripts/setup_env.sh

# 2. Build Rust connector
./scripts/quick_rust_build.sh

# 3. Run the app
./scripts/run_standalone.sh
```

### Development Workflow
```bash
# Run in standalone mode (no auth)
./scripts/run_standalone.sh

# Or use Docker standalone
MODE=standalone ./scripts/dev.sh up
```

### Production Deployment
```bash
# 1. Create secrets file
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit secrets.toml with your password

# 2. Run with authentication
./scripts/run_with_auth.sh

# Or use Docker production mode
MODE=prod ./scripts/dev.sh up
```

### Rust Development
```bash
# Quick incremental build (fast)
./scripts/quick_rust_build.sh

# Full clean rebuild (slower but safer)
./scripts/restart_rust.sh

# Restart everything (Rust + Python)
./scripts/restart_all.sh
```

## Environment Variables

### Authentication Control
- `ENABLE_AUTH=false` - Run without authentication (default)
- `ENABLE_AUTH=true` - Require password authentication

### Docker Mode Selection
- `MODE=standalone` - Standalone mode (no auth)
- `MODE=prod` - Production mode (with auth)
- `MODE=default` - Use standard docker-compose.yml

## Troubleshooting

### Script Not Found Errors
All scripts now navigate to the project root automatically. Run them from anywhere:
```bash
# From project root
./scripts/quick_rust_build.sh

# From scripts directory
cd scripts && ./quick_rust_build.sh

# From anywhere with absolute path
/path/to/rust-hft-arbitrage-lab/scripts/quick_rust_build.sh
```

### Virtual Environment Issues
```bash
# Activate venv manually
source scripts/activate.sh

# Or directly
source .venv/bin/activate
```

### Rust Build Issues
```bash
# Clean rebuild
./scripts/restart_rust.sh

# Check if rust_connector is importable
python -c "import rust_connector; print('✓ OK')"
```

### Docker Issues
```bash
# Stop all containers
./scripts/dev.sh down

# Rebuild from scratch
./scripts/dev.sh build
MODE=standalone ./scripts/dev.sh up
```

## All Scripts Summary

✅ **Fixed and Working:**
- `activate.sh` - Activates Python venv (source this)
- `clean_restart_streamlit.sh` - Restarts Streamlit cleanly
- `dev.sh` - Docker compose wrapper with mode selection
- `quick_rust_build.sh` - Fast incremental Rust builds
- `restart_all.sh` - Complete environment restart
- `restart_rust.sh` - Rebuild Rust connector
- `run_app.sh` - Main app launcher
- `run_standalone.sh` - Run without auth (NEW)
- `run_with_auth.sh` - Run with auth (NEW)
- `setup_env.sh` - Environment setup helper
- `start_jupyter.sh` - Start Jupyter server

📝 **Python Utilities:**
- `enrich_notebooks.py` - Notebook enrichment utility
- `update_notebooks_finnhub.py` - Update notebooks with Finnhub
- `setup_project.sh` - Project scaffolding (for new projects)
