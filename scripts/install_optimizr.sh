#!/usr/bin/env bash
# Install or update OptimizR from local repository

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OPTIMIZR_PATH="/Users/melvinalvarez/Documents/Workspace/optimiz-r"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}===================================${NC}"
echo -e "${GREEN}Installing OptimizR (Rust Backend)${NC}"
echo -e "${GREEN}===================================${NC}"

# Check if optimiz-r exists
if [ ! -d "$OPTIMIZR_PATH" ]; then
    echo -e "${RED}✗ OptimizR not found at: $OPTIMIZR_PATH${NC}"
    echo -e "${YELLOW}Please clone it first:${NC}"
    echo "  cd ~/Documents/Workspace"
    echo "  git clone https://github.com/ThotDjehuty/optimiz-r.git"
    exit 1
fi

# Activate virtual environment
if [ ! -d "$PROJECT_ROOT/.venv" ]; then
    echo -e "${YELLOW}⚠ Virtual environment not found. Creating one...${NC}"
    python3 -m venv "$PROJECT_ROOT/.venv"
fi

source "$PROJECT_ROOT/.venv/bin/activate"

# Check for maturin
if ! command -v maturin &> /dev/null; then
    echo -e "${YELLOW}Installing maturin (Rust-Python bridge)...${NC}"
    pip install maturin
fi

# Build and install optimizr
echo -e "${GREEN}Building OptimizR from source...${NC}"
cd "$OPTIMIZR_PATH"

# Pull latest changes
git pull origin main || echo -e "${YELLOW}Could not pull latest changes (continuing with local version)${NC}"

# Build with release optimizations
maturin develop --release

echo ""
echo -e "${GREEN}✓ OptimizR installed successfully!${NC}"
echo ""
echo "Test it with:"
echo "  source .venv/bin/activate"
echo "  python -c 'import optimizr; print(optimizr.__doc__)'"
echo ""
