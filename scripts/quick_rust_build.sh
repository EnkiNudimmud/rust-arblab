#!/bin/bash
# Quick Rust Build Script (no clean, faster for iterations)

set -e

echo "⚡ Quick Rust Build..."

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

cd "$(dirname "$0")/.."

# Activate venv
source .venv/bin/activate 2>/dev/null || source source/bin/activate
unset CONDA_PREFIX

echo -e "${YELLOW}🔨 Building Rust connector (incremental)...${NC}"

cd rust_connector
maturin develop --release

echo -e "${GREEN}✓ Build complete!${NC}"
python -c "import rust_connector; print('✓ rust_connector loaded successfully')"
