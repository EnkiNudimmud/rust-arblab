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

echo -e "${YELLOW}🔨 Building Rust components (incremental)...${NC}"
if command -v make &> /dev/null; then
	$(MAKE) build || true
else
	cd rust_connector
	maturin develop --release || true
	cd ..
fi

echo -e "${GREEN}✓ Build step finished (check logs for errors)${NC}"
python - <<'PY' || true
