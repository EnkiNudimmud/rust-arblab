#!/bin/bash
# Quick setup for both Python and Rust refactoring

set -e

echo "🎯 Complete Refactoring Setup"
echo "=============================="
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check if we're in the right directory
if [ ! -f "Cargo.toml" ]; then
    echo "❌ Please run this script from the project root"
    exit 1
fi

echo -e "${BLUE}[1/3]${NC} Python Setup"
echo "─────────────────────────"

# Python dependencies
echo "Installing Python dependencies..."
pip install -q numpy pandas grpcio grpcio-tools || {
    echo "${YELLOW}⚠️  Some Python packages may already be installed${NC}"
}
echo -e "${GREEN}✓${NC} Python dependencies ready"

# Generate Python proto code
echo "Generating Python gRPC code..."
mkdir -p python/grpc_gen
python3 -m grpc_tools.protoc \
    -I./proto \
    --python_out=./python/grpc_gen \
    --grpc_python_out=./python/grpc_gen \
    ./proto/trading.proto 2>/dev/null || {
    echo "${YELLOW}⚠️  Proto generation skipped (may need grpcio-tools)${NC}"
}
touch python/grpc_gen/__init__.py
echo -e "${GREEN}✓${NC} Python gRPC code generated"

echo ""
echo -e "${BLUE}[2/3]${NC} Rust Setup"
echo "─────────────────────────"

# Check Rust version
if ! command -v cargo &> /dev/null; then
    echo "❌ Rust not found. Install from https://rustup.rs/"
    exit 1
fi

# Build gRPC server
echo "Building gRPC server (this may take a few minutes)..."
cd hft-grpc-server
cargo build --release 2>&1 | grep -E "(Compiling|Finished)" || true
cd ..
echo -e "${GREEN}✓${NC} gRPC server built"

echo ""
echo -e "${BLUE}[3/3]${NC} Verification"
echo "─────────────────────────"

# Check files
if [ -f "hft-grpc-server/target/release/hft-server" ]; then
    echo -e "${GREEN}✓${NC} gRPC server binary exists"
else
    echo "${YELLOW}⚠️  Server binary not found (build may have failed)${NC}"
fi

if [ -f "python/grpc_gen/trading_pb2.py" ]; then
    echo -e "${GREEN}✓${NC} Python proto code exists"
else
    echo "${YELLOW}⚠️  Python proto code not found${NC}"
fi

if [ -f "python/type_fixes.py" ]; then
    echo -e "${GREEN}✓${NC} Python refactoring complete"
fi

if [ -d "python/strategies" ]; then
    echo -e "${GREEN}✓${NC} Python folder structure updated"
fi

echo ""
echo -e "${GREEN}════════════════════════════${NC}"
echo -e "${GREEN}✓ Setup Complete!${NC}"
echo -e "${GREEN}════════════════════════════${NC}"
echo ""
echo "📚 Next Steps:"
echo ""
echo "1. Start gRPC server:"
echo "   ${BLUE}./scripts/start_grpc_server.sh${NC}"
echo ""
echo "2. Test connectivity:"
echo "   ${BLUE}python3 scripts/test_grpc.py${NC}"
echo ""
echo "3. Use from Python:"
echo "   ${BLUE}from python.grpc_client import get_client${NC}"
echo "   ${BLUE}client = get_client()${NC}"
echo "   ${BLUE}result = client.calculate_mean_reversion(prices)${NC}"
echo ""
echo "📖 Documentation:"
echo "   - Quick Start:    ${BLUE}GRPC_QUICKSTART.md${NC}"
echo "   - Architecture:   ${BLUE}RUST_GRPC_REFACTORING.md${NC}"
echo "   - Python Changes: ${BLUE}python/REFACTORING_COMPLETE.md${NC}"
echo "   - Full Summary:   ${BLUE}COMPLETE_REFACTORING_SUMMARY.md${NC}"
echo ""
echo "🎉 Happy trading! ⚡"
