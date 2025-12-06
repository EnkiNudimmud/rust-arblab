#!/bin/bash
# Setup and build gRPC infrastructure

set -e

echo "🚀 Setting up HFT gRPC Infrastructure"
echo "======================================"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check Rust
echo -e "\n${BLUE}[1/5]${NC} Checking Rust installation..."
if ! command -v cargo &> /dev/null; then
    echo "❌ Rust not found. Install from https://rustup.rs/"
    exit 1
fi
echo -e "${GREEN}✓${NC} Rust $(rustc --version)"

# Check Python
echo -e "\n${BLUE}[2/5]${NC} Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found"
    exit 1
fi
echo -e "${GREEN}✓${NC} Python $(python3 --version)"

# Install Python gRPC tools
echo -e "\n${BLUE}[3/5]${NC} Installing Python gRPC tools..."
pip install grpcio grpcio-tools numpy || {
    echo "❌ Failed to install Python packages"
    exit 1
}
echo -e "${GREEN}✓${NC} Python gRPC tools installed"

# Generate Python proto code
echo -e "\n${BLUE}[4/5]${NC} Generating Python gRPC code..."
mkdir -p python/grpc_gen
python3 -m grpc_tools.protoc \
    -I./proto \
    --python_out=./python/grpc_gen \
    --grpc_python_out=./python/grpc_gen \
    ./proto/trading.proto || {
    echo "❌ Failed to generate Python code"
    exit 1
}
touch python/grpc_gen/__init__.py
echo -e "${GREEN}✓${NC} Python gRPC code generated"

# Build Rust gRPC server
echo -e "\n${BLUE}[5/5]${NC} Building Rust gRPC server..."
cd hft-grpc-server
cargo build --release || {
    echo "❌ Failed to build gRPC server"
    exit 1
}
cd ..
echo -e "${GREEN}✓${NC} gRPC server built: ./hft-grpc-server/target/release/hft-server"

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}✓ Setup complete!${NC}"
echo -e "\n${BLUE}To start the server:${NC}"
echo "  cd hft-grpc-server && cargo run --release"
echo "  # or"
echo "  ./hft-grpc-server/target/release/hft-server"
echo -e "\n${BLUE}To use from Python:${NC}"
echo "  from python.grpc_client import get_client"
echo "  client = get_client()"
echo "  result = client.calculate_mean_reversion(prices)"
