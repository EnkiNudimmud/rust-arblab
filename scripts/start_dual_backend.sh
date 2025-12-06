#!/bin/bash

# Dual Backend Launcher
# Starts both gRPC server and Streamlit app for side-by-side comparison

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  🚀 HFT Arbitrage Lab - Dual Backend Launcher${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Function to cleanup background processes
cleanup() {
    echo -e "\n${YELLOW}Cleaning up background processes...${NC}"
    if [ ! -z "$GRPC_PID" ]; then
        echo "Stopping gRPC server (PID: $GRPC_PID)"
        kill $GRPC_PID 2>/dev/null || true
    fi
    if [ ! -z "$STREAMLIT_PID" ]; then
        echo "Stopping Streamlit (PID: $STREAMLIT_PID)"
        kill $STREAMLIT_PID 2>/dev/null || true
    fi
    exit 0
}

trap cleanup SIGINT SIGTERM

# Kill any existing processes on our ports
echo -e "${YELLOW}🔍 Checking for existing processes...${NC}"

# Kill gRPC server on port 50051
if lsof -Pi :50051 -sTCP:LISTEN -t >/dev/null 2>&1; then
    EXISTING_GRPC_PID=$(lsof -Pi :50051 -sTCP:LISTEN -t)
    echo -e "${YELLOW}   Stopping existing gRPC server (PID: $EXISTING_GRPC_PID)${NC}"
    kill -9 $EXISTING_GRPC_PID 2>/dev/null || true
    sleep 1
fi

# Kill Streamlit on port 8501
if lsof -Pi :8501 -sTCP:LISTEN -t >/dev/null 2>&1; then
    EXISTING_STREAMLIT_PID=$(lsof -Pi :8501 -sTCP:LISTEN -t)
    echo -e "${YELLOW}   Stopping existing Streamlit (PID: $EXISTING_STREAMLIT_PID)${NC}"
    kill -9 $EXISTING_STREAMLIT_PID 2>/dev/null || true
    sleep 1
fi

# Also kill any Streamlit processes by name (in case they're on different ports)
pkill -9 -f "streamlit run" 2>/dev/null || true

echo -e "${GREEN}   ✓ Ports cleared${NC}"
echo ""

GRPC_ALREADY_RUNNING=false

# Start gRPC server
echo -e "${BLUE}[1/3]${NC} Starting gRPC server on port 50051..."

if [ ! -f "target/release/hft-server" ]; then
    echo -e "${YELLOW}   Building gRPC server (first time only)...${NC}"
    cd hft-grpc-server
    cargo build --release 2>&1 | tail -5
    cd ..
fi

RUST_LOG=info ./target/release/hft-server > grpc_server.log 2>&1 &
GRPC_PID=$!

echo -e "${GREEN}   ✓ gRPC server started (PID: $GRPC_PID)${NC}"
echo -e "${BLUE}     Logs: grpc_server.log${NC}"

# Wait for server to be ready
echo -e "${YELLOW}   Waiting for server to be ready...${NC}"
for i in {1..10}; do
    if lsof -Pi :50051 -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo -e "${GREEN}   ✓ Server is ready!${NC}"
        break
    fi
    sleep 1
done

echo ""

# Start Streamlit with gRPC backend
echo -e "${BLUE}[2/3]${NC} Starting Streamlit app (gRPC backend)..."

# Export backend configuration
export BACKEND_TYPE=grpc
export GRPC_HOST=localhost
export GRPC_PORT=50051

# Start Streamlit
cd app
streamlit run HFT_Arbitrage_Lab.py \
    --server.port 8501 \
    --server.address localhost \
    --server.headless true \
    --browser.gatherUsageStats false \
    > ../streamlit.log 2>&1 &

STREAMLIT_PID=$!

echo -e "${GREEN}   ✓ Streamlit started (PID: $STREAMLIT_PID)${NC}"
echo -e "${BLUE}     URL: http://localhost:8501${NC}"
echo -e "${BLUE}     Logs: streamlit.log${NC}"

cd ..

echo ""
echo -e "${BLUE}[3/3]${NC} Waiting for Streamlit to be ready..."

# Wait for Streamlit to be ready
for i in {1..20}; do
    if curl -s http://localhost:8501 > /dev/null 2>&1; then
        echo -e "${GREEN}   ✓ Streamlit is ready!${NC}"
        break
    fi
    sleep 1
done

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  ✅ Both backends are running!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "  📊 ${BLUE}Streamlit App:${NC} http://localhost:8501"
echo -e "  🚀 ${BLUE}gRPC Server:${NC}  localhost:50051"
echo ""
echo -e "${YELLOW}Backend Selection:${NC}"
echo -e "  • Use the sidebar in Streamlit to switch between backends"
echo -e "  • Click ${BLUE}'📊 Compare Backends'${NC} button to see performance comparison"
echo ""
echo -e "${YELLOW}Logs:${NC}"
echo -e "  • gRPC:     tail -f grpc_server.log"
echo -e "  • Streamlit: tail -f streamlit.log"
echo ""
echo -e "${RED}Press Ctrl+C to stop all services${NC}"
echo ""

# Keep script running
wait $GRPC_PID $STREAMLIT_PID
