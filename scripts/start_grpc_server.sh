#!/bin/bash
# Start gRPC server with proper logging and error handling

set -e

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/grpc_server_$TIMESTAMP.log"

echo "🚀 Starting HFT gRPC Server"
echo "=========================="
echo "📝 Logs: $LOG_FILE"
echo "🌐 Address: localhost:50051"
echo ""

export RUST_LOG=info

cd hft-grpc-server

# Build if needed
if [ ! -f "target/release/hft-server" ]; then
    echo "🔨 Building server (first time)..."
    cargo build --release
fi

# Start server with logging
echo "✓ Starting server..."
cargo run --release 2>&1 | tee "../$LOG_FILE"
