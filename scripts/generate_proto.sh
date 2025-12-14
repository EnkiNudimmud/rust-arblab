#!/usr/bin/env bash
# Generate Python gRPC stubs from protobuf definitions

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."

cd "$PROJECT_ROOT"

echo "🔧 Generating Python gRPC code from protobuf definitions..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Generate Python code from proto files
python -m grpc_tools.protoc \
    -I./proto \
    --python_out=./python \
    --grpc_python_out=./python \
    ./proto/pair_discovery.proto

echo "✅ Generated Python protobuf code:"
echo "   - python/pair_discovery_pb2.py"
echo "   - python/pair_discovery_pb2_grpc.py"
echo ""
echo "📦 You can now import with:"
echo "   from python import pair_discovery_pb2"
echo "   from python import pair_discovery_pb2_grpc"
