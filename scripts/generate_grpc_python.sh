#!/bin/bash
# Generate Python gRPC code from proto files

set -e

echo "Generating Python gRPC code..."

# Install grpcio-tools if not present
pip install -q grpcio-tools

# Generate from pair_discovery.proto
python -m grpc_tools.protoc \
    -I./proto \
    --python_out=./python \
    --grpc_python_out=./python \
    ./proto/pair_discovery.proto

echo "✅ Generated pair_discovery_pb2.py and pair_discovery_pb2_grpc.py"

# Fix imports in generated files (make them relative)
if [[ "$OSTYPE" == "darwin"* ]]; then
    sed -i '' 's/^import pair_discovery_pb2/from . import pair_discovery_pb2/' python/pair_discovery_pb2_grpc.py
else
    sed -i 's/^import pair_discovery_pb2/from . import pair_discovery_pb2/' python/pair_discovery_pb2_grpc.py
fi

echo "✅ Fixed imports in generated files"
echo "Done!"
