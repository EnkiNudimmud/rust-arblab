"""
gRPC Client Wrapper for Streamlit App

Handles import path resolution for both script and app contexts
"""

import sys
from pathlib import Path

# Ensure python directory is in path
project_root = Path(__file__).parent.parent.parent
python_dir = project_root / "python"
if str(python_dir) not in sys.path:
    sys.path.insert(0, str(python_dir))

# Now import the actual client
try:
    # Try app context first
    from python.grpc_client import TradingGrpcClient, GrpcConfig
except ImportError:
    try:
        # Try direct import
        from grpc_client import TradingGrpcClient, GrpcConfig
    except ImportError:
        # Last resort - add grpc_gen to path
        grpc_gen_dir = python_dir / "grpc_gen"
        if str(grpc_gen_dir) not in sys.path:
            sys.path.insert(0, str(grpc_gen_dir))
        from grpc_client import TradingGrpcClient, GrpcConfig

__all__ = ['TradingGrpcClient', 'GrpcConfig']
