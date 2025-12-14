"""Simple gRPC smoke test client.

Starts a short client that connects to the Rust gRPC server and calls
`CalculateMeanReversion` with a small synthetic price series.

Usage:
    python scripts/grpc_smoke_test.py

Make sure the gRPC server is running (e.g. `make run-server`) before running.
"""

import sys
import time
import numpy as np

# Ensure project python/ package is importable
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from python.grpc_client import TradingGrpcClient


def main():
    prices = np.array([100.0, 100.5, 99.8, 100.2, 100.1, 100.3, 100.6])
    client = TradingGrpcClient()
    try:
        print("Connecting to gRPC server...")
        client.connect()
        print("Connected. Running mean reversion calculation...")
        result = client.calculate_mean_reversion(prices, threshold=1.5, lookback=5)
        print("Result:")
        for k, v in result.items():
            print(f"  {k}: {v}")
    except Exception as e:
        print("Smoke test failed:", e)
        raise
    finally:
        client.close()


if __name__ == '__main__':
    main()
