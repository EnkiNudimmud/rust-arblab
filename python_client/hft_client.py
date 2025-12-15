# hft_client.py
"""Python client wrapper for the HFT gRPC services.

This module provides a high-level ``HFTClient`` class that abstracts the
individual gRPC service stubs generated from the ``*.proto`` files.  The
implementation below is a minimal skeleton – each method raises
``NotImplementedError`` and should be filled in with the appropriate call
to the generated stub (e.g. ``self.regime_client.calibrate(...)``).

The client can be instantiated once and reused across notebooks:

```python
from hft_client import HFTClient
client = HFTClient(host="127.0.0.1", port=50051)
# example usage (replace with real RPC names)
# result = client.calibrate_regime(...)
```
"""

import grpc
from . import regime_pb2_grpc, drift_pb2_grpc, superspace_pb2_grpc, options_pb2_grpc


class HFTClient:
    """High‑level wrapper around the generated gRPC stubs.

    The constructor creates a channel and stores a stub for each service.
    Individual RPC methods should be added as needed – they simply forward
    the request to the corresponding stub.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 50051):
        self.channel = grpc.insecure_channel(f"{host}:{port}")
        # Service stubs – replace with the actual service names from the .proto files
        self.regime_stub = regime_pb2_grpc.RegimeServiceStub(self.channel)
        self.drift_stub = drift_pb2_grpc.DriftServiceStub(self.channel)
        self.superspace_stub = superspace_pb2_grpc.SuperspaceServiceStub(self.channel)
        self.options_stub = options_pb2_grpc.OptionsServiceStub(self.channel)

    # ---------------------------------------------------------------------
    # Regime service methods
    # ---------------------------------------------------------------------
    def calibrate_regime(self, request):
        """Call the ``Calibrate`` RPC of the Regime service."""
        return self.regime_stub.Calibrate(request)

    def detect_regime_changes(self, request):
        """Call the ``DetectChanges`` RPC of the Regime service."""
        return self.regime_stub.DetectChanges(request)

    # ---------------------------------------------------------------------
    # Drift service methods
    # ---------------------------------------------------------------------
    def compute_drift(self, request):
        """Call the ``ComputeDrift`` RPC of the Drift service."""
        return self.drift_stub.ComputeDrift(request)

    def compute_volatility(self, request):
        """Call the ``ComputeVolatility`` RPC of the Drift service."""
        return self.drift_stub.ComputeVolatility(request)

    # ---------------------------------------------------------------------
    # Superspace service methods
    # ---------------------------------------------------------------------
    def detect_anomalies(self, request):
        """Call the ``DetectAnomalies`` RPC of the Superspace service.
        
        Args:
            request (superspace_pb2.DetectRequest): The request message.
            
        Returns:
            superspace_pb2.DetectResponse: The response message.
        """
        return self.superspace_stub.DetectAnomalies(request)

    def calculate_chern_simons(self, request):
        """Call the ``CalculateChernSimons`` RPC of the Superspace service."""
        return self.superspace_stub.CalculateChernSimons(request)

    def generate_ghost_field(self, request):
        """Call the ``GenerateGhostField`` RPC of the Superspace service."""
        return self.superspace_stub.GenerateGhostField(request)

    # ---------------------------------------------------------------------
    # Options service methods
    # ---------------------------------------------------------------------
    def price_option(self, request):
        """Call the ``PriceOption`` RPC of the Options service."""
        return self.options_stub.PriceOption(request)

    def calculate_greeks(self, request):
        """Call the ``CalculateGreeks`` RPC of the Options service."""
        return self.options_stub.CalculateGreeks(request)

    # ---------------------------------------------------------------------
    # Utility / cleanup
    # ---------------------------------------------------------------------
    def close(self):
        """Close the underlying gRPC channel."""
        self.channel.close()
