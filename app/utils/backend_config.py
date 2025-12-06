"""
Backend Configuration and Selection System

Supports multiple backends:
- Legacy: PyO3 bindings (original)
- gRPC: Ultra-low latency Rust gRPC server
"""

import os
from enum import Enum
from dataclasses import dataclass
from typing import Optional

class BackendType(Enum):
    """Available backend types"""
    LEGACY = "legacy"
    GRPC = "grpc"

@dataclass
class BackendConfig:
    """Backend configuration"""
    backend_type: BackendType
    grpc_host: str = "localhost"
    grpc_port: int = 50051
    legacy_available: bool = True
    grpc_available: bool = True
    
    @classmethod
    def from_env(cls) -> 'BackendConfig':
        """Load configuration from environment variables"""
        backend_str = os.getenv('BACKEND_TYPE', 'grpc').lower()
        backend_type = BackendType.GRPC if backend_str == 'grpc' else BackendType.LEGACY
        
        return cls(
            backend_type=backend_type,
            grpc_host=os.getenv('GRPC_HOST', 'localhost'),
            grpc_port=int(os.getenv('GRPC_PORT', '50051')),
        )
    
    def get_display_name(self) -> str:
        """Get human-readable backend name"""
        if self.backend_type == BackendType.GRPC:
            return f"gRPC ({self.grpc_host}:{self.grpc_port})"
        return "Legacy PyO3"
    
    def get_description(self) -> str:
        """Get backend description"""
        if self.backend_type == BackendType.GRPC:
            return "Ultra-low latency Rust gRPC server (100x faster)"
        return "Original PyO3 bindings (legacy)"

# Global configuration instance
_config: Optional[BackendConfig] = None

def get_backend_config() -> BackendConfig:
    """Get or create global backend configuration"""
    global _config
    if _config is None:
        _config = BackendConfig.from_env()
    return _config

def set_backend_config(config: BackendConfig):
    """Set global backend configuration"""
    global _config
    _config = config

def reset_backend_config():
    """Reset backend configuration"""
    global _config
    _config = None
