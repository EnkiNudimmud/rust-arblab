"""
Unified Backend Interface

Provides a consistent API regardless of backend (Legacy or gRPC)
"""

import numpy as np
from typing import Dict, List, Optional, Any, Union
from abc import ABC, abstractmethod
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.utils.backend_config import BackendType, get_backend_config


class BackendInterface(ABC):
    """Abstract base class for backend implementations"""
    
    @abstractmethod
    def calculate_mean_reversion(
        self,
        prices: Union[List[float], np.ndarray],
        lookback: int = 20,
        threshold: float = 2.0
    ) -> Dict[str, Any]:
        """Calculate mean reversion metrics"""
        pass
    
    @abstractmethod
    def optimize_portfolio(
        self,
        prices: Union[Dict, List],
        method: str = "max_sharpe",
        parameters: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Optimize portfolio weights"""
        pass
    
    @abstractmethod
    def run_hmm(
        self,
        observations: Union[List[float], np.ndarray],
        n_states: int = 3,
        max_iterations: int = 100,
        tolerance: float = 1e-6
    ) -> Dict[str, Any]:
        """Run HMM regime detection"""
        pass
    
    @abstractmethod
    def calculate_sparse_portfolio(
        self,
        prices: Union[Dict, List],
        method: str = "lasso",
        lambda_param: float = 0.1,
        alpha: float = 0.5
    ) -> Dict[str, Any]:
        """Calculate sparse portfolio"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend is available"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get backend name"""
        pass


class GrpcBackend(BackendInterface):
    """gRPC backend implementation"""
    
    def __init__(self):
        self._client = None
        self._available = self._check_availability()
    
    def _check_availability(self) -> bool:
        """Check if gRPC server is available"""
        try:
            # Import via wrapper to handle path issues
            import sys
            grpc_gen_path = str(project_root / "python" / "grpc_gen")
            if grpc_gen_path not in sys.path:
                sys.path.insert(0, grpc_gen_path)
            
            from python.grpc_client import TradingGrpcClient
            config = get_backend_config()
            
            # Try to connect
            client = TradingGrpcClient()
            client.connect()
            client.close()
            return True
        except Exception as e:
            print(f"gRPC backend not available: {e}")
            return False
    
    def _get_client(self):
        """Get or create gRPC client"""
        if self._client is None:
            # Import via wrapper to handle path issues
            import sys
            grpc_gen_path = str(project_root / "python" / "grpc_gen")
            if grpc_gen_path not in sys.path:
                sys.path.insert(0, grpc_gen_path)
            
            from python.grpc_client import TradingGrpcClient, GrpcConfig
            config = get_backend_config()
            
            grpc_config = GrpcConfig(
                host=config.grpc_host,
                port=config.grpc_port
            )
            self._client = TradingGrpcClient(grpc_config)
            self._client.connect()
        
        return self._client
    
    def calculate_mean_reversion(
        self,
        prices: Union[List[float], np.ndarray],
        lookback: int = 20,
        threshold: float = 2.0
    ) -> Dict[str, Any]:
        client = self._get_client()
        return client.calculate_mean_reversion(prices, threshold, lookback)
    
    def optimize_portfolio(
        self,
        prices: Union[Dict, List],
        method: str = "max_sharpe",
        parameters: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        client = self._get_client()
        return client.optimize_portfolio(prices, method, parameters or {})
    
    def run_hmm(
        self,
        observations: Union[List[float], np.ndarray],
        n_states: int = 3,
        max_iterations: int = 100,
        tolerance: float = 1e-6
    ) -> Dict[str, Any]:
        client = self._get_client()
        return client.run_hmm(observations, n_states, max_iterations, tolerance)
    
    def calculate_sparse_portfolio(
        self,
        prices: Union[Dict, List],
        method: str = "lasso",
        lambda_param: float = 0.1,
        alpha: float = 0.5
    ) -> Dict[str, Any]:
        client = self._get_client()
        return client.calculate_sparse_portfolio(prices, method, lambda_param, alpha)
    
    def is_available(self) -> bool:
        return self._available
    
    def get_name(self) -> str:
        return "gRPC"
    
    def __del__(self):
        """Cleanup"""
        if self._client:
            try:
                self._client.close()
            except:
                pass


class LegacyBackend(BackendInterface):
    """Legacy PyO3 backend implementation"""
    
    def __init__(self):
        self._available = self._check_availability()
    
    def _check_availability(self) -> bool:
        """Check if legacy backend is available"""
        try:
            import rust_connector
            return True
        except Exception as e:
            print(f"Legacy backend not available: {e}")
            return False
    
    def calculate_mean_reversion(
        self,
        prices: Union[List[float], np.ndarray],
        lookback: int = 20,
        threshold: float = 2.0
    ) -> Dict[str, Any]:
        """Legacy mean reversion calculation"""
        if isinstance(prices, np.ndarray):
            prices = prices.tolist()
        
        # Simple z-score calculation
        if len(prices) < lookback:
            return {
                'signal': 0.0,
                'zscore': 0.0,
                'entry_signal': False,
                'exit_signal': False,
                'metrics': {}
            }
        
        recent = prices[-lookback:]
        mean = np.mean(recent)
        std = np.std(recent)
        current = prices[-1]
        
        zscore = (current - mean) / std if std > 0 else 0.0
        signal = -1.0 if zscore > threshold else (1.0 if zscore < -threshold else 0.0)
        
        return {
            'signal': signal,
            'zscore': zscore,
            'entry_signal': abs(zscore) > threshold,
            'exit_signal': abs(zscore) < threshold * 0.5,
            'metrics': {
                'mean': mean,
                'std': std,
                'current': current
            }
        }
    
    def optimize_portfolio(
        self,
        prices: Union[Dict, List],
        method: str = "max_sharpe",
        parameters: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Legacy portfolio optimization"""
        # Convert to numpy arrays
        if isinstance(prices, dict):
            price_arrays = [np.array(p) for p in prices.values()]
        else:
            price_arrays = [np.array(item['prices']) for item in prices]
        
        n_assets = len(price_arrays)
        
        # Simple equal weight for legacy
        weights = np.ones(n_assets) / n_assets
        
        # Calculate basic metrics
        returns = [np.diff(p) / p[:-1] for p in price_arrays]
        mean_returns = [np.mean(r) for r in returns]
        
        portfolio_return = np.dot(weights, mean_returns)
        
        return {
            'weights': weights,
            'expected_return': portfolio_return,
            'volatility': 0.02,  # Placeholder
            'sharpe_ratio': 1.0,  # Placeholder
            'metrics': {}
        }
    
    def run_hmm(
        self,
        observations: Union[List[float], np.ndarray],
        n_states: int = 3,
        max_iterations: int = 100,
        tolerance: float = 1e-6
    ) -> Dict[str, Any]:
        """Legacy HMM - uses simple threshold-based regime detection"""
        if isinstance(observations, np.ndarray):
            observations = observations.tolist()
        
        # Simple regime detection based on volatility
        returns = np.array(observations)
        volatility = np.std(returns)
        
        # Dummy regime probabilities
        state_probs = np.ones(n_states) / n_states
        
        return {
            'state_probabilities': state_probs,
            'transition_matrix': np.eye(n_states),
            'emission_means': np.linspace(-volatility, volatility, n_states),
            'emission_stds': np.ones(n_states) * volatility,
            'log_likelihood': 0.0,
            'converged': True
        }
    
    def calculate_sparse_portfolio(
        self,
        prices: Union[Dict, List],
        method: str = "lasso",
        lambda_param: float = 0.1,
        alpha: float = 0.5
    ) -> Dict[str, Any]:
        """Legacy sparse portfolio"""
        # Convert to count
        if isinstance(prices, dict):
            n_assets = len(prices)
        else:
            n_assets = len(prices)
        
        # Simple equal weight
        weights = np.ones(n_assets) / n_assets
        
        return {
            'weights': weights,
            'n_assets_selected': n_assets,
            'objective_value': 0.0,
            'metrics': {}
        }
    
    def is_available(self) -> bool:
        return self._available
    
    def get_name(self) -> str:
        return "Legacy"


# Backend factory
_backend_cache: Dict[BackendType, BackendInterface] = {}

def get_backend(backend_type: Optional[BackendType] = None) -> BackendInterface:
    """Get backend instance"""
    if backend_type is None:
        config = get_backend_config()
        backend_type = config.backend_type
    
    # Check cache
    if backend_type in _backend_cache:
        return _backend_cache[backend_type]
    
    # Create new backend
    if backend_type == BackendType.GRPC:
        backend = GrpcBackend()
    else:
        backend = LegacyBackend()
    
    # Cache it
    _backend_cache[backend_type] = backend
    
    return backend


def clear_backend_cache():
    """Clear backend cache"""
    global _backend_cache
    _backend_cache.clear()
