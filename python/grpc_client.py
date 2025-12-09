"""
Python gRPC client for low-latency Rust communication.

This module provides a high-performance interface to Rust-based trading algorithms
via gRPC, eliminating serialization overhead and ensuring microsecond-level latency.
"""

import grpc
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass

# Generated proto files
from python.grpc_gen import trading_pb2, trading_pb2_grpc

logger = logging.getLogger(__name__)


@dataclass
class GrpcConfig:
    """Configuration for gRPC client."""
    host: str = "localhost"
    port: int = 50051
    max_retries: int = 3
    timeout: float = 30.0
    compression: bool = False  # Disabled by default for Rust tonic compatibility


class TradingGrpcClient:
    """High-performance gRPC client for Rust trading services."""
    
    def __init__(self, config: Optional[GrpcConfig] = None):
        self.config = config or GrpcConfig()
        self.address = f"{self.config.host}:{self.config.port}"
        self._channel: Optional[grpc.Channel] = None
        self._stub: Optional[trading_pb2_grpc.TradingServiceStub] = None
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def connect(self):
        """Establish gRPC connection to Rust server."""
        options = [
            ('grpc.max_send_message_length', 50 * 1024 * 1024),  # 50MB
            ('grpc.max_receive_message_length', 50 * 1024 * 1024),
        ]
        
        if self.config.compression:
            self._channel = grpc.insecure_channel(
                self.address,
                options=options,
                compression=grpc.Compression.Gzip
            )
        else:
            self._channel = grpc.insecure_channel(self.address, options=options)
        
        self._stub = trading_pb2_grpc.TradingServiceStub(self._channel)
        logger.info(f"Connected to gRPC server at {self.address}")
    
    def close(self):
        """Close gRPC connection."""
        if self._channel:
            self._channel.close()
            self._channel = None
            self._stub = None
    
    def calculate_mean_reversion(
        self,
        prices,
        threshold: float = 2.0,
        lookback: int = 20
    ) -> Dict:
        """
        Calculate mean reversion signals using Rust implementation.
        
        Args:
            prices: Price series (list, tuple, or np.ndarray)
            threshold: Z-score threshold for signals
            lookback: Lookback period
            
        Returns:
            Dictionary with signal, zscore, entry/exit signals, and metrics
        """
        if self._stub is None:
            raise RuntimeError("Not connected. Call connect() first.")
        
        # Convert to list if needed
        if isinstance(prices, np.ndarray):
            prices = prices.tolist()
        elif not isinstance(prices, (list, tuple)):
            prices = list(prices)
        
        request = trading_pb2.MeanReversionRequest(
            prices=prices,
            threshold=threshold,
            lookback=lookback
        )
        
        response = self._stub.CalculateMeanReversion(
            request,
            timeout=self.config.timeout
        )
        
        return {
            'signal': response.signal,
            'zscore': response.zscore,
            'entry_signal': response.entry_signal,
            'exit_signal': response.exit_signal,
            'metrics': dict(response.metrics)
        }
    
    def optimize_portfolio(
        self,
        prices,
        method: str = "markowitz",
        parameters: Optional[Dict[str, float]] = None
    ) -> Dict:
        """
        Optimize portfolio weights using Rust implementation.
        
        Args:
            prices: Dictionary of symbol -> price series OR list of dicts with 'symbol' and 'prices'
            method: Optimization method (markowitz, risk_parity, etc.)
            parameters: Method-specific parameters
            
        Returns:
            Dictionary with weights, returns, volatility, sharpe ratio
        """
        if self._stub is None:
            raise RuntimeError("Not connected. Call connect() first.")
        
        # Handle both dict and list formats
        if isinstance(prices, dict):
            price_vectors = [
                trading_pb2.PriceVector(symbol=sym, prices=p.tolist() if isinstance(p, np.ndarray) else p)
                for sym, p in prices.items()
            ]
        elif isinstance(prices, list):
            price_vectors = [
                trading_pb2.PriceVector(
                    symbol=item.get('symbol', f'Asset_{i}'),
                    prices=item['prices'].tolist() if isinstance(item['prices'], np.ndarray) else item['prices']
                )
                for i, item in enumerate(prices)
            ]
        else:
            raise ValueError("prices must be dict or list")
        
        request = trading_pb2.PortfolioOptimizationRequest(
            prices=price_vectors,
            method=method,
            parameters=parameters or {}
        )
        
        response = self._stub.OptimizePortfolio(
            request,
            timeout=self.config.timeout
        )
        
        return {
            'weights': np.array(response.weights),
            'expected_return': response.expected_return,
            'volatility': response.volatility,
            'sharpe_ratio': response.sharpe_ratio,
            'metrics': dict(response.metrics)
        }
    
    def detect_regime(
        self,
        returns: np.ndarray,
        n_regimes: int = 3,
        n_iterations: int = 100
    ) -> Dict:
        """
        Detect market regimes using HMM in Rust.
        
        Args:
            returns: Return series
            n_regimes: Number of regimes
            n_iterations: Maximum iterations
            
        Returns:
            Dictionary with current regime, probabilities, and parameters
        """
        if self._stub is None:
            raise RuntimeError("Not connected. Call connect() first.")
        
        request = trading_pb2.RegimeDetectionRequest(
            returns=returns.tolist(),
            n_regimes=n_regimes,
            n_iterations=n_iterations
        )
        
        response = self._stub.DetectRegime(
            request,
            timeout=self.config.timeout
        )
        
        regimes = [
            {
                'mean': r.mean,
                'volatility': r.volatility,
                'persistence': r.persistence
            }
            for r in response.regimes
        ]
        
        return {
            'current_regime': response.current_regime,
            'probabilities': np.array(response.regime_probabilities),
            'regimes': regimes
        }
    
    def stream_market_data(
        self,
        symbols: List[str],
        exchange: str = "binance",
        interval_ms: int = 100
    ):
        """
        Stream real-time market data from Rust server.
        
        Args:
            symbols: List of symbols to stream
            exchange: Exchange name
            interval_ms: Update interval in milliseconds
            
        Yields:
            Market data updates
        """
        if self._stub is None:
            raise RuntimeError("Not connected. Call connect() first.")
        
        request = trading_pb2.StreamRequest(
            symbols=symbols,
            exchange=exchange,
            interval_ms=interval_ms
        )
        
        for update in self._stub.StreamMarketData(request):
            yield {
                'symbol': update.symbol,
                'timestamp': update.timestamp,
                'bid': update.bid,
                'ask': update.ask,
                'mid': update.mid,
                'volume': update.volume
            }
    
    def run_hmm(
        self,
        observations,
        n_states: int = 3,
        max_iterations: int = 100,
        tolerance: float = 1e-6
    ) -> Dict:
        """
        Run Hidden Markov Model in Rust.
        
        Args:
            observations: Observation sequence (list, tuple, or np.ndarray)
            n_states: Number of hidden states
            max_iterations: Maximum EM iterations
            tolerance: Convergence tolerance
            
        Returns:
            HMM results with state probabilities and parameters
        """
        if self._stub is None:
            raise RuntimeError("Not connected. Call connect() first.")
        
        # Convert to list if needed
        if isinstance(observations, np.ndarray):
            observations = observations.tolist()
        elif not isinstance(observations, (list, tuple)):
            observations = list(observations)
        
        request = trading_pb2.HMMRequest(
            observations=observations,
            n_states=n_states,
            max_iterations=max_iterations,
            tolerance=tolerance
        )
        
        response = self._stub.RunHMM(
            request,
            timeout=self.config.timeout
        )
        
        n = n_states
        transition_matrix = np.array(response.transition_matrix).reshape((n, n))
        
        return {
            'state_probabilities': np.array(response.state_probabilities),
            'transition_matrix': transition_matrix,
            'emission_means': np.array(response.emission_means),
            'emission_stds': np.array(response.emission_stds),
            'log_likelihood': response.log_likelihood,
            'converged': response.converged
        }
    
    def run_mcmc(
        self,
        parameters: Dict[str, Tuple[float, float]],
        n_iterations: int = 1000,
        burn_in: int = 100,
        step_size: float = 0.1
    ) -> Dict:
        """
        Run MCMC optimization in Rust.
        
        Args:
            parameters: Dict of param_name -> (min, max)
            n_iterations: Number of MCMC iterations
            burn_in: Burn-in period
            step_size: Proposal step size
            
        Returns:
            MCMC results with samples and best parameters
        """
        if self._stub is None:
            raise RuntimeError("Not connected. Call connect() first.")
        
        param_spaces = [
            trading_pb2.ParameterSpace(name=name, min=bounds[0], max=bounds[1])
            for name, bounds in parameters.items()
        ]
        
        request = trading_pb2.MCMCRequest(
            parameters=param_spaces,
            n_iterations=n_iterations,
            burn_in=burn_in,
            step_size=step_size
        )
        
        response = self._stub.RunMCMC(
            request,
            timeout=self.config.timeout
        )
        
        samples = [
            {'values': dict(s.values), 'score': s.score}
            for s in response.samples
        ]
        
        return {
            'samples': samples,
            'best_params': dict(response.best_params),
            'best_score': response.best_score,
            'acceptance_rate': response.acceptance_rate
        }
    
    def calculate_sparse_portfolio(
        self,
        prices,
        method: str = "lasso",
        lambda_param: float = 0.1,
        alpha: float = 0.5
    ) -> Dict:
        """
        Calculate sparse portfolio using Rust implementation.
        
        Args:
            prices: Dictionary of symbol -> price series OR list of dicts with 'symbol' and 'prices'
            method: Sparse method (lasso, elastic_net, adaptive_lasso)
            lambda_param: Regularization parameter
            alpha: Elastic net mixing parameter
            
        Returns:
            Sparse portfolio weights and metrics
        """
        if self._stub is None:
            raise RuntimeError("Not connected. Call connect() first.")
        
        # Handle both dict and list formats
        if isinstance(prices, dict):
            price_vectors = [
                trading_pb2.PriceVector(symbol=sym, prices=p.tolist() if isinstance(p, np.ndarray) else p)
                for sym, p in prices.items()
            ]
        elif isinstance(prices, list):
            price_vectors = [
                trading_pb2.PriceVector(
                    symbol=item.get('symbol', f'Asset_{i}'),
                    prices=item['prices'].tolist() if isinstance(item['prices'], np.ndarray) else item['prices']
                )
                for i, item in enumerate(prices)
            ]
        else:
            raise ValueError("prices must be dict or list")
        
        request = trading_pb2.SparsePortfolioRequest(
            prices=price_vectors,
            method=method,
            **{'lambda': lambda_param},  # Using dict unpacking since 'lambda' is a Python keyword
            alpha=alpha
        )
        
        response = self._stub.CalculateSparsePortfolio(
            request,
            timeout=self.config.timeout
        )
        
        return {
            'weights': np.array(response.weights),
            'n_assets_selected': response.n_assets_selected,
            'objective_value': response.objective_value,
            'metrics': dict(response.metrics)
        }
    
    def box_tao_decomposition(
        self,
        prices: Dict[str, np.ndarray],
        lambda_param: float = 0.1,
        mu: float = 0.1,
        max_iterations: int = 1000,
        tolerance: float = 1e-6
    ) -> Dict:
        """
        Perform Box-Tao decomposition in Rust.
        
        Args:
            prices: Dictionary of symbol -> price series
            lambda_param: Nuclear norm penalty
            mu: Sparse penalty
            max_iterations: Maximum ADMM iterations
            tolerance: Convergence tolerance
            
        Returns:
            Low-rank, sparse, and noise matrices
        """
        if self._stub is None:
            raise RuntimeError("Not connected. Call connect() first.")
        
        price_vectors = [
            trading_pb2.PriceVector(symbol=sym, prices=p.tolist())
            for sym, p in prices.items()
        ]
        
        request = trading_pb2.BoxTaoRequest(
            prices=price_vectors,
            lambda_=lambda_param,
            mu=mu,
            max_iterations=max_iterations,
            tolerance=tolerance
        )
        
        response = self._stub.BoxTaoDecomposition(
            request,
            timeout=self.config.timeout
        )
        
        def matrix_to_numpy(mat):
            return np.array(mat.data).reshape((mat.rows, mat.cols))
        
        return {
            'low_rank': matrix_to_numpy(response.low_rank),
            'sparse': matrix_to_numpy(response.sparse),
            'noise': matrix_to_numpy(response.noise),
            'iterations': response.iterations,
            'converged': response.converged
        }


# Singleton client for convenience
_default_client: Optional[TradingGrpcClient] = None


def get_client(config: Optional[GrpcConfig] = None) -> TradingGrpcClient:
    """Get or create default gRPC client."""
    global _default_client
    if _default_client is None:
        _default_client = TradingGrpcClient(config)
        _default_client.connect()
    return _default_client


def close_client():
    """Close default gRPC client."""
    global _default_client
    if _default_client:
        _default_client.close()
        _default_client = None
