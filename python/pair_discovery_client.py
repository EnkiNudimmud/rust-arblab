"""
Pair Discovery gRPC Client
===========================

High-performance client for pair discovery using Rust backend via gRPC.

Usage:
    from python.pair_discovery_client import PairDiscoveryClient
    
    client = PairDiscoveryClient()
    result = client.test_pair("AAPL", "MSFT", prices_aapl, prices_msft)
"""

import grpc
import numpy as np
from typing import List, Dict, Optional, Iterator, Tuple
import os

# Import generated protobuf code
try:
    from . import pair_discovery_pb2
    from . import pair_discovery_pb2_grpc
except ImportError:
    # Fallback for direct execution
    import pair_discovery_pb2
    import pair_discovery_pb2_grpc


class PairDiscoveryClient:
    """Client for Rust-based pair discovery gRPC service"""
    
    def __init__(self, host: Optional[str] = None, port: Optional[int] = None):
        """
        Initialize gRPC client.
        
        Args:
            host: gRPC server host (default: from env or 'localhost')
            port: gRPC server port (default: from env or 50051)
        """
        self.host = host or os.getenv('GRPC_HOST', 'localhost')
        self.port = port or int(os.getenv('GRPC_PORT', '50051'))
        self.address = f'{self.host}:{self.port}'
        self.channel = grpc.insecure_channel(self.address)
        self.stub = pair_discovery_pb2_grpc.PairDiscoveryServiceStub(self.channel)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def close(self):
        """Close the gRPC channel"""
        if hasattr(self, 'channel'):
            self.channel.close()
    
    def test_pair(
        self,
        prices_x: np.ndarray,
        prices_y: np.ndarray,
        pair_name: Optional[str] = None,
        symbol1: Optional[str] = None,
        symbol2: Optional[str] = None,
        significance: float = 0.05,
        min_hurst: float = 0.45,
        transaction_cost: float = 0.001,
    ) -> Optional[Dict]:
        """
        Test a single pair for cointegration and mean-reversion.
        
        Args:
            prices_x: Price series for first asset
            prices_y: Price series for second asset  
            pair_name: Optional name for the pair (e.g., "BTC-ETH")
            symbol1: First symbol (deprecated, use pair_name)
            symbol2: Second symbol (deprecated, use pair_name)
            significance: Cointegration p-value threshold
            min_hurst: Maximum Hurst exponent for mean-reversion
            transaction_cost: Transaction cost per trade
        
        Returns:
            Dictionary with test results or None if pair is invalid
        """
        # Handle legacy API
        if symbol1 and symbol2:
            pair_name = f"{symbol1}-{symbol2}"
        elif not pair_name:
            pair_name = "ASSET_X-ASSET_Y"
            
        request = pair_discovery_pb2.PairTestRequest(
            symbol1=pair_name.split('-')[0] if '-' in pair_name else 'X',
            symbol2=pair_name.split('-')[1] if '-' in pair_name else 'Y',
            prices1=prices_x.astype(np.float64).tolist(),
            prices2=prices_y.astype(np.float64).tolist(),
            significance=significance,
            min_hurst=min_hurst,
            transaction_cost=transaction_cost,
        )
        
        try:
            response = self.stub.TestPair(request, timeout=60.0)
            
            if not response.success:
                return None
            
            return {
                'pair': f'{response.symbol1}-{response.symbol2}',
                'symbol1': response.symbol1,
                'symbol2': response.symbol2,
                'is_cointegrated': response.is_cointegrated,
                'beta': response.beta,
                'hedge_ratio': response.beta,  # Alias
                'p_value': response.p_value,
                'hurst': response.hurst,
                'hurst_exponent': response.hurst,  # Alias
                'kappa': response.kappa,
                'theta': response.theta,
                'sigma': response.sigma,
                'ou_params': {
                    'theta': response.theta,
                    'mu': response.theta, 
                    'sigma': response.sigma,
                },
                'half_life': response.half_life,
                'lower_boundary': response.lower_boundary,
                'upper_boundary': response.upper_boundary,
                'entry_threshold': response.upper_boundary,  # Alias
                'exit_threshold': response.lower_boundary,   # Alias
                'total_return': response.total_return,
                'sharpe_ratio': response.sharpe_ratio,
                'max_drawdown': response.max_drawdown,
                'num_trades': response.num_trades,
                'win_rate': response.win_rate,
                'coint_score': response.coint_score,
                'meanrev_score': response.meanrev_score,
                'profit_score': response.profit_score,
                'combined_score': response.combined_score,
                'error': response.error if response.error else None,
            }
        
        except grpc.RpcError as e:
            print(f"gRPC error testing {symbol1}-{symbol2}: {e.details()}")
            return None
    
    def discover_pairs(
        self,
        data: Dict[str, np.ndarray],
        significance: float = 0.05,
        min_hurst: float = 0.45,
        transaction_cost: float = 0.001,
        max_pairs: int = 0,
        parallel_workers: int = 0,
    ) -> Iterator[Dict]:
        """
        Discover cointegrated pairs with streaming results.
        
        Args:
            data: Dictionary mapping symbol -> price series
            significance: Cointegration p-value threshold
            min_hurst: Maximum Hurst exponent
            transaction_cost: Transaction cost per trade
            max_pairs: Maximum pairs to test (0 = all)
            parallel_workers: Number of parallel workers (0 = auto)
        
        Yields:
            Dictionary with progress and results
        """
        symbols = []
        for symbol, prices in data.items():
            symbols.append(
                pair_discovery_pb2.SymbolData(
                    symbol=symbol,
                    prices=prices.tolist(),
                )
            )
        
        request = pair_discovery_pb2.BatchDiscoveryRequest(
            symbols=symbols,
            significance=significance,
            min_hurst=min_hurst,
            transaction_cost=transaction_cost,
            max_pairs=max_pairs,
            parallel_workers=parallel_workers,
        )
        
        try:
            for update in self.stub.DiscoverPairs(request, timeout=300.0):
                result_dict = {
                    'pairs_tested': update.pairs_tested,
                    'pairs_found': update.pairs_found,
                    'progress': update.progress,
                    'is_final': update.is_final,
                }
                
                if update.HasField('result') and update.result.success:
                    result_dict['result'] = {
                        'pair': f'{update.result.symbol1}-{update.result.symbol2}',
                        'symbol1': update.result.symbol1,
                        'symbol2': update.result.symbol2,
                        'beta': update.result.beta,
                        'p_value': update.result.p_value,
                        'hurst': update.result.hurst,
                        'kappa': update.result.kappa,
                        'theta': update.result.theta,
                        'sigma': update.result.sigma,
                        'half_life': update.result.half_life,
                        'lower_boundary': update.result.lower_boundary,
                        'upper_boundary': update.result.upper_boundary,
                        'total_return': update.result.total_return,
                        'sharpe_ratio': update.result.sharpe_ratio,
                        'max_dd': update.result.max_drawdown,
                        'num_trades': update.result.num_trades,
                        'win_rate': update.result.win_rate,
                        'combined_score': update.result.combined_score,
                    }
                
                yield result_dict
        
        except grpc.RpcError as e:
            print(f"gRPC error in discovery: {e.details()}")
    
    def solve_hjb(
        self,
        kappa: float,
        theta: float,
        sigma: float,
        rho: float = 0.04,
        transaction_cost: float = 0.001,
        n_points: int = 200,
        max_iter: int = 2000,
        tolerance: float = 1e-6,
        n_std: float = 4.0,
    ) -> Tuple[float, float, float, int]:
        """
        Solve HJB equation for optimal boundaries.
        
        Returns:
            (lower_boundary, upper_boundary, residual, iterations)
        """
        request = pair_discovery_pb2.HJBRequest(
            kappa=kappa,
            theta=theta,
            sigma=sigma,
            rho=rho,
            transaction_cost=transaction_cost,
            n_points=n_points,
            max_iter=max_iter,
            tolerance=tolerance,
            n_std=n_std,
        )
        
        try:
            response = self.stub.SolveHJB(request, timeout=30.0)
            return (
                response.lower_boundary,
                response.upper_boundary,
                response.residual,
                response.iterations,
            )
        except grpc.RpcError as e:
            raise RuntimeError(f"HJB solver failed: {e.details()}")
    
    def estimate_ou_params(
        self,
        spread: np.ndarray,
        dt: float = 1.0 / 252.0,
        use_mle: bool = False,
    ) -> Dict[str, float]:
        """
        Estimate Ornstein-Uhlenbeck parameters.
        
        Returns:
            Dictionary with kappa, theta, sigma, half_life
        """
        request = pair_discovery_pb2.OUEstimationRequest(
            spread=spread.tolist(),
            dt=dt,
            use_mle=use_mle,
        )
        
        try:
            response = self.stub.EstimateOU(request, timeout=10.0)
            return {
                'kappa': response.kappa,
                'theta': response.theta,
                'sigma': response.sigma,
                'half_life': response.half_life,
            }
        except grpc.RpcError as e:
            raise RuntimeError(f"OU estimation failed: {e.details()}")
    
    def test_cointegration(
        self,
        y: np.ndarray,
        x: np.ndarray,
        significance: float = 0.05,
    ) -> Dict:
        """
        Test cointegration (Engle-Granger).
        
        Returns:
            Dictionary with beta, adf_statistic, p_value, is_cointegrated, spread
        """
        request = pair_discovery_pb2.CointegrationRequest(
            y=y.tolist(),
            x=x.tolist(),
            significance=significance,
        )
        
        try:
            response = self.stub.TestCointegration(request, timeout=10.0)
            return {
                'beta': response.beta,
                'adf_statistic': response.adf_statistic,
                'p_value': response.p_value,
                'is_cointegrated': response.is_cointegrated,
                'spread': np.array(response.spread),
            }
        except grpc.RpcError as e:
            raise RuntimeError(f"Cointegration test failed: {e.details()}")
    
    def calculate_hurst(
        self,
        series: np.ndarray,
        max_lag: int = 20,
        use_dfa: bool = False,
    ) -> float:
        """
        Calculate Hurst exponent.
        
        Returns:
            Hurst exponent (< 0.5 = mean-reverting)
        """
        request = pair_discovery_pb2.HurstRequest(
            series=series.tolist(),
            max_lag=max_lag,
            use_dfa=use_dfa,
        )
        
        try:
            response = self.stub.CalculateHurst(request, timeout=10.0)
            return response.hurst
        except grpc.RpcError as e:
            raise RuntimeError(f"Hurst calculation failed: {e.details()}")
    
    def backtest_strategy(
        self,
        spread: np.ndarray,
        lower_bound: float,
        upper_bound: float,
        transaction_cost: float = 0.001,
    ) -> Dict:
        """
        Backtest optimal switching strategy.
        
        Returns:
            Dictionary with performance metrics
        """
        request = pair_discovery_pb2.BacktestRequest(
            spread=spread.tolist(),
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            transaction_cost=transaction_cost,
        )
        
        try:
            response = self.stub.BacktestStrategy(request, timeout=30.0)
            return {
                'total_return': response.total_return,
                'sharpe_ratio': response.sharpe_ratio,
                'max_drawdown': response.max_drawdown,
                'num_trades': response.num_trades,
                'win_rate': response.win_rate,
                'pnl': np.array(response.pnl),
                'avg_holding_period': response.avg_holding_period,
                'profit_factor': response.profit_factor,
            }
        except grpc.RpcError as e:
            raise RuntimeError(f"Backtest failed: {e.details()}")


# Convenience functions
def test_pair_fast(
    prices_x: np.ndarray,
    prices_y: np.ndarray,
    pair_name: Optional[str] = None,
    **kwargs
) -> Optional[Dict]:
    """Quick pair test using gRPC backend"""
    with PairDiscoveryClient() as client:
        return client.test_pair(prices_x, prices_y, pair_name=pair_name, **kwargs)


def discover_pairs_fast(
    data: Dict[str, np.ndarray],
    **kwargs
) -> List[Dict]:
    """Quick pair discovery using gRPC backend"""
    results = []
    with PairDiscoveryClient() as client:
        for update in client.discover_pairs(data, **kwargs):
            if 'result' in update:
                results.append(update['result'])
    return results
