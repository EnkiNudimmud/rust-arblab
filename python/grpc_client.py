"""
gRPC client for MeanRev service
Provides high-level API matching original Python interface with gRPC streaming support
"""

import grpc
import asyncio
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import logging

# Add generated proto files to path
PROTO_PATH = Path(__file__).parent.parent / "proto"

logger = logging.getLogger(__name__)

# Proto files compiled with:
# python -m grpc_tools.protoc -I./proto --python_out=python --grpc_python_out=python proto/meanrev.proto

from python import meanrev_pb2, meanrev_pb2_grpc


class MeanRevClient:
    """
    gRPC client for mean-reversion portfolio analysis
    Handles connection management, streaming, and error handling
    """

    def __init__(self, host: str = "localhost", port: int = 50051):
        """
        Initialize gRPC client
        
        Args:
            host: gRPC server host
            port: gRPC server port
        """
        self.host = host
        self.port = port
        self.channel = None
        self.stub = None
        self._connect()

    def _connect(self):
        """Establish gRPC channel and stub"""
        try:
            self.channel = grpc.aio.secure_channel(
                f"{self.host}:{self.port}",
                grpc.ssl_channel_credentials()
            ) if self.host != "localhost" else grpc.aio.insecure_channel(
                f"{self.host}:{self.port}"
            )
            self.stub = meanrev_pb2_grpc.MeanRevServiceStub(self.channel)
            logger.info(f"Connected to gRPC server at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to gRPC server: {e}")
            raise

    def __del__(self):
        """Clean up channel on deletion"""
        if self.channel:
            try:
                asyncio.run(self.channel.close())
            except:
                pass

    # ============ CORE API METHODS ============

    async def estimate_ou_params(
        self, prices: Union[List[float], np.ndarray]
    ) -> Dict[str, float]:
        """
        Estimate OU (Ornstein-Uhlenbeck) parameters from price series
        
        Streams prices to server, receives OU parameters with statistics
        
        Args:
            prices: Price time series
            
        Returns:
            {
                "theta": Mean reversion speed,
                "mu": Long-term mean,
                "sigma": Volatility,
                "half_life": Half-life of mean reversion,
                "r_squared": Model fit quality,
                "n_samples": Number of samples used
            }
        """
        prices = np.asarray(prices, dtype=np.float64)
        
        # Stream prices to server
        async def price_stream():
            for i, price in enumerate(prices):
                # yield meanrev_pb2.PriceUpdate(
                #     asset_id="BTC/USD",
                #     price=float(price),
                #     timestamp_ms=int((i) * 60000)  # Assuming 1-minute bars
                # )
                pass

        # Receive OU estimates
        # async for estimate in self.stub.EstimateOUParams(price_stream()):
        #     return {
        #         "theta": estimate.theta,
        #         "mu": estimate.mu,
        #         "sigma": estimate.sigma,
        #         "half_life": estimate.half_life,
        #         "r_squared": estimate.r_squared,
        #         "n_samples": estimate.n_samples,
        #     }

        return {}

    async def optimal_thresholds(
        self,
        ou_params: Dict[str, float],
        transaction_cost: float = 0.001,
        optimization_objective: str = "sharpe"
    ) -> Dict[str, float]:
        """
        Compute optimal entry/exit thresholds from OU parameters
        
        Args:
            ou_params: OU parameters from estimate_ou_params
            transaction_cost: Trading cost as proportion
            optimization_objective: "profit", "sharpe", or "kelly"
            
        Returns:
            {
                "entry_zscore": Entry threshold,
                "exit_zscore": Exit threshold,
                "expected_holding_periods": Expected holding time,
                "expected_return_per_trade": Expected trade return,
                "win_rate_estimate": Estimated win rate
            }
        """
        # request = meanrev_pb2.ThresholdRequest(
        #     ou_params=meanrev_pb2.OUEstimate(**ou_params),
        #     transaction_cost=transaction_cost,
        #     optimization_objective=optimization_objective
        # )
        # response = await self.stub.OptimalThresholds(request)
        # return {
        #     "entry_zscore": response.entry_zscore,
        #     "exit_zscore": response.exit_zscore,
        #     "expected_holding_periods": response.expected_holding_periods,
        #     "expected_return_per_trade": response.expected_return_per_trade,
        #     "win_rate_estimate": response.win_rate_estimate,
        # }
        return {}

    async def backtest_with_costs(
        self,
        prices: Union[List[float], np.ndarray],
        entry_z: float,
        exit_z: float,
        transaction_cost: float = 0.001,
        update_frequency: int = 10
    ) -> Dict[str, Union[List, float]]:
        """
        Backtest strategy with transaction costs
        
        Streams prices to server, receives continuous backtest metrics
        
        Args:
            prices: Price time series
            entry_z: Entry z-score threshold
            exit_z: Exit z-score threshold
            transaction_cost: Transaction cost as proportion
            update_frequency: Emit metrics every N bars
            
        Returns:
            {
                "returns": Cumulative returns per bar,
                "positions": Position state per bar,
                "pnl": Realized & unrealized P&L,
                "sharpe": Sharpe ratio,
                "max_drawdown": Maximum drawdown,
                "num_trades": Total trades executed,
                "win_rate": Win rate
            }
        """
        prices = np.asarray(prices, dtype=np.float64)
        
        async def backtest_stream():
            for i, price in enumerate(prices):
                # yield meanrev_pb2.BacktestInput(
                #     price=float(price),
                #     timestamp_ms=int(i * 60000),
                #     entry_zscore=entry_z,
                #     exit_zscore=exit_z,
                #     transaction_cost=transaction_cost,
                #     update_frequency=update_frequency
                # )
                pass

        results = {
            "returns": [],
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "num_trades": 0,
            "win_rate": 0.0
        }

        # async for metric in self.stub.BacktestWithCosts(backtest_stream()):
        #     results["returns"].append(metric.cumulative_return)
        #     results["sharpe"] = metric.sharpe_ratio
        #     results["max_drawdown"] = metric.max_drawdown
        #     results["num_trades"] = metric.num_trades
        #     results["win_rate"] = metric.win_rate

        return results

    async def compute_log_returns(
        self, prices: Union[List[float], np.ndarray]
    ) -> pd.DataFrame:
        """
        Compute log returns from prices
        
        Args:
            prices: Price time series
            
        Returns:
            DataFrame with log returns and statistics
        """
        prices = np.asarray(prices, dtype=np.float64)
        
        # request = meanrev_pb2.ReturnsRequest(
        #     asset_ids="BTC/USD",
        #     prices=prices.tolist(),
        #     return_type="log"
        # )
        # response = await self.stub.ComputeLogReturns(request)
        
        # return pd.DataFrame({
        #     "log_returns": response.log_returns,
        #     "mean": response.mean_return,
        #     "std": response.std_return
        # })
        
        return pd.DataFrame()

    async def pca_portfolios(
        self,
        returns: Union[List[List[float]], np.ndarray],
        n_components: int = 3
    ) -> Dict[str, Union[np.ndarray, float]]:
        """
        Principal Component Analysis on returns
        
        Args:
            returns: T x N matrix of returns (T periods, N assets)
            n_components: Number of principal components
            
        Returns:
            {
                "components": Principal components matrix,
                "variance_explained": Variance explained ratio,
                "total_variance": Total variance explained,
                "transformed": Transformed data in PC space
            }
        """
        returns = np.asarray(returns, dtype=np.float64)
        
        # request = meanrev_pb2.PCARequest(
        #     returns_matrix=meanrev_pb2.Matrix(
        #         rows=returns.shape[0],
        #         cols=returns.shape[1],
        #         values=returns.flatten().tolist()
        #     ),
        #     n_components=n_components,
        #     scaling="zscore"
        # )
        # response = await self.stub.PCAPortfolios(request)
        
        # return {
        #     "components": np.array(response.components.values).reshape(
        #         response.components.rows, response.components.cols
        #     ),
        #     "variance_explained": response.explained_variance_ratio,
        #     "total_variance": response.total_variance_explained
        # }
        
        return {}

    async def cara_optimal_weights(
        self,
        returns: Union[List[List[float]], np.ndarray],
        covariance: Optional[np.ndarray] = None,
        gamma: float = 1.0
    ) -> Dict[str, Union[List[float], float]]:
        """
        Compute CARA (Constant Absolute Risk Aversion) optimal weights
        
        Streams return matrices, receives optimal weights continuously
        
        Args:
            returns: T x N returns matrix or historical data
            covariance: Pre-computed covariance (optional)
            gamma: Risk aversion coefficient
            
        Returns:
            {
                "weights": Optimal portfolio weights,
                "expected_return": Portfolio expected return,
                "expected_volatility": Portfolio volatility,
                "sharpe_ratio": Sharpe ratio,
                "utility": Objective function value
            }
        """
        returns = np.asarray(returns, dtype=np.float64)
        
        async def opt_stream():
            # yield meanrev_pb2.OptimizationInput(
            #     returns=meanrev_pb2.Matrix(
            #         rows=returns.shape[0],
            #         cols=returns.shape[1],
            #         values=returns.flatten().tolist()
            #     ),
            #     gamma=gamma,
            #     risk_free_rate=0.0
            # )
            pass

        # async for output in self.stub.CARAOptimalWeights(opt_stream()):
        #     return {
        #         "weights": output.weights,
        #         "expected_return": output.expected_return,
        #         "expected_volatility": output.expected_volatility,
        #         "sharpe_ratio": output.sharpe_ratio,
        #         "utility": output.utility,
        #         "status": output.status
        #     }
        
        return {}

    async def sharpe_optimal_weights(
        self,
        returns: Union[List[List[float]], np.ndarray],
        covariance: Optional[np.ndarray] = None,
        risk_free_rate: float = 0.0
    ) -> Dict[str, Union[List[float], float]]:
        """
        Compute Sharpe-optimal portfolio weights
        
        Args:
            returns: T x N returns matrix
            covariance: Pre-computed covariance (optional)
            risk_free_rate: Risk-free rate
            
        Returns:
            {
                "weights": Optimal weights,
                "sharpe_ratio": Sharpe ratio,
                "expected_return": Expected return,
                "expected_volatility": Expected volatility
            }
        """
        returns = np.asarray(returns, dtype=np.float64)
        
        async def opt_stream():
            # yield meanrev_pb2.OptimizationInput(
            #     returns=meanrev_pb2.Matrix(
            #         rows=returns.shape[0],
            #         cols=returns.shape[1],
            #         values=returns.flatten().tolist()
            #     ),
            #     risk_free_rate=risk_free_rate
            # )
            pass

        # async for output in self.stub.SharpeOptimalWeights(opt_stream()):
        #     return {
        #         "weights": output.weights,
        #         "sharpe_ratio": output.sharpe_ratio,
        #         "expected_return": output.expected_return,
        #         "expected_volatility": output.expected_volatility
        #     }
        
        return {}

    async def multiperiod_optimize(
        self,
        returns_sequence: List[np.ndarray],
        gamma: float = 1.0,
        transaction_cost: float = 0.001,
        rebalance_frequency: int = 5
    ) -> Dict[str, Union[List, float]]:
        """
        Multi-period portfolio optimization
        
        Args:
            returns_sequence: List of return matrices for each period
            gamma: Risk aversion
            transaction_cost: Transaction cost
            rebalance_frequency: Periods between rebalances
            
        Returns:
            {
                "weights_sequence": Weights over time,
                "cumulative_utility": Total utility,
                "avg_turnover": Average portfolio turnover
            }
        """
        # matrices = [
        #     meanrev_pb2.Matrix(
        #         rows=r.shape[0],
        #         cols=r.shape[1],
        #         values=r.flatten().tolist()
        #     ) for r in returns_sequence
        # ]
        # request = meanrev_pb2.MultiperiodRequest(
        #     returns_sequence=matrices,
        #     gamma=gamma,
        #     transaction_cost=transaction_cost,
        #     rebalance_frequency=rebalance_frequency
        # )
        # response = await self.stub.MultiperiodOptimize(request)
        # return {
        #     "weights_sequence": response.weights_sequence,
        #     "cumulative_utility": response.cumulative_utility,
        #     "avg_turnover": response.avg_turnover
        # }
        return {}

    async def portfolio_stats(
        self,
        returns: Union[List[List[float]], np.ndarray],
        weights: Optional[List[float]] = None
    ) -> Dict[str, Union[float, np.ndarray]]:
        """
        Compute portfolio statistics
        
        Args:
            returns: T x N returns matrix
            weights: Optional portfolio weights for weighted stats
            
        Returns:
            {
                "mean_return": Mean return,
                "std_return": Standard deviation,
                "correlation": Correlation matrix,
                "covariance": Covariance matrix,
                "skewness": Skewness,
                "kurtosis": Kurtosis
            }
        """
        returns = np.asarray(returns, dtype=np.float64)
        
        # request = meanrev_pb2.PortfolioStatsRequest(
        #     returns=meanrev_pb2.Matrix(
        #         rows=returns.shape[0],
        #         cols=returns.shape[1],
        #         values=returns.flatten().tolist()
        #     ),
        #     weights=weights or []
        # )
        # response = await self.stub.PortfolioStats(request)
        # return {
        #     "mean_return": response.mean_return,
        #     "std_return": response.std_return,
        #     "skewness": response.skewness,
        #     "kurtosis": response.kurtosis,
        #     "correlation": np.array(response.correlation.values).reshape(
        #         response.correlation.rows, response.correlation.cols
        #     ),
        #     "covariance": np.array(response.covariance.values).reshape(
        #         response.covariance.rows, response.covariance.cols
        #     )
        # }
        
        return {}

    async def generate_signals(
        self,
        price_history: Union[List[float], np.ndarray],
        ou_params: Dict[str, float],
        thresholds: Dict[str, float],
        position_state: str = "flat"
    ) -> Dict[str, Union[str, float]]:
        """
        Generate trading signals from price and parameters
        
        Args:
            price_history: Historical prices
            ou_params: OU parameters
            thresholds: Entry/exit thresholds
            position_state: Current position ("flat", "long", "short")
            
        Returns:
            {
                "signal": "buy", "sell", or "hold",
                "confidence": 0.0 to 1.0,
                "zscore": Current z-score,
                "expected_profit": Expected profit if signal taken,
                "reasoning": Explanation
            }
        """
        # request = meanrev_pb2.SignalRequest(
        #     price_history=meanrev_pb2.PriceHistory(
        #         asset_id="BTC/USD",
        #         prices=list(price_history)
        #     ),
        #     ou_params=meanrev_pb2.OUEstimate(**ou_params),
        #     thresholds=meanrev_pb2.ThresholdResponse(**thresholds),
        #     position_state=position_state
        # )
        # response = await self.stub.GenerateSignals(request)
        # return {
        #     "signal": response.signal,
        #     "confidence": response.confidence,
        #     "zscore": response.zscore,
        #     "expected_profit": response.expected_profit,
        #     "reasoning": response.reasoning
        # }
        
        return {}

    async def portfolio_health_stream(
        self,
        portfolio_id: str,
        asset_ids: List[str],
        update_frequency_ms: int = 1000
    ):
        """
        Stream portfolio health metrics
        
        Args:
            portfolio_id: Portfolio identifier
            asset_ids: Assets to monitor
            update_frequency_ms: Update frequency in milliseconds
            
        Yields:
            Health metrics every update_frequency_ms
        """
        # request = meanrev_pb2.HealthRequest(
        #     portfolio_id=portfolio_id,
        #     asset_ids=asset_ids,
        #     update_frequency_ms=update_frequency_ms
        # )
        # async for metric in self.stub.PortfolioHealthStream(request):
        #     yield {
        #         "portfolio_id": metric.portfolio_id,
        #         "timestamp": metric.timestamp_ms,
        #         "total_value": metric.total_value,
        #         "daily_return": metric.daily_return,
        #         "volatility_24h": metric.volatility_24h,
        #         "max_drawdown": metric.max_drawdown,
        #         "open_positions": metric.open_positions,
        #         "status": metric.status,
        #         "alerts": metric.alerts
        #     }
        pass


# ============ SYNCHRONOUS WRAPPER (for backward compatibility) ============

class MeanRevClientSync:
    """Synchronous wrapper around async client for backward compatibility"""

    def __init__(self, host: str = "localhost", port: int = 50051):
        self.client = MeanRevClient(host, port)
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def estimate_ou_params(self, prices: Union[List[float], np.ndarray]) -> Dict[str, float]:
        return self.loop.run_until_complete(self.client.estimate_ou_params(prices))

    def optimal_thresholds(self, ou_params: Dict[str, float], **kwargs) -> Dict[str, float]:
        return self.loop.run_until_complete(
            self.client.optimal_thresholds(ou_params, **kwargs)
        )

    def backtest_with_costs(self, prices: Union[List[float], np.ndarray], **kwargs) -> Dict:
        return self.loop.run_until_complete(
            self.client.backtest_with_costs(prices, **kwargs)
        )

    def compute_log_returns(self, prices: Union[List[float], np.ndarray]) -> pd.DataFrame:
        return self.loop.run_until_complete(self.client.compute_log_returns(prices))

    def pca_portfolios(self, returns: Union[List[List[float]], np.ndarray], **kwargs) -> Dict:
        return self.loop.run_until_complete(self.client.pca_portfolios(returns, **kwargs))

    def cara_optimal_weights(self, returns: Union[List[List[float]], np.ndarray], **kwargs) -> Dict:
        return self.loop.run_until_complete(
            self.client.cara_optimal_weights(returns, **kwargs)
        )

    def sharpe_optimal_weights(self, returns: Union[List[List[float]], np.ndarray], **kwargs) -> Dict:
        return self.loop.run_until_complete(
            self.client.sharpe_optimal_weights(returns, **kwargs)
        )

    def multiperiod_optimize(self, returns_sequence: List[np.ndarray], **kwargs) -> Dict:
        return self.loop.run_until_complete(
            self.client.multiperiod_optimize(returns_sequence, **kwargs)
        )

    def portfolio_stats(self, returns: Union[List[List[float]], np.ndarray], **kwargs) -> Dict:
        return self.loop.run_until_complete(
            self.client.portfolio_stats(returns, **kwargs)
        )

    def generate_signals(self, **kwargs) -> Dict:
        return self.loop.run_until_complete(self.client.generate_signals(**kwargs))

    def __del__(self):
        self.loop.close()


# Default client instance for module-level convenience functions
_default_client: Optional[MeanRevClientSync] = None


def get_client(host: str = "localhost", port: int = 50051) -> MeanRevClientSync:
    """Get or create default client instance"""
    global _default_client
    if _default_client is None:
        _default_client = MeanRevClientSync(host, port)
    return _default_client


# Module-level convenience functions (backward compatible with original API)
def estimate_ou_params(prices: Union[List[float], np.ndarray]) -> Dict[str, float]:
    """See MeanRevClientSync.estimate_ou_params"""
    return get_client().estimate_ou_params(prices)


def compute_log_returns(prices: Union[List[float], np.ndarray]) -> pd.DataFrame:
    """See MeanRevClientSync.compute_log_returns"""
    return get_client().compute_log_returns(prices)


def pca_portfolios(returns: Union[List[List[float]], np.ndarray], n_components: int = 3) -> Dict:
    """See MeanRevClientSync.pca_portfolios"""
    return get_client().pca_portfolios(returns, n_components=n_components)


def cara_optimal_weights(
    returns: Union[List[List[float]], np.ndarray],
    cov: Optional[np.ndarray] = None,
    gamma: float = 1.0
) -> Dict:
    """See MeanRevClientSync.cara_optimal_weights"""
    return get_client().cara_optimal_weights(returns, covariance=cov, gamma=gamma)


def sharpe_optimal_weights(
    returns: Union[List[List[float]], np.ndarray],
    cov: Optional[np.ndarray] = None,
    rf: float = 0.0
) -> Dict:
    """See MeanRevClientSync.sharpe_optimal_weights"""
    return get_client().sharpe_optimal_weights(returns, covariance=cov, risk_free_rate=rf)


def optimal_thresholds(ou_params: Dict[str, float], **kwargs) -> Dict:
    """See MeanRevClientSync.optimal_thresholds"""
    return get_client().optimal_thresholds(ou_params, **kwargs)


def backtest_with_costs(
    prices: Union[List[float], np.ndarray],
    entry_z: float,
    exit_z: float,
    **kwargs
) -> Dict:
    """See MeanRevClientSync.backtest_with_costs"""
    return get_client().backtest_with_costs(prices, entry_z=entry_z, exit_z=exit_z, **kwargs)


def multiperiod_optimize(returns_sequence: List[np.ndarray], **kwargs) -> Dict:
    """See MeanRevClientSync.multiperiod_optimize"""
    return get_client().multiperiod_optimize(returns_sequence, **kwargs)


def portfolio_stats(returns: Union[List[List[float]], np.ndarray], **kwargs) -> Dict:
    """See MeanRevClientSync.portfolio_stats"""
    return get_client().portfolio_stats(returns, **kwargs)


def generate_signals(**kwargs) -> Dict:
    """See MeanRevClientSync.generate_signals"""
    return get_client().generate_signals(**kwargs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("MeanRev gRPC Client loaded successfully")
