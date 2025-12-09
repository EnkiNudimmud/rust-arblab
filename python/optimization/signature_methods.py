"""
Signature Methods for Portfolio Optimization

This module provides Python wrappers for high-performance Rust implementations
of signature-based portfolio methods from Stochastic Portfolio Theory.

All computationally intensive operations (signature computation, covariance 
estimation, optimization) are performed in Rust for maximum performance.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)

# Try to import Rust implementation
try:
    from hft_py.signature import (
        compute_signature,
        compute_log_signature,
        expected_signature,
        signature_kernel,
        signature_covariance,
        signature_portfolio_weights,
        rank_based_portfolio,
        portfolio_metrics,
        signature_optimal_stopping,
        randomized_signature_features
    )
    RUST_AVAILABLE = True
    logger.info("Rust signature methods available")
except ImportError as e:
    RUST_AVAILABLE = False
    logger.warning(f"Rust signature methods not available: {e}")


class SignaturePortfolio:
    """
    Portfolio optimization using signature methods from Stochastic Portfolio Theory.
    
    This class implements the methodology from:
    - "Signature Methods in Stochastic Portfolio Theory" (Lyons et al.)
    - "Randomized Signature Methods in Optimal Portfolio Selection"
    
    All heavy computations are performed in Rust for performance.
    """
    
    def __init__(self, signature_level: int = 3, risk_aversion: float = 1.0):
        """
        Initialize signature-based portfolio optimizer.
        
        Args:
            signature_level: Truncation level for signature computation (typically 2-4)
            risk_aversion: Risk aversion parameter λ in mean-variance optimization
        """
        if not RUST_AVAILABLE:
            raise RuntimeError("Rust signature methods not available. Please rebuild with maturin.")
        
        self.signature_level = signature_level
        self.risk_aversion = risk_aversion
        self.weights_history = []
        self.returns_history = []
        
    def compute_path_signature(self, path: np.ndarray) -> np.ndarray:
        """
        Compute truncated path signature.
        
        For a path X in R^d, computes S(X) = (1, S^1(X), S^2(X), ..., S^N(X))
        where S^k(X) are the level-k iterated integrals.
        
        Args:
            path: Array of shape (T, d) representing the path
            
        Returns:
            Signature vector of length ∑_{k=0}^N d^k
        """
        path_list = path.tolist()
        sig = compute_signature(path_list, self.signature_level)
        return np.array(sig)
    
    def compute_signature_kernel(self, path1: np.ndarray, path2: np.ndarray) -> float:
        """
        Compute signature kernel K(X, Y) = <S(X), S(Y)>.
        
        This provides a measure of path similarity that respects
        the geometry of path space.
        
        Args:
            path1: First path, shape (T1, d)
            path2: Second path, shape (T2, d)
            
        Returns:
            Kernel value (similarity measure)
        """
        p1 = path1.tolist()
        p2 = path2.tolist()
        return signature_kernel(p1, p2, self.signature_level)
    
    def optimize_portfolio(
        self,
        returns: np.ndarray,
        allow_short: bool = False
    ) -> np.ndarray:
        """
        Optimize portfolio using signature-based mean-variance optimization.
        
        Solves:
            max w^T μ_sig - (λ/2) w^T Σ_sig w
            subject to: ∑ w_i = 1, w_i ≥ 0 (if long-only)
        
        where μ_sig and Σ_sig are computed from path signatures.
        
        Args:
            returns: Asset returns matrix, shape (n_assets, n_timesteps)
            allow_short: Whether to allow short positions
            
        Returns:
            Optimal weights vector, shape (n_assets,)
        """
        returns_list = returns.tolist()
        weights = signature_portfolio_weights(
            returns_list,
            self.signature_level,
            self.risk_aversion,
            allow_short
        )
        return np.array(weights)
    
    def rank_weighted_portfolio(
        self,
        returns: np.ndarray,
        method: str = "diversity"
    ) -> np.ndarray:
        """
        Compute rank-based portfolio weights from Stochastic Portfolio Theory.
        
        Portfolio weights depend on asset rank:
            π_i(t) = g(rank(R_i(t)) / N)
        
        Args:
            returns: Current period returns for each asset
            method: Generating function type:
                - "market": Equal weight (1/N)
                - "entropy": Entropy-weighted (-u log u)
                - "diversity": Diversity-weighted (log u)
                - "linear": Momentum (u)
                - "contrarian": Contrarian (1-u)
                
        Returns:
            Portfolio weights vector
        """
        returns_list = returns.tolist()
        weights = rank_based_portfolio(returns_list, method)
        return np.array(weights)
    
    def compute_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """
        Compute portfolio performance metrics.
        
        Returns:
            Dictionary containing:
                - total_return: Cumulative return
                - sharpe_ratio: Annualized Sharpe ratio
                - max_drawdown: Maximum drawdown
                - volatility: Return volatility
        """
        returns_list = returns.tolist()
        return portfolio_metrics(returns_list)
    
    def optimal_stopping_time(
        self,
        path: np.ndarray,
        threshold: float = 1.0,
        window: int = 20
    ) -> int:
        """
        Determine optimal stopping time using signature features.
        
        Uses rolling signature features to detect when to rebalance
        or close positions based on learned patterns.
        
        Args:
            path: Price path, shape (T, d)
            threshold: Stopping threshold
            window: Rolling window size
            
        Returns:
            Optimal stopping time (index)
        """
        path_list = path.tolist()
        return signature_optimal_stopping(
            path_list,
            self.signature_level,
            threshold,
            window
        )
    
    def randomized_features(
        self,
        path: np.ndarray,
        n_features: int = 100,
        seed: int = 42
    ) -> np.ndarray:
        """
        Compute randomized signature features for dimensionality reduction.
        
        Uses random projections to reduce signature dimension while
        preserving essential information for portfolio optimization.
        
        Args:
            path: Price path, shape (T, d)
            n_features: Number of random features
            seed: Random seed for reproducibility
            
        Returns:
            Feature vector of length n_features
        """
        path_list = path.tolist()
        features = randomized_signature_features(
            path_list,
            self.signature_level,
            n_features,
            seed
        )
        return np.array(features)
    
    def backtest(
        self,
        returns: np.ndarray,
        rebalance_freq: int = 20,
        allow_short: bool = False
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Backtest signature-based portfolio strategy.
        
        Args:
            returns: Asset returns, shape (n_assets, n_timesteps)
            rebalance_freq: Rebalancing frequency (in timesteps)
            allow_short: Allow short positions
            
        Returns:
            Tuple of (portfolio_returns, metrics_dict)
        """
        n_assets, n_timesteps = returns.shape
        portfolio_returns = []
        
        # Initialize with equal weights
        weights = np.ones(n_assets) / n_assets
        
        for t in range(n_timesteps):
            # Rebalance periodically
            if t % rebalance_freq == 0 and t > 0:
                # Use historical data up to current time
                historical_returns = returns[:, max(0, t-100):t]
                if historical_returns.shape[1] > 10:
                    weights = self.optimize_portfolio(historical_returns, allow_short)
                    self.weights_history.append(weights.copy())
            
            # Compute portfolio return
            period_returns = returns[:, t]
            portfolio_return = np.dot(weights, period_returns)
            portfolio_returns.append(portfolio_return)
        
        portfolio_returns = np.array(portfolio_returns)
        self.returns_history = portfolio_returns
        
        # Compute metrics
        metrics = self.compute_metrics(portfolio_returns)
        
        return portfolio_returns, metrics


class StochasticPortfolioTheory:
    """
    Implementation of Stochastic Portfolio Theory methods.
    
    Focuses on rank-based portfolios and relative arbitrage opportunities
    as described in the SPT literature.
    """
    
    def __init__(self):
        if not RUST_AVAILABLE:
            raise RuntimeError("Rust signature methods not available")
    
    def compute_relative_returns(
        self,
        returns: np.ndarray
    ) -> np.ndarray:
        """
        Compute relative returns (market-relative performance).
        
        Args:
            returns: Asset returns, shape (n_assets, n_timesteps)
            
        Returns:
            Relative returns matrix
        """
        market_returns = returns.mean(axis=0)
        return returns - market_returns[np.newaxis, :]
    
    def diversity_weighted_portfolio(
        self,
        returns: np.ndarray
    ) -> np.ndarray:
        """
        Compute diversity-weighted portfolio from SPT.
        
        This portfolio maximizes diversity in the market,
        giving higher weights to outperforming assets.
        
        Args:
            returns: Current period returns
            
        Returns:
            Portfolio weights
        """
        return rank_based_portfolio(returns.tolist(), "diversity")
    
    def entropy_weighted_portfolio(
        self,
        returns: np.ndarray
    ) -> np.ndarray:
        """
        Compute entropy-weighted portfolio from SPT.
        
        This portfolio is related to the market entropy and
        provides a balanced approach to portfolio weighting.
        
        Args:
            returns: Current period returns
            
        Returns:
            Portfolio weights
        """
        return rank_based_portfolio(returns.tolist(), "entropy")


# Convenience functions
def compute_signature_rust(path: np.ndarray, level: int = 3) -> np.ndarray:
    """Compute path signature (convenience function)."""
    if not RUST_AVAILABLE:
        raise RuntimeError("Rust not available")
    sig = compute_signature(path.tolist(), level)
    return np.array(sig)


def optimize_portfolio_rust(
    returns: np.ndarray,
    level: int = 3,
    risk_aversion: float = 1.0,
    allow_short: bool = False
) -> np.ndarray:
    """Optimize portfolio using signature methods (convenience function)."""
    if not RUST_AVAILABLE:
        raise RuntimeError("Rust not available")
    weights = signature_portfolio_weights(
        returns.tolist(),
        level,
        risk_aversion,
        allow_short
    )
    return np.array(weights)


def compute_portfolio_metrics_rust(returns: np.ndarray) -> Dict[str, float]:
    """Compute portfolio metrics (convenience function)."""
    if not RUST_AVAILABLE:
        raise RuntimeError("Rust not available")
    return portfolio_metrics(returns.tolist())


__all__ = [
    'SignaturePortfolio',
    'StochasticPortfolioTheory',
    'compute_signature_rust',
    'optimize_portfolio_rust',
    'compute_portfolio_metrics_rust',
    'RUST_AVAILABLE'
]
