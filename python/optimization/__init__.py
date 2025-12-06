"""Optimization algorithms and techniques."""

from .advanced_optimization import (
    HMMRegimeDetector,
    MCMCOptimizer,
    RUST_AVAILABLE
)
from .signature_methods import (
    SignaturePortfolio,
    StochasticPortfolioTheory,
    compute_signature_rust,
    optimize_portfolio_rust,
    compute_portfolio_metrics_rust
)

# Alias for backward compatibility
SignatureMethods = SignaturePortfolio

__all__ = [
    'HMMRegimeDetector',
    'MCMCOptimizer',
    'SignaturePortfolio',
    'SignatureMethods',
    'StochasticPortfolioTheory',
    'compute_signature_rust',
    'optimize_portfolio_rust',
    'compute_portfolio_metrics_rust',
    'RUST_AVAILABLE'
]
