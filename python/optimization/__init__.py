"""Optimization algorithms and techniques."""

from .advanced_optimization import (
    HMMRegimeDetector,
    MCMCOptimizer,
    RUST_AVAILABLE
)

# Try to import signature methods, but don't fail if Rust bindings not available
try:
    from .signature_methods import (
        SignaturePortfolio,
        StochasticPortfolioTheory,
        compute_signature_rust,
        optimize_portfolio_rust,
        compute_portfolio_metrics_rust
    )
    # Alias for backward compatibility
    SignatureMethods = SignaturePortfolio
    SIGNATURE_AVAILABLE = True
except ImportError:
    # Create dummy classes if Rust bindings not available
    SignaturePortfolio = None
    StochasticPortfolioTheory = None
    SignatureMethods = None
    compute_signature_rust = None
    optimize_portfolio_rust = None
    compute_portfolio_metrics_rust = None
    SIGNATURE_AVAILABLE = False

__all__ = [
    'HMMRegimeDetector',
    'MCMCOptimizer',
    'SignaturePortfolio',
    'SignatureMethods',
    'StochasticPortfolioTheory',
    'compute_signature_rust',
    'optimize_portfolio_rust',
    'compute_portfolio_metrics_rust',
    'RUST_AVAILABLE',
    'SIGNATURE_AVAILABLE'
]
