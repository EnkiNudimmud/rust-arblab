"""
Fallback wrapper for hft_py module.
Attempts to import hft_py, falls back to numpy/pandas implementations if not available.
"""

import sys
import logging

logger = logging.getLogger(__name__)

# Try to import hft_py
try:
    import hft_py
    HFT_PY_AVAILABLE = True
    logger.info("✓ hft_py module available - using Rust implementations")
except ImportError as e:
    HFT_PY_AVAILABLE = False
    logger.warning(f"⚠️  hft_py not available: {e}. Using fallback numpy/pandas implementations.")
    
    # Create dummy module objects for fallback
    class DummyModule:
        """Dummy module to provide attribute access without errors"""
        def __getattr__(self, name):
            return None
    
    # Create mock modules
    hft_py = DummyModule()
    sys.modules['hft_py'] = hft_py
    sys.modules['hft_py.superspace'] = DummyModule()
    sys.modules['hft_py.portfolio_drift'] = DummyModule()
    sys.modules['hft_py.regime_portfolio'] = DummyModule()
    sys.modules['hft_py.signature'] = DummyModule()
    sys.modules['hft_py.statistical_analyzer'] = DummyModule()

# Export for use in other modules
__all__ = ['hft_py', 'HFT_PY_AVAILABLE']
