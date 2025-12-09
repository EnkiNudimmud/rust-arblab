"""
Python Version Compatibility Module
====================================

Ensures compatibility across Python 3.7+ versions.
Provides fallbacks for features not available in older versions.
"""

import sys

# Check Python version
PYTHON_VERSION = sys.version_info
MIN_PYTHON_VERSION = (3, 7)
RECOMMENDED_PYTHON_VERSION = (3, 8)

def check_python_version(min_version=MIN_PYTHON_VERSION, recommended_version=RECOMMENDED_PYTHON_VERSION):
    """Check if Python version meets requirements"""
    
    if PYTHON_VERSION < min_version:
        raise RuntimeError(
            f"Python {min_version[0]}.{min_version[1]}+ is required. "
            f"You are using Python {PYTHON_VERSION[0]}.{PYTHON_VERSION[1]}.{PYTHON_VERSION[2]}"
        )
    
    if PYTHON_VERSION < recommended_version:
        import warnings
        warnings.warn(
            f"Python {recommended_version[0]}.{recommended_version[1]}+ is recommended for best performance. "
            f"You are using Python {PYTHON_VERSION[0]}.{PYTHON_VERSION[1]}.{PYTHON_VERSION[2]}",
            UserWarning
        )
    
    return True

# Typing compatibility
# Python 3.9+ supports using dict, list, tuple directly in type hints
# Python 3.7-3.8 require typing.Dict, typing.List, typing.Tuple
# For compatibility, we always use typing module
from typing import Dict, List, Tuple, Optional, Any

USE_TYPING_MODULE = True
    
__all__ = [
    'check_python_version',
    'Dict',
    'List', 
    'Tuple',
    'Optional',
    'Any',
    'PYTHON_VERSION',
    'MIN_PYTHON_VERSION',
    'RECOMMENDED_PYTHON_VERSION'
]
