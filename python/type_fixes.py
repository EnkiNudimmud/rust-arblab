"""
Quick fixes for common Pylance type errors.

Import this module to fix type compatibility issues between pandas and numpy.
"""

import numpy as np
import pandas as pd
from typing import Union, Any

# Type alias that handles pandas ExtensionArray
ArrayLike = Union[np.ndarray, pd.Series, pd.DataFrame, list, tuple, Any]


def safe_mean(data: ArrayLike) -> float:
    """Calculate mean, handling pandas types correctly."""
    if isinstance(data, (pd.Series, pd.DataFrame)):
        return float(data.to_numpy().mean())
    return float(np.mean(data))


def safe_std(data: ArrayLike) -> float:
    """Calculate standard deviation, handling pandas types correctly."""
    if isinstance(data, (pd.Series, pd.DataFrame)):
        return float(data.to_numpy().std())
    return float(np.std(data))


def safe_median(data: ArrayLike) -> float:
    """Calculate median, handling pandas types correctly."""
    if isinstance(data, (pd.Series, pd.DataFrame)):
        return float(data.to_numpy().median() if hasattr(data.to_numpy(), 'median') else np.median(data.to_numpy()))
    return float(np.median(data))


def as_numpy(data: ArrayLike) -> np.ndarray:
    """Convert any array-like to numpy array."""
    if isinstance(data, pd.DataFrame):
        return data.values
    elif isinstance(data, pd.Series):
        return data.to_numpy()
    elif isinstance(data, np.ndarray):
        return data
    else:
        return np.asarray(data)
