"""
Type definitions and conversion utilities.

Provides consistent type handling across the codebase,
fixing Pylance errors related to pandas/numpy type compatibility.
"""

import numpy as np
import pandas as pd
from typing import Union, Protocol, runtime_checkable
from numpy.typing import NDArray

# Type aliases for clarity
ArrayLike = Union[np.ndarray, pd.Series, pd.DataFrame, list, tuple]
FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]


@runtime_checkable
class SupportsArray(Protocol):
    """Protocol for objects that can be converted to numpy arrays."""
    
    def __array__(self) -> np.ndarray:
        """Convert to numpy array."""
        ...


def to_numpy_array(data: ArrayLike, dtype: type = np.float64, copy: bool = False) -> np.ndarray:
    """
    Convert array-like data to numpy array with proper type handling.
    
    Fixes Pylance errors by ensuring proper type conversion from pandas
    ExtensionArray types to numpy arrays.
    
    Parameters
    ----------
    data : ArrayLike
        Input data (pandas Series, DataFrame, numpy array, list, etc.)
    dtype : type, default np.float64
        Target numpy dtype
    copy : bool, default False
        Whether to force a copy
        
    Returns
    -------
    np.ndarray
        Numpy array with specified dtype
        
    Examples
    --------
    >>> series = pd.Series([1, 2, 3])
    >>> arr = to_numpy_array(series)
    >>> type(arr)
    <class 'numpy.ndarray'>
    """
    if isinstance(data, pd.DataFrame):
        return data.values.astype(dtype, copy=copy)
    elif isinstance(data, pd.Series):
        # Explicitly convert pandas Series to numpy array to avoid ExtensionArray issues
        return data.to_numpy(dtype=dtype, copy=copy)
    elif isinstance(data, np.ndarray):
        if data.dtype != dtype or copy:
            return data.astype(dtype, copy=copy)
        return data
    else:
        # Handle lists, tuples, and other array-like objects
        return np.asarray(data, dtype=dtype)


def validate_array(
    data: ArrayLike,
    name: str = "data",
    min_length: int = 1,
    allow_nan: bool = False,
    allow_inf: bool = False
) -> np.ndarray:
    """
    Validate and convert array-like data to numpy array.
    
    Parameters
    ----------
    data : ArrayLike
        Input data to validate
    name : str, default "data"
        Name for error messages
    min_length : int, default 1
        Minimum required length
    allow_nan : bool, default False
        Whether to allow NaN values
    allow_inf : bool, default False
        Whether to allow infinite values
        
    Returns
    -------
    np.ndarray
        Validated numpy array
        
    Raises
    ------
    ValueError
        If validation fails
    """
    from .errors import ValidationError
    
    arr = to_numpy_array(data)
    
    if arr.size == 0:
        raise ValidationError(f"{name} cannot be empty")
    
    if len(arr) < min_length:
        raise ValidationError(
            f"{name} must have at least {min_length} elements, got {len(arr)}"
        )
    
    if not allow_nan and np.any(np.isnan(arr)):
        raise ValidationError(f"{name} contains NaN values")
    
    if not allow_inf and np.any(np.isinf(arr)):
        raise ValidationError(f"{name} contains infinite values")
    
    return arr


def ensure_2d(arr: np.ndarray) -> np.ndarray:
    """Ensure array is 2D (add dimension if 1D)."""
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def ensure_1d(arr: np.ndarray) -> np.ndarray:
    """Ensure array is 1D (flatten if higher dimensional)."""
    return arr.flatten()
