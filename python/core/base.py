"""
Base classes for strategies, models, and optimizers.

Provides abstract base classes that enforce consistent interfaces
and enable polymorphism through the Strategy pattern.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from datetime import datetime


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.
    
    Implements the Strategy design pattern for interchangeable algorithms.
    """
    
    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        """
        Initialize strategy.
        
        Parameters
        ----------
        name : str
            Strategy name
        params : dict, optional
            Strategy parameters
        """
        self.name = name
        self.params = params or {}
        self.state: Dict[str, Any] = {}
        
    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> int:
        """
        Generate trading signal from market data.
        
        Parameters
        ----------
        data : DataFrame
            Market data
            
        Returns
        -------
        int
            Signal: 1 (buy), -1 (sell), 0 (hold)
        """
        pass
    
    def update_state(self, **kwargs):
        """Update internal strategy state."""
        self.state.update(kwargs)
    
    def reset(self):
        """Reset strategy state."""
        self.state = {}
    
    def get_params(self) -> Dict[str, Any]:
        """Get strategy parameters."""
        return self.params.copy()
    
    def set_params(self, **params):
        """Update strategy parameters."""
        self.params.update(params)


class BaseModel(ABC):
    """
    Abstract base class for statistical/ML models.
    
    Provides consistent interface for model training and prediction.
    """
    
    def __init__(self, name: str):
        """
        Initialize model.
        
        Parameters
        ----------
        name : str
            Model name
        """
        self.name = name
        self.is_fitted = False
        self.metadata: Dict[str, Any] = {}
    
    @abstractmethod
    def fit(self, data: np.ndarray, **kwargs):
        """
        Fit model to data.
        
        Parameters
        ----------
        data : ndarray
            Training data
        **kwargs
            Additional parameters
        """
        pass
    
    @abstractmethod
    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Parameters
        ----------
        data : ndarray
            Input data
            
        Returns
        -------
        ndarray
            Predictions
        """
        pass
    
    def _check_is_fitted(self):
        """Check if model is fitted."""
        if not self.is_fitted:
            raise RuntimeError(f"{self.name} must be fitted before prediction")


class BaseOptimizer(ABC):
    """
    Abstract base class for optimization algorithms.
    
    Implements the Strategy pattern for different optimization approaches.
    """
    
    def __init__(self, name: str, max_iter: int = 1000, tol: float = 1e-6):
        """
        Initialize optimizer.
        
        Parameters
        ----------
        name : str
            Optimizer name
        max_iter : int
            Maximum iterations
        tol : float
            Convergence tolerance
        """
        self.name = name
        self.max_iter = max_iter
        self.tol = tol
        self.result: Optional[Dict[str, Any]] = None
        
    @abstractmethod
    def optimize(self, objective, initial_guess: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Run optimization.
        
        Parameters
        ----------
        objective : callable
            Objective function to minimize
        initial_guess : ndarray
            Starting point
        **kwargs
            Additional parameters
            
        Returns
        -------
        dict
            Optimization result with keys: 'x' (solution), 'fun' (objective value),
            'success' (bool), 'message' (str), 'nit' (iterations)
        """
        pass
    
    def get_result(self) -> Dict[str, Any]:
        """Get last optimization result."""
        if self.result is None:
            raise RuntimeError("Optimizer has not been run yet")
        return self.result
