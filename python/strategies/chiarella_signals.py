"""
Chiarella Model Signal Generator for Live Trading

Based on: Kurth & Bouchaud (2025), arXiv:2511.13277
"Stationary Distributions of the Mode-switching Chiarella Model"
"""

import numpy as np
from typing import Dict, Optional, Tuple
from datetime import datetime


class ChiarellaSignalGenerator:
    """
    Generate trading signals using the Mode-Switching Chiarella Model
    
    The model describes market dynamics as competition between:
    - Fundamentalists: traders who believe prices revert to fundamental value
    - Chartists: momentum/trend-following traders
    
    Equations:
        dp/dt = α·trend(t) - β·mispricing(t) + noise
        dtrend/dt = γ·Δp(t) - δ·trend(t) + noise
    """
    
    def __init__(self, fundamental_price: float, 
                 alpha: float = 0.3, beta: float = 0.5,
                 gamma: float = 0.4, delta: float = 0.2):
        """
        Initialize signal generator
        
        Args:
            fundamental_price: Equilibrium/fair value price
            alpha: Chartist strength (trend feedback)
            beta: Fundamentalist strength (mean reversion)
            gamma: Trend formation speed
            delta: Trend decay rate
        """
        self.fundamental_price = fundamental_price
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        
        # State
        self.current_price = fundamental_price
        self.trend = 0.0
        
        # History for statistics
        self.price_history = [fundamental_price]
        self.trend_history = [0.0]
        self.max_history = 100
        
    def update(self, new_price: float) -> None:
        """
        Update model with new market price
        
        Args:
            new_price: Latest observed market price
        """
        # Calculate price change
        delta_p = new_price - self.current_price
        
        # Update trend: dtrend/dt = γ·Δp - δ·trend
        trend_drift = self.gamma * delta_p - self.delta * self.trend
        self.trend += trend_drift
        
        # Update price
        self.current_price = new_price
        
        # Store history
        self.price_history.append(new_price)
        self.trend_history.append(self.trend)
        
        # Trim history
        if len(self.price_history) > self.max_history:
            self.price_history.pop(0)
            self.trend_history.pop(0)
    
    def get_regime(self) -> Tuple[str, float]:
        """
        Determine current market regime
        
        Returns:
            (regime_name, bifurcation_parameter)
        """
        # Bifurcation parameter: Λ = (α·γ) / (β·δ)
        Lambda = (self.alpha * self.gamma) / (self.beta * self.delta) if self.delta > 0 else 1.0
        
        if Lambda < 0.67:
            regime = "mean_reverting"
        elif Lambda > 1.5:
            regime = "trending"
        else:
            regime = "mixed"
        
        return regime, Lambda
    
    def generate_signal(self) -> Dict:
        """
        Generate trading signal based on current state
        
        Returns:
            Dictionary with signal components:
            - strength: Signal strength [-1, 1] (-1=strong sell, 1=strong buy)
            - position_size: Recommended position [0, 1]
            - confidence: Signal confidence [0, 1]
            - regime: Current market regime
            - mispricing: Current mispricing (p - p_fundamental)
            - trend: Current trend estimate
            - expected_return: Expected return estimate
            - risk: Risk estimate (volatility)
        """
        mispricing = self.current_price - self.fundamental_price
        mispricing_pct = mispricing / self.fundamental_price if self.fundamental_price > 0 else 0
        
        # Component signals
        # 1. Fundamentalist signal: buy when undervalued
        signal_fundamental = -self.beta * mispricing_pct
        
        # 2. Chartist signal: follow trend
        signal_chartist = self.alpha * (self.trend / self.fundamental_price) if self.fundamental_price > 0 else 0
        
        # Get regime
        regime, Lambda = self.get_regime()
        
        # Regime-adaptive weighting
        if regime == "mean_reverting":
            w_f, w_c = 0.8, 0.2  # Fundamentalists dominate
        elif regime == "trending":
            w_f, w_c = 0.2, 0.8  # Chartists dominate
        else:
            w_f, w_c = 0.5, 0.5  # Balanced
        
        # Combined signal
        raw_signal = w_f * signal_fundamental + w_c * signal_chartist
        signal_strength = np.tanh(raw_signal)  # Normalize to [-1, 1]
        
        # Calculate confidence
        confidence = self._calculate_confidence()
        
        # Calculate risk
        risk = self._calculate_risk()
        
        # Position sizing (Kelly criterion approximation)
        expected_return = raw_signal
        kelly_fraction = abs(expected_return / (risk ** 2)) if risk > 0 else 0
        position_size = min(1.0, kelly_fraction * confidence)
        
        return {
            'strength': signal_strength,
            'position_size': position_size,
            'confidence': confidence,
            'regime': regime,
            'bifurcation_parameter': Lambda,
            'mispricing': mispricing,
            'mispricing_pct': mispricing_pct * 100,  # As percentage
            'trend': self.trend,
            'expected_return': expected_return,
            'risk': risk,
            'signal_fundamental': signal_fundamental,
            'signal_chartist': signal_chartist,
            'weight_fundamental': w_f,
            'weight_chartist': w_c,
        }
    
    def _calculate_confidence(self) -> float:
        """Calculate signal confidence based on trend consistency"""
        if len(self.trend_history) < 20:
            return 0.5
        
        recent_trends = self.trend_history[-20:]
        trend_mean = np.mean(recent_trends)
        trend_std = np.std(recent_trends)
        
        # Confidence increases with trend consistency
        if abs(trend_mean) > 0:
            consistency = 1.0 - min(1.0, trend_std / (abs(trend_mean) + 0.001))
        else:
            consistency = 0.5
        
        return max(0.3, min(0.95, consistency))
    
    def _calculate_risk(self) -> float:
        """Calculate risk as realized volatility"""
        if len(self.price_history) < 2:
            return 0.02
        
        # Calculate returns
        prices = np.array(self.price_history[-20:])
        returns = np.diff(prices) / prices[:-1]
        
        if len(returns) > 0:
            return max(0.01, np.std(returns))
        else:
            return 0.02
    
    def update_fundamental(self, new_fundamental: float) -> None:
        """
        Update fundamental price estimate
        
        Args:
            new_fundamental: New fundamental price (e.g., from moving average, DCF, etc.)
        """
        self.fundamental_price = new_fundamental
    
    def get_state(self) -> Dict:
        """Get current model state"""
        regime, Lambda = self.get_regime()
        
        return {
            'price': self.current_price,
            'fundamental_price': self.fundamental_price,
            'trend': self.trend,
            'mispricing': self.current_price - self.fundamental_price,
            'regime': regime,
            'bifurcation_parameter': Lambda,
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma,
            'delta': self.delta,
        }


def estimate_fundamental_price(prices: list, method: str = 'ma') -> float:
    """
    Estimate fundamental price from historical prices
    
    Args:
        prices: List of historical prices
        method: Estimation method ('ma', 'ema', 'median')
    
    Returns:
        Estimated fundamental price
    """
    if len(prices) == 0:
        return 100.0
    
    prices_array = np.array(prices)
    
    if method == 'ma':
        # Moving average
        return np.mean(prices_array[-50:]) if len(prices_array) >= 50 else np.mean(prices_array)
    elif method == 'ema':
        # Exponential moving average
        alpha_ema = 2.0 / (50 + 1)
        ema = prices_array[0]
        for price in prices_array[1:]:
            ema = alpha_ema * price + (1 - alpha_ema) * ema
        return ema
    elif method == 'median':
        # Median (robust to outliers)
        return np.median(prices_array[-50:]) if len(prices_array) >= 50 else np.median(prices_array)
    else:
        return np.mean(prices_array)
