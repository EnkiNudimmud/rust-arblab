"""
Regime Detection Module for Mean Reversion and Trend Following Strategies

Detects market regimes using statistical methods:
- Mean-reverting regime: High autocorrelation, low volatility
- Trending regime: Strong momentum, moderate volatility
- High volatility regime: Large price swings, low predictability
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from scipy import stats
from sklearn.covariance import EmpiricalCovariance


class RegimeDetector:
    """
    Detect market regimes using multiple statistical indicators.
    """
    
    def __init__(self, lookback_window: int = 50):
        """
        Args:
            lookback_window: Number of periods for regime calculation
        """
        self.lookback_window = lookback_window
        
    def detect_regime(self, returns: pd.Series) -> str:
        """
        Detect current market regime.
        
        Args:
            returns: Time series of returns
            
        Returns:
            Regime string: 'mean_reverting', 'trending', or 'high_volatility'
        """
        if len(returns) < self.lookback_window:
            return 'unknown'
        
        recent = returns.iloc[-self.lookback_window:]
        
        # Calculate regime indicators
        volatility = recent.std()
        autocorr = self._calculate_autocorr(recent)
        trend_strength = self._calculate_trend_strength(recent)
        hurst = self._calculate_hurst_exponent(recent)
        
        # Regime classification
        if hurst < 0.45 and autocorr < -0.1:
            return 'mean_reverting'
        elif trend_strength > 0.6 and hurst > 0.55:
            return 'trending'
        elif volatility > recent.rolling(100).std().iloc[-1] * 1.5:
            return 'high_volatility'
        elif autocorr < 0 and trend_strength < 0.4:
            return 'mean_reverting'
        else:
            return 'mixed'
    
    def detect_multi_regime(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect regimes for multiple assets.
        
        Args:
            returns_df: DataFrame with returns for multiple assets (columns)
            
        Returns:
            DataFrame with regime for each asset
        """
        regimes = {}
        metrics = {}
        
        for col in returns_df.columns:
            regime = self.detect_regime(returns_df[col])
            regimes[col] = regime
            
            # Calculate metrics for this asset
            recent = returns_df[col].iloc[-self.lookback_window:]
            metrics[col] = {
                'regime': regime,
                'volatility': recent.std(),
                'autocorr': self._calculate_autocorr(recent),
                'trend_strength': self._calculate_trend_strength(recent),
                'hurst': self._calculate_hurst_exponent(recent)
            }
        
        return pd.DataFrame(metrics).T
    
    def _calculate_autocorr(self, series: pd.Series, lag: int = 1) -> float:
        """Calculate autocorrelation at given lag."""
        try:
            return series.autocorr(lag=lag)
        except:
            return 0.0
    
    def _calculate_trend_strength(self, series: pd.Series) -> float:
        """
        Calculate trend strength using linear regression R².
        
        Returns:
            R² value (0-1), higher means stronger trend
        """
        if len(series) < 2:
            return 0.0
        
        x = np.arange(len(series))
        y = series.values
        
        # Remove NaN
        mask = ~np.isnan(y)
        if mask.sum() < 2:
            return 0.0
        
        x_clean = x[mask]
        y_clean = y[mask]
        
        # Linear regression
        slope, intercept, r_value, _, _ = stats.linregress(x_clean, y_clean)
        
        return abs(r_value ** 2)
    
    def _calculate_hurst_exponent(self, series: pd.Series) -> float:
        """
        Calculate Hurst exponent using R/S analysis.
        
        H < 0.5: Mean reverting
        H = 0.5: Random walk
        H > 0.5: Trending
        
        Returns:
            Hurst exponent (0-1)
        """
        try:
            ts = series.dropna().values
            if len(ts) < 20:
                return 0.5
            
            # Calculate log returns
            lags = range(2, min(20, len(ts) // 2))
            
            # R/S analysis
            tau = []
            rs_values = []
            
            for lag in lags:
                # Divide into subseries
                subseries = [ts[i:i+lag] for i in range(0, len(ts), lag) if len(ts[i:i+lag]) == lag]
                
                if len(subseries) < 2:
                    continue
                
                rs = []
                for sub in subseries:
                    # Mean-adjusted series
                    mean_sub = sub - np.mean(sub)
                    cumsum = np.cumsum(mean_sub)
                    
                    # Range
                    R = np.max(cumsum) - np.min(cumsum)
                    
                    # Standard deviation
                    S = np.std(sub)
                    
                    if S > 0:
                        rs.append(R / S)
                
                if rs:
                    tau.append(lag)
                    rs_values.append(np.mean(rs))
            
            if len(tau) < 2:
                return 0.5
            
            # Log-log regression
            tau_log = np.log(tau)
            rs_log = np.log(rs_values)
            
            slope, _, _, _, _ = stats.linregress(tau_log, rs_log)
            
            return max(0, min(1, slope))
        
        except:
            return 0.5
    
    def calculate_regime_probabilities(self, returns: pd.Series) -> Dict[str, float]:
        """
        Calculate probability of being in each regime.
        
        Args:
            returns: Time series of returns
            
        Returns:
            Dict with probabilities for each regime
        """
        if len(returns) < self.lookback_window:
            return {'mean_reverting': 0.33, 'trending': 0.33, 'high_volatility': 0.33}
        
        recent = returns.iloc[-self.lookback_window:]
        
        # Calculate indicators
        vol = recent.std()
        autocorr = self._calculate_autocorr(recent)
        trend = self._calculate_trend_strength(recent)
        hurst = self._calculate_hurst_exponent(recent)
        
        # Score each regime
        mean_rev_score = (1 - hurst) * 2 + max(0, -autocorr) + (1 - trend)
        trending_score = hurst * 2 + trend + max(0, autocorr)
        high_vol_score = vol / (recent.rolling(100).std().iloc[-1] + 1e-6)
        
        # Normalize to probabilities
        total = mean_rev_score + trending_score + high_vol_score
        
        return {
            'mean_reverting': mean_rev_score / total,
            'trending': trending_score / total,
            'high_volatility': high_vol_score / total
        }


class AdaptiveStrategySelector:
    """
    Select optimal strategy based on detected regime.
    """
    
    def __init__(self):
        self.regime_detector = RegimeDetector()
        
    def get_strategy_weights(self, returns: pd.Series) -> Dict[str, float]:
        """
        Get strategy weights based on regime.
        
        Returns:
            Dict with weights for 'mean_reversion', 'momentum', 'market_making'
        """
        regime = self.regime_detector.detect_regime(returns)
        
        weights = {
            'mean_reverting': {'mean_reversion': 0.8, 'momentum': 0.1, 'market_making': 0.1},
            'trending': {'mean_reversion': 0.1, 'momentum': 0.8, 'market_making': 0.1},
            'high_volatility': {'mean_reversion': 0.3, 'momentum': 0.2, 'market_making': 0.5},
            'mixed': {'mean_reversion': 0.5, 'momentum': 0.4, 'market_making': 0.1},
            'unknown': {'mean_reversion': 0.33, 'momentum': 0.33, 'market_making': 0.34}
        }
        
        return weights.get(regime, weights['unknown'])
    
    def get_position_sizing(self, returns: pd.Series, base_size: float = 1.0) -> float:
        """
        Adjust position sizing based on regime.
        
        Args:
            returns: Time series of returns
            base_size: Base position size
            
        Returns:
            Adjusted position size
        """
        regime = self.regime_detector.detect_regime(returns)
        
        multipliers = {
            'mean_reverting': 1.2,  # Increase size in mean-reverting
            'trending': 1.0,         # Normal size in trending
            'high_volatility': 0.5,  # Reduce size in high vol
            'mixed': 0.8,
            'unknown': 0.7
        }
        
        return base_size * multipliers.get(regime, 0.7)


def detect_regime_simple(returns: pd.Series, window: int = 50) -> str:
    """
    Simple regime detection function for quick use.
    
    Args:
        returns: Time series of returns
        window: Lookback window
        
    Returns:
        Regime string
    """
    detector = RegimeDetector(lookback_window=window)
    return detector.detect_regime(returns)


def get_regime_metrics(returns: pd.Series, window: int = 50) -> Dict:
    """
    Get all regime metrics for analysis.
    
    Args:
        returns: Time series of returns
        window: Lookback window
        
    Returns:
        Dict with regime and all metrics
    """
    detector = RegimeDetector(lookback_window=window)
    
    if len(returns) < window:
        return {'regime': 'unknown'}
    
    recent = returns.iloc[-window:]
    
    return {
        'regime': detector.detect_regime(returns),
        'volatility': recent.std(),
        'autocorr': detector._calculate_autocorr(recent),
        'trend_strength': detector._calculate_trend_strength(recent),
        'hurst_exponent': detector._calculate_hurst_exponent(recent),
        'probabilities': detector.calculate_regime_probabilities(returns)
    }


if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)
    
    # Mean-reverting series
    mean_rev = pd.Series(np.cumsum(np.random.randn(200) * 0.01))
    mean_rev = mean_rev - mean_rev.rolling(20).mean().fillna(0) * 0.5
    
    # Trending series
    trend = pd.Series(np.cumsum(np.random.randn(200) * 0.01 + 0.002))
    
    # High volatility series
    high_vol = pd.Series(np.random.randn(200) * 0.05)
    
    detector = RegimeDetector()
    
    print("Mean-reverting series:", detector.detect_regime(mean_rev))
    print("Trending series:", detector.detect_regime(trend))
    print("High volatility series:", detector.detect_regime(high_vol))
    
    print("\nRegime probabilities (mean-reverting):")
    print(detector.calculate_regime_probabilities(mean_rev))
