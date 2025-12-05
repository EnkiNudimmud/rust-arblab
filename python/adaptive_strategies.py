"""
Adaptive Trading Strategies with HMM Regime Detection
======================================================

Implements regime-adaptive trading strategies that automatically
adjust parameters based on detected market conditions.

Features:
- HMM-based regime detection
- Automatic parameter adaptation per regime
- Real-time regime monitoring
- Strategy performance by regime
- Rust-accelerated computations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime
import logging

from python.advanced_optimization import HMMRegimeDetector, RUST_AVAILABLE

logger = logging.getLogger(__name__)


@dataclass
class RegimeConfig:
    """Configuration for a specific market regime"""
    regime_id: int
    name: str
    entry_threshold: float
    exit_threshold: float
    position_size: float
    stop_loss: float
    take_profit: float
    max_holding_period: int
    
    def to_dict(self) -> Dict:
        return {
            'regime_id': self.regime_id,
            'name': self.name,
            'entry_threshold': self.entry_threshold,
            'exit_threshold': self.exit_threshold,
            'position_size': self.position_size,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'max_holding_period': self.max_holding_period
        }


@dataclass
class RegimeStats:
    """Statistics for a regime"""
    regime_id: int
    mean_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_trade_duration: float
    total_trades: int


class AdaptiveStrategy:
    """
    Base class for regime-adaptive trading strategies.
    
    Automatically detects market regimes using HMM and adapts
    strategy parameters accordingly.
    """
    
    def __init__(
        self,
        n_regimes: int = 3,
        lookback_period: int = 500,
        update_frequency: int = 100,
        base_config: Optional[Dict] = None
    ):
        """
        Args:
            n_regimes: Number of market regimes to detect
            lookback_period: Historical bars for HMM training
            update_frequency: Bars between HMM retraining
            base_config: Base parameter configuration
        """
        self.n_regimes = n_regimes
        self.lookback_period = lookback_period
        self.update_frequency = update_frequency
        self.base_config = base_config or self._default_config()
        
        # HMM detector
        self.hmm_detector = HMMRegimeDetector(n_states=n_regimes)
        self.hmm_trained = False
        self.bars_since_update = 0
        
        # Regime configurations
        self.regime_configs: Dict[int, RegimeConfig] = {}
        self._initialize_regime_configs()
        
        # Performance tracking
        self.regime_stats: Dict[int, RegimeStats] = {}
        self.trade_history: List[Dict] = []
        self.current_regime: Optional[int] = None
        
        logger.info(f"Initialized adaptive strategy with {n_regimes} regimes (Rust: {RUST_AVAILABLE})")
    
    def _default_config(self) -> Dict:
        """Default base configuration"""
        return {
            'entry_threshold': 2.0,
            'exit_threshold': 0.5,
            'position_size': 1.0,
            'stop_loss': 0.02,
            'take_profit': 0.05,
            'max_holding_period': 20
        }
    
    def _initialize_regime_configs(self):
        """Initialize regime-specific configurations"""
        regime_names = ["Bear Market", "Sideways Market", "Bull Market"]
        if self.n_regimes > 3:
            regime_names.extend([f"Regime {i}" for i in range(3, self.n_regimes)])
        
        # Define regime-specific adjustments
        adjustments = [
            # Bear: defensive, tight stops, smaller positions
            {
                'entry_threshold': 1.3,
                'exit_threshold': 1.2,
                'position_size': 0.6,
                'stop_loss': 0.8,
                'take_profit': 0.8,
                'max_holding_period': 0.7
            },
            # Sideways: mean reversion focused
            {
                'entry_threshold': 1.0,
                'exit_threshold': 1.0,
                'position_size': 1.0,
                'stop_loss': 1.0,
                'take_profit': 1.0,
                'max_holding_period': 1.0
            },
            # Bull: aggressive, wider stops, larger positions
            {
                'entry_threshold': 0.8,
                'exit_threshold': 0.8,
                'position_size': 1.3,
                'stop_loss': 1.3,
                'take_profit': 1.3,
                'max_holding_period': 1.3
            }
        ]
        
        for i in range(self.n_regimes):
            adj = adjustments[i] if i < len(adjustments) else adjustments[1]  # Default to sideways
            
            self.regime_configs[i] = RegimeConfig(
                regime_id=i,
                name=regime_names[i],
                entry_threshold=self.base_config['entry_threshold'] * adj['entry_threshold'],
                exit_threshold=self.base_config['exit_threshold'] * adj['exit_threshold'],
                position_size=self.base_config['position_size'] * adj['position_size'],
                stop_loss=self.base_config['stop_loss'] * adj['stop_loss'],
                take_profit=self.base_config['take_profit'] * adj['take_profit'],
                max_holding_period=int(self.base_config['max_holding_period'] * adj['max_holding_period'])
            )
    
    def update_regimes(self, returns: np.ndarray) -> bool:
        """
        Update HMM model with new data.
        
        Args:
            returns: Recent returns data
            
        Returns:
            True if model was updated
        """
        self.bars_since_update += 1
        
        # Only update at specified frequency
        if self.bars_since_update < self.update_frequency and self.hmm_trained:
            return False
        
        if len(returns) < self.lookback_period:
            logger.warning(f"Insufficient data for HMM: {len(returns)} < {self.lookback_period}")
            return False
        
        try:
            # Train HMM on recent data
            train_data = returns[-self.lookback_period:]
            
            # Remove NaN/inf values that can break HMM
            train_data = train_data[np.isfinite(train_data)]
            
            if len(train_data) < 20:  # Need minimum data
                logger.warning(f"Insufficient valid data after cleaning: {len(train_data)}")
                return False
            
            self.hmm_detector.fit(train_data, n_iterations=50)
            
            self.hmm_trained = True
            self.bars_since_update = 0
            
            # Update current regime
            if self.hmm_detector.state_sequence is not None:
                self.current_regime = self.hmm_detector.state_sequence[-1]
                logger.info(f"✓ HMM updated, current regime: {self.current_regime}")
            
            return True
            
        except Exception as e:
            logger.error(f"HMM update failed: {e}")
            return False
    
    def get_current_config(self) -> RegimeConfig:
        """Get configuration for current regime"""
        if self.current_regime is None:
            # Return base config as default
            return RegimeConfig(
                regime_id=-1,
                name="Default",
                **self.base_config
            )
        
        return self.regime_configs[self.current_regime]
    
    def detect_regime(self, recent_returns: np.ndarray) -> int:
        """
        Detect current market regime.
        
        Args:
            recent_returns: Recent returns for prediction
            
        Returns:
            Regime ID
        """
        if not self.hmm_trained or self.hmm_detector.state_sequence is None:
            return 1  # Default to sideways
        
        try:
            # Use HMM to predict regime
            regime = self.hmm_detector.predict_regime(recent_returns[-20:])
            self.current_regime = regime
            return regime
        except Exception as e:
            logger.warning(f"Regime detection failed: {e}")
            return self.current_regime if self.current_regime is not None else 1
    
    def generate_signal(
        self,
        data: pd.DataFrame,
        symbol: str,
        current_positions: Dict
    ) -> Optional[Dict]:
        """
        Generate trading signal with regime-adaptive parameters.
        
        Args:
            data: Market data with OHLCV
            symbol: Trading symbol
            current_positions: Current portfolio positions
            
        Returns:
            Signal dict or None
        """
        # Get returns
        if 'close' not in data.columns:
            return None
        
        returns = data['close'].pct_change().dropna().values
        
        # Update regimes periodically
        if len(returns) >= self.lookback_period:
            self.update_regimes(returns)
        
        # Detect current regime
        regime = self.detect_regime(returns)
        config = self.regime_configs[regime]
        
        # Strategy-specific logic (override in subclass)
        signal = self._compute_signal(data, config, current_positions.get(symbol))
        
        if signal:
            signal['regime'] = regime
            signal['regime_name'] = config.name
            signal['config'] = config.to_dict()
        
        return signal
    
    def _compute_signal(
        self,
        data: pd.DataFrame,
        config: RegimeConfig,
        current_position: Optional[Dict]
    ) -> Optional[Dict]:
        """
        Compute trading signal (to be overridden by subclasses).
        
        Args:
            data: Market data
            config: Regime-specific configuration
            current_position: Current position if any
            
        Returns:
            Signal dict with action, size, etc.
        """
        raise NotImplementedError("Subclass must implement _compute_signal")
    
    def record_trade(
        self,
        symbol: str,
        action: str,
        regime: int,
        entry_price: float,
        exit_price: Optional[float] = None,
        pnl: Optional[float] = None
    ):
        """Record trade for performance tracking"""
        trade = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'action': action,
            'regime': regime,
            'regime_name': self.regime_configs[regime].name,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl
        }
        
        self.trade_history.append(trade)
    
    def get_regime_performance(self) -> pd.DataFrame:
        """Get performance metrics by regime"""
        if not self.trade_history:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.trade_history)
        
        # Calculate metrics per regime
        metrics = []
        for regime_id in range(self.n_regimes):
            regime_trades = df[df['regime'] == regime_id]
            
            if len(regime_trades) > 0 and 'pnl' in regime_trades.columns:
                pnl_series = regime_trades['pnl'].dropna()
                
                if len(pnl_series) > 0:
                    metrics.append({
                        'regime': self.regime_configs[regime_id].name,
                        'total_trades': len(regime_trades),
                        'win_rate': (pnl_series > 0).mean() * 100,
                        'avg_pnl': pnl_series.mean(),
                        'total_pnl': pnl_series.sum(),
                        'sharpe': pnl_series.mean() / pnl_series.std() if pnl_series.std() > 0 else 0
                    })
        
        return pd.DataFrame(metrics)
    
    def get_transition_probabilities(self) -> Optional[np.ndarray]:
        """Get HMM transition matrix"""
        if self.hmm_detector.transition_matrix is not None:
            return self.hmm_detector.transition_matrix
        return None
    
    def get_emission_params(self) -> Optional[List[Tuple[float, float]]]:
        """Get HMM emission parameters (mean, variance) per state"""
        if self.hmm_detector.emission_params is not None:
            return self.hmm_detector.emission_params
        return None


class AdaptiveMeanReversion(AdaptiveStrategy):
    """Regime-adaptive mean reversion strategy"""
    
    def _compute_signal(
        self,
        data: pd.DataFrame,
        config: RegimeConfig,
        current_position: Optional[Dict]
    ) -> Optional[Dict]:
        """Mean reversion signal with regime-adaptive parameters"""
        
        if len(data) < 20:
            return None
        
        # Calculate z-score
        lookback = int(config.max_holding_period * 2)
        prices = data['close'].values
        
        mean = np.mean(prices[-lookback:])
        std = np.std(prices[-lookback:])
        
        if std == 0:
            return None
        
        current_price = prices[-1]
        z_score = (current_price - mean) / std
        
        # Check for exit signal first
        if current_position:
            position_side = current_position.get('side', 'long')
            entry_price = current_position.get('entry_price', current_price)
            
            # Exit conditions
            should_exit = False
            exit_reason = None
            
            # Mean reversion exit
            if abs(z_score) < config.exit_threshold:
                should_exit = True
                exit_reason = "mean_reversion"
            
            # Stop loss
            pnl_pct = (current_price - entry_price) / entry_price
            if position_side == 'long' and pnl_pct < -config.stop_loss:
                should_exit = True
                exit_reason = "stop_loss"
            elif position_side == 'short' and pnl_pct > config.stop_loss:
                should_exit = True
                exit_reason = "stop_loss"
            
            # Take profit
            if position_side == 'long' and pnl_pct > config.take_profit:
                should_exit = True
                exit_reason = "take_profit"
            elif position_side == 'short' and pnl_pct < -config.take_profit:
                should_exit = True
                exit_reason = "take_profit"
            
            if should_exit:
                return {
                    'action': 'close',
                    'reason': exit_reason,
                    'z_score': z_score
                }
        
        # Entry signals
        if z_score > config.entry_threshold:
            # Price too high - short
            return {
                'action': 'open',
                'side': 'short',
                'size': config.position_size,
                'z_score': z_score,
                'reason': 'overvalued'
            }
        elif z_score < -config.entry_threshold:
            # Price too low - long
            return {
                'action': 'open',
                'side': 'long',
                'size': config.position_size,
                'z_score': z_score,
                'reason': 'undervalued'
            }
        
        return None


class AdaptiveMomentum(AdaptiveStrategy):
    """Regime-adaptive momentum strategy"""
    
    def _compute_signal(
        self,
        data: pd.DataFrame,
        config: RegimeConfig,
        current_position: Optional[Dict]
    ) -> Optional[Dict]:
        """Momentum signal with regime-adaptive parameters"""
        
        if len(data) < 50:
            return None
        
        prices = data['close'].values
        
        # Calculate momentum indicators
        fast_ma = np.mean(prices[-10:])
        slow_ma = np.mean(prices[-30:])
        momentum = (fast_ma - slow_ma) / slow_ma
        
        # Calculate RSI
        delta = pd.Series(prices).diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        
        current_price = prices[-1]
        
        # Check for exit
        if current_position:
            position_side = current_position.get('side', 'long')
            entry_price = current_position.get('entry_price', current_price)
            
            pnl_pct = (current_price - entry_price) / entry_price
            
            # Exit conditions
            if position_side == 'long':
                if momentum < -config.exit_threshold or current_rsi < 30:
                    return {'action': 'close', 'reason': 'momentum_reversal'}
                if pnl_pct < -config.stop_loss:
                    return {'action': 'close', 'reason': 'stop_loss'}
                if pnl_pct > config.take_profit:
                    return {'action': 'close', 'reason': 'take_profit'}
            
            elif position_side == 'short':
                if momentum > config.exit_threshold or current_rsi > 70:
                    return {'action': 'close', 'reason': 'momentum_reversal'}
                if pnl_pct > config.stop_loss:
                    return {'action': 'close', 'reason': 'stop_loss'}
                if pnl_pct < -config.take_profit:
                    return {'action': 'close', 'reason': 'take_profit'}
        
        # Entry signals
        threshold = config.entry_threshold * 0.01  # Scale down
        
        if momentum > threshold and current_rsi > 50:
            return {
                'action': 'open',
                'side': 'long',
                'size': config.position_size,
                'momentum': momentum,
                'rsi': current_rsi,
                'reason': 'uptrend'
            }
        elif momentum < -threshold and current_rsi < 50:
            return {
                'action': 'open',
                'side': 'short',
                'size': config.position_size,
                'momentum': momentum,
                'rsi': current_rsi,
                'reason': 'downtrend'
            }
        
        return None


class AdaptiveStatArb(AdaptiveStrategy):
    """Regime-adaptive statistical arbitrage (pairs trading)"""
    
    def __init__(self, *args, pair_symbols: Optional[List[str]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pair_symbols = pair_symbols or []
        self.spread_params = {}
    
    def _compute_signal(
        self,
        data: pd.DataFrame,
        config: RegimeConfig,
        current_position: Optional[Dict]
    ) -> Optional[Dict]:
        """Statistical arbitrage signal with regime adaptation"""
        
        # This is a simplified version - real stat arb needs pair data
        if len(data) < 50:
            return None
        
        # Calculate spread statistics (would use pair data in reality)
        prices = data['close'].values
        spread = prices  # Simplified
        
        spread_mean = np.mean(spread[-50:])
        spread_std = np.std(spread[-50:])
        
        if spread_std == 0:
            return None
        
        current_spread = spread[-1]
        z_score = (current_spread - spread_mean) / spread_std
        
        # Exit logic
        if current_position:
            if abs(z_score) < config.exit_threshold:
                return {'action': 'close', 'reason': 'spread_converged', 'z_score': z_score}
        
        # Entry logic
        if z_score > config.entry_threshold:
            return {
                'action': 'open',
                'side': 'short',
                'size': config.position_size,
                'z_score': z_score,
                'reason': 'spread_divergence_high'
            }
        elif z_score < -config.entry_threshold:
            return {
                'action': 'open',
                'side': 'long',
                'size': config.position_size,
                'z_score': z_score,
                'reason': 'spread_divergence_low'
            }
        
        return None
