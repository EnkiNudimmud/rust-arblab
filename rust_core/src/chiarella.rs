/// Mode-Switching Chiarella Model Implementation
/// 
/// Based on: "Stationary Distributions of the Mode-switching Chiarella Model"
/// Kurth & Bouchaud (2025), arXiv:2511.13277
/// 
/// The Chiarella model describes financial market dynamics as competition between:
/// - Fundamentalists: traders who believe price reverts to fundamental value
/// - Chartists: momentum/trend-following traders
/// 
/// Core equations:
/// dp/dt = α·trend(t) - β·mispricing(t) + σ·noise(t)
/// dtrend/dt = γ·[p(t) - p(t-1)] - δ·trend(t) + η·noise(t)
/// 
/// Where:
/// - mispricing(t) = p(t) - p_fundamental
/// - α: chartist strength (trend feedback)
/// - β: fundamentalist strength (mean reversion)
/// - γ: trend formation speed
/// - δ: trend decay rate
/// - σ, η: noise terms

use std::f64::consts::PI;

#[derive(Debug, Clone)]
pub struct ChiarellaModel {
    /// Fundamental price (equilibrium)
    pub fundamental_price: f64,
    
    /// Current market price
    pub current_price: f64,
    
    /// Current trend estimate
    pub trend: f64,
    
    /// Chartist strength (trend feedback coefficient)
    pub alpha: f64,
    
    /// Fundamentalist strength (mean reversion coefficient)
    pub beta: f64,
    
    /// Trend formation speed
    pub gamma: f64,
    
    /// Trend decay rate
    pub delta: f64,
    
    /// Price noise standard deviation
    pub sigma: f64,
    
    /// Trend noise standard deviation
    pub eta: f64,
    
    /// Time step (dt)
    pub dt: f64,
    
    /// Historical prices (for statistics)
    price_history: Vec<f64>,
    trend_history: Vec<f64>,
    
    /// Regime state
    regime: RegimeState,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RegimeState {
    /// Mean-reversion dominant (fundamentalists winning)
    MeanReverting,
    
    /// Trending dominant (chartists winning)
    Trending,
    
    /// Mixed regime
    Mixed,
}

#[derive(Debug, Clone)]
pub struct TradingSignal {
    /// Signal strength [-1, 1]: -1 = strong sell, 0 = neutral, 1 = strong buy
    pub strength: f64,
    
    /// Recommended position size [0, 1]
    pub position_size: f64,
    
    /// Confidence level [0, 1]
    pub confidence: f64,
    
    /// Current regime
    pub regime: RegimeState,
    
    /// Mispricing (p - p_fundamental)
    pub mispricing: f64,
    
    /// Current trend
    pub trend: f64,
    
    /// Expected return
    pub expected_return: f64,
    
    /// Risk estimate (volatility)
    pub risk: f64,
}

impl ChiarellaModel {
    /// Create new Chiarella model with default parameters
    /// 
    /// Default parameters chosen for stable dynamics:
    /// - α = 0.3 (moderate chartist influence)
    /// - β = 0.5 (stronger fundamentalist influence)
    /// - γ = 0.4 (moderate trend formation)
    /// - δ = 0.2 (slow trend decay)
    pub fn new(fundamental_price: f64) -> Self {
        Self {
            fundamental_price,
            current_price: fundamental_price,
            trend: 0.0,
            alpha: 0.3,
            beta: 0.5,
            gamma: 0.4,
            delta: 0.2,
            sigma: 0.02,
            eta: 0.01,
            dt: 0.01,
            price_history: vec![fundamental_price],
            trend_history: vec![0.0],
            regime: RegimeState::Mixed,
        }
    }
    
    /// Create with custom parameters
    pub fn with_parameters(
        fundamental_price: f64,
        alpha: f64,
        beta: f64,
        gamma: f64,
        delta: f64,
        sigma: f64,
        eta: f64,
    ) -> Self {
        Self {
            fundamental_price,
            current_price: fundamental_price,
            trend: 0.0,
            alpha,
            beta,
            gamma,
            delta,
            sigma,
            eta,
            dt: 0.01,
            price_history: vec![fundamental_price],
            trend_history: vec![0.0],
            regime: RegimeState::Mixed,
        }
    }
    
    /// Update the model by one time step
    /// 
    /// Uses Euler-Maruyama discretization:
    /// p(t+dt) = p(t) + [α·trend(t) - β·mispricing(t)]·dt + σ·√dt·N(0,1)
    /// trend(t+dt) = trend(t) + [γ·Δp(t) - δ·trend(t)]·dt + η·√dt·N(0,1)
    pub fn step(&mut self, noise_price: f64, noise_trend: f64) {
        let mispricing = self.current_price - self.fundamental_price;
        
        // Price dynamics: dp/dt = α·trend - β·mispricing + noise
        let price_drift = self.alpha * self.trend - self.beta * mispricing;
        let price_diffusion = self.sigma * (self.dt.sqrt()) * noise_price;
        let new_price = self.current_price + price_drift * self.dt + price_diffusion;
        
        // Trend dynamics: dtrend/dt = γ·Δp - δ·trend + noise
        let delta_price = new_price - self.current_price;
        let trend_drift = self.gamma * delta_price - self.delta * self.trend;
        let trend_diffusion = self.eta * (self.dt.sqrt()) * noise_trend;
        let new_trend = self.trend + trend_drift * self.dt + trend_diffusion;
        
        // Update state
        self.current_price = new_price;
        self.trend = new_trend;
        
        // Store history (keep last 1000 points)
        self.price_history.push(new_price);
        self.trend_history.push(new_trend);
        if self.price_history.len() > 1000 {
            self.price_history.remove(0);
            self.trend_history.remove(0);
        }
        
        // Update regime
        self.update_regime();
    }
    
    /// Update with real market data
    pub fn update_with_price(&mut self, new_price: f64) {
        // Calculate implied trend from price change
        let delta_price = new_price - self.current_price;
        let implied_trend_change = self.gamma * delta_price;
        
        // Update trend with decay
        self.trend = self.trend * (1.0 - self.delta * self.dt) + implied_trend_change;
        
        // Update price
        self.current_price = new_price;
        
        // Store history
        self.price_history.push(new_price);
        self.trend_history.push(self.trend);
        if self.price_history.len() > 1000 {
            self.price_history.remove(0);
            self.trend_history.remove(0);
        }
        
        // Update regime
        self.update_regime();
    }
    
    /// Detect current regime based on relative strengths
    fn update_regime(&mut self) {
        let mispricing = (self.current_price - self.fundamental_price).abs();
        let trend_strength = self.trend.abs();
        
        // Regime detection based on dominant force
        let fundamentalist_force = self.beta * mispricing;
        let chartist_force = self.alpha * trend_strength;
        
        let ratio = if fundamentalist_force > 0.0 {
            chartist_force / fundamentalist_force
        } else {
            10.0 // Default to trending if no mean reversion
        };
        
        self.regime = if ratio > 1.5 {
            RegimeState::Trending
        } else if ratio < 0.67 {
            RegimeState::MeanReverting
        } else {
            RegimeState::Mixed
        };
    }
    
    /// Generate trading signal
    pub fn generate_signal(&self) -> TradingSignal {
        let mispricing = self.current_price - self.fundamental_price;
        let mispricing_pct = mispricing / self.fundamental_price;
        
        // Signal components
        // 1. Fundamentalist signal: buy when undervalued, sell when overvalued
        let fundamental_signal = -mispricing_pct * self.beta;
        
        // 2. Chartist signal: follow the trend
        let trend_signal = self.trend * self.alpha / self.fundamental_price;
        
        // Combined signal (weighted by regime)
        let raw_signal = match self.regime {
            RegimeState::MeanReverting => {
                // Fundamentalists dominate: 80% fundamental, 20% trend
                0.8 * fundamental_signal + 0.2 * trend_signal
            }
            RegimeState::Trending => {
                // Chartists dominate: 20% fundamental, 80% trend
                0.2 * fundamental_signal + 0.8 * trend_signal
            }
            RegimeState::Mixed => {
                // Balanced: 50-50
                0.5 * fundamental_signal + 0.5 * trend_signal
            }
        };
        
        // Normalize signal to [-1, 1] using tanh
        let strength = raw_signal.tanh();
        
        // Calculate confidence based on signal consistency
        let confidence = self.calculate_confidence();
        
        // Position sizing using Kelly criterion approximation
        let expected_return = raw_signal;
        let risk = self.calculate_risk();
        let kelly_fraction = if risk > 0.0 {
            (expected_return / risk).abs().min(1.0)
        } else {
            0.0
        };
        let position_size = (kelly_fraction * confidence).min(1.0);
        
        TradingSignal {
            strength,
            position_size,
            confidence,
            regime: self.regime.clone(),
            mispricing,
            trend: self.trend,
            expected_return,
            risk,
        }
    }
    
    /// Calculate signal confidence based on historical consistency
    fn calculate_confidence(&self) -> f64 {
        if self.price_history.len() < 20 {
            return 0.5; // Low confidence with insufficient data
        }
        
        // Use trend consistency and mispricing stability
        let recent_trends: Vec<f64> = self.trend_history.iter()
            .rev()
            .take(20)
            .copied()
            .collect();
        
        let trend_mean = recent_trends.iter().sum::<f64>() / recent_trends.len() as f64;
        let trend_std = {
            let variance = recent_trends.iter()
                .map(|t| (t - trend_mean).powi(2))
                .sum::<f64>() / recent_trends.len() as f64;
            variance.sqrt()
        };
        
        // Confidence increases with trend consistency (low std relative to mean)
        let trend_consistency = if trend_mean.abs() > 0.0 {
            1.0 - (trend_std / (trend_mean.abs() + 0.001)).min(1.0)
        } else {
            0.5
        };
        
        trend_consistency.max(0.3).min(0.95)
    }
    
    /// Calculate risk (realized volatility)
    fn calculate_risk(&self) -> f64 {
        if self.price_history.len() < 20 {
            return 0.1; // Default risk
        }
        
        let recent_prices: Vec<f64> = self.price_history.iter()
            .rev()
            .take(20)
            .copied()
            .collect();
        
        // Calculate log returns
        let returns: Vec<f64> = recent_prices.windows(2)
            .map(|w| (w[0] / w[1]).ln())
            .collect();
        
        if returns.is_empty() {
            return 0.1;
        }
        
        // Calculate standard deviation of returns
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / returns.len() as f64;
        
        variance.sqrt().max(0.01)
    }
    
    /// Calculate stationary distribution statistics
    /// Based on the paper's analysis of unimodal/bimodal regimes
    pub fn stationary_statistics(&self) -> StationaryStats {
        let mispricing = self.current_price - self.fundamental_price;
        
        // P-bifurcation condition: β·δ vs α·γ
        let mean_reversion_strength = self.beta * self.delta;
        let trend_strength = self.alpha * self.gamma;
        let bifurcation_param = trend_strength / mean_reversion_strength.max(0.001);
        
        // Distribution is unimodal when mean-reversion is fast (β·δ > α·γ)
        // Distribution is bimodal when mean-reversion is slow (β·δ < α·γ)
        let is_bimodal = bifurcation_param > 1.0;
        
        StationaryStats {
            mispricing_mean: mispricing,
            trend_mean: self.trend,
            bifurcation_parameter: bifurcation_param,
            is_bimodal,
            regime: self.regime.clone(),
        }
    }
    
    /// Get current state
    pub fn get_state(&self) -> ModelState {
        ModelState {
            price: self.current_price,
            fundamental_price: self.fundamental_price,
            trend: self.trend,
            mispricing: self.current_price - self.fundamental_price,
            regime: self.regime.clone(),
        }
    }
    
    /// Update fundamental price (e.g., based on new information)
    pub fn update_fundamental(&mut self, new_fundamental: f64) {
        self.fundamental_price = new_fundamental;
    }
}

#[derive(Debug, Clone)]
pub struct StationaryStats {
    pub mispricing_mean: f64,
    pub trend_mean: f64,
    pub bifurcation_parameter: f64,
    pub is_bimodal: bool,
    pub regime: RegimeState,
}

#[derive(Debug, Clone)]
pub struct ModelState {
    pub price: f64,
    pub fundamental_price: f64,
    pub trend: f64,
    pub mispricing: f64,
    pub regime: RegimeState,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_initialization() {
        let model = ChiarellaModel::new(100.0);
        assert_eq!(model.fundamental_price, 100.0);
        assert_eq!(model.current_price, 100.0);
        assert_eq!(model.trend, 0.0);
    }
    
    #[test]
    fn test_regime_detection() {
        let mut model = ChiarellaModel::new(100.0);
        
        // Strong fundamentalist setup (high β, low α)
        model.beta = 1.0;
        model.alpha = 0.1;
        model.current_price = 110.0; // 10% overvalued
        model.update_regime();
        assert_eq!(model.regime, RegimeState::MeanReverting);
        
        // Strong chartist setup (low β, high α)
        model.beta = 0.1;
        model.alpha = 1.0;
        model.trend = 5.0;
        model.update_regime();
        assert_eq!(model.regime, RegimeState::Trending);
    }
    
    #[test]
    fn test_signal_generation() {
        let mut model = ChiarellaModel::new(100.0);
        model.current_price = 95.0; // Undervalued
        
        let signal = model.generate_signal();
        
        // Should generate buy signal when undervalued
        assert!(signal.strength > 0.0, "Signal should be positive for undervalued stock");
        assert!(signal.confidence >= 0.0 && signal.confidence <= 1.0);
        assert!(signal.position_size >= 0.0 && signal.position_size <= 1.0);
    }
    
    #[test]
    fn test_bifurcation() {
        let mut model = ChiarellaModel::new(100.0);
        
        // Strong mean-reversion: β·δ > α·γ (unimodal)
        model.alpha = 0.2;
        model.beta = 1.0;
        model.gamma = 0.3;
        model.delta = 0.8;
        
        let stats = model.stationary_statistics();
        assert!(!stats.is_bimodal, "Should be unimodal with strong mean-reversion");
        
        // Weak mean-reversion: α·γ > β·δ (bimodal)
        model.alpha = 1.0;
        model.beta = 0.2;
        model.gamma = 0.8;
        model.delta = 0.3;
        
        let stats = model.stationary_statistics();
        assert!(stats.is_bimodal, "Should be bimodal with weak mean-reversion");
    }
}
