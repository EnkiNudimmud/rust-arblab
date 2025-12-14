//! Regime-Switching Portfolio Optimization with Jumps
//! ===================================================
//!
//! Portfolio optimization using Markov Regime Switching Jump Diffusion (MRSJD).
//! 
//! # Problem Formulation
//!
//! ## Wealth Dynamics
//!
//! Log-wealth: W_t = log(Wealth_t)
//! dW_t = [r^{α_t} + π_t(μ^{α_t} - r^{α_t}) - (σ^{α_t})²π_t²/2]dt 
//!        + σ^{α_t}π_t dB_t + dJ_t^{α_t}
//!
//! where:
//! - α_t: market regime (bull/bear/normal)
//! - π_t: fraction invested in risky asset
//! - r^i: risk-free rate in regime i
//! - μ^i, σ^i: risky asset parameters in regime i
//! - J_t^i: jumps in regime i
//!
//! ## Objective
//!
//! Maximize utility: E[U(W_T)] with CRRA utility U(w) = w^(1-γ)/(1-γ)
//!
//! ## HJB Equation
//!
//! ρV^i(w,t) = sup_{π} [b^i(w,π)·∂V^i/∂w + (σ^i(π))²/2·∂²V^i/∂w²
//!             + λ^i∫[V^i(w+y) - V^i(w)]F^i(dy)
//!             + Σ_{j≠i} q_{ij}[V^j(w) - V^i(w)]]

use ndarray::{Array1, Array2};
use optimizr::optimal_control::{
    MRSJDSolver, MRSJDConfig, MRSJDResult, RegimeJumpParameters, JumpDistribution
};
use optimizr::optimal_control::OptimalControlError;
use statrs::distribution::{Normal, ContinuousCDF};
use serde::{Serialize, Deserialize};

/// Market regime definition
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum MarketRegime {
    Bull = 0,
    Normal = 1,
    Bear = 2,
}

/// Portfolio optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioConfig {
    /// Risk aversion parameter (CRRA)
    pub gamma: f64,
    /// Time horizon (years)
    pub time_horizon: f64,
    /// Discount rate
    pub rho: f64,
    /// Transaction cost (proportional)
    pub transaction_cost: f64,
    
    // Bull regime parameters
    pub r_bull: f64,
    pub mu_bull: f64,
    pub sigma_bull: f64,
    pub lambda_bull: f64,
    pub jump_mean_bull: f64,
    pub jump_std_bull: f64,
    
    // Normal regime parameters
    pub r_normal: f64,
    pub mu_normal: f64,
    pub sigma_normal: f64,
    pub lambda_normal: f64,
    pub jump_mean_normal: f64,
    pub jump_std_normal: f64,
    
    // Bear regime parameters
    pub r_bear: f64,
    pub mu_bear: f64,
    pub sigma_bear: f64,
    pub lambda_bear: f64,
    pub jump_mean_bear: f64,
    pub jump_std_bear: f64,
    
    // Transition rates (annualized)
    pub q_bull_normal: f64,
    pub q_bull_bear: f64,
    pub q_normal_bull: f64,
    pub q_normal_bear: f64,
    pub q_bear_bull: f64,
    pub q_bear_normal: f64,
}

impl Default for PortfolioConfig {
    fn default() -> Self {
        Self {
            // Utility parameters
            gamma: 2.0,      // Moderate risk aversion
            time_horizon: 10.0,
            rho: 0.05,
            transaction_cost: 0.001,
            
            // Bull regime (good times)
            r_bull: 0.03,
            mu_bull: 0.15,
            sigma_bull: 0.18,
            lambda_bull: 0.05,  // Few jumps
            jump_mean_bull: -0.02,
            jump_std_bull: 0.03,
            
            // Normal regime (typical)
            r_normal: 0.025,
            mu_normal: 0.08,
            sigma_normal: 0.20,
            lambda_normal: 0.10,
            jump_mean_normal: -0.05,
            jump_std_normal: 0.05,
            
            // Bear regime (crisis)
            r_bear: 0.01,
            mu_bear: -0.05,
            sigma_bear: 0.35,
            lambda_bear: 0.30,  // Frequent jumps
            jump_mean_bear: -0.15,
            jump_std_bear: 0.10,
            
            // Transition rates (per year)
            q_bull_normal: 0.3,
            q_bull_bear: 0.05,
            q_normal_bull: 0.2,
            q_normal_bear: 0.1,
            q_bear_bull: 0.1,
            q_bear_normal: 0.4,
        }
    }
}

/// Portfolio optimization result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioResult {
    /// Wealth grid
    pub wealth: Vec<f64>,
    /// Value functions for each regime
    pub values: Vec<Vec<f64>>,
    /// Optimal portfolio weights for each regime
    pub portfolio_weights: Vec<Vec<f64>>,
    /// Stationary probabilities of regimes
    pub stationary_probs: Vec<f64>,
    /// Expected optimal weight (stationary average)
    pub expected_weight: f64,
    /// Value function at initial wealth
    pub initial_value: f64,
    /// Convergence info
    pub iterations: usize,
    pub residual: f64,
}

/// Regime-switching portfolio optimizer
pub struct RegimeSwitchingPortfolio {
    config: PortfolioConfig,
}

impl RegimeSwitchingPortfolio {
    /// Create new optimizer with configuration
    pub fn new(config: PortfolioConfig) -> Self {
        Self { config }
    }
    
    /// Solve optimal portfolio problem
    pub fn optimize(&self) -> Result<PortfolioResult, OptimalControlError> {
        let cfg = &self.config;
        
        // Setup transition rate matrix (3x3 for Bull/Normal/Bear)
        let mut q = Array2::<f64>::zeros((3, 3));
        q[[0, 1]] = cfg.q_bull_normal;
        q[[0, 2]] = cfg.q_bull_bear;
        q[[1, 0]] = cfg.q_normal_bull;
        q[[1, 2]] = cfg.q_normal_bear;
        q[[2, 0]] = cfg.q_bear_bull;
        q[[2, 1]] = cfg.q_bear_normal;
        
        // MRSJD configuration
        let mrsjd_config = MRSJDConfig {
            n_regimes: 3,
            transition_rates: q,
            rho: cfg.rho,
            transaction_cost: cfg.transaction_cost,
            state_bounds: (-2.0, 8.0), // Log-wealth bounds
            n_points: 500,
            max_iter: 5000,
            tolerance: 1e-6,
        };
        
        // Create regime-specific parameters
        let regime_params = vec![
            self.create_regime_params(
                cfg.r_bull, cfg.mu_bull, cfg.sigma_bull, cfg.gamma,
                cfg.lambda_bull, cfg.jump_mean_bull, cfg.jump_std_bull
            ),
            self.create_regime_params(
                cfg.r_normal, cfg.mu_normal, cfg.sigma_normal, cfg.gamma,
                cfg.lambda_normal, cfg.jump_mean_normal, cfg.jump_std_normal
            ),
            self.create_regime_params(
                cfg.r_bear, cfg.mu_bear, cfg.sigma_bear, cfg.gamma,
                cfg.lambda_bear, cfg.jump_mean_bear, cfg.jump_std_bear
            ),
        ];
        
        // Solve MRSJD system
        let solver = MRSJDSolver::new(mrsjd_config, regime_params)?;
        let result = solver.solve()?;
        
        // Extract results
        let wealth: Vec<f64> = result.x.iter().map(|&w| w.exp()).collect();
        
        let mut values = Vec::new();
        let mut portfolio_weights = Vec::new();
        
        for i in 0..3 {
            values.push(result.values.row(i).to_vec());
            portfolio_weights.push(result.controls.row(i).to_vec());
        }
        
        let stationary_probs = result.stationary_distribution.to_vec();
        
        // Compute expected optimal weight
        let mid_idx = result.x.len() / 2;
        let expected_weight: f64 = (0..3)
            .map(|i| portfolio_weights[i][mid_idx] * stationary_probs[i])
            .sum();
        
        // Initial value (at wealth = 1, i.e., log-wealth = 0)
        let w0_idx = result.x.iter()
            .position(|&w| (w - 0.0).abs() < 0.01)
            .unwrap_or(mid_idx);
        
        let initial_value: f64 = (0..3)
            .map(|i| values[i][w0_idx] * stationary_probs[i])
            .sum();
        
        Ok(PortfolioResult {
            wealth,
            values,
            portfolio_weights,
            stationary_probs,
            expected_weight,
            initial_value,
            iterations: result.iterations,
            residual: result.residual,
        })
    }
    
    /// Create regime-specific parameters for MRSJD
    fn create_regime_params(
        &self,
        r: f64,
        mu: f64,
        sigma: f64,
        gamma: f64,
        lambda: f64,
        jump_mean: f64,
        jump_std: f64,
    ) -> RegimeJumpParameters {
        // Merton optimal portfolio (without jumps baseline)
        let pi_merton = (mu - r) / (gamma * sigma * sigma);
        
        RegimeJumpParameters {
            // Drift: r + π(μ-r) - γπ²σ²/2
            drift: Box::new(move |w| {
                let pi = pi_merton.max(-2.0).min(2.0); // Bounded
                r + pi * (mu - r) - 0.5 * gamma * pi * pi * sigma * sigma
            }),
            
            // Diffusion: σπ  
            diffusion: Box::new(move |_w| {
                let pi = pi_merton.max(-2.0).min(2.0);
                sigma * pi.abs()
            }),
            
            // Cost: transaction costs (simplified)
            cost: Box::new(move |_w, _u| {
                // Could add rebalancing costs here
                0.0
            }),
            
            jump_intensity: lambda,
            jump_distribution: JumpDistribution::Normal {
                mean: jump_mean,
                std: jump_std,
            },
        }
    }
    
    /// Estimate current market regime from returns data
    pub fn estimate_regime(returns: &[f64]) -> MarketRegime {
        if returns.is_empty() {
            return MarketRegime::Normal;
        }
        
        let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance: f64 = returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / returns.len() as f64;
        let volatility = variance.sqrt();
        
        // Simple heuristic
        if mean > 0.10 && volatility < 0.25 {
            MarketRegime::Bull
        } else if mean < -0.05 || volatility > 0.35 {
            MarketRegime::Bear
        } else {
            MarketRegime::Normal
        }
    }
    
    /// Simulate portfolio path under optimal policy
    pub fn simulate_path(
        &self,
        result: &PortfolioResult,
        initial_wealth: f64,
        n_steps: usize,
        dt: f64,
    ) -> Result<(Vec<f64>, Vec<f64>, Vec<usize>), OptimalControlError> {
        use rand::thread_rng;
        use rand::Rng;
        
        let mut rng = thread_rng();
        let mut times = Vec::with_capacity(n_steps + 1);
        let mut wealths = Vec::with_capacity(n_steps + 1);
        let mut regimes = Vec::with_capacity(n_steps + 1);
        
        let mut w = initial_wealth.ln();
        let mut regime: usize = 1; // Start in normal
        let mut t = 0.0;
        
        times.push(t);
        wealths.push(w.exp());
        regimes.push(regime);
        
        let sqrt_dt = dt.sqrt();
        
        // Transition rate matrix
        let cfg = &self.config;
        let mut q = Array2::<f64>::zeros((3, 3));
        q[[0, 1]] = cfg.q_bull_normal;
        q[[0, 2]] = cfg.q_bull_bear;
        q[[1, 0]] = cfg.q_normal_bull;
        q[[1, 2]] = cfg.q_normal_bear;
        q[[2, 0]] = cfg.q_bear_bull;
        q[[2, 1]] = cfg.q_bear_normal;
        
        for _ in 0..n_steps {
            // Get current regime parameters
            let (r, mu, sigma, lambda, jump_mean, jump_std) = match regime {
                0 => (cfg.r_bull, cfg.mu_bull, cfg.sigma_bull, 
                      cfg.lambda_bull, cfg.jump_mean_bull, cfg.jump_std_bull),
                1 => (cfg.r_normal, cfg.mu_normal, cfg.sigma_normal,
                      cfg.lambda_normal, cfg.jump_mean_normal, cfg.jump_std_normal),
                2 => (cfg.r_bear, cfg.mu_bear, cfg.sigma_bear,
                      cfg.lambda_bear, cfg.jump_mean_bear, cfg.jump_std_bear),
                _ => unreachable!(),
            };
            
            // Get optimal portfolio weight at current wealth
            let w_idx = result.wealth.iter()
                .position(|&wealth| (wealth.ln() - w).abs() < 0.1)
                .unwrap_or(result.wealth.len() / 2);
            let pi = result.portfolio_weights[regime][w_idx];
            
            // Wealth dynamics
            let drift = r + pi * (mu - r) - 0.5 * pi * pi * sigma * sigma;
            let diffusion = sigma * pi;
            
            let dw_norm: f64 = rng.sample(Normal::new(0.0, sqrt_dt).unwrap());
            let dw = drift * dt + diffusion * dw_norm;
            
            // Jump
            let jump_prob = lambda * dt;
            let jump = if rng.gen::<f64>() < jump_prob {
                rng.sample(Normal::new(jump_mean, jump_std).unwrap())
            } else {
                0.0
            };
            
            w += dw + jump;
            
            // Regime transition
            for j in 0..3 {
                if j != regime {
                    let trans_prob = q[[regime, j]] * dt;
                    if rng.gen::<f64>() < trans_prob {
                        regime = j;
                        break;
                    }
                }
            }
            
            t += dt;
            times.push(t);
            wealths.push(w.exp());
            regimes.push(regime);
        }
        
        Ok((times, wealths, regimes))
    }
}

/// Calibrate model parameters from historical data
pub fn calibrate_from_data(
    returns: &[f64],
    regimes: Option<&[usize]>,
) -> PortfolioConfig {
    // Simple moment matching for now
    let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
    let var: f64 = returns.iter()
        .map(|r| (r - mean).powi(2))
        .sum::<f64>() / returns.len() as f64;
    let sigma = var.sqrt();
    
    let mut config = PortfolioConfig::default();
    
    // If regimes provided, estimate per-regime parameters
    if let Some(reg) = regimes {
        for regime_id in 0..3 {
            let regime_returns: Vec<f64> = returns.iter()
                .zip(reg.iter())
                .filter_map(|(&r, &rid)| if rid == regime_id { Some(r) } else { None })
                .collect();
            
            if !regime_returns.is_empty() {
                let r_mean = regime_returns.iter().sum::<f64>() / regime_returns.len() as f64;
                let r_var = regime_returns.iter()
                    .map(|r| (r - r_mean).powi(2))
                    .sum::<f64>() / regime_returns.len() as f64;
                let r_sigma = r_var.sqrt();
                
                match regime_id {
                    0 => { // Bull
                        config.mu_bull = r_mean * 252.0; // Annualize
                        config.sigma_bull = r_sigma * (252.0_f64).sqrt();
                    }
                    1 => { // Normal
                        config.mu_normal = r_mean * 252.0;
                        config.sigma_normal = r_sigma * (252.0_f64).sqrt();
                    }
                    2 => { // Bear
                        config.mu_bear = r_mean * 252.0;
                        config.sigma_bear = r_sigma * (252.0_f64).sqrt();
                    }
                    _ => {}
                }
            }
        }
    } else {
        // Use overall statistics for normal regime
        config.mu_normal = mean * 252.0;
        config.sigma_normal = sigma * (252.0_f64).sqrt();
    }
    
    config
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_portfolio_optimization() {
        let config = PortfolioConfig::default();
        let optimizer = RegimeSwitchingPortfolio::new(config);
        
        let result = optimizer.optimize().unwrap();
        
        assert!(result.stationary_probs.iter().sum::<f64>() - 1.0 < 1e-6);
        assert!(result.iterations > 0);
        assert!(result.residual < 1e-6);
        
        // Portfolio weights should be reasonable
        assert!(result.expected_weight.abs() < 5.0);
    }
    
    #[test]
    fn test_regime_estimation() {
        let bull_returns = vec![0.01, 0.015, 0.012, 0.018, 0.011];
        assert!(matches!(
            RegimeSwitchingPortfolio::estimate_regime(&bull_returns),
            MarketRegime::Bull
        ));
        
        let bear_returns = vec![-0.03, -0.02, -0.04, -0.025, -0.035];
        assert!(matches!(
            RegimeSwitchingPortfolio::estimate_regime(&bear_returns),
            MarketRegime::Bear
        ));
    }
}
