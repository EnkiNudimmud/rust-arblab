//! Portfolio optimization under drift uncertainty (Bismuth, Guéant, Pu)
//!
//! This module implements techniques from "Portfolio choice, portfolio liquidation, 
//! and portfolio transition under drift uncertainty" focusing on robust portfolio 
//! optimization when the drift parameter is uncertain.
//!
//! Key features:
//! - Worst-case portfolio optimization (robust Markowitz)
//! - Optimal liquidation under drift uncertainty
//! - Portfolio transition with ambiguity aversion
//! - CARA utility with exponential preferences

use std::f64::consts::PI;

/// Portfolio optimization result under drift uncertainty
#[derive(Debug, Clone)]
pub struct PortfolioResult {
    pub weights: Vec<f64>,
    pub expected_return: f64,
    pub worst_case_return: f64,
    pub variance: f64,
    pub utility: f64,
}

/// Liquidation strategy result
#[derive(Debug, Clone)]
pub struct LiquidationResult {
    pub trading_schedule: Vec<f64>,  // Positions over time
    pub trading_rates: Vec<f64>,     // dq/dt at each time step
    pub expected_cost: f64,
    pub worst_case_cost: f64,
    pub times: Vec<f64>,
}

/// Portfolio transition result (moving from initial to target portfolio)
#[derive(Debug, Clone)]
pub struct TransitionResult {
    pub trajectory: Vec<Vec<f64>>,  // Portfolio weights over time
    pub trading_rates: Vec<Vec<f64>>, // Trading velocities
    pub expected_cost: f64,
    pub worst_case_cost: f64,
    pub times: Vec<f64>,
}

/// Worst-case portfolio choice under drift uncertainty
/// 
/// Solves: max_q min_μ E[U(wealth)] where U is CARA utility
/// μ ∈ [μ_min, μ_max] represents drift uncertainty
/// 
/// # Arguments
/// * `expected_returns` - Best estimate of asset returns
/// * `covariance` - Covariance matrix (flattened row-major)
/// * `risk_aversion` - Risk aversion parameter γ (CARA utility)
/// * `drift_uncertainty` - Half-width of drift uncertainty interval
/// * `n_assets` - Number of assets
/// 
/// # Returns
/// Optimal portfolio weights that maximize worst-case utility
pub fn portfolio_choice_robust(
    expected_returns: &[f64],
    covariance: &[f64],
    risk_aversion: f64,
    drift_uncertainty: f64,
    n_assets: usize,
) -> PortfolioResult {
    assert_eq!(expected_returns.len(), n_assets);
    assert_eq!(covariance.len(), n_assets * n_assets);
    assert!(risk_aversion > 0.0);
    assert!(drift_uncertainty >= 0.0);

    // For CARA utility with drift uncertainty, the worst-case optimization yields:
    // q* = (1/γ) * Σ^{-1} * (μ - κ * |Σ^{-1} μ|)
    // where κ relates to drift uncertainty
    
    // Compute covariance inverse (simplified - assumes well-conditioned)
    let cov_inv = invert_matrix(covariance, n_assets);
    
    // Compute Σ^{-1} * μ
    let mut sigma_inv_mu = vec![0.0; n_assets];
    for i in 0..n_assets {
        for j in 0..n_assets {
            sigma_inv_mu[i] += cov_inv[i * n_assets + j] * expected_returns[j];
        }
    }
    
    // Compute norm of Σ^{-1} * μ
    let norm_sigma_inv_mu = sigma_inv_mu.iter().map(|x| x * x).sum::<f64>().sqrt();
    
    // Adjust for drift uncertainty (robust formulation)
    let kappa = drift_uncertainty * norm_sigma_inv_mu.sqrt();
    
    // Optimal weights: q* = (1/γ) * (Σ^{-1} μ - κ * e)
    // where e is direction of uncertainty
    let mut weights = Vec::with_capacity(n_assets);
    for i in 0..n_assets {
        let adjustment = if norm_sigma_inv_mu > 1e-10 {
            kappa * sigma_inv_mu[i] / norm_sigma_inv_mu
        } else {
            0.0
        };
        weights.push((sigma_inv_mu[i] - adjustment) / risk_aversion);
    }
    
    // Compute expected return
    let expected_return: f64 = weights.iter()
        .zip(expected_returns.iter())
        .map(|(w, r)| w * r)
        .sum();
    
    // Compute worst-case return
    let worst_case_return = expected_return - drift_uncertainty * 
        weights.iter().map(|w| w.abs()).sum::<f64>();
    
    // Compute variance: w' Σ w
    let mut variance = 0.0;
    for i in 0..n_assets {
        for j in 0..n_assets {
            variance += weights[i] * covariance[i * n_assets + j] * weights[j];
        }
    }
    
    // CARA utility: -exp(-γ * (μ'q - 0.5 * γ * q'Σq))
    let utility = -(- risk_aversion * (worst_case_return - 0.5 * risk_aversion * variance)).exp();
    
    PortfolioResult {
        weights,
        expected_return,
        worst_case_return,
        variance,
        utility,
    }
}

/// Optimal liquidation under drift uncertainty
/// 
/// Solves the problem of liquidating a position q_0 over time T
/// minimizing worst-case cost considering price impact and drift uncertainty
/// 
/// # Arguments
/// * `initial_position` - Initial position to liquidate
/// * `time_horizon` - Liquidation horizon T
/// * `n_steps` - Number of time steps
/// * `drift` - Expected drift parameter
/// * `drift_uncertainty` - Uncertainty in drift
/// * `volatility` - Price volatility
/// * `temporary_impact` - Temporary price impact coefficient
/// * `permanent_impact` - Permanent price impact coefficient
/// * `risk_aversion` - Risk aversion parameter
/// 
/// # Returns
/// Optimal liquidation schedule
pub fn optimal_liquidation(
    initial_position: f64,
    time_horizon: f64,
    n_steps: usize,
    drift: f64,
    drift_uncertainty: f64,
    volatility: f64,
    temporary_impact: f64,
    permanent_impact: f64,
    risk_aversion: f64,
) -> LiquidationResult {
    assert!(n_steps > 0);
    assert!(time_horizon > 0.0);
    assert!(initial_position != 0.0);
    
    let dt = time_horizon / n_steps as f64;
    let mut times = Vec::with_capacity(n_steps + 1);
    let mut trading_schedule = Vec::with_capacity(n_steps + 1);
    let mut trading_rates = Vec::with_capacity(n_steps);
    
    // For drift uncertainty, we use worst-case optimization
    // The optimal strategy balances trading speed vs. market risk
    
    // Key parameters
    let kappa = permanent_impact;
    let eta = temporary_impact;
    let gamma = risk_aversion;
    let sigma = volatility;
    
    // Worst-case drift (conservative)
    let mu_worst = if initial_position > 0.0 {
        drift - drift_uncertainty  // Selling into falling market
    } else {
        drift + drift_uncertainty  // Buying into rising market
    };
    
    // Compute optimal decay rate λ from HJB equation
    // λ = sqrt(γ * σ² / η) (simplified)
    let lambda = if eta > 1e-10 {
        (gamma * sigma * sigma / eta).sqrt()
    } else {
        1.0 / time_horizon
    };
    
    // Optimal trajectory: q(t) = q_0 * exp(-λ * t)
    trading_schedule.push(initial_position);
    times.push(0.0);
    
    for i in 0..n_steps {
        let t = (i as f64 + 1.0) * dt;
        times.push(t);
        
        let q_t = initial_position * (-lambda * t).exp();
        trading_schedule.push(q_t);
        
        // Trading rate: dq/dt = -λ * q(t)
        let rate = -lambda * q_t;
        trading_rates.push(rate);
    }
    
    // Compute expected cost (includes drift, impact, and risk)
    let mut expected_cost = 0.0;
    let mut worst_case_cost = 0.0;
    
    for i in 0..n_steps {
        let q_t = trading_schedule[i];
        let dq = trading_schedule[i + 1] - trading_schedule[i];
        
        // Cost components:
        // 1. Price drift cost
        expected_cost += drift * q_t * dt;
        worst_case_cost += mu_worst * q_t * dt;
        
        // 2. Temporary impact
        let temp_cost = eta * dq * dq / dt;
        expected_cost += temp_cost;
        worst_case_cost += temp_cost;
        
        // 3. Permanent impact
        let perm_cost = kappa * dq.abs() * q_t;
        expected_cost += perm_cost;
        worst_case_cost += perm_cost;
        
        // 4. Risk penalty (variance)
        let risk_penalty = 0.5 * gamma * sigma * sigma * q_t * q_t * dt;
        expected_cost += risk_penalty;
        worst_case_cost += risk_penalty;
    }
    
    LiquidationResult {
        trading_schedule,
        trading_rates,
        expected_cost,
        worst_case_cost,
        times,
    }
}

/// Portfolio transition from initial to target weights under drift uncertainty
/// 
/// Optimally transitions from q_initial to q_target over time T
/// accounting for transaction costs and drift uncertainty
/// 
/// # Arguments
/// * `initial_weights` - Starting portfolio weights
/// * `target_weights` - Desired final weights
/// * `time_horizon` - Transition horizon
/// * `n_steps` - Number of time steps
/// * `expected_returns` - Expected returns
/// * `covariance` - Covariance matrix (flattened)
/// * `drift_uncertainty` - Drift uncertainty parameter
/// * `transaction_cost` - Linear transaction cost
/// * `risk_aversion` - Risk aversion
/// 
/// # Returns
/// Optimal transition trajectory
pub fn portfolio_transition(
    initial_weights: &[f64],
    target_weights: &[f64],
    time_horizon: f64,
    n_steps: usize,
    expected_returns: &[f64],
    covariance: &[f64],
    drift_uncertainty: f64,
    transaction_cost: f64,
    risk_aversion: f64,
) -> TransitionResult {
    let n_assets = initial_weights.len();
    assert_eq!(target_weights.len(), n_assets);
    assert_eq!(expected_returns.len(), n_assets);
    assert_eq!(covariance.len(), n_assets * n_assets);
    
    let dt = time_horizon / n_steps as f64;
    let mut times = Vec::with_capacity(n_steps + 1);
    let mut trajectory = Vec::with_capacity(n_steps + 1);
    let mut trading_rates = Vec::with_capacity(n_steps);
    
    // Compute difference to rebalance
    let mut delta: Vec<f64> = target_weights.iter()
        .zip(initial_weights.iter())
        .map(|(t, i)| t - i)
        .collect();
    
    // Optimal transition uses exponential decay adjusted for drift uncertainty
    // λ = sqrt(γ * σ² / c) where c is transaction cost
    let avg_var: f64 = (0..n_assets).map(|i| covariance[i * n_assets + i]).sum::<f64>() / n_assets as f64;
    let lambda = if transaction_cost > 1e-10 {
        (risk_aversion * avg_var / transaction_cost).sqrt()
    } else {
        5.0 / time_horizon
    };
    
    trajectory.push(initial_weights.to_vec());
    times.push(0.0);
    
    for i in 0..n_steps {
        let t = (i as f64 + 1.0) * dt;
        times.push(t);
        
        // Exponential transition: q(t) = q_target + (q_0 - q_target) * exp(-λt)
        let decay = (-lambda * t).exp();
        let mut weights_t = Vec::with_capacity(n_assets);
        let mut rates_t = Vec::with_capacity(n_assets);
        
        for j in 0..n_assets {
            let w_t = target_weights[j] + delta[j] * decay;
            weights_t.push(w_t);
            
            // Trading rate: dq/dt = -λ * (q(t) - q_target)
            let rate = -lambda * delta[j] * decay;
            rates_t.push(rate);
        }
        
        trajectory.push(weights_t);
        trading_rates.push(rates_t);
    }
    
    // Compute costs
    let mut expected_cost = 0.0;
    let mut worst_case_cost = 0.0;
    
    for i in 0..n_steps {
        let weights = &trajectory[i];
        
        // Holding cost (portfolio variance)
        let mut variance = 0.0;
        for j in 0..n_assets {
            for k in 0..n_assets {
                variance += weights[j] * covariance[j * n_assets + k] * weights[k];
            }
        }
        let holding_cost = 0.5 * risk_aversion * variance * dt;
        expected_cost += holding_cost;
        worst_case_cost += holding_cost;
        
        // Transaction costs
        if i < trading_rates.len() {
            let rates = &trading_rates[i];
            let trade_cost: f64 = rates.iter().map(|r| r.abs()).sum::<f64>() * transaction_cost;
            expected_cost += trade_cost;
            worst_case_cost += trade_cost;
        }
        
        // Drift uncertainty cost
        let position_size: f64 = weights.iter().map(|w| w.abs()).sum();
        let uncertainty_cost = drift_uncertainty * position_size * dt;
        worst_case_cost += uncertainty_cost;
    }
    
    TransitionResult {
        trajectory,
        trading_rates,
        expected_cost,
        worst_case_cost,
        times,
    }
}

/// Value-at-Risk under drift uncertainty
/// 
/// Computes VaR considering both volatility and drift uncertainty
pub fn value_at_risk_robust(
    weights: &[f64],
    expected_returns: &[f64],
    covariance: &[f64],
    drift_uncertainty: f64,
    confidence_level: f64,
    time_horizon: f64,
) -> f64 {
    let n = weights.len();
    assert_eq!(expected_returns.len(), n);
    assert_eq!(covariance.len(), n * n);
    
    // Portfolio expected return
    let mu_p: f64 = weights.iter().zip(expected_returns.iter()).map(|(w, r)| w * r).sum();
    
    // Portfolio variance
    let mut var_p = 0.0;
    for i in 0..n {
        for j in 0..n {
            var_p += weights[i] * covariance[i * n + j] * weights[j];
        }
    }
    let sigma_p = var_p.sqrt();
    
    // Worst-case drift
    let mu_worst = mu_p - drift_uncertainty * weights.iter().map(|w| w.abs()).sum::<f64>();
    
    // VaR = -(μ_worst * T - z_α * σ * sqrt(T))
    let z_alpha = normal_inv_cdf(confidence_level);
    let var = -(mu_worst * time_horizon - z_alpha * sigma_p * time_horizon.sqrt());
    
    var
}

/// Expected shortfall (CVaR) under drift uncertainty
pub fn expected_shortfall_robust(
    weights: &[f64],
    expected_returns: &[f64],
    covariance: &[f64],
    drift_uncertainty: f64,
    confidence_level: f64,
    time_horizon: f64,
) -> f64 {
    let n = weights.len();
    
    let mu_p: f64 = weights.iter().zip(expected_returns.iter()).map(|(w, r)| w * r).sum();
    
    let mut var_p = 0.0;
    for i in 0..n {
        for j in 0..n {
            var_p += weights[i] * covariance[i * n + j] * weights[j];
        }
    }
    let sigma_p = var_p.sqrt();
    
    let mu_worst = mu_p - drift_uncertainty * weights.iter().map(|w| w.abs()).sum::<f64>();
    
    let z_alpha = normal_inv_cdf(confidence_level);
    let phi_z = normal_pdf(z_alpha);
    
    // ES = μ_worst * T - σ * sqrt(T) * φ(z_α) / (1 - α)
    let es = -(mu_worst * time_horizon - sigma_p * time_horizon.sqrt() * phi_z / (1.0 - confidence_level));
    
    es
}

// Helper functions

fn invert_matrix(matrix: &[f64], n: usize) -> Vec<f64> {
    // Simplified matrix inversion using Cholesky decomposition
    // For production, use a proper linear algebra library
    let mut result = vec![0.0; n * n];
    
    // Identity matrix initialization
    for i in 0..n {
        result[i * n + i] = 1.0;
    }
    
    // Gauss-Jordan elimination (simplified)
    let mut a = matrix.to_vec();
    
    for i in 0..n {
        // Pivot
        let pivot = a[i * n + i];
        if pivot.abs() < 1e-10 {
            // Singular matrix, return pseudo-inverse (identity scaled)
            for j in 0..n {
                result[j * n + j] = 1.0 / (matrix[j * n + j] + 1e-6);
            }
            return result;
        }
        
        // Scale pivot row
        for j in 0..n {
            a[i * n + j] /= pivot;
            result[i * n + j] /= pivot;
        }
        
        // Eliminate column
        for k in 0..n {
            if k != i {
                let factor = a[k * n + i];
                for j in 0..n {
                    a[k * n + j] -= factor * a[i * n + j];
                    result[k * n + j] -= factor * result[i * n + j];
                }
            }
        }
    }
    
    result
}

fn normal_inv_cdf(p: f64) -> f64 {
    // Approximation of inverse normal CDF (Beasley-Springer-Moro algorithm)
    assert!(p > 0.0 && p < 1.0);
    
    let a = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    
    let b = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    
    let c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    
    let d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    
    let p_low = 0.02425;
    let p_high = 1.0 - p_low;
    
    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
}

fn normal_pdf(x: f64) -> f64 {
    (1.0 / (2.0 * PI).sqrt()) * (-0.5 * x * x).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_portfolio_choice_robust() {
        let returns = vec![0.10, 0.12, 0.08];
        let cov = vec![
            0.04, 0.01, 0.005,
            0.01, 0.05, 0.01,
            0.005, 0.01, 0.03,
        ];
        let result = portfolio_choice_robust(&returns, &cov, 2.0, 0.02, 3);
        
        assert!(result.weights.iter().sum::<f64>().abs() <= 2.0);
        assert!(result.expected_return >= result.worst_case_return);
    }

    #[test]
    fn test_optimal_liquidation() {
        let result = optimal_liquidation(
            100.0,  // initial position
            1.0,    // time horizon
            10,     // steps
            0.0,    // drift
            0.01,   // drift uncertainty
            0.2,    // volatility
            0.1,    // temporary impact
            0.01,   // permanent impact
            1.0,    // risk aversion
        );
        
        assert_eq!(result.trading_schedule.len(), 11);
        assert!(result.trading_schedule[0] == 100.0);
        assert!(result.trading_schedule[10].abs() < 1.0);
    }

    #[test]
    fn test_portfolio_transition() {
        let initial = vec![0.3, 0.3, 0.4];
        let target = vec![0.4, 0.35, 0.25];
        let returns = vec![0.08, 0.10, 0.12];
        let cov = vec![
            0.04, 0.01, 0.005,
            0.01, 0.05, 0.01,
            0.005, 0.01, 0.03,
        ];
        
        let result = portfolio_transition(
            &initial, &target, 1.0, 10,
            &returns, &cov, 0.01, 0.001, 2.0
        );
        
        assert_eq!(result.trajectory.len(), 11);
        assert_eq!(result.trajectory[0], initial);
        
        // Final weights should be close to target
        let final_weights = &result.trajectory[10];
        for i in 0..3 {
            assert!((final_weights[i] - target[i]).abs() < 0.1);
        }
    }
}
