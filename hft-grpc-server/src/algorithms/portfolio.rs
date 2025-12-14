/// Portfolio Optimization Algorithms
/// 
/// Efficient frontier, mean-variance optimization, risk parity

#[allow(unused_imports)]
use nalgebra::{DMatrix, DVector};
use std::collections::HashMap;

/// Optimize portfolio using specified method
pub fn optimize_portfolio(
    prices: &[Vec<f64>],
    method: &str,
    risk_free_rate: f64,
    _target_return: Option<f64>,
) -> PortfolioResult {
    if prices.is_empty() || prices[0].is_empty() {
        return PortfolioResult::default();
    }
    
    let n_assets = prices.len();
    let _n_periods = prices[0].len();
    
    // Calculate returns
    let returns = calculate_returns(prices);
    
    // Calculate statistics
    let mean_returns = calculate_mean_returns(&returns);
    let cov_matrix = calculate_covariance(&returns);
    
    // Optimize based on method
    let weights = match method {
        "max_sharpe" => optimize_max_sharpe(&mean_returns, &cov_matrix, risk_free_rate),
        "min_variance" => optimize_min_variance(&cov_matrix),
        "risk_parity" => optimize_risk_parity(&cov_matrix),
        "equal_weight" => vec![1.0 / n_assets as f64; n_assets],
        _ => vec![1.0 / n_assets as f64; n_assets],
    };
    
    // Calculate portfolio metrics
    let expected_return = weights.iter().zip(mean_returns.iter())
        .map(|(w, r)| w * r)
        .sum();
    
    let variance = calculate_portfolio_variance(&weights, &cov_matrix);
    let volatility = variance.sqrt();
    let sharpe_ratio = if volatility > 1e-10 {
        (expected_return - risk_free_rate) / volatility
    } else {
        0.0
    };
    
    let mut metrics = HashMap::new();
    metrics.insert("expected_return".to_string(), expected_return);
    metrics.insert("volatility".to_string(), volatility);
    metrics.insert("sharpe_ratio".to_string(), sharpe_ratio);
    metrics.insert("n_assets".to_string(), n_assets as f64);
    
    PortfolioResult {
        weights,
        expected_return,
        volatility,
        sharpe_ratio,
        metrics,
    }
}

fn calculate_returns(prices: &[Vec<f64>]) -> Vec<Vec<f64>> {
    prices.iter().map(|asset_prices| {
        (1..asset_prices.len())
            .map(|i| (asset_prices[i] / asset_prices[i-1]) - 1.0)
            .collect()
    }).collect()
}

fn calculate_mean_returns(returns: &[Vec<f64>]) -> Vec<f64> {
    returns.iter().map(|asset_returns| {
        asset_returns.iter().sum::<f64>() / asset_returns.len() as f64
    }).collect()
}

fn calculate_covariance(returns: &[Vec<f64>]) -> DMatrix<f64> {
    let n_assets = returns.len();
    let n_periods = returns[0].len();
    
    let means = calculate_mean_returns(returns);
    let mut cov = DMatrix::zeros(n_assets, n_assets);
    
    for i in 0..n_assets {
        for j in 0..n_assets {
            let mut sum = 0.0;
            for t in 0..n_periods {
                sum += (returns[i][t] - means[i]) * (returns[j][t] - means[j]);
            }
            cov[(i, j)] = sum / (n_periods - 1) as f64;
        }
    }
    
    cov
}

fn calculate_portfolio_variance(weights: &[f64], cov: &DMatrix<f64>) -> f64 {
    let n = weights.len();
    let mut variance = 0.0;
    
    for i in 0..n {
        for j in 0..n {
            variance += weights[i] * weights[j] * cov[(i, j)];
        }
    }
    
    variance
}

/// Maximize Sharpe ratio using analytical solution
fn optimize_max_sharpe(
    mean_returns: &[f64],
    cov: &DMatrix<f64>,
    risk_free_rate: f64,
) -> Vec<f64> {
    let n = mean_returns.len();
    
    // Excess returns
    let excess: Vec<f64> = mean_returns.iter()
        .map(|&r| r - risk_free_rate)
        .collect();
    
    // Try to invert covariance matrix
    match cov.clone().try_inverse() {
        Some(cov_inv) => {
            // w ∝ Σ^(-1) * (μ - rf)
            let mut weights = Vec::with_capacity(n);
            for i in 0..n {
                let mut w = 0.0;
                for j in 0..n {
                    w += cov_inv[(i, j)] * excess[j];
                }
                weights.push(w);
            }
            
            // Normalize to sum to 1
            let sum: f64 = weights.iter().sum();
            if sum.abs() > 1e-10 {
                weights.iter_mut().for_each(|w| *w /= sum);
            } else {
                // Fallback to equal weights
                return vec![1.0 / n as f64; n];
            }
            
            weights
        }
        None => {
            // Singular matrix - use equal weights
            vec![1.0 / n as f64; n]
        }
    }
}

/// Minimize variance (Global Minimum Variance Portfolio)
fn optimize_min_variance(cov: &DMatrix<f64>) -> Vec<f64> {
    let n = cov.nrows();
    
    match cov.clone().try_inverse() {
        Some(cov_inv) => {
            // w ∝ Σ^(-1) * 1
            let mut weights = Vec::with_capacity(n);
            for i in 0..n {
                let mut w = 0.0;
                for j in 0..n {
                    w += cov_inv[(i, j)];
                }
                weights.push(w);
            }
            
            // Normalize
            let sum: f64 = weights.iter().sum();
            if sum.abs() > 1e-10 {
                weights.iter_mut().for_each(|w| *w /= sum);
            } else {
                return vec![1.0 / n as f64; n];
            }
            
            weights
        }
        None => vec![1.0 / n as f64; n],
    }
}

/// Risk Parity: Equal risk contribution from each asset
fn optimize_risk_parity(cov: &DMatrix<f64>) -> Vec<f64> {
    let n = cov.nrows();
    
    // Iterative algorithm (simplified)
    let mut weights = vec![1.0 / n as f64; n];
    
    for _ in 0..50 {
        // Calculate marginal risk contributions
        let portfolio_vol = calculate_portfolio_variance(&weights, cov).sqrt();
        
        if portfolio_vol < 1e-10 {
            break;
        }
        
        let mut risk_contributions = vec![0.0; n];
        for i in 0..n {
            let mut marginal = 0.0;
            for j in 0..n {
                marginal += weights[j] * cov[(i, j)];
            }
            risk_contributions[i] = weights[i] * marginal / portfolio_vol;
        }
        
        // Update weights to equalize risk contributions
        let target_risk = 1.0 / n as f64;
        for i in 0..n {
            if risk_contributions[i] > 1e-10 {
                weights[i] *= target_risk / risk_contributions[i];
            }
        }
        
        // Normalize
        let sum: f64 = weights.iter().sum();
        if sum > 1e-10 {
            weights.iter_mut().for_each(|w| *w /= sum);
        }
    }
    
    weights
}

#[derive(Debug, Clone, Default)]
pub struct PortfolioResult {
    pub weights: Vec<f64>,
    pub expected_return: f64,
    pub volatility: f64,
    pub sharpe_ratio: f64,
    pub metrics: HashMap<String, f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_equal_weight() {
        let prices = vec![
            vec![100.0, 101.0, 102.0],
            vec![50.0, 51.0, 52.0],
        ];
        let result = optimize_portfolio(&prices, "equal_weight", 0.02, None);
        assert_eq!(result.weights.len(), 2);
        assert!((result.weights[0] - 0.5).abs() < 0.01);
    }
}
