/// Sparse Portfolio Optimization
/// 
/// Implementations of sparse PCA and sparse mean-reversion portfolios

use nalgebra::{DMatrix, DVector};
use std::collections::HashMap;

/// Sparse PCA using Iterative Soft-Thresholding
pub fn sparse_pca(
    returns: &[Vec<f64>],
    lambda: f64,
    max_iter: usize,
    tol: f64,
) -> SparsePCAResult {
    if returns.is_empty() || returns[0].is_empty() {
        return SparsePCAResult::default();
    }
    
    let n_assets = returns.len();
    let _n_periods = returns[0].len();
    
    // Compute covariance matrix
    let cov = compute_covariance(returns);
    
    // Initialize with uniform weights
    let mut w = DVector::from_element(n_assets, 1.0 / (n_assets as f64).sqrt());
    
    let mut iter = 0;
    let mut converged = false;
    
    while iter < max_iter && !converged {
        let w_old = w.clone();
        
        // Gradient step: w_new = Σw
        let mut w_new = &cov * &w;
        
        // Soft thresholding
        for i in 0..n_assets {
            let val = w_new[i];
            w_new[i] = val.signum() * (val.abs() - lambda).max(0.0);
        }
        
        // Normalize
        let norm = w_new.norm();
        if norm > 1e-10 {
            w_new /= norm;
        } else {
            // Restart if all weights shrunk to zero
            w_new = DVector::from_element(n_assets, 1.0 / (n_assets as f64).sqrt());
        }
        
        w = w_new;
        
        // Check convergence
        let diff = (&w - &w_old).norm();
        if diff < tol {
            converged = true;
        }
        
        iter += 1;
    }
    
    // Compute variance explained
    let var = w.transpose() * &cov * &w;
    let variance_explained = var[(0, 0)];
    
    // Compute sparsity
    let non_zero = w.iter().filter(|&&x| x.abs() > 1e-6).count();
    let sparsity = non_zero as f64 / n_assets as f64;
    
    let weights: Vec<f64> = w.iter().copied().collect();
    
    SparsePCAResult {
        weights,
        variance_explained,
        sparsity,
        iterations: iter,
        converged,
    }
}

/// Find sparse mean-reverting portfolio
pub fn sparse_meanrev_portfolio(
    prices: &[Vec<f64>],
    lambda: f64,
    lookback: usize,
) -> SparsePortfolioResult {
    if prices.is_empty() || prices[0].is_empty() {
        return SparsePortfolioResult::default();
    }
    
    let _n_assets = prices.len();
    let n_periods = prices[0].len();
    
    // Validate lookback
    if n_periods < 2 {
        return SparsePortfolioResult::default();
    }
    
    // Calculate returns (use all available data or lookback, whichever is smaller)
    let actual_lookback = lookback.min(n_periods - 1).max(1);
    let returns: Vec<Vec<f64>> = prices.iter().map(|asset_prices| {
        let start_idx = if asset_prices.len() > actual_lookback + 1 {
            asset_prices.len() - actual_lookback - 1
        } else {
            0
        };
        (start_idx + 1..asset_prices.len())
            .map(|i| {
                if asset_prices[i-1].abs() > 1e-10 {
                    (asset_prices[i] / asset_prices[i-1]).ln()
                } else {
                    0.0
                }
            })
            .collect()
    }).collect();
    
    // Validate we have returns
    if returns.is_empty() || returns[0].is_empty() {
        return SparsePortfolioResult::default();
    }
    
    // Use sparse PCA to find mean-reverting combinations
    let result = sparse_pca(&returns, lambda, 100, 1e-4);
    
    // Test for mean reversion
    let portfolio_prices = compute_portfolio_prices(prices, &result.weights);
    let half_life = estimate_half_life(&portfolio_prices);
    
    // Calculate additional metrics
    let mut metrics = HashMap::new();
    metrics.insert("half_life".to_string(), half_life);
    metrics.insert("sparsity".to_string(), result.sparsity);
    metrics.insert("variance_explained".to_string(), result.variance_explained);
    
    // Count selected assets
    let n_selected = result.weights.iter().filter(|&&w| w.abs() > 1e-6).count();
    
    SparsePortfolioResult {
        weights: result.weights,
        n_assets_selected: n_selected as i32,
        objective_value: half_life,
        metrics,
    }
}

fn compute_covariance(returns: &[Vec<f64>]) -> DMatrix<f64> {
    let n_assets = returns.len();
    
    if n_assets == 0 || returns[0].is_empty() {
        return DMatrix::zeros(1, 1);
    }
    
    let n_periods = returns[0].len();
    
    // Calculate means
    let means: Vec<f64> = returns.iter()
        .map(|r| {
            if r.is_empty() {
                0.0
            } else {
                r.iter().sum::<f64>() / r.len() as f64
            }
        })
        .collect();
    
    // Compute covariance
    let mut cov = DMatrix::zeros(n_assets, n_assets);
    
    if n_periods < 2 {
        // Not enough data for covariance - return identity matrix
        for i in 0..n_assets {
            cov[(i, i)] = 1.0;
        }
        return cov;
    }
    
    for i in 0..n_assets {
        for j in 0..n_assets {
            let mut sum = 0.0;
            let valid_periods = n_periods.min(returns[i].len()).min(returns[j].len());
            for t in 0..valid_periods {
                sum += (returns[i][t] - means[i]) * (returns[j][t] - means[j]);
            }
            cov[(i, j)] = sum / ((valid_periods - 1).max(1) as f64);
        }
    }
    
    cov
}

fn compute_portfolio_prices(prices: &[Vec<f64>], weights: &[f64]) -> Vec<f64> {
    let n_periods = prices[0].len();
    let mut portfolio_prices = vec![0.0; n_periods];
    
    for t in 0..n_periods {
        let mut value = 0.0;
        for (i, asset_prices) in prices.iter().enumerate() {
            value += weights[i] * asset_prices[t];
        }
        portfolio_prices[t] = value;
    }
    
    portfolio_prices
}

fn estimate_half_life(prices: &[f64]) -> f64 {
    if prices.len() < 3 {
        return f64::INFINITY;
    }
    
    // Estimate OU process
    let mut diffs = Vec::new();
    let mut lags = Vec::new();
    
    for i in 1..prices.len() {
        diffs.push(prices[i] - prices[i-1]);
        lags.push(prices[i-1]);
    }
    
    let mean_lag = lags.iter().sum::<f64>() / lags.len() as f64;
    let mean_diff = diffs.iter().sum::<f64>() / diffs.len() as f64;
    
    let mut num = 0.0;
    let mut den = 0.0;
    
    for i in 0..lags.len() {
        num += (lags[i] - mean_lag) * (diffs[i] - mean_diff);
        den += (lags[i] - mean_lag).powi(2);
    }
    
    let slope = if den.abs() > 1e-10 { num / den } else { 0.0 };
    let theta = -slope;
    
    if theta > 1e-10 {
        (2.0_f64).ln() / theta
    } else {
        f64::INFINITY
    }
}

#[derive(Debug, Clone, Default)]
pub struct SparsePCAResult {
    pub weights: Vec<f64>,
    pub variance_explained: f64,
    pub sparsity: f64,
    pub iterations: usize,
    pub converged: bool,
}

#[derive(Debug, Clone, Default)]
pub struct SparsePortfolioResult {
    pub weights: Vec<f64>,
    pub n_assets_selected: i32,
    pub objective_value: f64,
    pub metrics: HashMap<String, f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sparse_pca() {
        let returns = vec![
            vec![0.01, -0.02, 0.015],
            vec![0.02, -0.01, 0.01],
        ];
        let result = sparse_pca(&returns, 0.1, 100, 1e-4);
        assert_eq!(result.weights.len(), 2);
        assert!(result.sparsity <= 1.0);
    }
}
