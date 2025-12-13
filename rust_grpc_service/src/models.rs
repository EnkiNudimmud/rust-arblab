/// Mathematical models and utilities for mean-reversion analysis

/// Compute log returns from price series
pub fn compute_log_returns(prices: &[f64]) -> Vec<f64> {
    if prices.len() < 2 {
        return vec![];
    }
    
    prices
        .windows(2)
        .map(|w| (w[1] / w[0]).ln())
        .collect()
}

/// Estimate Ornstein-Uhlenbeck process parameters
pub fn estimate_ou_params(prices: &[f64]) -> (f64, f64, f64, f64, f64, usize) {
    let n = prices.len();
    if n < 2 {
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0);
    }
    
    // Estimate mu as mean price
    let mu = prices.iter().sum::<f64>() / n as f64;
    
    // Compute price differences
    let diffs: Vec<f64> = prices
        .windows(2)
        .map(|w| w[1] - w[0])
        .collect();
    
    if diffs.is_empty() {
        return (0.0, mu, 0.0, 0.0, 0.0, n);
    }
    
    let lags = &prices[..n - 1];
    
    // Regression: diffs = intercept + slope * lags + residuals
    let mean_lag = lags.iter().sum::<f64>() / lags.len() as f64;
    let mean_diff = diffs.iter().sum::<f64>() / diffs.len() as f64;
    
    let num: f64 = lags
        .iter()
        .zip(diffs.iter())
        .map(|(lag, diff)| (lag - mean_lag) * (diff - mean_diff))
        .sum();
    
    let den: f64 = lags
        .iter()
        .map(|lag| (lag - mean_lag).powi(2))
        .sum();
    
    let slope = if den > 1e-10 { num / den } else { 0.0 };
    let theta = -slope;
    
    // Compute sigma from residuals
    let intercept = mean_diff - slope * mean_lag;
    let residuals: Vec<f64> = lags
        .iter()
        .zip(diffs.iter())
        .map(|(lag, diff)| diff - (intercept + slope * lag))
        .collect();
    
    let sigma_sq = residuals.iter().map(|r| r.powi(2)).sum::<f64>() / residuals.len() as f64;
    let sigma = sigma_sq.sqrt();
    
    // Half-life
    let half_life = if theta > 1e-10 {
        2.0_f64.ln() / theta
    } else {
        f64::INFINITY
    };
    
    // R-squared
    let ss_res = residuals.iter().map(|r| r.powi(2)).sum::<f64>();
    let ss_tot = diffs
        .iter()
        .map(|d| (d - mean_diff).powi(2))
        .sum::<f64>();
    
    let r_squared = if ss_tot > 1e-10 {
        1.0 - (ss_res / ss_tot)
    } else {
        0.0
    };
    
    (theta, mu, sigma, half_life, r_squared, n)
}

/// Compute mean and standard deviation
pub fn mean_and_std(values: &[f64]) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|v| (v - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64;
    
    (mean, variance.sqrt())
}

/// Compute covariance matrix from returns (T x N)
pub fn compute_covariance(returns: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if returns.is_empty() || returns[0].is_empty() {
        return vec![];
    }
    
    let n_assets = returns[0].len();
    let n_periods = returns.len();
    
    // Compute means
    let means: Vec<f64> = (0..n_assets)
        .map(|j| {
            returns.iter().map(|r| r[j]).sum::<f64>() / n_periods as f64
        })
        .collect();
    
    // Compute covariance matrix
    let mut cov = vec![vec![0.0; n_assets]; n_assets];
    
    for i in 0..n_assets {
        for j in 0..n_assets {
            let cov_ij = returns
                .iter()
                .map(|r| (r[i] - means[i]) * (r[j] - means[j]))
                .sum::<f64>()
                / n_periods as f64;
            cov[i][j] = cov_ij;
        }
    }
    
    cov
}

/// Matrix inversion using Gaussian elimination with regularization
pub fn matrix_invert(matrix: &[Vec<f64>], regularization: f64) -> Option<Vec<Vec<f64>>> {
    let n = matrix.len();
    if n == 0 || matrix[0].len() != n {
        return None;
    }
    
    // Create augmented matrix [A | I]
    let mut aug = Vec::new();
    for i in 0..n {
        let mut row = matrix[i].clone();
        for j in 0..n {
            row.push(if i == j { 1.0 } else { 0.0 });
        }
        // Add regularization to diagonal
        row[i] += regularization;
        aug.push(row);
    }
    
    // Forward elimination
    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        for k in (i + 1)..n {
            if aug[k][i].abs() > aug[max_row][i].abs() {
                max_row = k;
            }
        }
        
        // Swap rows
        if aug[max_row][i].abs() < 1e-14 {
            return None; // Singular matrix
        }
        aug.swap(i, max_row);
        
        // Make all rows below this one 0 in current column
        for k in (i + 1)..n {
            let c = aug[k][i] / aug[i][i];
            for j in i..2 * n {
                if i == j {
                    aug[k][j] = 0.0;
                } else {
                    aug[k][j] -= c * aug[i][j];
                }
            }
        }
    }
    
    // Back substitution
    for i in (0..n).rev() {
        let c = aug[i][i];
        for j in i..2 * n {
            aug[i][j] /= c;
        }
        for k in 0..i {
            let c = aug[k][i];
            for j in 0..2 * n {
                aug[k][j] -= c * aug[i][j];
            }
        }
    }
    
    // Extract inverse
    let mut inv = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            inv[i][j] = aug[i][n + j];
        }
    }
    
    Some(inv)
}

/// Matrix-vector multiplication
pub fn matrix_vec_multiply(matrix: &[Vec<f64>], vector: &[f64]) -> Vec<f64> {
    matrix
        .iter()
        .map(|row| row.iter().zip(vector.iter()).map(|(a, b)| a * b).sum())
        .collect()
}

/// Vector dot product
pub fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| x * y)
        .sum()
}

/// Compute optimal threshold for mean-reversion trading
pub fn optimal_thresholds(theta: f64, _mu: f64, sigma: f64, transaction_cost: f64) -> (f64, f64, f64) {
    if theta <= 0.0 || sigma <= 0.0 {
        return (2.0, 0.5, 10.0);
    }
    
    let half_life = 2.0_f64.ln() / theta;
    let cost_adjustment = (1.0 + 100.0 * transaction_cost).sqrt();
    
    let entry_z = 1.5 * cost_adjustment;
    let exit_z = 0.3 * cost_adjustment.sqrt();
    let expected_holding = half_life * 0.5;
    
    (entry_z, exit_z, expected_holding)
}

/// CARA optimal portfolio weights
pub fn cara_optimal_weights(
    expected_returns: &[f64],
    covariance: &[Vec<f64>],
    gamma: f64,
) -> Vec<f64> {
    let n = expected_returns.len();
    if n == 0 {
        return vec![];
    }
    
    // Add regularization to covariance matrix
    let mut cov_reg = covariance.to_vec();
    for i in 0..n {
        cov_reg[i][i] += 1e-8;
    }
    
    // Invert covariance matrix
    if let Some(sigma_inv) = matrix_invert(&cov_reg, 0.0) {
        // weights = Σ^-1 @ μ / γ
        let inv_times_returns = matrix_vec_multiply(&sigma_inv, expected_returns);
        let weights: Vec<f64> = inv_times_returns.iter().map(|w| w / gamma).collect();
        weights
    } else {
        // Fallback to equal-weight
        vec![1.0 / n as f64; n]
    }
}

/// Sharpe optimal portfolio weights
pub fn sharpe_optimal_weights(
    expected_returns: &[f64],
    covariance: &[Vec<f64>],
    risk_free_rate: f64,
) -> Vec<f64> {
    let n = expected_returns.len();
    if n == 0 {
        return vec![];
    }
    
    // Excess returns
    let excess_returns: Vec<f64> = expected_returns
        .iter()
        .map(|r| r - risk_free_rate)
        .collect();
    
    // Add regularization
    let mut cov_reg = covariance.to_vec();
    for i in 0..n {
        cov_reg[i][i] += 1e-8;
    }
    
    // Invert and multiply
    if let Some(sigma_inv) = matrix_invert(&cov_reg, 0.0) {
        let mut weights = matrix_vec_multiply(&sigma_inv, &excess_returns);
        let total: f64 = weights.iter().sum();
        if total.abs() > 1e-10 {
            weights.iter_mut().for_each(|w| *w /= total);
        } else {
            weights = vec![1.0 / n as f64; n];
        }
        weights
    } else {
        vec![1.0 / n as f64; n]
    }
}

/// Backtest mean-reversion strategy
pub fn backtest_with_costs(
    prices: &[f64],
    entry_z: f64,
    exit_z: f64,
    transaction_cost: f64,
) -> BacktestResult {
    let n = prices.len();
    let window = (n / 4).max(20);
    
    let mut positions = vec![0i32; n];
    let mut pnl = vec![0.0; n];
    let mut returns = vec![0.0; n];
    
    let mut current_position: i32 = 0;
    let mut cash = 100000.0;
    let mut portfolio_value = cash;
    let mut peak_value = cash;
    let mut max_drawdown = 0.0;
    let mut total_costs = 0.0;
    let mut num_trades = 0;
    
    for i in window..n {
        let window_prices = &prices[i - window..i];
        let (mean, std) = mean_and_std(window_prices);
        
        if std < 1e-10 {
            positions[i] = current_position;
            continue;
        }
        
        let z_score = (prices[i] - mean) / std;
        let prev_position = current_position;
        
        // Trading logic
        if z_score < -entry_z && current_position == 0 {
            current_position = 1;
        } else if z_score > entry_z && current_position == 0 {
            current_position = -1;
        } else if z_score.abs() < exit_z && current_position != 0 {
            current_position = 0;
        }
        
        positions[i] = current_position;
        
        // Compute returns with costs
        if i > 0 {
            let price_return = (prices[i] - prices[i - 1]) / prices[i - 1];
            
            let mut cost = 0.0;
            if prev_position != current_position {
                let position_change = (prev_position - current_position).abs() as f64;
                cost = transaction_cost * prices[i] * position_change;
                total_costs += cost;
                num_trades += 1;
            }
            
            returns[i] = price_return * prev_position as f64 - (cost / portfolio_value);
            portfolio_value *= 1.0 + returns[i];
            pnl[i] = portfolio_value - cash;
            
            if portfolio_value > peak_value {
                peak_value = portfolio_value;
            }
            
            let drawdown = (peak_value - portfolio_value) / peak_value;
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }
        }
    }
    
    // Compute Sharpe ratio
    let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance = returns
        .iter()
        .map(|r| (r - mean_return).powi(2))
        .sum::<f64>()
        / returns.len() as f64;
    let return_std = variance.sqrt();
    let sharpe = if return_std > 1e-10 {
        (mean_return / return_std) * (252.0_f64.sqrt())
    } else {
        0.0
    };
    
    BacktestResult {
        cumulative_return: (portfolio_value - cash) / cash,
        sharpe_ratio: sharpe,
        max_drawdown,
        num_trades,
        total_costs,
        win_rate: compute_win_rate(&returns),
        avg_profit_per_trade: if num_trades > 0 {
            (portfolio_value - cash) / num_trades as f64
        } else {
            0.0
        },
    }
}

pub struct BacktestResult {
    pub cumulative_return: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub num_trades: i32,
    pub total_costs: f64,
    pub win_rate: f64,
    pub avg_profit_per_trade: f64,
}

fn compute_win_rate(returns: &[f64]) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }
    
    let winning = returns.iter().filter(|r| **r > 1e-10).count();
    winning as f64 / returns.len() as f64
}

/// Simple PCA using power iteration method
pub fn compute_pca(data: &[Vec<f64>], n_components: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    if data.is_empty() || data[0].is_empty() {
        return (vec![], vec![]);
    }
    
    let n_features = data[0].len();
    let n_components = n_components.min(n_features);
    
    // Compute covariance matrix
    let cov = compute_covariance(data);
    
    // Initialize components and explained variance
    let mut components = vec![];
    let mut explained_variance = vec![];
    let total_variance = cov.iter().enumerate().map(|(i, row)| row[i]).sum::<f64>();
    
    // Power iteration to find top components (simplified for initial MVP)
    // In production, use eigendecomposition
    for k in 0..n_components {
        let mut component = vec![1.0 / (n_features as f64).sqrt(); n_features];
        
        // Simple power iteration (5 iterations for stability)
        for _ in 0..5 {
            let new_component = matrix_vec_multiply(&cov, &component);
            let norm = (new_component.iter().map(|x| x.powi(2)).sum::<f64>()).sqrt();
            if norm > 1e-10 {
                component = new_component.iter().map(|x| x / norm).collect();
            }
        }
        
        // Compute variance explained
        let variance = dot_product(&component, &matrix_vec_multiply(&cov, &component));
        let explained_var_ratio = if total_variance > 1e-10 {
            variance / total_variance
        } else {
            1.0 / n_components as f64
        };
        
        components.push(component);
        explained_variance.push(explained_var_ratio);
    }
    
    (components, explained_variance)
}
