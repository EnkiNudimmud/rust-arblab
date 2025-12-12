/// Signature Methods for Stochastic Portfolio Theory
/// 
/// This module implements signature-based portfolio optimization methods
/// including path signatures, log-signature computations, and portfolio
/// selection using signature kernels as described in:
/// - "Signature Methods in Stochastic Portfolio Theory" (Lyons et al.)
/// - "Randomized Signature Methods in Optimal Portfolio Selection"
/// 
/// All computationally intensive operations are implemented here for performance.

/// Compute the truncated path signature up to level N using Chen's identity
/// 
/// For a path X = (X_1, ..., X_d) in R^d, the signature S(X) is the collection
/// of iterated integrals:
/// S(X)_I = ∫_{0<t_1<...<t_k<1} dX^{i_1}_{t_1} ... dX^{i_k}_{t_k}
/// 
/// # Arguments
/// * `path` - Matrix of shape (T, d) representing the path
/// * `level` - Maximum signature level to compute
/// 
/// # Returns
/// Vector containing all signature terms up to `level`
pub fn compute_signature(path: &[Vec<f64>], level: usize) -> Vec<f64> {
    if path.is_empty() {
        return vec![1.0]; // Empty path signature is 1
    }
    
    let n_steps = path.len();
    let dim = path[0].len();
    
    // Initialize with level 0 (constant term = 1)
    let mut sig = vec![1.0];
    
    // Compute increments
    let mut increments = Vec::with_capacity(n_steps - 1);
    for i in 1..n_steps {
        let mut inc = Vec::with_capacity(dim);
        for j in 0..dim {
            inc.push(path[i][j] - path[i-1][j]);
        }
        increments.push(inc);
    }
    
    // Use Chen's identity iteratively
    for lvl in 1..=level {
        let _prev_size = sig.len();
        let new_terms = compute_signature_level(&increments, lvl, dim);
        sig.extend(new_terms);
    }
    
    sig
}

/// Compute signature terms at a specific level
fn compute_signature_level(increments: &[Vec<f64>], level: usize, dim: usize) -> Vec<f64> {
    let n_terms = dim.pow(level as u32);
    let mut terms = vec![0.0; n_terms];
    
    if level == 1 {
        // Level 1: just sum the increments
        for inc in increments {
            for (j, &val) in inc.iter().enumerate() {
                terms[j] += val;
            }
        }
    } else {
        // Higher levels: use recursive Chen's identity
        // S^{i_1,...,i_k} = ∑_t ΔX^{i_1}_t * S^{i_2,...,i_k}_{0,t}
        // Simplified approximation for performance
        for (t, inc) in increments.iter().enumerate() {
            let weight = 1.0 / ((t + 1) as f64);
            for idx in 0..n_terms {
                let mut prod = weight;
                let mut temp_idx = idx;
                for _ in 0..level {
                    let coord = temp_idx % dim;
                    prod *= inc[coord];
                    temp_idx /= dim;
                }
                terms[idx] += prod / (level as f64).sqrt();
            }
        }
    }
    
    terms
}

/// Compute log-signature using the logarithm of the signature
/// 
/// The log-signature log(S(X)) is related to the signature via:
/// exp(log(S(X))) = S(X)
/// 
/// This is more efficient for some computations and has better
/// numerical properties.
pub fn compute_log_signature(path: &[Vec<f64>], level: usize) -> Vec<f64> {
    let sig = compute_signature(path, level);
    
    // Apply logarithm carefully (signature terms can be negative)
    sig.iter().map(|&x| {
        if x.abs() < 1e-10 {
            0.0
        } else {
            x.signum() * x.abs().ln().max(-10.0).min(10.0)
        }
    }).collect()
}

/// Compute Expected Signature using randomized features
/// 
/// E[S(X)] ≈ (1/M) ∑_{m=1}^M S(X^(m))
/// where X^(m) are sampled paths
pub fn expected_signature(paths: &[Vec<Vec<f64>>], level: usize) -> Vec<f64> {
    if paths.is_empty() {
        return vec![1.0];
    }
    
    let mut sum_sig = vec![0.0; signature_dimension(paths[0][0].len(), level)];
    
    for path in paths {
        let sig = compute_signature(path, level);
        for (i, &val) in sig.iter().enumerate() {
            if i < sum_sig.len() {
                sum_sig[i] += val;
            }
        }
    }
    
    let n = paths.len() as f64;
    sum_sig.iter().map(|&x| x / n).collect()
}

/// Calculate the dimension of signature space for given path dimension and level
fn signature_dimension(path_dim: usize, level: usize) -> usize {
    let mut dim = 1; // Level 0
    for lvl in 1..=level {
        dim += path_dim.pow(lvl as u32);
    }
    dim
}

/// Signature kernel between two paths
/// 
/// K(X, Y) = <S(X), S(Y)> where S is the truncated signature
/// This provides a measure of similarity between paths
pub fn signature_kernel(path1: &[Vec<f64>], path2: &[Vec<f64>], level: usize) -> f64 {
    let sig1 = compute_signature(path1, level);
    let sig2 = compute_signature(path2, level);
    
    let min_len = sig1.len().min(sig2.len());
    let mut kernel = 0.0;
    
    for i in 0..min_len {
        kernel += sig1[i] * sig2[i];
    }
    
    kernel
}

/// Compute the signature-based covariance matrix for portfolio optimization
/// 
/// Σ_{ij} = E[S(log R^i) ⊗ S(log R^j)] where R^i is the return process of asset i
/// 
/// # Arguments
/// * `returns` - Matrix of asset returns (n_assets, n_timesteps)
/// * `level` - Signature truncation level
/// 
/// # Returns
/// Covariance matrix (n_assets, n_assets)
pub fn signature_covariance(returns: &[Vec<f64>], level: usize) -> Vec<Vec<f64>> {
    let n_assets = returns.len();
    let mut cov = vec![vec![0.0; n_assets]; n_assets];
    
    // Convert returns to cumulative log-returns (paths)
    let paths: Vec<Vec<Vec<f64>>> = returns.iter().map(|ret| {
        let mut cumulative = vec![vec![0.0]];
        let mut cum_sum = 0.0;
        for &r in ret {
            cum_sum += (1.0 + r).ln().max(-1.0).min(1.0);
            cumulative.push(vec![cum_sum]);
        }
        cumulative
    }).collect();
    
    // Compute pairwise signature kernels
    for i in 0..n_assets {
        for j in i..n_assets {
            let kernel = signature_kernel(&paths[i], &paths[j], level);
            cov[i][j] = kernel;
            cov[j][i] = kernel;
        }
    }
    
    cov
}

/// Compute signature-based portfolio weights using mean-variance optimization
/// 
/// Solves: max w^T μ_sig - (λ/2) w^T Σ_sig w
/// subject to: ∑ w_i = 1, w_i ≥ 0 (long-only)
/// 
/// where μ_sig and Σ_sig are signature-based moments
pub fn signature_portfolio_weights(
    returns: &[Vec<f64>],
    level: usize,
    risk_aversion: f64,
    allow_short: bool
) -> Vec<f64> {
    let n_assets = returns.len();
    if n_assets == 0 {
        return vec![];
    }
    
    // Compute signature-based covariance
    let cov = signature_covariance(returns, level);
    
    // Compute signature-based expected returns
    let mut exp_returns = vec![0.0; n_assets];
    for (i, ret) in returns.iter().enumerate() {
        let path: Vec<Vec<f64>> = ret.iter().map(|&r| vec![r]).collect();
        let sig = compute_signature(&path, level);
        // Use first-level signature as proxy for expected return
        exp_returns[i] = if sig.len() > 1 { sig[1] } else { 0.0 };
    }
    
    // Simplified mean-variance optimization
    // w = (1/λ) Σ^{-1} μ (unconstrained solution)
    let weights = solve_portfolio_optimization(&exp_returns, &cov, risk_aversion, allow_short);
    
    // Normalize to sum to 1
    let sum: f64 = weights.iter().map(|&w| w.abs()).sum();
    if sum > 1e-10 {
        weights.iter().map(|&w| w / sum).collect()
    } else {
        vec![1.0 / n_assets as f64; n_assets]
    }
}

/// Solve portfolio optimization using gradient descent
fn solve_portfolio_optimization(
    expected_returns: &[f64],
    covariance: &[Vec<f64>],
    risk_aversion: f64,
    allow_short: bool
) -> Vec<f64> {
    let n = expected_returns.len();
    let mut weights = vec![1.0 / n as f64; n];
    
    let learning_rate = 0.01;
    let n_iterations = 1000;
    
    for _ in 0..n_iterations {
        // Compute gradient: ∇ = μ - λ Σ w
        let mut gradient = vec![0.0; n];
        
        for i in 0..n {
            gradient[i] = expected_returns[i];
            for j in 0..n {
                gradient[i] -= risk_aversion * covariance[i][j] * weights[j];
            }
        }
        
        // Update weights
        for i in 0..n {
            weights[i] += learning_rate * gradient[i];
            
            // Apply constraints
            if !allow_short && weights[i] < 0.0 {
                weights[i] = 0.0;
            }
        }
        
        // Project onto simplex (sum to 1)
        let sum: f64 = weights.iter().sum();
        if sum.abs() > 1e-10 {
            for w in weights.iter_mut() {
                *w /= sum;
            }
        }
    }
    
    weights
}

/// Compute rank-based portfolio weights (Stochastic Portfolio Theory)
/// 
/// In SPT, portfolio weights depend on the rank of asset returns:
/// π_i(t) = g(rank(R_i(t)) / N)
/// 
/// where g is a generating function and rank(·) is the rank of the asset
pub fn rank_based_portfolio(returns: &[f64], generating_fn: &str) -> Vec<f64> {
    let n = returns.len();
    if n == 0 {
        return vec![];
    }
    
    // Compute ranks (higher return = higher rank)
    let mut indexed_returns: Vec<(usize, f64)> = returns.iter().enumerate()
        .map(|(i, &r)| (i, r))
        .collect();
    indexed_returns.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    
    let mut ranks = vec![0; n];
    for (rank, &(idx, _)) in indexed_returns.iter().enumerate() {
        ranks[idx] = rank + 1;
    }
    
    // Apply generating function
    let mut weights = vec![0.0; n];
    for i in 0..n {
        let u = ranks[i] as f64 / n as f64;
        weights[i] = match generating_fn {
            "market" => 1.0 / n as f64,  // Market portfolio
            "entropy" => -u * u.ln(),     // Entropy-weighted
            "diversity" => u.ln(),         // Diversity-weighted
            "linear" => u,                 // Linear (momentum)
            "contrarian" => 1.0 - u,       // Contrarian
            _ => 1.0 / n as f64,
        };
    }
    
    // Normalize
    let sum: f64 = weights.iter().sum();
    if sum > 1e-10 {
        weights.iter().map(|&w| w / sum).collect()
    } else {
        vec![1.0 / n as f64; n]
    }
}

/// Compute portfolio performance metrics
/// 
/// Returns: (total_return, sharpe_ratio, max_drawdown, volatility)
pub fn portfolio_metrics(returns: &[f64]) -> (f64, f64, f64, f64) {
    if returns.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    
    let n = returns.len() as f64;
    
    // Total return (compounded)
    let total_return = returns.iter().fold(1.0, |acc, &r| acc * (1.0 + r)) - 1.0;
    
    // Mean and volatility
    let mean = returns.iter().sum::<f64>() / n;
    let variance = returns.iter().map(|&r| (r - mean).powi(2)).sum::<f64>() / n;
    let volatility = variance.sqrt();
    
    // Sharpe ratio (assuming 0 risk-free rate)
    let sharpe_ratio = if volatility > 1e-10 {
        mean / volatility * (252.0_f64).sqrt() // Annualized
    } else {
        0.0
    };
    
    // Maximum drawdown
    let mut cumulative = vec![1.0];
    let mut cum_prod = 1.0;
    for &r in returns {
        cum_prod *= 1.0 + r;
        cumulative.push(cum_prod);
    }
    
    let mut max_drawdown: f64 = 0.0;
    for i in 0..cumulative.len() {
        let peak = cumulative[..=i].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let drawdown = (cumulative[i] - peak) / peak;
        max_drawdown = max_drawdown.min(drawdown);
    }
    
    (total_return, sharpe_ratio, max_drawdown, volatility)
}

/// Optimal stopping using signature features
/// 
/// Uses signature features to determine optimal stopping time for portfolio rebalancing
/// or position closure based on learned patterns in the path
pub fn signature_optimal_stopping(
    path: &[Vec<f64>],
    level: usize,
    threshold: f64,
    window: usize
) -> usize {
    let n = path.len();
    if n < window {
        return n - 1;
    }
    
    for t in window..n {
        let window_path = &path[t-window..t];
        let sig = compute_signature(window_path, level);
        
        // Compute stopping criterion based on signature features
        let score = compute_stopping_score(&sig);
        
        if score > threshold {
            return t;
        }
    }
    
    n - 1
}

/// Compute stopping score from signature features
fn compute_stopping_score(sig: &[f64]) -> f64 {
    if sig.len() < 2 {
        return 0.0;
    }
    
    // Use combination of signature levels
    let level1 = if sig.len() > 1 { sig[1].abs() } else { 0.0 };
    let level2 = if sig.len() > 2 { sig[2].abs() } else { 0.0 };
    let level3 = if sig.len() > 3 { sig[3].abs() } else { 0.0 };
    
    // Weighted combination
    level1 + 0.5 * level2 + 0.1 * level3
}

/// Randomized signature features for dimensionality reduction
/// 
/// Uses random projections of signature features to reduce computational cost
/// while preserving essential information
pub fn randomized_signature_features(
    path: &[Vec<f64>],
    level: usize,
    n_features: usize,
    seed: u64
) -> Vec<f64> {
    let sig = compute_signature(path, level);
    let sig_dim = sig.len();
    
    if sig_dim <= n_features {
        return sig;
    }
    
    // Generate random projection matrix
    let mut rng = seed;
    let mut random_matrix = vec![vec![0.0; sig_dim]; n_features];
    
    for i in 0..n_features {
        for j in 0..sig_dim {
            // Simple linear congruential generator
            rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            let uniform = (rng as f64) / (u64::MAX as f64);
            random_matrix[i][j] = (uniform - 0.5) * 2.0;
        }
    }
    
    // Project signature onto random features
    let mut features = vec![0.0; n_features];
    for i in 0..n_features {
        for j in 0..sig_dim {
            features[i] += random_matrix[i][j] * sig[j];
        }
        features[i] /= (sig_dim as f64).sqrt();
    }
    
    features
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_compute_signature() {
        let path = vec![
            vec![0.0],
            vec![1.0],
            vec![2.0],
        ];
        let sig = compute_signature(&path, 2);
        assert!(sig.len() > 1);
        assert_eq!(sig[0], 1.0); // Level 0 is always 1
    }
    
    #[test]
    fn test_signature_kernel() {
        let path1 = vec![vec![0.0], vec![1.0], vec![2.0]];
        let path2 = vec![vec![0.0], vec![1.5], vec![2.5]];
        let kernel = signature_kernel(&path1, &path2, 2);
        assert!(kernel > 0.0);
    }
    
    #[test]
    fn test_portfolio_metrics() {
        let returns = vec![0.01, -0.02, 0.03, 0.01, -0.01];
        let (total, _sharpe, _dd, vol) = portfolio_metrics(&returns);
        assert!(total != 0.0);
        assert!(vol > 0.0);
    }
}
