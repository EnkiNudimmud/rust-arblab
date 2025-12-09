/// Mean Reversion Analysis
/// 
/// Pure Rust implementation optimized for gRPC latency

#[allow(unused_imports)]
use nalgebra::{DMatrix, DVector, SVD};
use std::collections::HashMap;

/// Calculate mean reversion metrics with z-score
pub fn calculate_mean_reversion(
    prices: &[f64],
    lookback: usize,
    threshold: f64,
) -> MeanReversionResult {
    if prices.is_empty() || prices.len() < lookback {
        return MeanReversionResult::default();
    }
    
    // Use most recent data
    let recent = &prices[prices.len() - lookback..];
    
    // Calculate statistics
    let mean: f64 = recent.iter().sum::<f64>() / lookback as f64;
    let variance: f64 = recent.iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>() / lookback as f64;
    let std = variance.sqrt();
    
    let current = prices[prices.len() - 1];
    let zscore = if std > 1e-10 { (current - mean) / std } else { 0.0 };
    
    // Generate trading signals
    let signal = if zscore > threshold {
        -1.0  // Overbought - Sell
    } else if zscore < -threshold {
        1.0   // Oversold - Buy
    } else {
        0.0   // Hold
    };
    
    let entry_signal = zscore.abs() > threshold;
    let exit_signal = zscore.abs() < threshold * 0.5;
    
    // Additional metrics
    let mut metrics = HashMap::new();
    metrics.insert("mean".to_string(), mean);
    metrics.insert("std".to_string(), std);
    metrics.insert("current".to_string(), current);
    metrics.insert("variance".to_string(), variance);
    
    MeanReversionResult {
        signal,
        zscore,
        entry_signal,
        exit_signal,
        metrics,
    }
}

/// Estimate Ornstein-Uhlenbeck process parameters
/// Returns (theta, mu, sigma, half_life)
pub fn estimate_ou_process(prices: &[f64]) -> OUParameters {
    if prices.len() < 3 {
        return OUParameters::default();
    }
    
    let n = prices.len();
    
    // Compute differences: dX = X[t] - X[t-1]
    let mut diffs = Vec::with_capacity(n - 1);
    let mut lags = Vec::with_capacity(n - 1);
    
    for i in 1..n {
        diffs.push(prices[i] - prices[i-1]);
        lags.push(prices[i-1]);
    }
    
    // Estimate mu as mean of prices
    let mu = prices.iter().sum::<f64>() / (n as f64);
    
    // Regression: dX = a + b * X[t-1] + epsilon
    let mean_lag = lags.iter().sum::<f64>() / lags.len() as f64;
    let mean_diff = diffs.iter().sum::<f64>() / diffs.len() as f64;
    
    let mut num = 0.0;
    let mut den = 0.0;
    
    for i in 0..lags.len() {
        let lag_dev = lags[i] - mean_lag;
        let diff_dev = diffs[i] - mean_diff;
        num += lag_dev * diff_dev;
        den += lag_dev * lag_dev;
    }
    
    let slope = if den.abs() > 1e-10 { num / den } else { 0.0 };
    let theta = -slope;
    
    // Estimate sigma from residuals
    let intercept = mean_diff - slope * mean_lag;
    let mut residuals_sq = 0.0;
    for i in 0..lags.len() {
        let predicted = intercept + slope * lags[i];
        let residual = diffs[i] - predicted;
        residuals_sq += residual * residual;
    }
    
    let sigma = (residuals_sq / (lags.len() as f64)).sqrt();
    
    // Half-life: ln(2) / theta
    let half_life = if theta > 1e-10 {
        (2.0_f64).ln() / theta
    } else {
        f64::INFINITY
    };
    
    OUParameters {
        theta,
        mu,
        sigma,
        half_life,
    }
}

/// Cointegration test using Engle-Granger method
pub fn cointegration_test(prices_x: &[f64], prices_y: &[f64]) -> CointegrationResult {
    if prices_x.len() != prices_y.len() || prices_x.len() < 10 {
        return CointegrationResult::default();
    }
    
    let n = prices_x.len();
    
    // Step 1: OLS regression y = a + b*x
    let mean_x = prices_x.iter().sum::<f64>() / n as f64;
    let mean_y = prices_y.iter().sum::<f64>() / n as f64;
    
    let mut num = 0.0;
    let mut den = 0.0;
    
    for i in 0..n {
        let x_dev = prices_x[i] - mean_x;
        let y_dev = prices_y[i] - mean_y;
        num += x_dev * y_dev;
        den += x_dev * x_dev;
    }
    
    let beta = if den.abs() > 1e-10 { num / den } else { 0.0 };
    let alpha = mean_y - beta * mean_x;
    
    // Step 2: Test residuals for stationarity
    let residuals: Vec<f64> = (0..n)
        .map(|i| prices_y[i] - (alpha + beta * prices_x[i]))
        .collect();
    
    // ADF test approximation using simple autocorrelation
    let mut diffs = Vec::with_capacity(n - 1);
    let mut lags = Vec::with_capacity(n - 1);
    
    for i in 1..n {
        diffs.push(residuals[i] - residuals[i-1]);
        lags.push(residuals[i-1]);
    }
    
    let mean_lag = lags.iter().sum::<f64>() / lags.len() as f64;
    let mean_diff = diffs.iter().sum::<f64>() / diffs.len() as f64;
    
    let mut num = 0.0;
    let mut den = 0.0;
    
    for i in 0..lags.len() {
        num += (lags[i] - mean_lag) * (diffs[i] - mean_diff);
        den += (lags[i] - mean_lag).powi(2);
    }
    
    let adf_stat = if den > 1e-10 { num / den } else { 0.0 };
    
    // Critical values (5% significance)
    let critical_value = -2.86; // Approximate for n > 50
    let is_cointegrated = adf_stat < critical_value;
    let p_value = if is_cointegrated { 0.03 } else { 0.15 }; // Approximate
    
    CointegrationResult {
        statistic: adf_stat,
        p_value,
        is_cointegrated,
        hedge_ratio: beta,
        intercept: alpha,
    }
}

#[derive(Debug, Clone, Default)]
pub struct MeanReversionResult {
    pub signal: f64,
    pub zscore: f64,
    pub entry_signal: bool,
    pub exit_signal: bool,
    pub metrics: HashMap<String, f64>,
}

#[derive(Debug, Clone, Default)]
pub struct OUParameters {
    pub theta: f64,
    pub mu: f64,
    pub sigma: f64,
    pub half_life: f64,
}

#[derive(Debug, Clone, Default)]
pub struct CointegrationResult {
    pub statistic: f64,
    pub p_value: f64,
    pub is_cointegrated: bool,
    pub hedge_ratio: f64,
    pub intercept: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_mean_reversion() {
        let prices = vec![100.0, 101.0, 99.0, 102.0, 98.0, 103.0];
        let result = calculate_mean_reversion(&prices, 5, 1.5);
        assert!(result.zscore.abs() < 3.0);
    }
    
    #[test]
    fn test_ou_process() {
        let prices = vec![100.0, 100.5, 100.2, 100.8, 100.3];
        let params = estimate_ou_process(&prices);
        assert!(params.mu > 99.0 && params.mu < 102.0);
        assert!(params.sigma > 0.0);
    }
}
