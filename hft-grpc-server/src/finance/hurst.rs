//! Hurst Exponent Calculation

use crate::finance::{FinanceError, Result};

/// Calculate Hurst exponent using Rescaled Range (R/S) analysis
pub fn hurst_exponent(series: &[f64], max_lag: usize) -> Result<f64> {
    if series.len() < max_lag * 2 {
        return Err(FinanceError::InsufficientData(max_lag * 2));
    }
    
    let min_lag = 2;
    let mut lags = Vec::new();
    let mut rs_values = Vec::new();
    
    for lag in min_lag..=max_lag {
        let n_blocks = series.len() / lag;
        if n_blocks < 2 {
            continue;
        }
        
        let mut block_rs = Vec::new();
        
        for block_idx in 0..n_blocks {
            let start = block_idx * lag;
            let end = start + lag;
            let block = &series[start..end];
            
            let block_mean: f64 = block.iter().sum::<f64>() / lag as f64;
            let mut cum_sum = vec![0.0; lag];
            cum_sum[0] = block[0] - block_mean;
            
            for i in 1..lag {
                cum_sum[i] = cum_sum[i - 1] + (block[i] - block_mean);
            }
            
            let max_cum = cum_sum.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let min_cum = cum_sum.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let range = max_cum - min_cum;
            
            let variance: f64 = block.iter()
                .map(|&x| (x - block_mean).powi(2))
                .sum::<f64>() / lag as f64;
            let std = variance.sqrt();
            
            if std > 1e-10 && range > 1e-10 {
                block_rs.push(range / std);
            }
        }
        
        if !block_rs.is_empty() {
            lags.push(lag as f64);
            let mean_rs: f64 = block_rs.iter().sum::<f64>() / block_rs.len() as f64;
            rs_values.push(mean_rs);
        }
    }
    
    if lags.len() < 3 {
        return Err(FinanceError::NumericalError(
            "Insufficient lags for Hurst calculation".to_string()
        ));
    }
    
    // Linear regression: log(R/S) = log(c) + H * log(lag)
    let log_lags: Vec<f64> = lags.iter().map(|x| x.ln()).collect();
    let log_rs: Vec<f64> = rs_values.iter().map(|x| x.ln()).collect();
    
    let n = log_lags.len();
    let x_mean: f64 = log_lags.iter().sum::<f64>() / n as f64;
    let y_mean: f64 = log_rs.iter().sum::<f64>() / n as f64;
    
    let mut cov_xy = 0.0;
    let mut var_x = 0.0;
    
    for i in 0..n {
        let dx = log_lags[i] - x_mean;
        let dy = log_rs[i] - y_mean;
        cov_xy += dx * dy;
        var_x += dx * dx;
    }
    
    if var_x < 1e-10 {
        return Err(FinanceError::NumericalError(
            "Insufficient variance for Hurst calculation".to_string()
        ));
    }
    
    let hurst = cov_xy / var_x;
    
    Ok(hurst.max(0.0).min(1.0))
}

/// Alias for hurst_exponent
pub fn hurst_rs(series: &[f64], max_lag: Option<usize>) -> Result<f64> {
    let max_lag = max_lag.unwrap_or(20.min(series.len() / 4));
    hurst_exponent(series, max_lag)
}

/// Detrended Fluctuation Analysis
pub fn hurst_dfa(series: &[f64], min_window: usize, max_window: usize) -> Result<f64> {
    if series.len() < max_window * 2 {
        return Err(FinanceError::InsufficientData(max_window * 2));
    }
    
    let mean: f64 = series.iter().sum::<f64>() / series.len() as f64;
    let mut y = vec![0.0; series.len()];
    y[0] = series[0] - mean;
    for i in 1..series.len() {
        y[i] = y[i - 1] + (series[i] - mean);
    }
    
    let mut window_sizes = Vec::new();
    let mut fluctuations = Vec::new();
    
    let mut window_size = min_window;
    while window_size <= max_window {
        let n_windows = series.len() / window_size;
        if n_windows < 2 {
            break;
        }
        
        let mut window_vars = Vec::new();
        
        for w in 0..n_windows {
            let start = w * window_size;
            let end = start + window_size;
            let window = &y[start..end];
            
            let x: Vec<f64> = (0..window_size).map(|i| i as f64).collect();
            let x_mean = (window_size - 1) as f64 / 2.0;
            let y_mean: f64 = window.iter().sum::<f64>() / window_size as f64;
            
            let mut cov = 0.0;
            let mut var = 0.0;
            for i in 0..window_size {
                cov += (x[i] - x_mean) * (window[i] - y_mean);
                var += (x[i] - x_mean).powi(2);
            }
            
            let slope = cov / var;
            let intercept = y_mean - slope * x_mean;
            
            let mut mse = 0.0;
            for i in 0..window_size {
                let trend = intercept + slope * x[i];
                mse += (window[i] - trend).powi(2);
            }
            mse /= window_size as f64;
            
            window_vars.push(mse);
        }
        
        let avg_fluctuation: f64 = window_vars.iter().sum::<f64>() / n_windows as f64;
        
        window_sizes.push(window_size as f64);
        fluctuations.push(avg_fluctuation.sqrt());
        
        window_size = (window_size as f64 * 1.5) as usize;
    }
    
    if window_sizes.len() < 3 {
        return Err(FinanceError::NumericalError(
            "Insufficient windows for DFA".to_string()
        ));
    }
    
    let log_n: Vec<f64> = window_sizes.iter().map(|x| x.ln()).collect();
    let log_f: Vec<f64> = fluctuations.iter().map(|x| x.ln()).collect();
    
    let n = log_n.len();
    let x_mean: f64 = log_n.iter().sum::<f64>() / n as f64;
    let y_mean: f64 = log_f.iter().sum::<f64>() / n as f64;
    
    let mut cov = 0.0;
    let mut var = 0.0;
    
    for i in 0..n {
        let dx = log_n[i] - x_mean;
        let dy = log_f[i] - y_mean;
        cov += dx * dy;
        var += dx * dx;
    }
    
    if var < 1e-10 {
        return Err(FinanceError::NumericalError(
            "Insufficient variance for DFA".to_string()
        ));
    }
    
    let alpha = cov / var;
    
    Ok(alpha.max(0.0).min(1.0))
}
