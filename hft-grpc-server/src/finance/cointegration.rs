//! Cointegration Testing

use crate::finance::{FinanceError, Result};

/// Result from Engle-Granger cointegration test
#[derive(Debug, Clone)]
pub struct EngleGrangerResult {
    pub beta: f64,
    pub adf_statistic: f64,
    pub p_value: f64,
    pub is_cointegrated: bool,
    pub spread: Vec<f64>,
}

/// Engle-Granger two-step cointegration test
pub fn engle_granger_test(y: &[f64], x: &[f64], significance: f64) -> Result<EngleGrangerResult> {
    if y.len() != x.len() {
        return Err(FinanceError::InvalidParameters(
            "Series must have same length".to_string()
        ));
    }
    
    if y.len() < 20 {
        return Err(FinanceError::InsufficientData(20));
    }
    
    let n = y.len();
    
    // Step 1: OLS regression
    let x_mean: f64 = x.iter().sum::<f64>() / n as f64;
    let y_mean: f64 = y.iter().sum::<f64>() / n as f64;
    
    let mut cov_xy = 0.0;
    let mut var_x = 0.0;
    
    for i in 0..n {
        let dx = x[i] - x_mean;
        let dy = y[i] - y_mean;
        cov_xy += dx * dy;
        var_x += dx * dx;
    }
    
    if var_x < 1e-10 {
        return Err(FinanceError::NumericalError(
            "Insufficient variance in x".to_string()
        ));
    }
    
    let beta = cov_xy / var_x;
    
    // Calculate residuals (spread)
    let spread: Vec<f64> = y.iter()
        .zip(x.iter())
        .map(|(yi, xi)| yi - beta * xi)
        .collect();
    
    // Step 2: ADF test on residuals
    let (adf_statistic, p_value) = adf_test(&spread)?;
    
    let is_cointegrated = p_value < significance;
    
    Ok(EngleGrangerResult {
        beta,
        adf_statistic,
        p_value,
        is_cointegrated,
        spread,
    })
}

/// Augmented Dickey-Fuller test for unit root
fn adf_test(series: &[f64]) -> Result<(f64, f64)> {
    if series.len() < 10 {
        return Err(FinanceError::InsufficientData(10));
    }
    
    let n = series.len() - 1;
    
    // First differences
    let mut diff: Vec<f64> = Vec::with_capacity(n);
    for i in 1..series.len() {
        diff.push(series[i] - series[i - 1]);
    }
    
    let lagged = &series[..n];
    
    // OLS: Δy_t = α + β*y_{t-1} + ε
    let y_mean: f64 = diff.iter().sum::<f64>() / n as f64;
    let x_mean: f64 = lagged.iter().sum::<f64>() / n as f64;
    
    let mut cov_xy = 0.0;
    let mut var_x = 0.0;
    
    for i in 0..n {
        let dx = lagged[i] - x_mean;
        let dy = diff[i] - y_mean;
        cov_xy += dx * dy;
        var_x += dx * dx;
    }
    
    if var_x < 1e-10 {
        return Err(FinanceError::NumericalError(
            "Insufficient variance for ADF test".to_string()
        ));
    }
    
    let beta = cov_xy / var_x;
    let alpha = y_mean - beta * x_mean;
    
    // Calculate residuals
    let mut sse = 0.0;
    for i in 0..n {
        let residual = diff[i] - (alpha + beta * lagged[i]);
        sse += residual * residual;
    }
    
    let mse = sse / (n as f64 - 2.0);
    let se_beta = (mse / var_x).sqrt();
    
    // ADF statistic
    let adf_statistic = beta / se_beta;
    
    // Approximate p-value using MacKinnon critical values
    let p_value = adf_p_value(adf_statistic, n);
    
    Ok((adf_statistic, p_value))
}

/// Approximate p-value for ADF test
fn adf_p_value(statistic: f64, n: usize) -> f64 {
    let cv_5pct = -2.86 - 2.0 / n as f64;
    let cv_1pct = -3.43 - 4.0 / n as f64;
    
    if statistic > cv_5pct {
        0.10
    } else if statistic > cv_1pct {
        0.03
    } else {
        0.005
    }
}
