use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, PyReadonlyArray2};
use ndarray::{Array1, Array2, s};

/// Compute correlation matrix from returns matrix (Rust implementation for speed)
#[pyfunction]
fn compute_correlation_matrix(returns: PyReadonlyArray2<f64>) -> PyResult<Py<PyArray2<f64>>> {
    Python::with_gil(|py| {
        let returns_array = returns.as_array();
        let n_assets = returns_array.ncols();
        let n_periods = returns_array.nrows();
        
        // Compute means
        let means: Vec<f64> = (0..n_assets)
            .map(|i| {
                let col = returns_array.column(i);
                col.sum() / n_periods as f64
            })
            .collect();
        
        // Compute correlation matrix
        let mut corr = Array2::<f64>::zeros((n_assets, n_assets));
        
        for i in 0..n_assets {
            for j in i..n_assets {
                let col_i = returns_array.column(i);
                let col_j = returns_array.column(j);
                
                let mut cov = 0.0;
                let mut var_i = 0.0;
                let mut var_j = 0.0;
                
                for k in 0..n_periods {
                    let dev_i = col_i[k] - means[i];
                    let dev_j = col_j[k] - means[j];
                    cov += dev_i * dev_j;
                    var_i += dev_i * dev_i;
                    var_j += dev_j * dev_j;
                }
                
                let std_i = (var_i / n_periods as f64).sqrt();
                let std_j = (var_j / n_periods as f64).sqrt();
                
                let corr_val = if std_i > 1e-10 && std_j > 1e-10 {
                    (cov / n_periods as f64) / (std_i * std_j)
                } else {
                    if i == j { 1.0 } else { 0.0 }
                };
                
                corr[[i, j]] = corr_val;
                corr[[j, i]] = corr_val;
            }
        }
        
        Ok(PyArray2::from_array_bound(py, &corr).unbind())
    })
}

/// Compute simple covariance matrix (PCA moved to Python with scipy)
#[pyfunction]
fn compute_covariance_matrix(returns: PyReadonlyArray2<f64>) -> PyResult<Py<PyArray2<f64>>> {
    Python::with_gil(|py| {
        let returns_array = returns.as_array();
        let n_assets = returns_array.ncols();
        let n_periods = returns_array.nrows();
        
        // Compute means
        let means: Vec<f64> = (0..n_assets)
            .map(|i| {
                let col = returns_array.column(i);
                col.sum() / n_periods as f64
            })
            .collect();
        
        // Compute covariance matrix
        let mut cov = Array2::<f64>::zeros((n_assets, n_assets));
        
        for i in 0..n_assets {
            for j in i..n_assets {
                let col_i = returns_array.column(i);
                let col_j = returns_array.column(j);
                
                let mut covariance = 0.0;
                
                for k in 0..n_periods {
                    let dev_i = col_i[k] - means[i];
                    let dev_j = col_j[k] - means[j];
                    covariance += dev_i * dev_j;
                }
                
                covariance /= n_periods as f64 - 1.0;
                
                cov[[i, j]] = covariance;
                cov[[j, i]] = covariance;
            }
        }
        
        Ok(PyArray2::from_array_bound(py, &cov).unbind())
    })
}

/// Compute portfolio weights using mean-variance optimization (Rust for speed)
#[pyfunction]
fn optimize_portfolio_weights(
    returns: PyReadonlyArray2<f64>,
    target_return: Option<f64>,
    risk_aversion: Option<f64>
) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let returns_array = returns.as_array();
        let n_assets = returns_array.ncols();
        let n_periods = returns_array.nrows();
        
        // Compute mean returns
        let means: Vec<f64> = (0..n_assets)
            .map(|i| returns_array.column(i).mean().unwrap_or(0.0))
            .collect();
        
        // Compute covariance matrix
        let mut cov = Array2::<f64>::zeros((n_assets, n_assets));
        for i in 0..n_assets {
            for j in i..n_assets {
                let col_i = returns_array.column(i);
                let col_j = returns_array.column(j);
                
                let mut covariance = 0.0;
                for k in 0..n_periods {
                    let dev_i = col_i[k] - means[i];
                    let dev_j = col_j[k] - means[j];
                    covariance += dev_i * dev_j;
                }
                covariance /= n_periods as f64 - 1.0;
                
                cov[[i, j]] = covariance;
                cov[[j, i]] = covariance;
            }
        }
        
        // Simple mean-variance optimization (equal weight for now, could add optimization library)
        // For production, would use quadprog or similar
        let risk_av = risk_aversion.unwrap_or(1.0);
        
        // Inverse covariance (simplified - use identity if singular)
        let mut weights = Array1::<f64>::from_elem(n_assets, 1.0 / n_assets as f64);
        
        // Adjust based on returns if target specified
        if let Some(target_ret) = target_return {
            for i in 0..n_assets {
                if means[i] > target_ret {
                    weights[i] *= 1.0 + (means[i] - target_ret) / risk_av;
                } else {
                    weights[i] *= 1.0 - (target_ret - means[i]) / risk_av;
                }
            }
            
            // Normalize
            let sum: f64 = weights.sum();
            if sum > 1e-10 {
                weights /= sum;
            }
        }
        
        Ok(PyArray1::from_array_bound(py, &weights).unbind())
    })
}

/// Compute rolling statistics efficiently in Rust
#[pyfunction]
fn compute_rolling_stats(
    data: PyReadonlyArray2<f64>,
    window: usize
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray2<f64>>)> {
    Python::with_gil(|py| {
        let data_array = data.as_array();
        let n_rows = data_array.nrows();
        let n_cols = data_array.ncols();
        
        if n_rows < window {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Window size larger than data"
            ));
        }
        
        let n_windows = n_rows - window + 1;
        let mut means = Array2::<f64>::zeros((n_windows, n_cols));
        let mut stds = Array2::<f64>::zeros((n_windows, n_cols));
        
        for col in 0..n_cols {
            for i in 0..n_windows {
                let slice = data_array.slice(s![i..i+window, col]);
                let mean = slice.mean().unwrap_or(0.0);
                let variance = slice.iter()
                    .map(|&x| (x - mean).powi(2))
                    .sum::<f64>() / window as f64;
                
                means[[i, col]] = mean;
                stds[[i, col]] = variance.sqrt();
            }
        }
        
        Ok((
            PyArray2::from_array_bound(py, &means).unbind(),
            PyArray2::from_array_bound(py, &stds).unbind()
        ))
    })
}

/// Compute z-scores for mean reversion (vectorized in Rust)
#[pyfunction]
fn compute_zscores(
    prices: PyReadonlyArray2<f64>,
    window: usize
) -> PyResult<Py<PyArray2<f64>>> {
    Python::with_gil(|py| {
        let prices_array = prices.as_array();
        let n_rows = prices_array.nrows();
        let n_cols = prices_array.ncols();
        
        let mut zscores = Array2::<f64>::zeros((n_rows, n_cols));
        
        for col in 0..n_cols {
            for i in window..n_rows {
                let slice = prices_array.slice(s![i-window..i, col]);
                let mean = slice.mean().unwrap_or(0.0);
                let std = slice.std(0.0);
                
                if std > 1e-10 {
                    zscores[[i, col]] = (prices_array[[i, col]] - mean) / std;
                }
            }
        }
        
        Ok(PyArray2::from_array_bound(py, &zscores).unbind())
    })
}

/// Fast batch correlation computation for pairs trading
#[pyfunction]
fn compute_pairwise_correlations(
    returns: PyReadonlyArray2<f64>,
    window: usize
) -> PyResult<Py<PyArray2<f64>>> {
    Python::with_gil(|py| {
        let returns_array = returns.as_array();
        let n_assets = returns_array.ncols();
        let n_periods = returns_array.nrows();
        
        if n_periods < window {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Not enough periods for window"
            ));
        }
        
        // Use most recent window
        let recent = returns_array.slice(s![n_periods-window..n_periods, ..]);
        
        // Compute correlations
        let mut corr = Array2::<f64>::zeros((n_assets, n_assets));
        
        for i in 0..n_assets {
            for j in i..n_assets {
                let col_i = recent.column(i);
                let col_j = recent.column(j);
                
                let mean_i = col_i.mean().unwrap_or(0.0);
                let mean_j = col_j.mean().unwrap_or(0.0);
                
                let mut cov = 0.0;
                let mut var_i = 0.0;
                let mut var_j = 0.0;
                
                for k in 0..window {
                    let dev_i = col_i[k] - mean_i;
                    let dev_j = col_j[k] - mean_j;
                    cov += dev_i * dev_j;
                    var_i += dev_i * dev_i;
                    var_j += dev_j * dev_j;
                }
                
                let std_i = (var_i / window as f64).sqrt();
                let std_j = (var_j / window as f64).sqrt();
                
                let corr_val = if std_i > 1e-10 && std_j > 1e-10 {
                    (cov / window as f64) / (std_i * std_j)
                } else {
                    if i == j { 1.0 } else { 0.0 }
                };
                
                corr[[i, j]] = corr_val;
                corr[[j, i]] = corr_val;
            }
        }
        
        Ok(PyArray2::from_array_bound(py, &corr).unbind())
    })
}

pub fn register_analytics_module(py: Python, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let analytics = PyModule::new_bound(py, "analytics")?;
    
    analytics.add_function(wrap_pyfunction!(compute_correlation_matrix, &analytics)?)?;
    analytics.add_function(wrap_pyfunction!(compute_covariance_matrix, &analytics)?)?;
    analytics.add_function(wrap_pyfunction!(compute_rolling_stats, &analytics)?)?;
    analytics.add_function(wrap_pyfunction!(compute_zscores, &analytics)?)?;
    analytics.add_function(wrap_pyfunction!(compute_pairwise_correlations, &analytics)?)?;
    analytics.add_function(wrap_pyfunction!(optimize_portfolio_weights, &analytics)?)?;
    
    parent.add_submodule(&analytics)?;
    
    Ok(())
}
