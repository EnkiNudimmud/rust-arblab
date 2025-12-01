use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rust_core::signature_portfolio;

/// Compute path signature up to specified level
#[pyfunction]
fn compute_signature(py: Python, path: Vec<Vec<f64>>, level: usize) -> PyResult<Vec<f64>> {
    py.allow_threads(|| {
        Ok(signature_portfolio::compute_signature(&path, level))
    })
}

/// Compute log-signature
#[pyfunction]
fn compute_log_signature(py: Python, path: Vec<Vec<f64>>, level: usize) -> PyResult<Vec<f64>> {
    py.allow_threads(|| {
        Ok(signature_portfolio::compute_log_signature(&path, level))
    })
}

/// Compute expected signature from multiple paths
#[pyfunction]
fn expected_signature(py: Python, paths: Vec<Vec<Vec<f64>>>, level: usize) -> PyResult<Vec<f64>> {
    py.allow_threads(|| {
        Ok(signature_portfolio::expected_signature(&paths, level))
    })
}

/// Compute signature kernel between two paths
#[pyfunction]
fn signature_kernel(py: Python, path1: Vec<Vec<f64>>, path2: Vec<Vec<f64>>, level: usize) -> PyResult<f64> {
    py.allow_threads(|| {
        Ok(signature_portfolio::signature_kernel(&path1, &path2, level))
    })
}

/// Compute signature-based covariance matrix
#[pyfunction]
fn signature_covariance(py: Python, returns: Vec<Vec<f64>>, level: usize) -> PyResult<Vec<Vec<f64>>> {
    py.allow_threads(|| {
        Ok(signature_portfolio::signature_covariance(&returns, level))
    })
}

/// Compute optimal portfolio weights using signature methods
#[pyfunction]
fn signature_portfolio_weights(
    py: Python,
    returns: Vec<Vec<f64>>,
    level: usize,
    risk_aversion: f64,
    allow_short: bool
) -> PyResult<Vec<f64>> {
    py.allow_threads(|| {
        Ok(signature_portfolio::signature_portfolio_weights(
            &returns,
            level,
            risk_aversion,
            allow_short
        ))
    })
}

/// Compute rank-based portfolio weights
#[pyfunction]
fn rank_based_portfolio(py: Python, returns: Vec<f64>, generating_fn: String) -> PyResult<Vec<f64>> {
    py.allow_threads(|| {
        Ok(signature_portfolio::rank_based_portfolio(&returns, &generating_fn))
    })
}

/// Compute portfolio performance metrics
#[pyfunction]
fn portfolio_metrics(py: Python, returns: Vec<f64>) -> PyResult<PyObject> {
    let (total_return, sharpe_ratio, max_drawdown, volatility) = py.allow_threads(|| {
        signature_portfolio::portfolio_metrics(&returns)
    });
    
    let dict = PyDict::new_bound(py);
    dict.set_item("total_return", total_return)?;
    dict.set_item("sharpe_ratio", sharpe_ratio)?;
    dict.set_item("max_drawdown", max_drawdown)?;
    dict.set_item("volatility", volatility)?;
    
    Ok(dict.into())
}

/// Optimal stopping using signature features
#[pyfunction]
fn signature_optimal_stopping(
    py: Python,
    path: Vec<Vec<f64>>,
    level: usize,
    threshold: f64,
    window: usize
) -> PyResult<usize> {
    py.allow_threads(|| {
        Ok(signature_portfolio::signature_optimal_stopping(
            &path,
            level,
            threshold,
            window
        ))
    })
}

/// Randomized signature features
#[pyfunction]
fn randomized_signature_features(
    py: Python,
    path: Vec<Vec<f64>>,
    level: usize,
    n_features: usize,
    seed: u64
) -> PyResult<Vec<f64>> {
    py.allow_threads(|| {
        Ok(signature_portfolio::randomized_signature_features(
            &path,
            level,
            n_features,
            seed
        ))
    })
}

/// Register signature portfolio functions with Python module
pub fn register_signature_portfolio(py: Python, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    parent.add_function(wrap_pyfunction!(compute_signature, parent)?)?;
    parent.add_function(wrap_pyfunction!(compute_log_signature, parent)?)?;
    parent.add_function(wrap_pyfunction!(expected_signature, parent)?)?;
    parent.add_function(wrap_pyfunction!(signature_kernel, parent)?)?;
    parent.add_function(wrap_pyfunction!(signature_covariance, parent)?)?;
    parent.add_function(wrap_pyfunction!(signature_portfolio_weights, parent)?)?;
    parent.add_function(wrap_pyfunction!(rank_based_portfolio, parent)?)?;
    parent.add_function(wrap_pyfunction!(portfolio_metrics, parent)?)?;
    parent.add_function(wrap_pyfunction!(signature_optimal_stopping, parent)?)?;
    parent.add_function(wrap_pyfunction!(randomized_signature_features, parent)?)?;
    
    Ok(())
}
