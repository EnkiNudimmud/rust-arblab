/// Python bindings for analytics module - Clean, functional interface
use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use ndarray::Array2;

mod analytics;
use analytics::{
    matrix::{CorrelationMatrix, CovarianceMatrix},
    rolling::{RollingMean, RollingStd, RollingZScore, RollingCorrelation},
    statistics::{Mean, Variance, StdDev, Skewness, Kurtosis},
    traits::{MatrixOperation, RollingWindow, StatisticalMetric},
    AnalyticsError,
};

/// Convert AnalyticsError to PyErr
impl From<AnalyticsError> for PyErr {
    fn from(err: AnalyticsError) -> PyErr {
        pyo3::exceptions::PyValueError::new_err(err.to_string())
    }
}

// ============================================================================
// Matrix Operations
// ============================================================================

/// Compute correlation matrix from returns data
/// 
/// Args:
///     returns: 2D array of shape (n_periods, n_assets) with return data
/// 
/// Returns:
///     2D array of shape (n_assets, n_assets) with correlations
#[pyfunction]
fn compute_correlation_matrix(returns: PyReadonlyArray2<f64>) -> PyResult<Py<PyArray2<f64>>> {
    Python::with_gil(|py| {
        let data = returns.as_array().to_owned();
        let result = CorrelationMatrix.compute(&data)?;
        Ok(PyArray2::from_array_bound(py, &result).unbind())
    })
}

/// Compute covariance matrix from returns data
/// 
/// Args:
///     returns: 2D array of shape (n_periods, n_assets)
///     unbiased: If True, use n-1 denominator (default: True)
/// 
/// Returns:
///     2D array of shape (n_assets, n_assets) with covariances
#[pyfunction]
#[pyo3(signature = (returns, unbiased=true))]
fn compute_covariance_matrix(
    returns: PyReadonlyArray2<f64>,
    unbiased: bool,
) -> PyResult<Py<PyArray2<f64>>> {
    Python::with_gil(|py| {
        let data = returns.as_array().to_owned();
        let cov_calc = if unbiased {
            CovarianceMatrix::unbiased()
        } else {
            CovarianceMatrix::biased()
        };
        let result = cov_calc.compute(&data)?;
        Ok(PyArray2::from_array_bound(py, &result).unbind())
    })
}

// ============================================================================
// Rolling Window Calculations
// ============================================================================

/// Compute rolling mean
/// 
/// Args:
///     data: 1D array of values
///     window: Window size for rolling calculation
/// 
/// Returns:
///     1D array with rolling means (NaN for first window-1 values)
#[pyfunction]
fn compute_rolling_mean(
    data: PyReadonlyArray1<f64>,
    window: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let arr = data.as_array().to_owned();
        let result = RollingMean.compute_rolling(&arr, window)?;
        Ok(PyArray1::from_array_bound(py, &result).unbind())
    })
}

/// Compute rolling standard deviation
/// 
/// Args:
///     data: 1D array of values
///     window: Window size for rolling calculation
///     ddof: Degrees of freedom (0=biased, 1=unbiased, default: 1)
/// 
/// Returns:
///     1D array with rolling standard deviations
#[pyfunction]
#[pyo3(signature = (data, window, ddof=1))]
fn compute_rolling_std(
    data: PyReadonlyArray1<f64>,
    window: usize,
    ddof: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let arr = data.as_array().to_owned();
        let std_calc = RollingStd::new(ddof);
        let result = std_calc.compute_rolling(&arr, window)?;
        Ok(PyArray1::from_array_bound(py, &result).unbind())
    })
}

/// Compute rolling z-scores
/// 
/// Args:
///     data: 1D array of values
///     window: Window size for rolling calculation
/// 
/// Returns:
///     1D array with rolling z-scores
#[pyfunction]
fn compute_rolling_zscores(
    data: PyReadonlyArray1<f64>,
    window: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let arr = data.as_array().to_owned();
        let zscore_calc = RollingZScore::new(window);
        let result = zscore_calc.compute(&arr)?;
        Ok(PyArray1::from_array_bound(py, &result).unbind())
    })
}

/// Compute rolling correlation between two series
/// 
/// Args:
///     x: First 1D array
///     y: Second 1D array (must have same length as x)
///     window: Window size for rolling calculation
/// 
/// Returns:
///     1D array with rolling correlations
#[pyfunction]
fn compute_rolling_correlation(
    x: PyReadonlyArray1<f64>,
    y: PyReadonlyArray1<f64>,
    window: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    Python::with_gil(|py| {
        let arr_x = x.as_array().to_owned();
        let arr_y = y.as_array().to_owned();
        let corr_calc = RollingCorrelation::new(window);
        let result = corr_calc.compute(&arr_x, &arr_y)?;
        Ok(PyArray1::from_array_bound(py, &result).unbind())
    })
}

// ============================================================================
// Statistical Metrics
// ============================================================================

/// Compute mean of data
#[pyfunction]
fn compute_mean(data: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let arr = data.as_array();
    let vec: Vec<f64> = arr.iter().copied().collect();
    Ok(Mean.compute(&vec)?)
}

/// Compute variance of data
/// 
/// Args:
///     data: 1D array of values
///     ddof: Degrees of freedom (0=biased, 1=unbiased, default: 1)
#[pyfunction]
#[pyo3(signature = (data, ddof=1))]
fn compute_variance(data: PyReadonlyArray1<f64>, ddof: usize) -> PyResult<f64> {
    let arr = data.as_array();
    let vec: Vec<f64> = arr.iter().copied().collect();
    let var_calc = if ddof == 1 {
        Variance::unbiased()
    } else {
        Variance::biased()
    };
    Ok(var_calc.compute(&vec)?)
}

/// Compute standard deviation of data
/// 
/// Args:
///     data: 1D array of values
///     ddof: Degrees of freedom (0=biased, 1=unbiased, default: 1)
#[pyfunction]
#[pyo3(signature = (data, ddof=1))]
fn compute_std(data: PyReadonlyArray1<f64>, ddof: usize) -> PyResult<f64> {
    let arr = data.as_array();
    let vec: Vec<f64> = arr.iter().copied().collect();
    let std_calc = if ddof == 1 {
        StdDev::unbiased()
    } else {
        StdDev::biased()
    };
    Ok(std_calc.compute(&vec)?)
}

/// Compute skewness of data
#[pyfunction]
fn compute_skewness(data: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let arr = data.as_array();
    let vec: Vec<f64> = arr.iter().copied().collect();
    Ok(Skewness.compute(&vec)?)
}

/// Compute kurtosis of data
/// 
/// Args:
///     data: 1D array of values
///     excess: If True, return excess kurtosis (normal=0), else raw (normal=3)
#[pyfunction]
#[pyo3(signature = (data, excess=true))]
fn compute_kurtosis(data: PyReadonlyArray1<f64>, excess: bool) -> PyResult<f64> {
    let arr = data.as_array();
    let vec: Vec<f64> = arr.iter().copied().collect();
    let kurt_calc = if excess {
        Kurtosis::excess()
    } else {
        Kurtosis::raw()
    };
    Ok(kurt_calc.compute(&vec)?)
}

// ============================================================================
// Batch Operations (for efficiency with large datasets)
// ============================================================================

/// Compute pairwise correlations for multiple pairs efficiently
/// 
/// Args:
///     returns: 2D array of shape (n_periods, n_assets)
///     window: Window size for rolling correlations
/// 
/// Returns:
///     3D array of shape (n_periods, n_assets, n_assets) with rolling correlation matrices
#[pyfunction]
fn compute_pairwise_rolling_correlations(
    returns: PyReadonlyArray2<f64>,
    window: usize,
) -> PyResult<Py<PyArray2<f64>>> {
    Python::with_gil(|py| {
        let data = returns.as_array();
        let (n_periods, n_assets) = data.dim();
        
        if n_periods < window {
            return Err(AnalyticsError::InvalidWindow {
                window,
                data_len: n_periods,
            }
            .into());
        }
        
        // Compute rolling correlations for each pair
        // Result matrix: each row is the correlation at that time
        let n_pairs = n_assets * (n_assets - 1) / 2;
        let mut result = Array2::zeros((n_periods - window + 1, n_pairs));
        
        let mut pair_idx = 0;
        for i in 0..n_assets {
            for j in (i + 1)..n_assets {
                let x = data.column(i).to_owned();
                let y = data.column(j).to_owned();
                let corr_calc = RollingCorrelation::new(window);
                let correlations = corr_calc.compute(&x, &y)?;
                
                // Extract non-NaN values
                for (t, &val) in correlations.iter().enumerate().skip(window - 1) {
                    result[[t - (window - 1), pair_idx]] = val;
                }
                pair_idx += 1;
            }
        }
        
        Ok(PyArray2::from_array_bound(py, &result).unbind())
    })
}

// ============================================================================
// Module Registration
// ============================================================================

#[pymodule]
pub fn register_analytics(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Matrix operations
    m.add_function(wrap_pyfunction!(compute_correlation_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(compute_covariance_matrix, m)?)?;
    
    // Rolling window calculations
    m.add_function(wrap_pyfunction!(compute_rolling_mean, m)?)?;
    m.add_function(wrap_pyfunction!(compute_rolling_std, m)?)?;
    m.add_function(wrap_pyfunction!(compute_rolling_zscores, m)?)?;
    m.add_function(wrap_pyfunction!(compute_rolling_correlation, m)?)?;
    
    // Statistical metrics
    m.add_function(wrap_pyfunction!(compute_mean, m)?)?;
    m.add_function(wrap_pyfunction!(compute_variance, m)?)?;
    m.add_function(wrap_pyfunction!(compute_std, m)?)?;
    m.add_function(wrap_pyfunction!(compute_skewness, m)?)?;
    m.add_function(wrap_pyfunction!(compute_kurtosis, m)?)?;
    
    // Batch operations
    m.add_function(wrap_pyfunction!(compute_pairwise_rolling_correlations, m)?)?;
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_module_creation() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let module = PyModule::new_bound(py, "analytics").unwrap();
            register_analytics(py, &module).unwrap();
            assert!(module.hasattr("compute_correlation_matrix").unwrap());
        });
    }
}
