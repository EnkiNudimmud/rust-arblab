/// Python bindings for high-performance parallel statistical analyzer
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use rust_core::statistical_analyzer::{
    StatisticalAnalyzer, CointegrationResult, BasketResult, VolatilityRanking,
};

/// Python wrapper for CointegrationResult
#[pyclass(name = "CointegrationResult")]
#[derive(Clone)]
pub struct PyCointegrationResult {
    #[pyo3(get)]
    pub symbol1: String,
    #[pyo3(get)]
    pub symbol2: String,
    #[pyo3(get)]
    pub p_value: f64,
    #[pyo3(get)]
    pub correlation: f64,
    #[pyo3(get)]
    pub test_statistic: f64,
}

#[pymethods]
impl PyCointegrationResult {
    fn __repr__(&self) -> String {
        format!(
            "CointegrationResult({}-{}, p={:.4}, corr={:.3})",
            self.symbol1, self.symbol2, self.p_value, self.correlation
        )
    }
}

/// Python wrapper for BasketResult
#[pyclass(name = "BasketResult")]
#[derive(Clone)]
pub struct PyBasketResult {
    #[pyo3(get)]
    pub symbols: Vec<String>,
    #[pyo3(get)]
    pub avg_correlation: f64,
    #[pyo3(get)]
    pub avg_volatility: f64,
}

#[pymethods]
impl PyBasketResult {
    fn __repr__(&self) -> String {
        format!(
            "BasketResult(n={}, corr={:.3}, vol={:.3})",
            self.symbols.len(),
            self.avg_correlation,
            self.avg_volatility
        )
    }
}

/// Python wrapper for VolatilityRanking
#[pyclass(name = "VolatilityRanking")]
#[derive(Clone)]
pub struct PyVolatilityRanking {
    #[pyo3(get)]
    pub symbol: String,
    #[pyo3(get)]
    pub volatility: f64,
    #[pyo3(get)]
    pub rank: usize,
}

#[pymethods]
impl PyVolatilityRanking {
    fn __repr__(&self) -> String {
        format!(
            "VolatilityRanking(#{} {}: {:.4})",
            self.rank, self.symbol, self.volatility
        )
    }
}

/// Progress tracker that can be called from Rust
#[pyclass(name = "ProgressTracker")]
pub struct PyProgressTracker {
    callback: Arc<Mutex<Option<PyObject>>>,
}

#[pymethods]
impl PyProgressTracker {
    #[new]
    fn new() -> Self {
        Self {
            callback: Arc::new(Mutex::new(None)),
        }
    }

    fn set_callback(&mut self, callback: PyObject) {
        *self.callback.lock().unwrap() = Some(callback);
    }

    fn update(&self, current: usize, total: usize, eta_seconds: f64) -> PyResult<()> {
        if let Some(ref cb) = *self.callback.lock().unwrap() {
            Python::with_gil(|py| {
                let _ = cb.call1(py, (current, total, eta_seconds));
            });
        }
        Ok(())
    }
}

/// Main Python-facing statistical analyzer
#[pyclass(name = "StatisticalAnalyzer")]
pub struct PyStatisticalAnalyzer {
    analyzer: StatisticalAnalyzer,
}

#[pymethods]
impl PyStatisticalAnalyzer {
    #[new]
    #[pyo3(signature = (min_correlation=0.7, min_data_points=200))]
    fn new(min_correlation: f64, min_data_points: usize) -> Self {
        Self {
            analyzer: StatisticalAnalyzer::new(min_correlation, min_data_points),
        }
    }

    /// Filter symbols with sufficient data
    /// 
    /// Args:
    ///     data: Dict[str, List[float]] - Symbol to price series mapping
    /// 
    /// Returns:
    ///     Dict[str, List[float]] - Filtered data
    fn filter_valid_symbols(&self, data: &PyDict) -> PyResult<PyObject> {
        let mut rust_data: HashMap<String, Vec<f64>> = HashMap::new();

        for (key, value) in data.iter() {
            let symbol: String = key.extract()?;
            let values: Vec<f64> = value.extract()?;
            rust_data.insert(symbol, values);
        }

        let filtered = self.analyzer.filter_valid_symbols(&rust_data);

        Python::with_gil(|py| {
            let result = PyDict::new_bound(py);
            for (symbol, values) in filtered {
                result.set_item(symbol, values)?;
            }
            Ok(result.into())
        })
    }

    /// Compute correlation matrix
    /// 
    /// Args:
    ///     data: Dict[str, List[float]] - Symbol to price series mapping
    /// 
    /// Returns:
    ///     Tuple[List[str], List[List[float]]] - (symbols, correlation_matrix)
    fn compute_correlation_matrix(&self, data: &PyDict) -> PyResult<PyObject> {
        let mut rust_data: HashMap<String, Vec<f64>> = HashMap::new();

        for (key, value) in data.iter() {
            let symbol: String = key.extract()?;
            let values: Vec<f64> = value.extract()?;
            rust_data.insert(symbol, values);
        }

        let (symbols, corr_matrix) = self
            .analyzer
            .compute_correlation_matrix(&rust_data)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;

        Python::with_gil(|py| {
            let n = corr_matrix.shape()[0];
            let matrix_list = PyList::empty_bound(py);

            for i in 0..n {
                let row = PyList::empty_bound(py);
                for j in 0..n {
                    row.append(corr_matrix[[i, j]])?;
                }
                matrix_list.append(row)?;
            }

            Ok((symbols, matrix_list).into_py(py))
        })
    }

    /// Find best cointegrated pairs with parallel processing
    /// 
    /// Args:
    ///     data: Dict[str, List[float]] - Symbol to price series mapping
    ///     correlation_matrix: List[List[float]] - Pre-computed correlation matrix
    ///     symbols: List[str] - Symbol names in order
    ///     progress_callback: Optional[Callable[[int, int, float], None]] - Progress updates
    /// 
    /// Returns:
    ///     List[CointegrationResult] - Found cointegrated pairs
    #[pyo3(signature = (data, correlation_matrix, symbols, progress_callback=None))]
    fn find_cointegrated_pairs(
        &self,
        data: &PyDict,
        correlation_matrix: &PyList,
        symbols: Vec<String>,
        progress_callback: Option<PyObject>,
    ) -> PyResult<Vec<PyCointegrationResult>> {
        // Convert Python data to Rust
        let mut rust_data: HashMap<String, Vec<f64>> = HashMap::new();
        for (key, value) in data.iter() {
            let symbol: String = key.extract()?;
            let values: Vec<f64> = value.extract()?;
            rust_data.insert(symbol, values);
        }

        // Convert correlation matrix
        let n = correlation_matrix.len();
        let mut corr_array = ndarray::Array2::<f64>::zeros((n, n));
        for (i, row) in correlation_matrix.iter().enumerate() {
            let row_list: &PyList = row.extract()?;
            for (j, val) in row_list.iter().enumerate() {
                corr_array[[i, j]] = val.extract()?;
            }
        }

        // Create progress callback wrapper
        let results = if let Some(cb) = progress_callback {
            let cb_clone = cb.clone();
            self.analyzer.find_cointegrated_pairs(
                &rust_data,
                &corr_array,
                &symbols,
                Some(move |current, total, eta| {
                    Python::with_gil(|py| {
                        let _ = cb_clone.call1(py, (current, total, eta));
                    });
                }),
            )
        } else {
            self.analyzer
                .find_cointegrated_pairs::<fn(usize, usize, f64)>(&rust_data, &corr_array, &symbols, None)
        };

        // Convert results to Python
        Ok(results
            .into_iter()
            .map(|r| PyCointegrationResult {
                symbol1: r.symbol1,
                symbol2: r.symbol2,
                p_value: r.p_value,
                correlation: r.correlation,
                test_statistic: r.test_statistic,
            })
            .collect())
    }

    /// Build optimal basket with parallel correlation analysis
    /// 
    /// Args:
    ///     correlation_matrix: List[List[float]] - Pre-computed correlation matrix
    ///     volatility: List[float] - Volatility for each symbol
    ///     symbols: List[str] - Symbol names in order
    ///     max_assets: int - Maximum basket size
    ///     progress_callback: Optional[Callable[[int, int, float], None]] - Progress updates
    /// 
    /// Returns:
    ///     BasketResult - Optimal basket configuration
    #[pyo3(signature = (correlation_matrix, volatility, symbols, max_assets, progress_callback=None))]
    fn build_optimal_basket(
        &self,
        correlation_matrix: &PyList,
        volatility: Vec<f64>,
        symbols: Vec<String>,
        max_assets: usize,
        progress_callback: Option<PyObject>,
    ) -> PyResult<PyBasketResult> {
        // Convert correlation matrix
        let n = correlation_matrix.len();
        let mut corr_array = ndarray::Array2::<f64>::zeros((n, n));
        for (i, row) in correlation_matrix.iter().enumerate() {
            let row_list: &PyList = row.extract()?;
            for (j, val) in row_list.iter().enumerate() {
                corr_array[[i, j]] = val.extract()?;
            }
        }

        // Build basket with optional progress callback
        let result = if let Some(cb) = progress_callback {
            let cb_clone = cb.clone();
            self.analyzer.build_optimal_basket(
                &corr_array,
                &volatility,
                &symbols,
                max_assets,
                Some(move |current, total, eta| {
                    Python::with_gil(|py| {
                        let _ = cb_clone.call1(py, (current, total, eta));
                    });
                }),
            )
        } else {
            self.analyzer.build_optimal_basket::<fn(usize, usize, f64)>(
                &corr_array,
                &volatility,
                &symbols,
                max_assets,
                None,
            )
        };

        Ok(PyBasketResult {
            symbols: result.symbols,
            avg_correlation: result.avg_correlation,
            avg_volatility: result.avg_volatility,
        })
    }

    /// Compute volatility rankings in parallel
    /// 
    /// Args:
    ///     data: Dict[str, List[float]] - Symbol to price series mapping
    /// 
    /// Returns:
    ///     List[VolatilityRanking] - Rankings sorted by volatility (descending)
    fn compute_volatility_rankings(&self, data: &PyDict) -> PyResult<Vec<PyVolatilityRanking>> {
        let mut rust_data: HashMap<String, Vec<f64>> = HashMap::new();

        for (key, value) in data.iter() {
            let symbol: String = key.extract()?;
            let values: Vec<f64> = value.extract()?;
            rust_data.insert(symbol, values);
        }

        let rankings = self.analyzer.compute_volatility_rankings(&rust_data);

        Ok(rankings
            .into_iter()
            .map(|r| PyVolatilityRanking {
                symbol: r.symbol,
                volatility: r.volatility,
                rank: r.rank,
            })
            .collect())
    }

    /// Find all suitable assets (combined approach)
    /// 
    /// Args:
    ///     data: Dict[str, List[float]] - Symbol to price series mapping
    ///     correlation_matrix: List[List[float]] - Pre-computed correlation matrix
    ///     symbols: List[str] - Symbol names in order
    ///     max_assets: int - Maximum number of assets to return
    ///     progress_callback: Optional[Callable[[int, int, float], None]] - Progress updates
    /// 
    /// Returns:
    ///     List[str] - Suitable asset symbols
    #[pyo3(signature = (data, correlation_matrix, symbols, max_assets, progress_callback=None))]
    fn find_all_suitable(
        &self,
        data: &PyDict,
        correlation_matrix: &PyList,
        symbols: Vec<String>,
        max_assets: usize,
        progress_callback: Option<PyObject>,
    ) -> PyResult<Vec<String>> {
        // Convert Python data to Rust
        let mut rust_data: HashMap<String, Vec<f64>> = HashMap::new();
        for (key, value) in data.iter() {
            let symbol: String = key.extract()?;
            let values: Vec<f64> = value.extract()?;
            rust_data.insert(symbol, values);
        }

        // Convert correlation matrix
        let n = correlation_matrix.len();
        let mut corr_array = ndarray::Array2::<f64>::zeros((n, n));
        for (i, row) in correlation_matrix.iter().enumerate() {
            let row_list: &PyList = row.extract()?;
            for (j, val) in row_list.iter().enumerate() {
                corr_array[[i, j]] = val.extract()?;
            }
        }

        // Find suitable assets
        let results = if let Some(cb) = progress_callback {
            let cb_clone = cb.clone();
            self.analyzer.find_all_suitable(
                &rust_data,
                &corr_array,
                &symbols,
                max_assets,
                Some(move |current, total, eta| {
                    Python::with_gil(|py| {
                        let _ = cb_clone.call1(py, (current, total, eta));
                    });
                }),
            )
        } else {
            self.analyzer
                .find_all_suitable::<fn(usize, usize, f64)>(&rust_data, &corr_array, &symbols, max_assets, None)
        };

        Ok(results)
    }

    fn __repr__(&self) -> String {
        format!(
            "StatisticalAnalyzer(min_corr={:.2}, min_points={})",
            self.analyzer.min_correlation, self.analyzer.min_data_points
        )
    }
}

/// Register the module
pub fn register_module<'py>(py: Python<'py>, parent_module: &Bound<'py, PyModule>) -> PyResult<()> {
    let stats_module = PyModule::new_bound(py, "statistical_analyzer")?;
    
    stats_module.add_class::<PyStatisticalAnalyzer>()?;
    stats_module.add_class::<PyCointegrationResult>()?;
    stats_module.add_class::<PyBasketResult>()?;
    stats_module.add_class::<PyVolatilityRanking>()?;
    stats_module.add_class::<PyProgressTracker>()?;
    
    parent_module.add_submodule(&stats_module)?;
    Ok(())
}
