use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::path::PathBuf;
use rust_core::flat_file_processor::*;

/// Python wrapper for S3 configuration
#[pyclass]
#[derive(Clone)]
pub struct PyS3Config {
    #[pyo3(get, set)]
    pub access_key_id: String,
    #[pyo3(get, set)]
    pub secret_access_key: String,
    #[pyo3(get, set)]
    pub endpoint: String,
    #[pyo3(get, set)]
    pub bucket: String,
    #[pyo3(get, set)]
    pub region: Option<String>,
}

#[pymethods]
impl PyS3Config {
    #[new]
    fn new(
        access_key_id: String,
        secret_access_key: String,
        endpoint: String,
        bucket: String,
        region: Option<String>,
    ) -> Self {
        Self {
            access_key_id,
            secret_access_key,
            endpoint,
            bucket,
            region,
        }
    }
}

impl PyS3Config {
    fn to_rust(&self) -> S3Config {
        S3Config {
            access_key_id: self.access_key_id.clone(),
            secret_access_key: self.secret_access_key.clone(),
            endpoint: self.endpoint.clone(),
            bucket: self.bucket.clone(),
            region: self.region.clone(),
        }
    }
}

/// Download and process flat file from S3 using Rust backend
///
/// Automatically uses:
/// - Polars for files < 1 GB (fast in-memory processing)
/// - DataFusion for files > 1 GB (streaming, memory-efficient)
///
/// Args:
///     s3_config: S3 configuration (access keys, endpoint, bucket)
///     s3_key: S3 object key (path in bucket)
///     local_dir: Local directory to download file
///     start_date: Optional start date filter (YYYY-MM-DD)
///     end_date: Optional end date filter (YYYY-MM-DD)
///     symbols: Optional list of symbols to filter
///
/// Returns:
///     Dictionary with processing results:
///     {
///         'success': bool,
///         'rows': int,
///         'columns': int,
///         'file_size_bytes': int,
///         'processing_time_ms': int,
///         'engine_used': str ('Polars' or 'DataFusion'),
///         'message': str
///     }
#[pyfunction]
fn download_and_process_flat_file(
    py: Python,
    s3_config: &PyS3Config,
    s3_key: String,
    local_dir: String,
    start_date: Option<String>,
    end_date: Option<String>,
    symbols: Option<Vec<String>>,
) -> PyResult<PyObject> {
    let rust_config = s3_config.to_rust();
    let local_path = PathBuf::from(local_dir);

    // Run async operation in tokio runtime (convert error to String for Send)
    let result: Result<FlatFileResult, String> = py.allow_threads(|| {
        tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(async {
                download_and_process_s3(
                    &rust_config,
                    &s3_key,
                    &local_path,
                    start_date.as_deref(),
                    end_date.as_deref(),
                    symbols,
                )
                .await
                .map_err(|e| e.to_string())
            })
    });

    match result {
        Ok(result) => {
            let dict = PyDict::new_bound(py);
            dict.set_item("success", result.success)?;
            dict.set_item("rows", result.rows)?;
            dict.set_item("columns", result.columns)?;
            dict.set_item("file_size_bytes", result.file_size_bytes)?;
            dict.set_item("processing_time_ms", result.processing_time_ms)?;
            dict.set_item("engine_used", result.engine_used)?;
            dict.set_item("message", result.message)?;
            Ok(dict.into())
        }
        Err(e) => {
            let dict = PyDict::new_bound(py);
            dict.set_item("success", false)?;
            dict.set_item("rows", 0)?;
            dict.set_item("columns", 0)?;
            dict.set_item("file_size_bytes", 0)?;
            dict.set_item("processing_time_ms", 0)?;
            dict.set_item("engine_used", "Error")?;
            dict.set_item("message", format!("Error: {}", e))?;
            Ok(dict.into())
        }
    }
}

/// Process local flat file using Rust backend
///
/// Args:
///     file_path: Path to local parquet file
///     start_date: Optional start date filter
///     end_date: Optional end date filter
///     symbols: Optional list of symbols to filter
///
/// Returns:
///     Dictionary with processing results (same format as download_and_process_flat_file)
#[pyfunction]
fn process_local_flat_file(
    py: Python,
    file_path: String,
    start_date: Option<String>,
    end_date: Option<String>,
    symbols: Option<Vec<String>>,
) -> PyResult<PyObject> {
    let path = PathBuf::from(file_path);

    let result: Result<FlatFileResult, String> = py.allow_threads(|| {
        tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(async {
                process_flat_file_smart(
                    &path,
                    start_date.as_deref(),
                    end_date.as_deref(),
                    symbols,
                )
                .await
                .map_err(|e| e.to_string())
            })
    });

    match result {
        Ok(result) => {
            let dict = PyDict::new_bound(py);
            dict.set_item("success", result.success)?;
            dict.set_item("rows", result.rows)?;
            dict.set_item("columns", result.columns)?;
            dict.set_item("file_size_bytes", result.file_size_bytes)?;
            dict.set_item("processing_time_ms", result.processing_time_ms)?;
            dict.set_item("engine_used", result.engine_used)?;
            dict.set_item("message", result.message)?;
            Ok(dict.into())
        }
        Err(e) => {
            let dict = PyDict::new_bound(py);
            dict.set_item("success", false)?;
            dict.set_item("rows", 0)?;
            dict.set_item("columns", 0)?;
            dict.set_item("file_size_bytes", 0)?;
            dict.set_item("processing_time_ms", 0)?;
            dict.set_item("engine_used", "Error")?;
            dict.set_item("message", format!("Error: {}", e))?;
            Ok(dict.into())
        }
    }
}

/// Get file size threshold for engine selection
#[pyfunction]
fn get_size_threshold_gb() -> f64 {
    SIZE_THRESHOLD_GB
}

/// Check which engine would be used for a given file size
#[pyfunction]
fn recommend_engine(file_size_bytes: u64) -> String {
    if file_size_bytes < SIZE_THRESHOLD_BYTES {
        "Polars".to_string()
    } else {
        "DataFusion".to_string()
    }
}

pub fn register_flat_file_module<'py>(py: Python<'py>, parent_module: &Bound<'py, PyModule>) -> PyResult<()> {
    let flat_file_module = PyModule::new_bound(py, "flat_file")?;
    
    flat_file_module.add_class::<PyS3Config>()?;
    flat_file_module.add_function(wrap_pyfunction!(download_and_process_flat_file, &flat_file_module)?)?;
    flat_file_module.add_function(wrap_pyfunction!(process_local_flat_file, &flat_file_module)?)?;
    flat_file_module.add_function(wrap_pyfunction!(get_size_threshold_gb, &flat_file_module)?)?;
    flat_file_module.add_function(wrap_pyfunction!(recommend_engine, &flat_file_module)?)?;
    
    parent_module.add_submodule(&flat_file_module)?;
    Ok(())
}
