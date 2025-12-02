// Python bindings for Alpha Vantage connector
// Provides access to Alpha Vantage REST API from Python

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

/// Alpha Vantage connector configuration
#[pyclass]
#[derive(Clone)]
pub struct PyAlphaVantageConfig {
    api_key: String,
}

#[pymethods]
impl PyAlphaVantageConfig {
    /// Create new Alpha Vantage configuration
    /// 
    /// Args:
    ///     api_key: Alpha Vantage API key
    #[new]
    fn new(api_key: String) -> Self {
        PyAlphaVantageConfig { api_key }
    }
    
    #[getter]
    fn api_key(&self) -> String {
        self.api_key.clone()
    }
    
    fn __repr__(&self) -> String {
        format!("AlphaVantageConfig(api_key='***{}***')", 
                if self.api_key.len() > 6 { 
                    &self.api_key[self.api_key.len()-6..] 
                } else { 
                    "****" 
                })
    }
}

/// Fetch intraday time series data
#[pyfunction]
fn fetch_intraday(
    py: Python,
    api_key: String,
    symbol: String,
    interval: String,
) -> PyResult<PyObject> {
    use connector_alpha_vantage::{AlphaVantageConfig, fetch_intraday as rust_fetch_intraday};
    
    let config = AlphaVantageConfig::new(api_key);
    
    let result = py.allow_threads(|| {
        pyo3_asyncio::tokio::get_runtime()
            .block_on(async {
                rust_fetch_intraday(&config, &symbol, &interval).await
            })
    });
    
    match result {
        Ok(data) => {
            let dict = PyDict::new_bound(py);
            dict.set_item("symbol", data.metadata.symbol)?;
            dict.set_item("last_refreshed", data.metadata.last_refreshed)?;
            
            // Convert time series to Python dict
            let time_series = PyDict::new_bound(py);
            for (ts_key, ts_data) in data.time_series.iter() {
                for (timestamp, price_data) in ts_data.iter() {
                    let price_dict = PyDict::new_bound(py);
                    price_dict.set_item("open", price_data.open.clone())?;
                    price_dict.set_item("high", price_data.high.clone())?;
                    price_dict.set_item("low", price_data.low.clone())?;
                    price_dict.set_item("close", price_data.close.clone())?;
                    price_dict.set_item("volume", price_data.volume.clone())?;
                    time_series.set_item(timestamp, price_dict)?;
                }
            }
            
            dict.set_item("time_series", time_series)?;
            Ok(dict.into())
        }
        Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            format!("Failed to fetch intraday data: {}", e)
        )),
    }
}

/// Fetch daily time series data
#[pyfunction]
fn fetch_daily(
    py: Python,
    api_key: String,
    symbol: String,
) -> PyResult<PyObject> {
    use connector_alpha_vantage::{AlphaVantageConfig, fetch_daily as rust_fetch_daily};
    
    let config = AlphaVantageConfig::new(api_key);
    
    let result = py.allow_threads(|| {
        pyo3_asyncio::tokio::get_runtime()
            .block_on(async {
                rust_fetch_daily(&config, &symbol).await
            })
    });
    
    match result {
        Ok(data) => {
            let dict = PyDict::new_bound(py);
            dict.set_item("symbol", data.metadata.symbol)?;
            dict.set_item("last_refreshed", data.metadata.last_refreshed)?;
            
            // Convert time series to Python dict
            let time_series = PyDict::new_bound(py);
            for (ts_key, ts_data) in data.time_series.iter() {
                for (timestamp, price_data) in ts_data.iter() {
                    let price_dict = PyDict::new_bound(py);
                    price_dict.set_item("open", price_data.open.clone())?;
                    price_dict.set_item("high", price_data.high.clone())?;
                    price_dict.set_item("low", price_data.low.clone())?;
                    price_dict.set_item("close", price_data.close.clone())?;
                    price_dict.set_item("volume", price_data.volume.clone())?;
                    time_series.set_item(timestamp, price_dict)?;
                }
            }
            
            dict.set_item("time_series", time_series)?;
            Ok(dict.into())
        }
        Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            format!("Failed to fetch daily data: {}", e)
        )),
    }
}

/// Fetch real-time quote
#[pyfunction]
fn fetch_quote(
    py: Python,
    api_key: String,
    symbol: String,
) -> PyResult<PyObject> {
    use connector_alpha_vantage::{AlphaVantageConfig, fetch_quote as rust_fetch_quote};
    
    let config = AlphaVantageConfig::new(api_key);
    
    let result = py.allow_threads(|| {
        pyo3_asyncio::tokio::get_runtime()
            .block_on(async {
                rust_fetch_quote(&config, &symbol).await
            })
    });
    
    match result {
        Ok(quote_response) => {
            let dict = PyDict::new_bound(py);
            let quote = &quote_response.global_quote;
            
            dict.set_item("symbol", quote.symbol.clone())?;
            dict.set_item("open", quote.open.clone())?;
            dict.set_item("high", quote.high.clone())?;
            dict.set_item("low", quote.low.clone())?;
            dict.set_item("price", quote.price.clone())?;
            dict.set_item("volume", quote.volume.clone())?;
            dict.set_item("latest_trading_day", quote.latest_trading_day.clone())?;
            dict.set_item("previous_close", quote.previous_close.clone())?;
            dict.set_item("change", quote.change.clone())?;
            dict.set_item("change_percent", quote.change_percent.clone())?;
            
            Ok(dict.into())
        }
        Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            format!("Failed to fetch quote: {}", e)
        )),
    }
}

/// Fetch forex intraday data
#[pyfunction]
fn fetch_forex_intraday(
    py: Python,
    api_key: String,
    from_symbol: String,
    to_symbol: String,
    interval: String,
) -> PyResult<PyObject> {
    use connector_alpha_vantage::{AlphaVantageConfig, fetch_forex_intraday as rust_fetch_forex};
    
    let config = AlphaVantageConfig::new(api_key);
    
    let result = py.allow_threads(|| {
        pyo3_asyncio::tokio::get_runtime()
            .block_on(async {
                rust_fetch_forex(&config, &from_symbol, &to_symbol, &interval).await
            })
    });
    
    match result {
        Ok(data) => {
            let dict = PyDict::new_bound(py);
            dict.set_item("from_symbol", data.metadata.from_symbol)?;
            dict.set_item("to_symbol", data.metadata.to_symbol)?;
            dict.set_item("last_refreshed", data.metadata.last_refreshed)?;
            
            // Convert time series to Python dict
            let time_series = PyDict::new_bound(py);
            for (ts_key, ts_data) in data.time_series.iter() {
                for (timestamp, price_data) in ts_data.iter() {
                    let price_dict = PyDict::new_bound(py);
                    price_dict.set_item("open", price_data.open.clone())?;
                    price_dict.set_item("high", price_data.high.clone())?;
                    price_dict.set_item("low", price_data.low.clone())?;
                    price_dict.set_item("close", price_data.close.clone())?;
                    time_series.set_item(timestamp, price_dict)?;
                }
            }
            
            dict.set_item("time_series", time_series)?;
            Ok(dict.into())
        }
        Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            format!("Failed to fetch forex data: {}", e)
        )),
    }
}

/// Fetch crypto daily data
#[pyfunction]
fn fetch_crypto_daily(
    py: Python,
    api_key: String,
    symbol: String,
    market: String,
) -> PyResult<PyObject> {
    use connector_alpha_vantage::{AlphaVantageConfig, fetch_crypto_daily as rust_fetch_crypto};
    
    let config = AlphaVantageConfig::new(api_key);
    
    let result = py.allow_threads(|| {
        pyo3_asyncio::tokio::get_runtime()
            .block_on(async {
                rust_fetch_crypto(&config, &symbol, &market).await
            })
    });
    
    match result {
        Ok(data) => {
            let dict = PyDict::new_bound(py);
            dict.set_item("digital_currency_code", data.metadata.digital_currency_code)?;
            dict.set_item("digital_currency_name", data.metadata.digital_currency_name)?;
            dict.set_item("market_code", data.metadata.market_code)?;
            dict.set_item("last_refreshed", data.metadata.last_refreshed)?;
            
            // Convert time series to Python dict
            let time_series = PyDict::new_bound(py);
            for (ts_key, ts_data) in data.time_series.iter() {
                for (timestamp, price_data) in ts_data.iter() {
                    let price_dict = PyDict::new_bound(py);
                    price_dict.set_item("open_usd", price_data.open_usd.clone())?;
                    price_dict.set_item("high_usd", price_data.high_usd.clone())?;
                    price_dict.set_item("low_usd", price_data.low_usd.clone())?;
                    price_dict.set_item("close_usd", price_data.close_usd.clone())?;
                    price_dict.set_item("volume", price_data.volume.clone())?;
                    price_dict.set_item("market_cap_usd", price_data.market_cap_usd.clone())?;
                    time_series.set_item(timestamp, price_dict)?;
                }
            }
            
            dict.set_item("time_series", time_series)?;
            Ok(dict.into())
        }
        Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            format!("Failed to fetch crypto data: {}", e)
        )),
    }
}

/// Get free tier rate limits
#[pyfunction]
fn get_rate_limits() -> PyResult<(u32, u32)> {
    use connector_alpha_vantage::{FREE_TIER_DAILY_LIMIT, FREE_TIER_PER_MINUTE_LIMIT};
    Ok((FREE_TIER_DAILY_LIMIT, FREE_TIER_PER_MINUTE_LIMIT))
}

/// Register Alpha Vantage functions with Python module
pub fn register_alpha_vantage(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyAlphaVantageConfig>()?;
    m.add_function(wrap_pyfunction!(fetch_intraday, m)?)?;
    m.add_function(wrap_pyfunction!(fetch_daily, m)?)?;
    m.add_function(wrap_pyfunction!(fetch_quote, m)?)?;
    m.add_function(wrap_pyfunction!(fetch_forex_intraday, m)?)?;
    m.add_function(wrap_pyfunction!(fetch_crypto_daily, m)?)?;
    m.add_function(wrap_pyfunction!(get_rate_limits, m)?)?;
    Ok(())
}
