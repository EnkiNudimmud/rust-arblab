//! Python bindings for Regime-Switching Portfolio Optimization
//!
//! Exposes the MRSJD portfolio optimizer to Python via PyO3.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule, PyModuleMethods};
use numpy::{PyArray1, PyArray2, ToPyArray};
use rust_core::regime_portfolio::{
    RegimeSwitchingPortfolio, PortfolioConfig, PortfolioResult, MarketRegime,
};

#[pyclass]
#[derive(Clone)]
pub struct PyPortfolioConfig {
    inner: PortfolioConfig,
}

#[pymethods]
impl PyPortfolioConfig {
    #[new]
    #[pyo3(signature = (
        gamma=2.0,
        time_horizon=10.0,
        rho=0.05,
        transaction_cost=0.001,
        r_bull=0.03,
        mu_bull=0.15,
        sigma_bull=0.18,
        lambda_bull=0.05,
        jump_mean_bull=-0.02,
        jump_std_bull=0.03,
        r_normal=0.025,
        mu_normal=0.08,
        sigma_normal=0.20,
        lambda_normal=0.10,
        jump_mean_normal=-0.05,
        jump_std_normal=0.05,
        r_bear=0.01,
        mu_bear=-0.05,
        sigma_bear=0.35,
        lambda_bear=0.30,
        jump_mean_bear=-0.15,
        jump_std_bear=0.10,
        q_bull_normal=0.3,
        q_bull_bear=0.05,
        q_normal_bull=0.2,
        q_normal_bear=0.1,
        q_bear_bull=0.1,
        q_bear_normal=0.4
    ))]
    fn new(
        gamma: f64,
        time_horizon: f64,
        rho: f64,
        transaction_cost: f64,
        r_bull: f64,
        mu_bull: f64,
        sigma_bull: f64,
        lambda_bull: f64,
        jump_mean_bull: f64,
        jump_std_bull: f64,
        r_normal: f64,
        mu_normal: f64,
        sigma_normal: f64,
        lambda_normal: f64,
        jump_mean_normal: f64,
        jump_std_normal: f64,
        r_bear: f64,
        mu_bear: f64,
        sigma_bear: f64,
        lambda_bear: f64,
        jump_mean_bear: f64,
        jump_std_bear: f64,
        q_bull_normal: f64,
        q_bull_bear: f64,
        q_normal_bull: f64,
        q_normal_bear: f64,
        q_bear_bull: f64,
        q_bear_normal: f64,
    ) -> Self {
        Self {
            inner: PortfolioConfig {
                gamma,
                time_horizon,
                rho,
                transaction_cost,
                r_bull,
                mu_bull,
                sigma_bull,
                lambda_bull,
                jump_mean_bull,
                jump_std_bull,
                r_normal,
                mu_normal,
                sigma_normal,
                lambda_normal,
                jump_mean_normal,
                jump_std_normal,
                r_bear,
                mu_bear,
                sigma_bear,
                lambda_bear,
                jump_mean_bear,
                jump_std_bear,
                q_bull_normal,
                q_bull_bear,
                q_normal_bull,
                q_normal_bear,
                q_bear_bull,
                q_bear_normal,
            }
        }
    }
    
    #[staticmethod]
    fn default_config() -> Self {
        Self {
            inner: PortfolioConfig::default()
        }
    }
    
    fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        let cfg = &self.inner;
        
        dict.set_item("gamma", cfg.gamma)?;
        dict.set_item("time_horizon", cfg.time_horizon)?;
        dict.set_item("rho", cfg.rho)?;
        dict.set_item("transaction_cost", cfg.transaction_cost)?;
        
        dict.set_item("r_bull", cfg.r_bull)?;
        dict.set_item("mu_bull", cfg.mu_bull)?;
        dict.set_item("sigma_bull", cfg.sigma_bull)?;
        dict.set_item("lambda_bull", cfg.lambda_bull)?;
        
        dict.set_item("r_normal", cfg.r_normal)?;
        dict.set_item("mu_normal", cfg.mu_normal)?;
        dict.set_item("sigma_normal", cfg.sigma_normal)?;
        dict.set_item("lambda_normal", cfg.lambda_normal)?;
        
        dict.set_item("r_bear", cfg.r_bear)?;
        dict.set_item("mu_bear", cfg.mu_bear)?;
        dict.set_item("sigma_bear", cfg.sigma_bear)?;
        dict.set_item("lambda_bear", cfg.lambda_bear)?;
        
        Ok(dict.into())
    }
}

#[pyclass]
pub struct PyPortfolioResult {
    inner: PortfolioResult,
}

#[pymethods]
impl PyPortfolioResult {
    #[getter]
    fn wealth<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
        self.inner.wealth.to_pyarray(py)
    }
    
    #[getter]
    fn values<'py>(&self, py: Python<'py>) -> PyObject {
        let list = PyList::empty(py);
        for v in &self.inner.values {
            list.append(v.to_pyarray(py)).unwrap();
        }
        list.into()
    }
    
    #[getter]
    fn portfolio_weights<'py>(&self, py: Python<'py>) -> PyObject {
        let list = PyList::empty(py);
        for w in &self.inner.portfolio_weights {
            list.append(w.to_pyarray(py)).unwrap();
        }
        list.into()
    }
    
    #[getter]
    fn stationary_probs<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
        self.inner.stationary_probs.to_pyarray(py)
    }
    
    #[getter]
    fn expected_weight(&self) -> f64 {
        self.inner.expected_weight
    }
    
    #[getter]
    fn initial_value(&self) -> f64 {
        self.inner.initial_value
    }
    
    #[getter]
    fn iterations(&self) -> usize {
        self.inner.iterations
    }
    
    #[getter]
    fn residual(&self) -> f64 {
        self.inner.residual
    }
    
    fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        
        dict.set_item("wealth", self.wealth(py))?;
        dict.set_item("values", self.values(py))?;
        dict.set_item("portfolio_weights", self.portfolio_weights(py))?;
        dict.set_item("stationary_probs", self.stationary_probs(py))?;
        dict.set_item("expected_weight", self.expected_weight())?;
        dict.set_item("initial_value", self.initial_value())?;
        dict.set_item("iterations", self.iterations())?;
        dict.set_item("residual", self.residual())?;
        
        Ok(dict.into())
    }
}

#[pyclass]
pub struct PyRegimeSwitchingPortfolio {
    inner: RegimeSwitchingPortfolio,
}

#[pymethods]
impl PyRegimeSwitchingPortfolio {
    #[new]
    fn new(config: PyPortfolioConfig) -> Self {
        Self {
            inner: RegimeSwitchingPortfolio::new(config.inner)
        }
    }
    
    fn optimize(&self) -> PyResult<PyPortfolioResult> {
        match self.inner.optimize() {
            Ok(result) => Ok(PyPortfolioResult { inner: result }),
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Optimization failed: {}", e)
            ))
        }
    }
    
    fn simulate_path(
        &self,
        result: &PyPortfolioResult,
        initial_wealth: f64,
        n_steps: usize,
        dt: f64,
        py: Python,
    ) -> PyResult<PyObject> {
        match self.inner.simulate_path(&result.inner, initial_wealth, n_steps, dt) {
            Ok((times, wealths, regimes)) => {
                let dict = PyDict::new(py);
                dict.set_item("times", times.to_pyarray(py))?;
                dict.set_item("wealths", wealths.to_pyarray(py))?;
                dict.set_item("regimes", regimes.to_pyarray(py))?;
                Ok(dict.into())
            }
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Simulation failed: {}", e)
            ))
        }
    }
    
    #[staticmethod]
    fn estimate_regime(returns: Vec<f64>) -> String {
        let regime = RegimeSwitchingPortfolio::estimate_regime(&returns);
        match regime {
            MarketRegime::Bull => "Bull".to_string(),
            MarketRegime::Normal => "Normal".to_string(),
            MarketRegime::Bear => "Bear".to_string(),
        }
    }
}

#[pyfunction]
fn calibrate_model_from_data(
    returns: Vec<f64>,
    regimes: Option<Vec<usize>>,
) -> PyPortfolioConfig {
    let config = rust_core::regime_portfolio::calibrate_from_data(&returns, regimes.as_deref());
    PyPortfolioConfig { inner: config }
}

pub fn register_regime_portfolio<'py>(py: Python<'py>, parent_module: &Bound<'py, PyModule>) -> PyResult<()> {
    let regime_mod = PyModule::new_bound(py, "regime_portfolio")?;
    
    regime_mod.add_class::<PyPortfolioConfig>()?;
    regime_mod.add_class::<PyPortfolioResult>()?;
    regime_mod.add_class::<PyRegimeSwitchingPortfolio>()?;
    regime_mod.add_function(wrap_pyfunction!(calibrate_model_from_data, &regime_mod)?)?;
    
    parent_module.add_submodule(&regime_mod)?;
    
    Ok(())
}
