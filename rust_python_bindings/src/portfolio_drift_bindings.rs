use pyo3::prelude::*;
use pyo3::types::PyDict;
use rust_core::portfolio_drift_uncertainty::{
    portfolio_choice_robust, optimal_liquidation, portfolio_transition,
    value_at_risk_robust, expected_shortfall_robust,
    PortfolioResult, LiquidationResult, TransitionResult,
};

/// Python wrapper for PortfolioResult
#[pyclass(name = "PortfolioResult")]
#[derive(Clone)]
pub struct PyPortfolioResult {
    #[pyo3(get)]
    pub weights: Vec<f64>,
    #[pyo3(get)]
    pub expected_return: f64,
    #[pyo3(get)]
    pub worst_case_return: f64,
    #[pyo3(get)]
    pub variance: f64,
    #[pyo3(get)]
    pub utility: f64,
}

impl From<PortfolioResult> for PyPortfolioResult {
    fn from(result: PortfolioResult) -> Self {
        PyPortfolioResult {
            weights: result.weights,
            expected_return: result.expected_return,
            worst_case_return: result.worst_case_return,
            variance: result.variance,
            utility: result.utility,
        }
    }
}

#[pymethods]
impl PyPortfolioResult {
    fn __repr__(&self) -> String {
        format!(
            "PortfolioResult(weights={:?}, expected_return={:.4}, worst_case_return={:.4}, variance={:.6}, utility={:.6})",
            self.weights.iter().map(|w| format!("{:.3}", w)).collect::<Vec<_>>(),
            self.expected_return,
            self.worst_case_return,
            self.variance,
            self.utility
        )
    }

    fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        dict.set_item("weights", self.weights.clone())?;
        dict.set_item("expected_return", self.expected_return)?;
        dict.set_item("worst_case_return", self.worst_case_return)?;
        dict.set_item("variance", self.variance)?;
        dict.set_item("utility", self.utility)?;
        Ok(dict.into())
    }
}

/// Python wrapper for LiquidationResult
#[pyclass(name = "LiquidationResult")]
#[derive(Clone)]
pub struct PyLiquidationResult {
    #[pyo3(get)]
    pub trading_schedule: Vec<f64>,
    #[pyo3(get)]
    pub trading_rates: Vec<f64>,
    #[pyo3(get)]
    pub expected_cost: f64,
    #[pyo3(get)]
    pub worst_case_cost: f64,
    #[pyo3(get)]
    pub times: Vec<f64>,
}

impl From<LiquidationResult> for PyLiquidationResult {
    fn from(result: LiquidationResult) -> Self {
        PyLiquidationResult {
            trading_schedule: result.trading_schedule,
            trading_rates: result.trading_rates,
            expected_cost: result.expected_cost,
            worst_case_cost: result.worst_case_cost,
            times: result.times,
        }
    }
}

#[pymethods]
impl PyLiquidationResult {
    fn __repr__(&self) -> String {
        format!(
            "LiquidationResult(n_steps={}, expected_cost={:.6}, worst_case_cost={:.6})",
            self.trading_schedule.len() - 1,
            self.expected_cost,
            self.worst_case_cost
        )
    }

    fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        dict.set_item("trading_schedule", self.trading_schedule.clone())?;
        dict.set_item("trading_rates", self.trading_rates.clone())?;
        dict.set_item("expected_cost", self.expected_cost)?;
        dict.set_item("worst_case_cost", self.worst_case_cost)?;
        dict.set_item("times", self.times.clone())?;
        Ok(dict.into())
    }
}

/// Python wrapper for TransitionResult
#[pyclass(name = "TransitionResult")]
#[derive(Clone)]
pub struct PyTransitionResult {
    #[pyo3(get)]
    pub trajectory: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub trading_rates: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub expected_cost: f64,
    #[pyo3(get)]
    pub worst_case_cost: f64,
    #[pyo3(get)]
    pub times: Vec<f64>,
}

impl From<TransitionResult> for PyTransitionResult {
    fn from(result: TransitionResult) -> Self {
        PyTransitionResult {
            trajectory: result.trajectory,
            trading_rates: result.trading_rates,
            expected_cost: result.expected_cost,
            worst_case_cost: result.worst_case_cost,
            times: result.times,
        }
    }
}

#[pymethods]
impl PyTransitionResult {
    fn __repr__(&self) -> String {
        format!(
            "TransitionResult(n_steps={}, n_assets={}, expected_cost={:.6}, worst_case_cost={:.6})",
            self.trajectory.len() - 1,
            if self.trajectory.is_empty() { 0 } else { self.trajectory[0].len() },
            self.expected_cost,
            self.worst_case_cost
        )
    }

    fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        dict.set_item("trajectory", self.trajectory.clone())?;
        dict.set_item("trading_rates", self.trading_rates.clone())?;
        dict.set_item("expected_cost", self.expected_cost)?;
        dict.set_item("worst_case_cost", self.worst_case_cost)?;
        dict.set_item("times", self.times.clone())?;
        Ok(dict.into())
    }
}

/// Robust portfolio optimization under drift uncertainty
/// 
/// Solves worst-case portfolio choice problem with CARA utility
/// when the drift parameter μ is uncertain.
/// 
/// # Arguments
/// * `expected_returns` - Best estimate of expected returns for each asset
/// * `covariance` - Covariance matrix (flattened, row-major order)
/// * `risk_aversion` - CARA risk aversion parameter γ > 0
/// * `drift_uncertainty` - Half-width of drift uncertainty interval
/// 
/// # Returns
/// PyPortfolioResult with optimal weights and performance metrics
#[pyfunction]
#[pyo3(signature = (expected_returns, covariance, risk_aversion, drift_uncertainty))]
fn portfolio_choice_drift_uncertainty(
    expected_returns: Vec<f64>,
    covariance: Vec<f64>,
    risk_aversion: f64,
    drift_uncertainty: f64,
) -> PyResult<PyPortfolioResult> {
    let n_assets = expected_returns.len();
    
    if covariance.len() != n_assets * n_assets {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Covariance matrix size mismatch: expected {}x{}, got {}", n_assets, n_assets, covariance.len())
        ));
    }
    
    let result = portfolio_choice_robust(
        &expected_returns,
        &covariance,
        risk_aversion,
        drift_uncertainty,
        n_assets,
    );
    
    Ok(result.into())
}

/// Optimal liquidation strategy under drift uncertainty
/// 
/// Computes the optimal trading schedule to liquidate a position
/// while minimizing worst-case cost under drift uncertainty.
/// 
/// # Arguments
/// * `initial_position` - Initial position to liquidate (can be negative for buying)
/// * `time_horizon` - Liquidation time horizon T
/// * `n_steps` - Number of discrete time steps
/// * `drift` - Expected price drift
/// * `drift_uncertainty` - Uncertainty in drift parameter
/// * `volatility` - Price volatility σ
/// * `temporary_impact` - Temporary price impact coefficient η
/// * `permanent_impact` - Permanent price impact coefficient κ  
/// * `risk_aversion` - Risk aversion parameter γ
/// 
/// # Returns
/// PyLiquidationResult with trading schedule and costs
#[pyfunction]
#[pyo3(signature = (initial_position, time_horizon, n_steps, drift, drift_uncertainty, volatility, temporary_impact, permanent_impact, risk_aversion))]
fn liquidation_drift_uncertainty(
    initial_position: f64,
    time_horizon: f64,
    n_steps: usize,
    drift: f64,
    drift_uncertainty: f64,
    volatility: f64,
    temporary_impact: f64,
    permanent_impact: f64,
    risk_aversion: f64,
) -> PyResult<PyLiquidationResult> {
    if n_steps == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "n_steps must be positive"
        ));
    }
    
    if time_horizon <= 0.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time_horizon must be positive"
        ));
    }
    
    let result = optimal_liquidation(
        initial_position,
        time_horizon,
        n_steps,
        drift,
        drift_uncertainty,
        volatility,
        temporary_impact,
        permanent_impact,
        risk_aversion,
    );
    
    Ok(result.into())
}

/// Portfolio transition under drift uncertainty
/// 
/// Optimally rebalances from initial to target portfolio weights
/// over a given time horizon, accounting for transaction costs
/// and drift uncertainty.
/// 
/// # Arguments
/// * `initial_weights` - Current portfolio weights
/// * `target_weights` - Desired portfolio weights
/// * `time_horizon` - Transition time horizon T
/// * `n_steps` - Number of discrete time steps
/// * `expected_returns` - Expected returns for each asset
/// * `covariance` - Covariance matrix (flattened, row-major)
/// * `drift_uncertainty` - Drift uncertainty parameter
/// * `transaction_cost` - Linear transaction cost
/// * `risk_aversion` - Risk aversion parameter γ
/// 
/// # Returns
/// PyTransitionResult with optimal trajectory and costs
#[pyfunction]
#[pyo3(signature = (initial_weights, target_weights, time_horizon, n_steps, expected_returns, covariance, drift_uncertainty, transaction_cost, risk_aversion))]
fn transition_drift_uncertainty(
    initial_weights: Vec<f64>,
    target_weights: Vec<f64>,
    time_horizon: f64,
    n_steps: usize,
    expected_returns: Vec<f64>,
    covariance: Vec<f64>,
    drift_uncertainty: f64,
    transaction_cost: f64,
    risk_aversion: f64,
) -> PyResult<PyTransitionResult> {
    let n_assets = initial_weights.len();
    
    if target_weights.len() != n_assets {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "initial_weights and target_weights must have same length"
        ));
    }
    
    if expected_returns.len() != n_assets {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "expected_returns length must match number of assets"
        ));
    }
    
    if covariance.len() != n_assets * n_assets {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Covariance matrix size mismatch"
        ));
    }
    
    let result = portfolio_transition(
        &initial_weights,
        &target_weights,
        time_horizon,
        n_steps,
        &expected_returns,
        &covariance,
        drift_uncertainty,
        transaction_cost,
        risk_aversion,
    );
    
    Ok(result.into())
}

/// Compute Value-at-Risk under drift uncertainty
/// 
/// # Arguments
/// * `weights` - Portfolio weights
/// * `expected_returns` - Expected returns
/// * `covariance` - Covariance matrix (flattened)
/// * `drift_uncertainty` - Drift uncertainty parameter
/// * `confidence_level` - Confidence level (e.g., 0.95 for 95% VaR)
/// * `time_horizon` - Time horizon for VaR calculation
/// 
/// # Returns
/// VaR value (positive = potential loss)
#[pyfunction]
#[pyo3(signature = (weights, expected_returns, covariance, drift_uncertainty, confidence_level, time_horizon))]
fn var_drift_uncertainty(
    weights: Vec<f64>,
    expected_returns: Vec<f64>,
    covariance: Vec<f64>,
    drift_uncertainty: f64,
    confidence_level: f64,
    time_horizon: f64,
) -> PyResult<f64> {
    let n_assets = weights.len();
    
    if expected_returns.len() != n_assets || covariance.len() != n_assets * n_assets {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Dimension mismatch in inputs"
        ));
    }
    
    let var = value_at_risk_robust(
        &weights,
        &expected_returns,
        &covariance,
        drift_uncertainty,
        confidence_level,
        time_horizon,
    );
    
    Ok(var)
}

/// Compute Expected Shortfall (CVaR) under drift uncertainty
/// 
/// # Arguments
/// * `weights` - Portfolio weights
/// * `expected_returns` - Expected returns
/// * `covariance` - Covariance matrix (flattened)
/// * `drift_uncertainty` - Drift uncertainty parameter
/// * `confidence_level` - Confidence level (e.g., 0.95)
/// * `time_horizon` - Time horizon
/// 
/// # Returns
/// Expected Shortfall value (positive = potential loss)
#[pyfunction]
#[pyo3(signature = (weights, expected_returns, covariance, drift_uncertainty, confidence_level, time_horizon))]
fn cvar_drift_uncertainty(
    weights: Vec<f64>,
    expected_returns: Vec<f64>,
    covariance: Vec<f64>,
    drift_uncertainty: f64,
    confidence_level: f64,
    time_horizon: f64,
) -> PyResult<f64> {
    let n_assets = weights.len();
    
    if expected_returns.len() != n_assets || covariance.len() != n_assets * n_assets {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Dimension mismatch in inputs"
        ));
    }
    
    let cvar = expected_shortfall_robust(
        &weights,
        &expected_returns,
        &covariance,
        drift_uncertainty,
        confidence_level,
        time_horizon,
    );
    
    Ok(cvar)
}

/// Register portfolio drift uncertainty functions with Python module
pub fn register_portfolio_drift(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPortfolioResult>()?;
    m.add_class::<PyLiquidationResult>()?;
    m.add_class::<PyTransitionResult>()?;
    
    m.add_function(wrap_pyfunction!(portfolio_choice_drift_uncertainty, m)?)?;
    m.add_function(wrap_pyfunction!(liquidation_drift_uncertainty, m)?)?;
    m.add_function(wrap_pyfunction!(transition_drift_uncertainty, m)?)?;
    m.add_function(wrap_pyfunction!(var_drift_uncertainty, m)?)?;
    m.add_function(wrap_pyfunction!(cvar_drift_uncertainty, m)?)?;
    
    Ok(())
}
