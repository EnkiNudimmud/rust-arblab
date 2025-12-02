use pyo3::prelude::*;
use pyo3::types::PyDict;
use rust_core::options::{BlackScholesOption, DeltaHedgingStrategy};

#[pyclass(name = "BlackScholesOption")]
#[derive(Clone)]
pub struct PyBlackScholesOption {
    inner: BlackScholesOption,
}

#[pymethods]
impl PyBlackScholesOption {
    #[new]
    #[pyo3(signature = (spot, strike, time_to_expiry, risk_free_rate, dividend_yield, volatility, is_call=true))]
    fn new(
        spot: f64,
        strike: f64,
        time_to_expiry: f64,
        risk_free_rate: f64,
        dividend_yield: f64,
        volatility: f64,
        is_call: bool,
    ) -> Self {
        let inner = BlackScholesOption::new(
            spot,
            strike,
            time_to_expiry,
            risk_free_rate,
            dividend_yield,
            volatility,
            is_call,
        );
        PyBlackScholesOption { inner }
    }

    fn price(&self) -> f64 {
        self.inner.price()
    }

    fn delta(&self) -> f64 {
        self.inner.delta()
    }

    fn gamma(&self) -> f64 {
        self.inner.gamma()
    }

    fn vega(&self) -> f64 {
        self.inner.vega()
    }

    fn theta(&self) -> f64 {
        self.inner.theta()
    }

    fn rho(&self) -> f64 {
        self.inner.rho()
    }

    fn greeks(&self, py: Python) -> PyResult<PyObject> {
        let greeks = self.inner.greeks();
        let dict = PyDict::new_bound(py);
        dict.set_item("delta", greeks.delta)?;
        dict.set_item("gamma", greeks.gamma)?;
        dict.set_item("theta", greeks.theta)?;
        dict.set_item("vega", greeks.vega)?;
        dict.set_item("rho", greeks.rho)?;
        Ok(dict.into())
    }

    fn d1(&self) -> f64 {
        self.inner.d1()
    }

    fn d2(&self) -> f64 {
        self.inner.d2()
    }

    fn __repr__(&self) -> String {
        format!(
            "BlackScholesOption(spot={:.2}, strike={:.2}, T={:.3}, r={:.3}, σ={:.3}, type={})",
            self.inner.spot,
            self.inner.strike,
            self.inner.time_to_expiry,
            self.inner.risk_free_rate,
            self.inner.volatility,
            if self.inner.is_call { "Call" } else { "Put" }
        )
    }
}

#[pyclass(name = "DeltaHedgingStrategy")]
pub struct PyDeltaHedgingStrategy {
    inner: DeltaHedgingStrategy,
}

#[pymethods]
impl PyDeltaHedgingStrategy {
    #[new]
    #[pyo3(signature = (spot, strike, time_to_expiry, risk_free_rate, dividend_yield, implied_vol, actual_vol, hedging_vol, is_call=true))]
    fn new(
        spot: f64,
        strike: f64,
        time_to_expiry: f64,
        risk_free_rate: f64,
        dividend_yield: f64,
        implied_vol: f64,
        actual_vol: f64,
        hedging_vol: f64,
        is_call: bool,
    ) -> Self {
        let inner = DeltaHedgingStrategy::new(
            spot,
            strike,
            time_to_expiry,
            risk_free_rate,
            dividend_yield,
            implied_vol,
            actual_vol,
            hedging_vol,
            is_call,
        );
        PyDeltaHedgingStrategy { inner }
    }

    fn guaranteed_profit_actual_hedge(&self) -> f64 {
        self.inner.guaranteed_profit_actual_hedge()
    }

    fn mtm_pnl_implied_hedge(&self, dt: f64) -> f64 {
        self.inner.mtm_pnl_implied_hedge(dt)
    }

    fn mtm_pnl_custom_hedge(&self, dt: f64) -> f64 {
        self.inner.mtm_pnl_custom_hedge(dt)
    }

    fn expected_profit_simulation(
        &self,
        num_simulations: usize,
        num_steps: usize,
        drift: f64,
    ) -> (f64, f64) {
        self.inner.expected_profit_simulation(num_simulations, num_steps, drift)
    }

    fn __repr__(&self) -> String {
        format!(
            "DeltaHedgingStrategy(spot={:.2}, σ_implied={:.3}, σ_actual={:.3}, σ_hedge={:.3})",
            self.inner.option.spot,
            self.inner.implied_vol,
            self.inner.actual_vol,
            self.inner.hedging_vol
        )
    }
}

pub fn register_options(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBlackScholesOption>()?;
    m.add_class::<PyDeltaHedgingStrategy>()?;
    
    // Add helper functions
    m.add_function(wrap_pyfunction!(norm_cdf, m)?)?;
    m.add_function(wrap_pyfunction!(norm_pdf, m)?)?;
    
    Ok(())
}

#[pyfunction]
fn norm_cdf(x: f64) -> f64 {
    rust_core::options::norm_cdf(x)
}

#[pyfunction]
fn norm_pdf(x: f64) -> f64 {
    rust_core::options::norm_pdf(x)
}
