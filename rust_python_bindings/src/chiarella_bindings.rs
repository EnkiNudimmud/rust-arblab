/// Python bindings for Chiarella Model
use pyo3::prelude::*;
use rust_core::chiarella::{ChiarellaModel, TradingSignal, RegimeState, StationaryStats, ModelState};

/// Python wrapper for RegimeState
#[pyclass]
#[derive(Clone)]
pub struct PyRegimeState {
    #[pyo3(get)]
    pub state: String,
}

impl From<RegimeState> for PyRegimeState {
    fn from(regime: RegimeState) -> Self {
        let state = match regime {
            RegimeState::MeanReverting => "mean_reverting".to_string(),
            RegimeState::Trending => "trending".to_string(),
            RegimeState::Mixed => "mixed".to_string(),
        };
        PyRegimeState { state }
    }
}

/// Python wrapper for TradingSignal
#[pyclass]
#[derive(Clone)]
pub struct PyTradingSignal {
    #[pyo3(get)]
    pub strength: f64,
    
    #[pyo3(get)]
    pub position_size: f64,
    
    #[pyo3(get)]
    pub confidence: f64,
    
    #[pyo3(get)]
    pub regime: String,
    
    #[pyo3(get)]
    pub mispricing: f64,
    
    #[pyo3(get)]
    pub trend: f64,
    
    #[pyo3(get)]
    pub expected_return: f64,
    
    #[pyo3(get)]
    pub risk: f64,
}

impl From<TradingSignal> for PyTradingSignal {
    fn from(signal: TradingSignal) -> Self {
        let regime = match signal.regime {
            RegimeState::MeanReverting => "mean_reverting".to_string(),
            RegimeState::Trending => "trending".to_string(),
            RegimeState::Mixed => "mixed".to_string(),
        };
        
        PyTradingSignal {
            strength: signal.strength,
            position_size: signal.position_size,
            confidence: signal.confidence,
            regime,
            mispricing: signal.mispricing,
            trend: signal.trend,
            expected_return: signal.expected_return,
            risk: signal.risk,
        }
    }
}

/// Python wrapper for StationaryStats
#[pyclass]
#[derive(Clone)]
pub struct PyStationaryStats {
    #[pyo3(get)]
    pub mispricing_mean: f64,
    
    #[pyo3(get)]
    pub trend_mean: f64,
    
    #[pyo3(get)]
    pub bifurcation_parameter: f64,
    
    #[pyo3(get)]
    pub is_bimodal: bool,
    
    #[pyo3(get)]
    pub regime: String,
}

impl From<StationaryStats> for PyStationaryStats {
    fn from(stats: StationaryStats) -> Self {
        let regime = match stats.regime {
            RegimeState::MeanReverting => "mean_reverting".to_string(),
            RegimeState::Trending => "trending".to_string(),
            RegimeState::Mixed => "mixed".to_string(),
        };
        
        PyStationaryStats {
            mispricing_mean: stats.mispricing_mean,
            trend_mean: stats.trend_mean,
            bifurcation_parameter: stats.bifurcation_parameter,
            is_bimodal: stats.is_bimodal,
            regime,
        }
    }
}

/// Python wrapper for ModelState
#[pyclass]
#[derive(Clone)]
pub struct PyModelState {
    #[pyo3(get)]
    pub price: f64,
    
    #[pyo3(get)]
    pub fundamental_price: f64,
    
    #[pyo3(get)]
    pub trend: f64,
    
    #[pyo3(get)]
    pub mispricing: f64,
    
    #[pyo3(get)]
    pub regime: String,
}

impl From<ModelState> for PyModelState {
    fn from(state: ModelState) -> Self {
        let regime = match state.regime {
            RegimeState::MeanReverting => "mean_reverting".to_string(),
            RegimeState::Trending => "trending".to_string(),
            RegimeState::Mixed => "mixed".to_string(),
        };
        
        PyModelState {
            price: state.price,
            fundamental_price: state.fundamental_price,
            trend: state.trend,
            mispricing: state.mispricing,
            regime,
        }
    }
}

/// Python wrapper for ChiarellaModel
#[pyclass]
pub struct PyChiarellaModel {
    model: ChiarellaModel,
}

#[pymethods]
impl PyChiarellaModel {
    /// Create new Chiarella model with default parameters
    /// 
    /// Args:
    ///     fundamental_price: The equilibrium/fundamental price
    /// 
    /// Returns:
    ///     PyChiarellaModel: New model instance
    #[new]
    pub fn new(fundamental_price: f64) -> Self {
        PyChiarellaModel {
            model: ChiarellaModel::new(fundamental_price),
        }
    }
    
    /// Create with custom parameters
    /// 
    /// Args:
    ///     fundamental_price: Equilibrium price
    ///     alpha: Chartist strength (trend feedback)
    ///     beta: Fundamentalist strength (mean reversion)
    ///     gamma: Trend formation speed
    ///     delta: Trend decay rate
    ///     sigma: Price noise std dev
    ///     eta: Trend noise std dev
    #[staticmethod]
    pub fn with_parameters(
        fundamental_price: f64,
        alpha: f64,
        beta: f64,
        gamma: f64,
        delta: f64,
        sigma: f64,
        eta: f64,
    ) -> Self {
        PyChiarellaModel {
            model: ChiarellaModel::with_parameters(
                fundamental_price, alpha, beta, gamma, delta, sigma, eta
            ),
        }
    }
    
    /// Update model by one time step with noise
    /// 
    /// Args:
    ///     noise_price: Price noise sample from N(0,1)
    ///     noise_trend: Trend noise sample from N(0,1)
    pub fn step(&mut self, noise_price: f64, noise_trend: f64) {
        self.model.step(noise_price, noise_trend);
    }
    
    /// Update model with observed market price
    /// 
    /// Args:
    ///     new_price: Observed market price
    pub fn update_with_price(&mut self, new_price: f64) {
        self.model.update_with_price(new_price);
    }
    
    /// Generate trading signal based on current state
    /// 
    /// Returns:
    ///     PyTradingSignal: Trading signal with strength, position size, confidence, etc.
    pub fn generate_signal(&self) -> PyTradingSignal {
        self.model.generate_signal().into()
    }
    
    /// Get stationary distribution statistics
    /// 
    /// Returns:
    ///     PyStationaryStats: Statistics including bifurcation parameter
    pub fn stationary_statistics(&self) -> PyStationaryStats {
        self.model.stationary_statistics().into()
    }
    
    /// Get current model state
    /// 
    /// Returns:
    ///     PyModelState: Current price, trend, mispricing, regime
    pub fn get_state(&self) -> PyModelState {
        self.model.get_state().into()
    }
    
    /// Update fundamental price
    /// 
    /// Args:
    ///     new_fundamental: New fundamental/equilibrium price
    pub fn update_fundamental(&mut self, new_fundamental: f64) {
        self.model.update_fundamental(new_fundamental);
    }
    
    /// Get current price
    #[getter]
    pub fn current_price(&self) -> f64 {
        self.model.current_price
    }
    
    /// Get fundamental price
    #[getter]
    pub fn fundamental_price(&self) -> f64 {
        self.model.fundamental_price
    }
    
    /// Get current trend
    #[getter]
    pub fn trend(&self) -> f64 {
        self.model.trend
    }
    
    /// Get alpha parameter (chartist strength)
    #[getter]
    pub fn alpha(&self) -> f64 {
        self.model.alpha
    }
    
    /// Get beta parameter (fundamentalist strength)
    #[getter]
    pub fn beta(&self) -> f64 {
        self.model.beta
    }
    
    /// Get gamma parameter (trend formation)
    #[getter]
    pub fn gamma(&self) -> f64 {
        self.model.gamma
    }
    
    /// Get delta parameter (trend decay)
    #[getter]
    pub fn delta(&self) -> f64 {
        self.model.delta
    }
    
    /// String representation
    pub fn __repr__(&self) -> String {
        format!(
            "ChiarellaModel(price={:.2}, fundamental={:.2}, trend={:.4}, α={:.2}, β={:.2})",
            self.model.current_price,
            self.model.fundamental_price,
            self.model.trend,
            self.model.alpha,
            self.model.beta
        )
    }
}

/// Register Python module
pub fn register_chiarella_module(py: Python, parent_module: &PyModule) -> PyResult<()> {
    let chiarella_module = PyModule::new(py, "chiarella")?;
    
    chiarella_module.add_class::<PyChiarellaModel>()?;
    chiarella_module.add_class::<PyTradingSignal>()?;
    chiarella_module.add_class::<PyStationaryStats>()?;
    chiarella_module.add_class::<PyModelState>()?;
    chiarella_module.add_class::<PyRegimeState>()?;
    
    parent_module.add_submodule(chiarella_module)?;
    Ok(())
}
