//! Finance-specific statistical modules
//! ====================================

pub mod ou_estimator;
pub mod cointegration;
pub mod hurst;
pub mod backtest;

pub use ou_estimator::{estimate_ou_params, estimate_ou_params_mle};
pub use cointegration::engle_granger_test;
pub use hurst::{hurst_exponent, hurst_dfa};
pub use backtest::{
    backtest_optimal_switching, backtest_mean_reversion,
    BacktestResult, Trade, TradeType,
};

use thiserror::Error;

#[derive(Error, Debug)]
pub enum FinanceError {
    #[error("Numerical error: {0}")]
    NumericalError(String),
    
    #[error("Insufficient data: need at least {0} points")]
    InsufficientData(usize),
    
    #[error("Invalid parameters: {0}")]
    InvalidParameters(String),
}

pub type Result<T> = std::result::Result<T, FinanceError>;
