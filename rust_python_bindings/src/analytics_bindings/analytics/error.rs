/// Custom error types for analytics operations
use std::fmt;

#[derive(Debug, Clone)]
pub enum AnalyticsError {
    InvalidDimensions { expected: String, got: String },
    InvalidWindow { window: usize, data_len: usize },
    EmptyData,
    NumericalInstability { operation: String, reason: String },
    InvalidParameter { param: String, value: String, reason: String },
}

impl fmt::Display for AnalyticsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimensions { expected, got } => {
                write!(f, "Invalid dimensions: expected {}, got {}", expected, got)
            }
            Self::InvalidWindow { window, data_len } => {
                write!(f, "Invalid window size {}: data length is only {}", window, data_len)
            }
            Self::EmptyData => write!(f, "Cannot perform operation on empty data"),
            Self::NumericalInstability { operation, reason } => {
                write!(f, "Numerical instability in {}: {}", operation, reason)
            }
            Self::InvalidParameter { param, value, reason } => {
                write!(f, "Invalid parameter '{}' = '{}': {}", param, value, reason)
            }
        }
    }
}

impl std::error::Error for AnalyticsError {}

pub type AnalyticsResult<T> = Result<T, AnalyticsError>;
