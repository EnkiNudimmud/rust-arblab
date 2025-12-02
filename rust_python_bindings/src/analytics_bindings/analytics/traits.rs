/// Trait definitions for extensible analytics architecture
use ndarray::{Array1, Array2};
use super::error::AnalyticsResult;

/// Core analyzer trait - Strategy pattern for different algorithms
#[allow(dead_code)]
pub trait Analyzer {
    type Input;
    type Output;
    
    fn analyze(&self, input: Self::Input) -> AnalyticsResult<Self::Output>;
    fn validate_input(&self, input: &Self::Input) -> AnalyticsResult<()>;
}

/// Matrix operations trait for correlation, covariance, etc.
pub trait MatrixOperation {
    fn compute(&self, data: &Array2<f64>) -> AnalyticsResult<Array2<f64>>;
    #[allow(dead_code)]
    fn name(&self) -> &str;
}

/// Rolling window calculations trait
pub trait RollingWindow {
    fn compute_rolling(&self, data: &Array1<f64>, window: usize) -> AnalyticsResult<Array1<f64>>;
    #[allow(dead_code)]
    fn min_window_size(&self) -> usize { 2 }
}

/// Statistical metrics trait
pub trait StatisticalMetric {
    fn compute(&self, data: &[f64]) -> AnalyticsResult<f64>;
    #[allow(dead_code)]
    fn requires_min_samples(&self) -> usize { 1 }
}
