/// Rolling window calculations using functional programming
use ndarray::Array1;
use super::{traits::RollingWindow, error::{AnalyticsError, AnalyticsResult}};

/// Rolling mean calculator
pub struct RollingMean;

impl RollingWindow for RollingMean {
    fn compute_rolling(&self, data: &Array1<f64>, window: usize) -> AnalyticsResult<Array1<f64>> {
        validate_window(data.len(), window)?;
        
        // Functional approach: use windows and map
        let rolling_means: Vec<f64> = data
            .as_slice()
            .unwrap()
            .windows(window)
            .map(|w| w.iter().sum::<f64>() / window as f64)
            .collect();
        
        // Pad with NaN for alignment
        let mut result = vec![f64::NAN; window - 1];
        result.extend(rolling_means);
        
        Ok(Array1::from(result))
    }
    
    fn min_window_size(&self) -> usize {
        1
    }
}

/// Rolling standard deviation calculator
pub struct RollingStd {
    ddof: usize,
}

impl RollingStd {
    pub fn new(ddof: usize) -> Self {
        Self { ddof }
    }
    
    #[allow(dead_code)]
    pub fn unbiased() -> Self {
        Self::new(1)
    }
    
    #[allow(dead_code)]
    pub fn biased() -> Self {
        Self::new(0)
    }
}

impl RollingWindow for RollingStd {
    fn compute_rolling(&self, data: &Array1<f64>, window: usize) -> AnalyticsResult<Array1<f64>> {
        validate_window(data.len(), window)?;
        
        if window <= self.ddof {
            return Err(AnalyticsError::InvalidWindow {
                window,
                data_len: window,
            });
        }
        
        // Functional: compute rolling std
        let rolling_stds: Vec<f64> = data
            .as_slice()
            .unwrap()
            .windows(window)
            .map(|w| {
                let mean = w.iter().sum::<f64>() / window as f64;
                let variance = w.iter()
                    .map(|&x| (x - mean).powi(2))
                    .sum::<f64>() / (window - self.ddof) as f64;
                variance.sqrt()
            })
            .collect();
        
        let mut result = vec![f64::NAN; window - 1];
        result.extend(rolling_stds);
        
        Ok(Array1::from(result))
    }
    
    fn min_window_size(&self) -> usize {
        self.ddof + 1
    }
}

/// Rolling Z-score calculator (combines mean and std)
pub struct RollingZScore {
    window: usize,
}

impl RollingZScore {
    pub fn new(window: usize) -> Self {
        Self { window }
    }
    
    pub fn compute(&self, data: &Array1<f64>) -> AnalyticsResult<Array1<f64>> {
        validate_window(data.len(), self.window)?;
        
        let data_slice = data.as_slice().unwrap();
        
        // Functional approach: compute z-scores efficiently
        let zscores: Vec<f64> = data_slice
            .windows(self.window)
            .enumerate()
            .map(|(i, window)| {
                let mean = window.iter().sum::<f64>() / self.window as f64;
                let variance = window.iter()
                    .map(|&x| (x - mean).powi(2))
                    .sum::<f64>() / (self.window - 1) as f64;
                let std = variance.sqrt();
                
                let current_value = data_slice[i + self.window - 1];
                
                if std > 1e-10 {
                    (current_value - mean) / std
                } else {
                    0.0
                }
            })
            .collect();
        
        let mut result = vec![f64::NAN; self.window - 1];
        result.extend(zscores);
        
        Ok(Array1::from(result))
    }
}

/// Rolling correlation between two series
pub struct RollingCorrelation {
    window: usize,
}

impl RollingCorrelation {
    pub fn new(window: usize) -> Self {
        Self { window }
    }
    
    pub fn compute(&self, x: &Array1<f64>, y: &Array1<f64>) -> AnalyticsResult<Array1<f64>> {
        if x.len() != y.len() {
            return Err(AnalyticsError::InvalidDimensions {
                expected: format!("{}", x.len()),
                got: format!("{}", y.len()),
            });
        }
        
        validate_window(x.len(), self.window)?;
        
        let x_slice = x.as_slice().unwrap();
        let y_slice = y.as_slice().unwrap();
        
        // Functional: compute rolling correlations
        let correlations: Vec<f64> = (0..=(x.len() - self.window))
            .map(|i| {
                let x_window = &x_slice[i..i + self.window];
                let y_window = &y_slice[i..i + self.window];
                
                let mean_x = x_window.iter().sum::<f64>() / self.window as f64;
                let mean_y = y_window.iter().sum::<f64>() / self.window as f64;
                
                let (cov, var_x, var_y) = x_window.iter()
                    .zip(y_window.iter())
                    .fold((0.0, 0.0, 0.0), |(c, vx, vy), (&xi, &yi)| {
                        let dx = xi - mean_x;
                        let dy = yi - mean_y;
                        (c + dx * dy, vx + dx * dx, vy + dy * dy)
                    });
                
                let std_x = (var_x / self.window as f64).sqrt();
                let std_y = (var_y / self.window as f64).sqrt();
                
                if std_x > 1e-10 && std_y > 1e-10 {
                    (cov / self.window as f64) / (std_x * std_y)
                } else {
                    0.0
                }
            })
            .collect();
        
        let mut result = vec![f64::NAN; self.window - 1];
        result.extend(correlations);
        
        Ok(Array1::from(result))
    }
}

// Helper function
fn validate_window(data_len: usize, window: usize) -> AnalyticsResult<()> {
    if data_len == 0 {
        return Err(AnalyticsError::EmptyData);
    }
    if window == 0 || window > data_len {
        return Err(AnalyticsError::InvalidWindow { window, data_len });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;
    
    #[test]
    fn test_rolling_mean() {
        let data = arr1(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = RollingMean.compute_rolling(&data, 3).unwrap();
        assert!((result[2] - 2.0).abs() < 1e-10);
        assert!((result[3] - 3.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_rolling_zscore() {
        let data = arr1(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let zscore = RollingZScore::new(3);
        let result = zscore.compute(&data).unwrap();
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
    }
}
