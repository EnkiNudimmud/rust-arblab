/// Statistical metrics using functional programming
use super::{traits::StatisticalMetric, error::{AnalyticsError, AnalyticsResult}};

/// Mean calculator
pub struct Mean;

impl StatisticalMetric for Mean {
    fn compute(&self, data: &[f64]) -> AnalyticsResult<f64> {
        if data.is_empty() {
            return Err(AnalyticsError::EmptyData);
        }
        Ok(data.iter().sum::<f64>() / data.len() as f64)
    }
}

/// Variance calculator
pub struct Variance {
    ddof: usize,
}

impl Variance {
    pub fn unbiased() -> Self {
        Self { ddof: 1 }
    }
    
    pub fn biased() -> Self {
        Self { ddof: 0 }
    }
}

impl StatisticalMetric for Variance {
    fn compute(&self, data: &[f64]) -> AnalyticsResult<f64> {
        if data.is_empty() {
            return Err(AnalyticsError::EmptyData);
        }
        if data.len() <= self.ddof {
            return Err(AnalyticsError::InvalidParameter {
                param: "ddof".to_string(),
                value: self.ddof.to_string(),
                reason: format!("data length {} must be > ddof", data.len()),
            });
        }
        
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / (data.len() - self.ddof) as f64;
        
        Ok(variance)
    }
    
    fn requires_min_samples(&self) -> usize {
        self.ddof + 1
    }
}

/// Standard deviation calculator
pub struct StdDev {
    ddof: usize,
}

impl StdDev {
    pub fn unbiased() -> Self {
        Self { ddof: 1 }
    }
    
    pub fn biased() -> Self {
        Self { ddof: 0 }
    }
}

impl StatisticalMetric for StdDev {
    fn compute(&self, data: &[f64]) -> AnalyticsResult<f64> {
        let variance = Variance { ddof: self.ddof }.compute(data)?;
        Ok(variance.sqrt())
    }
    
    fn requires_min_samples(&self) -> usize {
        self.ddof + 1
    }
}

/// Skewness calculator
pub struct Skewness;

impl StatisticalMetric for Skewness {
    fn compute(&self, data: &[f64]) -> AnalyticsResult<f64> {
        if data.len() < 3 {
            return Err(AnalyticsError::InvalidParameter {
                param: "data_length".to_string(),
                value: data.len().to_string(),
                reason: "skewness requires at least 3 samples".to_string(),
            });
        }
        
        let mean = Mean.compute(data)?;
        let std = StdDev::unbiased().compute(data)?;
        
        if std < 1e-10 {
            return Ok(0.0);
        }
        
        let n = data.len() as f64;
        let m3 = data.iter()
            .map(|&x| ((x - mean) / std).powi(3))
            .sum::<f64>() / n;
        
        // Adjusted Fisher-Pearson standardized moment coefficient
        let skew = m3 * ((n * (n - 1.0)).sqrt() / (n - 2.0));
        
        Ok(skew)
    }
    
    fn requires_min_samples(&self) -> usize {
        3
    }
}

/// Kurtosis calculator
pub struct Kurtosis {
    excess: bool,
}

impl Kurtosis {
    pub fn excess() -> Self {
        Self { excess: true }
    }
    
    pub fn raw() -> Self {
        Self { excess: false }
    }
}

impl StatisticalMetric for Kurtosis {
    fn compute(&self, data: &[f64]) -> AnalyticsResult<f64> {
        if data.len() < 4 {
            return Err(AnalyticsError::InvalidParameter {
                param: "data_length".to_string(),
                value: data.len().to_string(),
                reason: "kurtosis requires at least 4 samples".to_string(),
            });
        }
        
        let mean = Mean.compute(data)?;
        let std = StdDev::unbiased().compute(data)?;
        
        if std < 1e-10 {
            return Ok(0.0);
        }
        
        let n = data.len() as f64;
        let m4 = data.iter()
            .map(|&x| ((x - mean) / std).powi(4))
            .sum::<f64>() / n;
        
        let kurt = if self.excess {
            // Excess kurtosis (normal distribution = 0)
            m4 - 3.0
        } else {
            // Raw kurtosis (normal distribution = 3)
            m4
        };
        
        Ok(kurt)
    }
    
    fn requires_min_samples(&self) -> usize {
        4
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_mean() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mean = Mean.compute(&data).unwrap();
        assert!((mean - 3.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_variance() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let var = Variance::unbiased().compute(&data).unwrap();
        assert!(var > 0.0);
    }
    
    #[test]
    fn test_empty_data() {
        let data: Vec<f64> = vec![];
        assert!(Mean.compute(&data).is_err());
    }
}
