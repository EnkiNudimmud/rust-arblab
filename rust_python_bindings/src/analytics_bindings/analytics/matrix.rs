/// Matrix operations using functional programming and the Strategy pattern
use ndarray::Array2;
use super::{traits::MatrixOperation, error::{AnalyticsError, AnalyticsResult}};

/// Correlation matrix calculator
pub struct CorrelationMatrix;

impl MatrixOperation for CorrelationMatrix {
    fn compute(&self, data: &Array2<f64>) -> AnalyticsResult<Array2<f64>> {
        let (n_periods, n_assets) = data.dim();
        
        if n_periods == 0 || n_assets == 0 {
            return Err(AnalyticsError::EmptyData);
        }
        
        if n_periods < 2 {
            return Err(AnalyticsError::InvalidWindow { 
                window: 2, 
                data_len: n_periods 
            });
        }
        
        // Functional approach: compute means using map
        let means: Vec<f64> = (0..n_assets)
            .map(|i| data.column(i).mean().unwrap_or(0.0))
            .collect();
        
        // Compute correlation matrix using functional style
        let correlations = (0..n_assets)
            .flat_map(|i| (i..n_assets).map(move |j| (i, j)))
            .map(|(i, j)| {
                let col_i: Vec<f64> = data.column(i).iter().copied().collect();
                let col_j: Vec<f64> = data.column(j).iter().copied().collect();
                let (corr_val, _) = self.compute_pair_correlation(
                    &col_i,
                    &col_j,
                    means[i],
                    means[j],
                )?;
                Ok(((i, j), corr_val))
            })
            .collect::<AnalyticsResult<Vec<_>>>()?;
        
        // Build symmetric matrix
        let mut matrix = Array2::zeros((n_assets, n_assets));
        for ((i, j), val) in correlations {
            matrix[[i, j]] = val;
            matrix[[j, i]] = val;
        }
        
        Ok(matrix)
    }
    
    fn name(&self) -> &str {
        "Correlation Matrix"
    }
}

impl CorrelationMatrix {
    /// Compute correlation between two series (functional helper)
    fn compute_pair_correlation(
        &self,
        x: &[f64],
        y: &[f64],
        mean_x: f64,
        mean_y: f64,
    ) -> AnalyticsResult<(f64, f64)> {
        let n = x.len();
        
        // Use fold for aggregation - functional approach
        let (cov, var_x, var_y) = x.iter()
            .zip(y.iter())
            .fold((0.0, 0.0, 0.0), |(cov_acc, vx_acc, vy_acc), (&xi, &yi)| {
                let dev_x = xi - mean_x;
                let dev_y = yi - mean_y;
                (
                    cov_acc + dev_x * dev_y,
                    vx_acc + dev_x * dev_x,
                    vy_acc + dev_y * dev_y,
                )
            });
        
        let std_x = (var_x / n as f64).sqrt();
        let std_y = (var_y / n as f64).sqrt();
        
        const MIN_STD: f64 = 1e-10;
        
        let correlation = if std_x > MIN_STD && std_y > MIN_STD {
            (cov / n as f64) / (std_x * std_y)
        } else if x == y {
            1.0
        } else {
            0.0
        };
        
        Ok((correlation, cov / n as f64))
    }
}

/// Covariance matrix calculator
pub struct CovarianceMatrix {
    unbiased: bool,
}

impl CovarianceMatrix {
    pub fn new(unbiased: bool) -> Self {
        Self { unbiased }
    }
    
    pub fn unbiased() -> Self {
        Self::new(true)
    }
    
    pub fn biased() -> Self {
        Self::new(false)
    }
}

impl MatrixOperation for CovarianceMatrix {
    fn compute(&self, data: &Array2<f64>) -> AnalyticsResult<Array2<f64>> {
        let (n_periods, n_assets) = data.dim();
        
        if n_periods == 0 || n_assets == 0 {
            return Err(AnalyticsError::EmptyData);
        }
        
        let min_periods = if self.unbiased { 2 } else { 1 };
        if n_periods < min_periods {
            return Err(AnalyticsError::InvalidWindow { 
                window: min_periods, 
                data_len: n_periods 
            });
        }
        
        // Functional: compute means
        let means: Vec<f64> = (0..n_assets)
            .map(|i| data.column(i).mean().unwrap_or(0.0))
            .collect();
        
        let divisor = if self.unbiased {
            n_periods as f64 - 1.0
        } else {
            n_periods as f64
        };
        
        // Functional: compute covariances
        let covariances = (0..n_assets)
            .flat_map(|i| (i..n_assets).map(move |j| (i, j)))
            .map(|(i, j)| {
                let col_i = data.column(i);
                let col_j = data.column(j);
                
                let cov = col_i.iter()
                    .zip(col_j.iter())
                    .map(|(&xi, &xj)| (xi - means[i]) * (xj - means[j]))
                    .sum::<f64>() / divisor;
                
                Ok(((i, j), cov))
            })
            .collect::<AnalyticsResult<Vec<_>>>()?;
        
        // Build symmetric matrix
        let mut matrix = Array2::zeros((n_assets, n_assets));
        for ((i, j), val) in covariances {
            matrix[[i, j]] = val;
            matrix[[j, i]] = val;
        }
        
        Ok(matrix)
    }
    
    fn name(&self) -> &str {
        if self.unbiased {
            "Unbiased Covariance Matrix"
        } else {
            "Biased Covariance Matrix"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;
    
    #[test]
    fn test_correlation_identity() {
        let data = arr2(&[[1.0], [2.0], [3.0]]);
        let corr = CorrelationMatrix.compute(&data).unwrap();
        assert!((corr[[0, 0]] - 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_covariance_matrix() {
        let data = arr2(&[[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]);
        let cov = CovarianceMatrix::unbiased().compute(&data).unwrap();
        assert!(cov[[0, 0]] > 0.0);
        assert!(cov[[0, 1]] > 0.0);
    }
}
