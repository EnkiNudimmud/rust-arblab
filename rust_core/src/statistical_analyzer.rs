/// High-performance parallel statistical analyzer for asset selection
/// Uses Rayon for CPU-bound parallel processing with generic trait patterns
/// 
/// Features:
/// - Parallel correlation matrix computation
/// - Parallel cointegration testing (Engle-Granger)
/// - Parallel volatility analysis
/// - Progress tracking with time estimation
/// - Zero-copy operations where possible

use std::collections::HashMap;
use std::time::Instant;
use rayon::prelude::*;
use ndarray::{Array1, Array2, Axis};
use statrs::statistics::{Statistics, OrderStatistics};

/// Generic trait for time series analysis
pub trait TimeSeriesAnalyzer: Send + Sync {
    fn analyze(&self, data: &[f64]) -> Result<f64, String>;
}

/// Correlation analyzer
pub struct CorrelationAnalyzer;

impl TimeSeriesAnalyzer for CorrelationAnalyzer {
    fn analyze(&self, data: &[f64]) -> Result<f64, String> {
        if data.len() < 2 {
            return Err("Insufficient data points".to_string());
        }
        Ok(data.iter().copied().std_dev())
    }
}

/// Cointegration test result
#[derive(Debug, Clone)]
pub struct CointegrationResult {
    pub symbol1: String,
    pub symbol2: String,
    pub p_value: f64,
    pub correlation: f64,
    pub test_statistic: f64,
}

/// Basket analysis result
#[derive(Debug, Clone)]
pub struct BasketResult {
    pub symbols: Vec<String>,
    pub avg_correlation: f64,
    pub avg_volatility: f64,
}

/// Volatility ranking result
#[derive(Debug, Clone)]
pub struct VolatilityRanking {
    pub symbol: String,
    pub volatility: f64,
    pub rank: usize,
}

/// Progress callback trait for time estimation
pub trait ProgressCallback: Send + Sync {
    fn update(&self, current: usize, total: usize, eta_seconds: f64);
}

/// Main statistical analyzer with parallel processing
pub struct StatisticalAnalyzer {
    pub min_correlation: f64,
    pub min_data_points: usize,
}

impl StatisticalAnalyzer {
    pub fn new(min_correlation: f64, min_data_points: usize) -> Self {
        Self {
            min_correlation,
            min_data_points,
        }
    }

    /// Filter symbols with sufficient data (parallel)
    pub fn filter_valid_symbols(
        &self,
        data: &HashMap<String, Vec<f64>>,
    ) -> HashMap<String, Vec<f64>> {
        data.par_iter()
            .filter(|(_, values)| values.len() >= self.min_data_points)
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    /// Compute correlation matrix in parallel using ndarray
    pub fn compute_correlation_matrix(
        &self,
        data: &HashMap<String, Vec<f64>>,
    ) -> Result<(Vec<String>, Array2<f64>), String> {
        let symbols: Vec<String> = data.keys().cloned().collect();
        let n_symbols = symbols.len();
        
        if n_symbols == 0 {
            return Err("No symbols provided".to_string());
        }

        // Find minimum length
        let min_len = data.values()
            .map(|v| v.len())
            .min()
            .ok_or("No data available")?;

        // Build data matrix (symbols x observations)
        let mut matrix = Array2::<f64>::zeros((n_symbols, min_len));
        for (i, symbol) in symbols.iter().enumerate() {
            if let Some(values) = data.get(symbol) {
                for j in 0..min_len {
                    matrix[[i, j]] = values[j];
                }
            }
        }

        // Compute correlation matrix in parallel
        let corr_matrix = self.parallel_correlation(&matrix);

        Ok((symbols, corr_matrix))
    }

    /// Parallel correlation computation
    fn parallel_correlation(&self, data: &Array2<f64>) -> Array2<f64> {
        let n = data.shape()[0];
        let mut corr = Array2::<f64>::zeros((n, n));

        // Compute means and std devs in parallel
        let stats: Vec<(f64, f64)> = (0..n)
            .into_par_iter()
            .map(|i| {
                let row = data.row(i);
                let n_points = row.len() as f64;
                let mean = row.sum() / n_points;
                let variance = row.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n_points;
                let std = variance.sqrt();
                (mean, std)
            })
            .collect();

        // Compute correlations in parallel (upper triangle)
        // Clone stats for sharing across threads
        let stats_clone = stats.clone();
        let pairs: Vec<(usize, usize, f64)> = (0..n)
            .into_par_iter()
            .flat_map(|i| {
                let stats_ref = &stats_clone;
                (i..n).into_par_iter().map(move |j| {
                    if i == j {
                        (i, j, 1.0)
                    } else {
                        let row_i = data.row(i);
                        let row_j = data.row(j);
                        let (mean_i, std_i) = stats_ref[i];
                        let (mean_j, std_j) = stats_ref[j];

                        if std_i == 0.0 || std_j == 0.0 {
                            return (i, j, 0.0);
                        }

                        let n_points = row_i.len() as f64;
                        let mut covariance = 0.0;
                        for k in 0..row_i.len() {
                            covariance += (row_i[k] - mean_i) * (row_j[k] - mean_j);
                        }
                        covariance /= n_points;

                        let correlation = covariance / (std_i * std_j);
                        (i, j, correlation)
                    }
                })
            })
            .collect();

        // Fill correlation matrix (symmetric)
        for (i, j, value) in pairs {
            corr[[i, j]] = value;
            if i != j {
                corr[[j, i]] = value;
            }
        }

        corr
    }

    /// Find best cointegrated pairs in parallel with progress tracking
    pub fn find_cointegrated_pairs<F>(
        &self,
        data: &HashMap<String, Vec<f64>>,
        correlation_matrix: &Array2<f64>,
        symbols: &[String],
        progress_callback: Option<F>,
    ) -> Vec<CointegrationResult>
    where
        F: Fn(usize, usize, f64) + Send + Sync,
    {
        let start_time = Instant::now();
        
        // Generate all pairs to test
        let pairs: Vec<(usize, usize)> = (0..symbols.len())
            .flat_map(|i| ((i + 1)..symbols.len()).map(move |j| (i, j)))
            .collect();

        let total_pairs = pairs.len();
        let processed = std::sync::atomic::AtomicUsize::new(0);

        // Parallel cointegration testing with progress
        let results: Vec<CointegrationResult> = pairs
            .par_iter()
            .filter_map(|&(i, j)| {
                let corr = correlation_matrix[[i, j]];
                
                if corr < self.min_correlation {
                    return None;
                }

                let sym1 = &symbols[i];
                let sym2 = &symbols[j];

                // Perform cointegration test
                if let (Some(series1), Some(series2)) = (data.get(sym1), data.get(sym2)) {
                    if let Ok(result) = self.engle_granger_test(series1, series2) {
                        // Update progress
                        let current = processed.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                        if let Some(ref cb) = progress_callback {
                            let elapsed = start_time.elapsed().as_secs_f64();
                            let eta = if current > 0 {
                                (elapsed / current as f64) * (total_pairs - current) as f64
                            } else {
                                0.0
                            };
                            cb(current, total_pairs, eta);
                        }

                        if result.0 < 0.05 {
                            return Some(CointegrationResult {
                                symbol1: sym1.clone(),
                                symbol2: sym2.clone(),
                                p_value: result.0,
                                correlation: corr,
                                test_statistic: result.1,
                            });
                        }
                    }
                }
                None
            })
            .collect();

        results
    }

    /// Engle-Granger cointegration test
    fn engle_granger_test(&self, y: &[f64], x: &[f64]) -> Result<(f64, f64), String> {
        if y.len() != x.len() || y.len() < 10 {
            return Err("Invalid series length".to_string());
        }

        let n = y.len();

        // OLS regression: y = alpha + beta * x + residuals
        let x_mean = x.iter().copied().mean();
        let y_mean = y.iter().copied().mean();

        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for i in 0..n {
            let x_dev = x[i] - x_mean;
            let y_dev = y[i] - y_mean;
            numerator += x_dev * y_dev;
            denominator += x_dev * x_dev;
        }

        if denominator == 0.0 {
            return Err("Singular matrix".to_string());
        }

        let beta = numerator / denominator;
        let alpha = y_mean - beta * x_mean;

        // Compute residuals
        let residuals: Vec<f64> = (0..n)
            .map(|i| y[i] - (alpha + beta * x[i]))
            .collect();

        // Augmented Dickey-Fuller test on residuals
        let adf_stat = self.adf_test(&residuals)?;

        // Critical values (simplified - should use proper lookup table)
        let critical_value = -3.34; // 5% significance for n=100
        let p_value = if adf_stat < critical_value { 0.01 } else { 0.1 };

        Ok((p_value, adf_stat))
    }

    /// Augmented Dickey-Fuller test statistic
    fn adf_test(&self, series: &[f64]) -> Result<f64, String> {
        if series.len() < 10 {
            return Err("Insufficient data for ADF test".to_string());
        }

        // First difference
        let diff: Vec<f64> = (1..series.len())
            .map(|i| series[i] - series[i - 1])
            .collect();

        let lagged: Vec<f64> = series[..series.len() - 1].to_vec();

        // Regression: diff = rho * lagged
        let lagged_mean = lagged.iter().copied().mean();
        let diff_mean = diff.iter().copied().mean();

        let mut numerator = 0.0;
        let mut denominator = 0.0;
        let mut sse = 0.0;

        for i in 0..diff.len() {
            let x_dev = lagged[i] - lagged_mean;
            let y_dev = diff[i] - diff_mean;
            numerator += x_dev * y_dev;
            denominator += x_dev * x_dev;
        }

        if denominator == 0.0 {
            return Err("Singular matrix in ADF".to_string());
        }

        let rho = numerator / denominator;
        
        // Compute standard error
        for i in 0..diff.len() {
            let predicted = rho * (lagged[i] - lagged_mean) + diff_mean;
            let residual = diff[i] - predicted;
            sse += residual * residual;
        }

        let se = (sse / (diff.len() as f64 - 1.0)).sqrt();
        let se_rho = se / denominator.sqrt();

        // t-statistic
        let t_stat = rho / se_rho;

        Ok(t_stat)
    }

    /// Build optimal basket with parallel correlation analysis
    pub fn build_optimal_basket<F>(
        &self,
        correlation_matrix: &Array2<f64>,
        volatility: &[f64],
        symbols: &[String],
        max_assets: usize,
        progress_callback: Option<F>,
    ) -> BasketResult
    where
        F: Fn(usize, usize, f64) + Send + Sync,
    {
        let start_time = Instant::now();
        
        // Find highest volatility as starting point
        let center_idx = volatility
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        let mut basket = vec![center_idx];
        let mut remaining: Vec<usize> = (0..symbols.len())
            .filter(|&i| i != center_idx)
            .collect();

        while basket.len() < max_assets && !remaining.is_empty() {
            // Parallel search for best next symbol
            let basket_clone = basket.clone();
            let best = remaining
                .par_iter()
                .map(|&idx| {
                    let avg_corr = basket_clone
                        .iter()
                        .map(|&b_idx| correlation_matrix[[idx, b_idx]])
                        .sum::<f64>()
                        / basket_clone.len() as f64;
                    (idx, avg_corr)
                })
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap());

            if let Some((best_idx, best_corr)) = best {
                if best_corr >= self.min_correlation {
                    basket.push(best_idx);
                    remaining.retain(|&x| x != best_idx);
                    
                    // Update progress
                    if let Some(ref cb) = progress_callback {
                        let elapsed = start_time.elapsed().as_secs_f64();
                        let eta = (elapsed / basket.len() as f64) * (max_assets - basket.len()) as f64;
                        cb(basket.len(), max_assets, eta);
                    }
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        // Calculate metrics
        let selected_symbols: Vec<String> = basket.iter().map(|&i| symbols[i].clone()).collect();
        let avg_correlation = self.compute_basket_correlation(correlation_matrix, &basket);
        let avg_volatility = basket.iter().map(|&i| volatility[i]).sum::<f64>() / basket.len() as f64;

        BasketResult {
            symbols: selected_symbols,
            avg_correlation,
            avg_volatility,
        }
    }

    /// Compute average intra-basket correlation
    fn compute_basket_correlation(&self, corr_matrix: &Array2<f64>, basket: &[usize]) -> f64 {
        if basket.len() < 2 {
            return 1.0;
        }

        let mut sum = 0.0;
        let mut count = 0;

        for i in 0..basket.len() {
            for j in (i + 1)..basket.len() {
                sum += corr_matrix[[basket[i], basket[j]]];
                count += 1;
            }
        }

        if count > 0 {
            sum / count as f64
        } else {
            0.0
        }
    }

    /// Compute volatility rankings in parallel
    pub fn compute_volatility_rankings(
        &self,
        data: &HashMap<String, Vec<f64>>,
    ) -> Vec<VolatilityRanking> {
        let mut rankings: Vec<VolatilityRanking> = data
            .par_iter()
            .map(|(symbol, values)| {
                // Compute returns
                let returns: Vec<f64> = (1..values.len())
                    .map(|i| (values[i] - values[i - 1]) / values[i - 1])
                    .collect();

                let volatility = returns.iter().copied().std_dev();

                VolatilityRanking {
                    symbol: symbol.clone(),
                    volatility,
                    rank: 0, // Will be set after sorting
                }
            })
            .collect();

        // Sort by volatility descending
        rankings.sort_by(|a, b| b.volatility.partial_cmp(&a.volatility).unwrap());

        // Assign ranks
        for (rank, ranking) in rankings.iter_mut().enumerate() {
            ranking.rank = rank + 1;
        }

        rankings
    }

    /// Find all suitable assets (combined approach) with parallel processing
    pub fn find_all_suitable<F>(
        &self,
        data: &HashMap<String, Vec<f64>>,
        correlation_matrix: &Array2<f64>,
        symbols: &[String],
        max_assets: usize,
        progress_callback: Option<F>,
    ) -> Vec<String>
    where
        F: Fn(usize, usize, f64) + Send + Sync,
    {
        // Parallel cointegration + volatility analysis
        let coint_results = self.find_cointegrated_pairs(data, correlation_matrix, symbols, progress_callback);
        
        let mut candidates: std::collections::HashSet<String> = coint_results
            .iter()
            .flat_map(|r| vec![r.symbol1.clone(), r.symbol2.clone()])
            .collect();

        // Add top volatile assets
        let volatility_rankings = self.compute_volatility_rankings(data);
        candidates.extend(
            volatility_rankings
                .iter()
                .take(5)
                .map(|r| r.symbol.clone())
        );

        candidates.into_iter().take(max_assets).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_valid_symbols() {
        let mut data = HashMap::new();
        data.insert("AAPL".to_string(), vec![1.0; 300]);
        data.insert("GOOGL".to_string(), vec![2.0; 150]);

        let analyzer = StatisticalAnalyzer::new(0.7, 200);
        let filtered = analyzer.filter_valid_symbols(&data);

        assert_eq!(filtered.len(), 1);
        assert!(filtered.contains_key("AAPL"));
    }

    #[test]
    fn test_correlation_matrix() {
        let mut data = HashMap::new();
        data.insert("A".to_string(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        data.insert("B".to_string(), vec![2.0, 4.0, 6.0, 8.0, 10.0]);

        let analyzer = StatisticalAnalyzer::new(0.5, 5);
        let (symbols, corr) = analyzer.compute_correlation_matrix(&data).unwrap();

        assert_eq!(symbols.len(), 2);
        assert!(corr[[0, 1]].abs() > 0.99); // Should be highly correlated
    }
}
