// Anomaly Detection using Superspace Framework

use super::ghost_fields::GhostFieldSystem;
use super::chern_simons::ChernSimonsCalculator;

#[derive(Debug, Clone)]
pub struct AnomalyDetector {
    pub threshold: f64,
    pub alpha: f64,  // Weight for ghost divergence
    pub beta: f64,   // Weight for CS changes
}

impl Default for AnomalyDetector {
    fn default() -> Self {
        Self {
            threshold: 2.5,
            alpha: 0.5,
            beta: 0.5,
        }
    }
}

impl AnomalyDetector {
    pub fn new(threshold: f64, alpha: f64, beta: f64) -> Self {
        Self { threshold, alpha, beta }
    }

    /// Detect anomalies from ghost field system and CS values
    pub fn detect(
        &self,
        ghost_system: &GhostFieldSystem,
        cs_values: &[f64],
    ) -> Vec<(usize, f64)> {
        let n = ghost_system.divergence.len().min(cs_values.len());
        let mut anomalies = Vec::new();

        // Compute z-scores
        let (div_mean, div_std) = Self::mean_std(&ghost_system.divergence[1..n]);
        let (cs_mean, cs_std) = Self::mean_std(&cs_values[1..n]);

        for t in 1..n {
            // Z-score for divergence
            let z_div = if div_std > 1e-10 {
                (ghost_system.divergence[t] - div_mean) / div_std
            } else {
                0.0
            };

            // Z-score for CS change
            let cs_change = (cs_values[t] - cs_values[t - 1]).abs();
            let z_cs = if cs_std > 1e-10 {
                (cs_change - cs_mean) / cs_std
            } else {
                0.0
            };

            // Combined anomaly score
            let score = self.alpha * z_div.abs() + self.beta * z_cs;

            if score > self.threshold {
                anomalies.push((t, score));
            }
        }

        anomalies
    }

    fn mean_std(data: &[f64]) -> (f64, f64) {
        let n = data.len() as f64;
        if n < 2.0 {
            return (0.0, 1.0);
        }

        let mean = data.iter().sum::<f64>() / n;
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let std = variance.sqrt();

        (mean, std.max(1e-10))
    }

    /// Generate trading signals from anomalies
    pub fn generate_signals(&self, anomalies: &[(usize, f64)]) -> Vec<(usize, i8)> {
        // Signal: +1 (buy), -1 (sell), 0 (hold)
        let mut signals = Vec::new();

        for (i, &(time, score)) in anomalies.iter().enumerate() {
            let signal = if i % 2 == 0 {
                // Even anomalies: contrarian buy
                1
            } else {
                // Odd anomalies: contrarian sell
                -1
            };
            signals.push((time, signal));
        }

        signals
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::superspace::ghost_fields::{GhostFieldParams, GhostFieldSystem};

    #[test]
    fn test_anomaly_detection() {
        let detector = AnomalyDetector::default();
        
        // Create mock ghost system
        let params = GhostFieldParams::default();
        let bosonic = vec![vec![1.0; 7]; 100];
        let ghost_system = GhostFieldSystem::from_bosonic_coords(&bosonic, params);
        
        // Create mock CS values with a spike
        let mut cs_values = vec![0.1; 100];
        cs_values[50] = 2.0; // Anomaly
        
        let anomalies = detector.detect(&ghost_system, &cs_values);
        assert!(!anomalies.is_empty());
    }

    #[test]
    fn test_signal_generation() {
        let detector = AnomalyDetector::default();
        let anomalies = vec![(10, 3.0), (20, 3.5), (30, 4.0)];
        let signals = detector.generate_signals(&anomalies);
        assert_eq!(signals.len(), 3);
        assert_eq!(signals[0].1, 1);  // Buy
        assert_eq!(signals[1].1, -1); // Sell
    }
}
