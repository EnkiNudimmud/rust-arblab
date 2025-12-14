// Chern-Simons Topological Invariants
// Detects topological phase transitions in market regime

use std::f64::consts::PI;

#[derive(Debug, Clone)]
pub struct ChernSimonsCalculator {
    pub coupling: f64,
    pub window: usize,
}

impl ChernSimonsCalculator {
    pub fn new(coupling: f64) -> Self {
        Self {
            coupling,
            window: 30,
        }
    }

    pub fn with_window(mut self, window: usize) -> Self {
        self.window = window;
        self
    }

    /// Calculate Chern-Simons invariant for a 3D connection
    /// CS(A) = k/(4π) ∫ Tr(A ∧ dA + 2/3 A ∧ A ∧ A)
    pub fn calculate(&self, field: &[f64]) -> Vec<f64> {
        let n = field.len();
        let mut cs_values = vec![0.0; n];

        for i in self.window..n {
            let window_data = &field[i - self.window..i];
            
            // Compute gauge connection A from field
            let a = self.gauge_connection(window_data);
            
            // Compute field strength F = dA
            let f = self.field_strength(&a);
            
            // CS invariant: simplified calculation
            let wedge_product = self.wedge_product(&a, &f);
            let cubic_term = self.cubic_term(&a);
            
            cs_values[i] = (self.coupling / (4.0 * PI)) * (wedge_product + (2.0 / 3.0) * cubic_term);
        }

        cs_values
    }

    fn gauge_connection(&self, data: &[f64]) -> Vec<f64> {
        // Connection as the log-derivative
        let mut conn = vec![0.0; data.len()];
        for i in 1..data.len() {
            if data[i - 1] != 0.0 {
                conn[i] = (data[i] / data[i - 1]).ln();
            }
        }
        conn
    }

    fn field_strength(&self, connection: &[f64]) -> Vec<f64> {
        // Field strength as derivative of connection
        let mut strength = vec![0.0; connection.len()];
        for i in 1..connection.len() {
            strength[i] = connection[i] - connection[i - 1];
        }
        strength
    }

    fn wedge_product(&self, a: &[f64], f: &[f64]) -> f64 {
        // Wedge product A ∧ F
        a.iter().zip(f.iter()).map(|(ai, fi)| ai * fi).sum()
    }

    fn cubic_term(&self, a: &[f64]) -> f64 {
        // Cubic term A ∧ A ∧ A (simplified)
        let sum_a: f64 = a.iter().sum();
        sum_a.powi(3) / (a.len() as f64).powi(2)
    }

    /// Detect topological transitions (sudden CS changes)
    pub fn detect_transitions(&self, cs_values: &[f64], threshold: f64) -> Vec<usize> {
        let mut transitions = Vec::new();
        
        for i in 1..cs_values.len() {
            let change = (cs_values[i] - cs_values[i - 1]).abs();
            if change > threshold {
                transitions.push(i);
            }
        }
        
        transitions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cs_calculator() {
        let calc = ChernSimonsCalculator::new(1.0);
        let field = vec![1.0, 1.1, 1.2, 1.3, 1.4, 1.5];
        let cs = calc.calculate(&field);
        assert_eq!(cs.len(), field.len());
    }

    #[test]
    fn test_transition_detection() {
        let calc = ChernSimonsCalculator::new(1.0);
        let cs_values = vec![0.1, 0.11, 0.12, 0.5, 0.51, 0.52]; // Jump at index 3
        let transitions = calc.detect_transitions(&cs_values, 0.2);
        assert!(!transitions.is_empty());
    }
}
