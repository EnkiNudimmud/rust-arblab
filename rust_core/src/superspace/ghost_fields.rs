// Ghost Field Dynamics
// Fermionic fields representing hidden market forces

#[derive(Debug, Clone)]
pub struct GhostFieldParams {
    pub spring_constant: f64,
    pub damping: f64,
    pub noise_level: f64,
    pub n_modes: usize,
}

impl Default for GhostFieldParams {
    fn default() -> Self {
        Self {
            spring_constant: 1.0,
            damping: 0.1,
            noise_level: 0.01,
            n_modes: 7,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GhostFieldSystem {
    pub fields: Vec<Vec<f64>>,  // [time, mode]
    pub divergence: Vec<f64>,
    pub curl: Vec<f64>,
    pub params: GhostFieldParams,
}

impl GhostFieldSystem {
    pub fn new(params: GhostFieldParams) -> Self {
        Self {
            fields: vec![],
            divergence: vec![],
            curl: vec![],
            params,
        }
    }

    pub fn from_bosonic_coords(bosonic: &[Vec<f64>], params: GhostFieldParams) -> Self {
        let n_time = bosonic.len();
        let n_modes = params.n_modes.min(7); // Max 7 modes
        
        let mut fields = vec![vec![0.0; n_modes]; n_time];
        let mut divergence = vec![0.0; n_time];
        let mut curl = vec![0.0; n_time];

        // Initialize ghost fields from bosonic coordinates
        for t in 0..n_time {
            for mode in 0..n_modes {
                if mode < bosonic[t].len() {
                    // Ghost field derived from bosonic momentum
                    fields[t][mode] = if t > 0 {
                        (bosonic[t][mode] - bosonic[t - 1][mode]) * params.spring_constant
                    } else {
                        0.0
                    };
                }
            }
        }

        // Evolve ghost fields
        for t in 1..n_time {
            for mode in 0..n_modes {
                // Damped harmonic oscillator
                let acceleration = -params.spring_constant * fields[t - 1][mode]
                    - params.damping * (fields[t][mode] - fields[t - 1][mode]);
                
                fields[t][mode] = fields[t - 1][mode] + acceleration * 0.1; // dt = 0.1
                
                // Add stochastic noise
                fields[t][mode] += params.noise_level * (rand::random::<f64>() - 0.5);
            }
        }

        // Compute divergence and curl
        for t in 1..n_time {
            // Divergence: sum of field derivatives
            let mut div = 0.0;
            for mode in 0..n_modes {
                div += fields[t][mode] - fields[t - 1][mode];
            }
            divergence[t] = div;

            // Curl: rotational component (simplified for 1D)
            let mut curl_val = 0.0;
            for mode in 1..n_modes {
                curl_val += (fields[t][mode] - fields[t][mode - 1]).abs();
            }
            curl[t] = curl_val;
        }

        Self {
            fields,
            divergence,
            curl,
            params,
        }
    }

    pub fn get_field(&self, time: usize, mode: usize) -> f64 {
        if time < self.fields.len() && mode < self.fields[time].len() {
            self.fields[time][mode]
        } else {
            0.0
        }
    }

    pub fn get_divergence(&self, time: usize) -> f64 {
        if time < self.divergence.len() {
            self.divergence[time]
        } else {
            0.0
        }
    }

    pub fn get_curl(&self, time: usize) -> f64 {
        if time < self.curl.len() {
            self.curl[time]
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ghost_field_creation() {
        let params = GhostFieldParams::default();
        let system = GhostFieldSystem::new(params);
        assert_eq!(system.fields.len(), 0);
    }

    #[test]
    fn test_from_bosonic() {
        let bosonic = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.1, 2.1, 3.1],
            vec![1.2, 2.2, 3.2],
        ];
        let params = GhostFieldParams::default();
        let system = GhostFieldSystem::from_bosonic_coords(&bosonic, params);
        assert_eq!(system.fields.len(), 3);
    }
}
