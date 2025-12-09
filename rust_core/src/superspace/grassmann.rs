// Grassmann Numbers (anticommuting fermionic coordinates)

use std::ops::{Add, Sub, Mul};

#[derive(Debug, Clone)]
pub struct GrassmannNumber {
    pub scalar: f64,
    pub grassmann: Vec<f64>,
}

impl GrassmannNumber {
    pub fn new(scalar: f64, grassmann: Vec<f64>) -> Self {
        Self { scalar, grassmann }
    }

    pub fn zero(dim: usize) -> Self {
        Self {
            scalar: 0.0,
            grassmann: vec![0.0; dim],
        }
    }

    pub fn from_scalar(scalar: f64) -> Self {
        Self {
            scalar,
            grassmann: vec![],
        }
    }

    pub fn norm(&self) -> f64 {
        let grass_norm: f64 = self.grassmann.iter().map(|x| x * x).sum();
        (self.scalar * self.scalar + grass_norm).sqrt()
    }
}

impl Add for GrassmannNumber {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let mut result_grass = self.grassmann.clone();
        for (i, &val) in other.grassmann.iter().enumerate() {
            if i < result_grass.len() {
                result_grass[i] += val;
            } else {
                result_grass.push(val);
            }
        }
        Self {
            scalar: self.scalar + other.scalar,
            grassmann: result_grass,
        }
    }
}

impl Sub for GrassmannNumber {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let mut result_grass = self.grassmann.clone();
        for (i, &val) in other.grassmann.iter().enumerate() {
            if i < result_grass.len() {
                result_grass[i] -= val;
            } else {
                result_grass.push(-val);
            }
        }
        Self {
            scalar: self.scalar - other.scalar,
            grassmann: result_grass,
        }
    }
}

impl Mul for GrassmannNumber {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        // Grassmann multiplication: θ₁ * θ₂ = -θ₂ * θ₁ (anticommutative)
        // θ * θ = 0 (nilpotent)
        let scalar_part = self.scalar * other.scalar;
        
        // Cross terms: scalar * grassmann
        let mut result_grass = vec![0.0; self.grassmann.len().max(other.grassmann.len())];
        for (i, &val) in self.grassmann.iter().enumerate() {
            result_grass[i] += val * other.scalar;
        }
        for (i, &val) in other.grassmann.iter().enumerate() {
            if i < result_grass.len() {
                result_grass[i] += self.scalar * val;
            }
        }

        Self {
            scalar: scalar_part,
            grassmann: result_grass,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grassmann_addition() {
        let g1 = GrassmannNumber::new(1.0, vec![0.5]);
        let g2 = GrassmannNumber::new(2.0, vec![0.3]);
        let result = g1 + g2;
        assert_eq!(result.scalar, 3.0);
        assert_eq!(result.grassmann[0], 0.8);
    }

    #[test]
    fn test_grassmann_multiplication() {
        let g1 = GrassmannNumber::new(2.0, vec![0.5]);
        let g2 = GrassmannNumber::new(3.0, vec![0.0]);
        let result = g1 * g2;
        assert_eq!(result.scalar, 6.0);
        assert_eq!(result.grassmann[0], 1.5);
    }
}
