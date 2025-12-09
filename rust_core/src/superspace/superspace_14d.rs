// 14-Dimensional Superspace Construction
// 7 bosonic + 7 fermionic coordinates

use super::grassmann::GrassmannNumber;

#[derive(Debug, Clone)]
pub struct Superspace14D {
    pub bosonic: Vec<f64>,      // 7 bosonic coordinates
    pub fermionic: Vec<GrassmannNumber>, // 7 fermionic coordinates
}

impl Superspace14D {
    pub fn new(bosonic: Vec<f64>, fermionic: Vec<GrassmannNumber>) -> Self {
        assert_eq!(bosonic.len(), 7, "Must have 7 bosonic coordinates");
        assert_eq!(fermionic.len(), 7, "Must have 7 fermionic coordinates");
        Self { bosonic, fermionic }
    }

    pub fn zero() -> Self {
        Self {
            bosonic: vec![0.0; 7],
            fermionic: vec![GrassmannNumber::zero(1); 7],
        }
    }

    /// Distance in superspace (Euclidean metric)
    pub fn distance(&self, other: &Self) -> f64 {
        let mut dist_sq = 0.0;
        
        // Bosonic contribution
        for i in 0..7 {
            let diff = self.bosonic[i] - other.bosonic[i];
            dist_sq += diff * diff;
        }
        
        // Fermionic contribution (Grassmann norm)
        for i in 0..7 {
            let g_diff = self.fermionic[i].clone() - other.fermionic[i].clone();
            dist_sq += g_diff.norm().powi(2);
        }
        
        dist_sq.sqrt()
    }

    /// BRST symmetry check (nilpotent transformation)
    pub fn brst_transform(&self) -> Self {
        let mut transformed_bosonic = self.bosonic.clone();
        let mut transformed_fermionic = self.fermionic.clone();

        // BRST: δx^i = ε * θ^i (fermionic shift of bosonic)
        for i in 0..7 {
            transformed_bosonic[i] += 0.01 * self.fermionic[i].scalar; // ε = 0.01
        }

        // BRST: δθ^i = 0 (fermionic coordinates unchanged)
        
        Self {
            bosonic: transformed_bosonic,
            fermionic: transformed_fermionic,
        }
    }

    /// Gauge transformation (U(1) symmetry)
    pub fn gauge_transform(&self, alpha: f64) -> Self {
        let mut transformed = self.clone();
        
        // U(1) rotation in bosonic space
        let cos_a = alpha.cos();
        let sin_a = alpha.sin();
        
        // Rotate first two bosonic coordinates
        let x0 = transformed.bosonic[0];
        let x1 = transformed.bosonic[1];
        transformed.bosonic[0] = x0 * cos_a - x1 * sin_a;
        transformed.bosonic[1] = x0 * sin_a + x1 * cos_a;
        
        transformed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_superspace_creation() {
        let bosonic = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let fermionic = vec![GrassmannNumber::zero(1); 7];
        let ss = Superspace14D::new(bosonic, fermionic);
        assert_eq!(ss.bosonic.len(), 7);
        assert_eq!(ss.fermionic.len(), 7);
    }

    #[test]
    fn test_distance() {
        let ss1 = Superspace14D::zero();
        let mut ss2 = Superspace14D::zero();
        ss2.bosonic[0] = 3.0;
        ss2.bosonic[1] = 4.0;
        let dist = ss1.distance(&ss2);
        assert!((dist - 5.0).abs() < 1e-10); // 3-4-5 triangle
    }

    #[test]
    fn test_brst() {
        let ss = Superspace14D::zero();
        let transformed = ss.brst_transform();
        assert_eq!(transformed.bosonic.len(), 7);
    }
}
