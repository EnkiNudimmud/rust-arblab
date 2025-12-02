// Rough Heston Model Implementation
// Based on "Which Free Lunch Would You Like Today, Sir?" and QM2024_3_Affine_models.ipynb
// Reference: Bourgey, F. (2024). Rough Volatility Workshop - Affine Models

/// Mittag-Leffler function E_{α,β}(z) using series expansion
/// For rough Heston kernel calculations
pub fn mittag_leffler_two(z: f64, alpha: f64, beta: f64) -> f64 {
    let max_terms = 100;
    let tol = 1e-12;
    
    let mut sum = 0.0;
    let mut term = 1.0;
    
    for k in 0..max_terms {
        let k_f = k as f64;
        // term = z^k / Gamma(alpha * k + beta)
        if k > 0 {
            term *= z / k_f;
        }
        
        let gamma_val = gamma(alpha * k_f + beta);
        let current = term / gamma_val;
        sum += current;
        
        if current.abs() < tol {
            break;
        }
    }
    
    sum
}

/// Gamma function approximation using Lanczos approximation
pub fn gamma(z: f64) -> f64 {
    if z <= 0.0 {
        return f64::INFINITY;
    }
    
    // Use libm gamma function
    libm::tgamma(z)
}

/// Rough Heston parameters
#[derive(Debug, Clone)]
pub struct RoughHestonParams {
    pub h: f64,           // Hurst parameter (0 < H < 0.5)
    pub nu: f64,          // Volatility of volatility
    pub rho: f64,         // Correlation (-1 < rho < 1)
    pub lambda: f64,      // Mean reversion speed
    pub theta: f64,       // Long-term variance
    pub v0: f64,          // Initial variance
}

impl RoughHestonParams {
    /// Create new rough Heston parameters with validation
    pub fn new(h: f64, nu: f64, rho: f64, lambda: f64, theta: f64, v0: f64) -> Result<Self, String> {
        if h <= 0.0 || h >= 0.5 {
            return Err("Hurst parameter H must be in (0, 0.5)".to_string());
        }
        if nu <= 0.0 {
            return Err("Volatility of volatility nu must be positive".to_string());
        }
        if rho <= -1.0 || rho >= 1.0 {
            return Err("Correlation rho must be in (-1, 1)".to_string());
        }
        if lambda < 0.0 {
            return Err("Mean reversion lambda must be non-negative".to_string());
        }
        if theta <= 0.0 {
            return Err("Long-term variance theta must be positive".to_string());
        }
        if v0 <= 0.0 {
            return Err("Initial variance v0 must be positive".to_string());
        }
        
        Ok(RoughHestonParams {
            h,
            nu,
            rho,
            lambda,
            theta,
            v0,
        })
    }
    
    /// Get alpha = H + 0.5 (fractional power)
    pub fn alpha(&self) -> f64 {
        self.h + 0.5
    }
    
    /// Get lambda' = lambda - rho * nu
    pub fn lambda_prime(&self) -> f64 {
        self.lambda - self.rho * self.nu
    }
}

/// Rough Heston kernel κ(τ) = η * τ^(α-1) * E_{α,α}(-λ * τ^α)
pub fn rough_heston_kernel(tau: f64, params: &RoughHestonParams) -> f64 {
    let alpha = params.alpha();
    let power = tau.powf(alpha - 1.0);
    
    if params.lambda == 0.0 {
        // Special case: λ = 0
        params.nu * power / gamma(alpha)
    } else {
        // General case with mean reversion
        let arg = -params.lambda * tau.powf(alpha);
        let ml = mittag_leffler_two(arg, alpha, alpha);
        params.nu * power * ml
    }
}

/// Normalized leverage contract for rough Heston with flat forward variance
/// L_t(T) = (ρ * ν / λ') * [1 - E_{α,2}(-λ' * τ^α)]
pub fn normalized_leverage_contract(
    tau: f64,
    params: &RoughHestonParams,
) -> f64 {
    let alpha = params.alpha();
    let lambda_prime = params.lambda_prime();
    
    if lambda_prime.abs() < 1e-10 {
        // Special case: λ' ≈ 0 (use series expansion)
        let arg = params.rho * params.nu * tau.powf(alpha);
        let mut sum = 0.0;
        for k in 1..=20 {
            let k_f = k as f64;
            sum += arg.powi(k as i32) / gamma(2.0 + k_f * alpha);
        }
        sum
    } else {
        // General case
        let arg = -lambda_prime * tau.powf(alpha);
        let ml = mittag_leffler_two(arg, alpha, 2.0);
        (params.rho * params.nu / lambda_prime) * (1.0 - ml)
    }
}

/// Rough Heston characteristic function via Riccati equation
/// ψ_t(T; a) = log E_t[exp(i*a*X_T)]
pub struct RoughHestonCharFunc {
    params: RoughHestonParams,
}

impl RoughHestonCharFunc {
    pub fn new(params: RoughHestonParams) -> Self {
        RoughHestonCharFunc { params }
    }
    
    /// Compute characteristic function for given maturity and frequency
    /// Returns complex value (real, imag)
    pub fn psi(&self, tau: f64, a_real: f64, a_imag: f64) -> (f64, f64) {
        // For now, implement simplified version
        // Full implementation requires solving convolution Riccati equation
        
        // Approximate using moment matching for short maturities
        let alpha = self.params.alpha();
        let v_bar = self.params.theta;
        
        // Drift term
        let drift = -0.5 * a_real * (a_real + a_imag) * v_bar * tau;
        
        // Correlation term (first order)
        let corr_term = self.params.rho * self.params.nu * a_real * v_bar * tau.powf(alpha + 1.0) 
            / gamma(alpha + 2.0);
        
        // Volatility of volatility term
        let vol_term = 0.5 * self.params.nu.powi(2) * v_bar * tau.powf(2.0 * alpha + 1.0) 
            / gamma(2.0 * alpha + 2.0);
        
        let log_cf = drift + corr_term + vol_term;
        
        (log_cf.cos() * log_cf.exp(), log_cf.sin() * log_cf.exp())
    }
    
    /// Variance swap value E_t[∫_t^T V_s ds]
    pub fn variance_swap(&self, tau: f64) -> f64 {
        // For flat forward variance curve: M_t(T) = θ * τ
        self.params.theta * tau
    }
    
    /// Leverage swap using normalized contract
    pub fn leverage_swap(&self, tau: f64) -> f64 {
        let norm_lev = normalized_leverage_contract(tau, &self.params);
        let var_swap = self.variance_swap(tau);
        norm_lev * var_swap
    }
}

/// ATM skew calculation using characteristic function
pub fn atm_skew(char_func: &RoughHestonCharFunc, tau: f64) -> f64 {
    // Approximate ATM skew = dσ/dk |_{k=0}
    // Using leverage formula: skew ≈ -ρ * η * θ^(-1/2) / (2*sqrt(τ))
    
    let params = &char_func.params;
    let sqrt_theta = params.theta.sqrt();
    let sqrt_tau = tau.sqrt();
    
    -params.rho * params.nu / (2.0 * sqrt_theta * sqrt_tau)
}

/// Skew-stickiness ratio (SSR)
pub fn skew_stickiness_ratio(char_func: &RoughHestonCharFunc, _tau: f64) -> f64 {
    // SSR = β / S where β is the sticky-delta coefficient
    // For rough Heston: SSR ≈ (1 + α) / 2
    
    let alpha = char_func.params.alpha();
    (1.0 + alpha) / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_mittag_leffler() {
        // E_{0.5, 1}(0) = 1
        let ml = mittag_leffler_two(0.0, 0.5, 1.0);
        assert!((ml - 1.0).abs() < 1e-10);
        
        // E_{1, 1}(x) = exp(x)
        let x = 0.5;
        let ml = mittag_leffler_two(x, 1.0, 1.0);
        assert!((ml - x.exp()).abs() < 1e-6);
    }
    
    #[test]
    fn test_rough_heston_params() {
        let params = RoughHestonParams::new(
            0.1,    // H
            0.3,    // nu
            -0.7,   // rho
            0.5,    // lambda
            0.04,   // theta
            0.04,   // v0
        ).unwrap();
        
        assert_eq!(params.alpha(), 0.6);
        assert!((params.lambda_prime() - 0.71).abs() < 1e-10);
    }
    
    #[test]
    fn test_leverage_contract() {
        let params = RoughHestonParams::new(
            0.1, 0.3, -0.7, 0.5, 0.04, 0.04
        ).unwrap();
        
        let tau = 1.0;
        let lev = normalized_leverage_contract(tau, &params);
        
        // Leverage should be negative (negative correlation)
        assert!(lev < 0.0);
        
        // Reasonable magnitude
        assert!(lev.abs() < 1.0);
    }
    
    #[test]
    fn test_char_func() {
        let params = RoughHestonParams::new(
            0.1, 0.3, -0.7, 0.5, 0.04, 0.04
        ).unwrap();
        
        let char_func = RoughHestonCharFunc::new(params);
        
        // ψ(τ; 0) = 0 (zero frequency)
        let (re, im) = char_func.psi(1.0, 0.0, 0.0);
        assert!((re - 1.0).abs() < 1e-6);
        assert!(im.abs() < 1e-6);
    }
    
    #[test]
    fn test_variance_swap() {
        let params = RoughHestonParams::new(
            0.1, 0.3, -0.7, 0.0, 0.04, 0.04
        ).unwrap();
        
        let char_func = RoughHestonCharFunc::new(params);
        let var_swap = char_func.variance_swap(1.0);
        
        // For flat curve: var_swap = θ * τ
        assert!((var_swap - 0.04).abs() < 1e-10);
    }
    
    #[test]
    fn test_atm_skew() {
        let params = RoughHestonParams::new(
            0.1, 0.3, -0.7, 0.0, 0.04, 0.04
        ).unwrap();
        
        let char_func = RoughHestonCharFunc::new(params);
        let skew = atm_skew(&char_func, 1.0);
        
        // Skew should be positive (negative rho)
        assert!(skew > 0.0);
    }
}
