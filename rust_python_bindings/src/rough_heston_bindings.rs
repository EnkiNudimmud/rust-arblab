// Python bindings for rough Heston model
// Reference: QM2024_3_Affine_models.ipynb (Bourgey, F. 2024)

use pyo3::prelude::*;
use rust_core::rough_heston::*;

/// Python wrapper for rough Heston parameters
#[pyclass]
#[derive(Clone)]
pub struct PyRoughHestonParams {
    inner: RoughHestonParams,
}

#[pymethods]
impl PyRoughHestonParams {
    /// Create new rough Heston parameters
    /// 
    /// Args:
    ///     h: Hurst parameter (0 < H < 0.5)
    ///     nu: Volatility of volatility (nu > 0)
    ///     rho: Correlation (-1 < rho < 1)
    ///     lambda_: Mean reversion speed (lambda >= 0)
    ///     theta: Long-term variance (theta > 0)
    ///     v0: Initial variance (v0 > 0)
    #[new]
    fn new(h: f64, nu: f64, rho: f64, lambda_: f64, theta: f64, v0: f64) -> PyResult<Self> {
        RoughHestonParams::new(h, nu, rho, lambda_, theta, v0)
            .map(|inner| PyRoughHestonParams { inner })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))
    }
    
    /// Get alpha = H + 0.5
    fn alpha(&self) -> f64 {
        self.inner.alpha()
    }
    
    /// Get lambda' = lambda - rho * nu
    fn lambda_prime(&self) -> f64 {
        self.inner.lambda_prime()
    }
    
    #[getter]
    fn h(&self) -> f64 {
        self.inner.h
    }
    
    #[getter]
    fn nu(&self) -> f64 {
        self.inner.nu
    }
    
    #[getter]
    fn rho(&self) -> f64 {
        self.inner.rho
    }
    
    #[getter]
    fn lambda(&self) -> f64 {
        self.inner.lambda
    }
    
    #[getter]
    fn theta(&self) -> f64 {
        self.inner.theta
    }
    
    #[getter]
    fn v0(&self) -> f64 {
        self.inner.v0
    }
    
    fn __repr__(&self) -> String {
        format!(
            "RoughHestonParams(H={:.4}, nu={:.4}, rho={:.4}, lambda={:.4}, theta={:.4}, v0={:.4})",
            self.inner.h, self.inner.nu, self.inner.rho, 
            self.inner.lambda, self.inner.theta, self.inner.v0
        )
    }
}

/// Python wrapper for rough Heston characteristic function
#[pyclass]
pub struct PyRoughHestonCharFunc {
    inner: RoughHestonCharFunc,
}

#[pymethods]
impl PyRoughHestonCharFunc {
    /// Create new rough Heston characteristic function
    ///
    /// Args:
    ///     params: RoughHestonParams object
    #[new]
    fn new(params: &PyRoughHestonParams) -> Self {
        PyRoughHestonCharFunc {
            inner: RoughHestonCharFunc::new(params.inner.clone()),
        }
    }
    
    /// Compute characteristic function ψ(τ; a)
    ///
    /// Args:
    ///     tau: Time to maturity
    ///     a_real: Real part of frequency
    ///     a_imag: Imaginary part of frequency
    ///
    /// Returns:
    ///     Tuple (real, imag) of characteristic function value
    fn psi(&self, tau: f64, a_real: f64, a_imag: f64) -> (f64, f64) {
        self.inner.psi(tau, a_real, a_imag)
    }
    
    /// Compute variance swap value
    ///
    /// Args:
    ///     tau: Time to maturity
    ///
    /// Returns:
    ///     Variance swap fair value
    fn variance_swap(&self, tau: f64) -> f64 {
        self.inner.variance_swap(tau)
    }
    
    /// Compute leverage swap value
    ///
    /// Args:
    ///     tau: Time to maturity
    ///
    /// Returns:
    ///     Leverage swap fair value
    fn leverage_swap(&self, tau: f64) -> f64 {
        self.inner.leverage_swap(tau)
    }
}

/// Mittag-Leffler function E_{α,β}(z)
/// 
/// Generalized Mittag-Leffler function used in rough Heston kernel
///
/// Args:
///     z: Argument
///     alpha: Order parameter α
///     beta: Additional parameter β
///
/// Returns:
///     Value of E_{α,β}(z)
#[pyfunction]
fn mittag_leffler_two(z: f64, alpha: f64, beta: f64) -> f64 {
    rust_core::rough_heston::mittag_leffler_two(z, alpha, beta)
}

/// Gamma function Γ(z)
///
/// Args:
///     z: Argument
///
/// Returns:
///     Value of Γ(z)
#[pyfunction]
fn gamma(z: f64) -> f64 {
    rust_core::rough_heston::gamma(z)
}

/// Rough Heston kernel κ(τ)
///
/// Kernel function for rough Heston model:
/// κ(τ) = η * τ^(α-1) * E_{α,α}(-λ * τ^α)
///
/// Args:
///     tau: Time argument
///     params: RoughHestonParams object
///
/// Returns:
///     Kernel value κ(τ)
#[pyfunction]
fn rough_heston_kernel(tau: f64, params: &PyRoughHestonParams) -> f64 {
    rust_core::rough_heston::rough_heston_kernel(tau, &params.inner)
}

/// Normalized leverage contract L_t(T)
///
/// For flat forward variance curve:
/// L_t(T) = (ρ * ν / λ') * [1 - E_{α,2}(-λ' * τ^α)]
///
/// Args:
///     tau: Time to maturity
///     params: RoughHestonParams object
///
/// Returns:
///     Normalized leverage contract value
#[pyfunction]
fn normalized_leverage_contract(tau: f64, params: &PyRoughHestonParams) -> f64 {
    rust_core::rough_heston::normalized_leverage_contract(tau, &params.inner)
}

/// ATM skew dσ/dk |_{k=0}
///
/// Approximate ATM implied volatility skew
///
/// Args:
///     char_func: RoughHestonCharFunc object
///     tau: Time to maturity
///
/// Returns:
///     ATM skew value
#[pyfunction]
fn atm_skew(char_func: &PyRoughHestonCharFunc, tau: f64) -> f64 {
    rust_core::rough_heston::atm_skew(&char_func.inner, tau)
}

/// Skew-stickiness ratio (SSR)
///
/// SSR = β / S where β is sticky-delta coefficient
///
/// Args:
///     char_func: RoughHestonCharFunc object
///     tau: Time to maturity
///
/// Returns:
///     SSR value
#[pyfunction]
fn skew_stickiness_ratio(char_func: &PyRoughHestonCharFunc, tau: f64) -> f64 {
    rust_core::rough_heston::skew_stickiness_ratio(&char_func.inner, tau)
}

/// Register rough Heston module
pub fn register_rough_heston(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent_module.py(), "rough_heston")?;
    
    m.add_class::<PyRoughHestonParams>()?;
    m.add_class::<PyRoughHestonCharFunc>()?;
    m.add_function(wrap_pyfunction!(mittag_leffler_two, &m)?)?;
    m.add_function(wrap_pyfunction!(gamma, &m)?)?;
    m.add_function(wrap_pyfunction!(rough_heston_kernel, &m)?)?;
    m.add_function(wrap_pyfunction!(normalized_leverage_contract, &m)?)?;
    m.add_function(wrap_pyfunction!(atm_skew, &m)?)?;
    m.add_function(wrap_pyfunction!(skew_stickiness_ratio, &m)?)?;
    
    parent_module.add_submodule(&m)?;
    Ok(())
}
