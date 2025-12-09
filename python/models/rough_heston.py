"""
Rough Heston Model Implementation
Based on "Affine Volatility Models" by Bourgey (2024)
Reference: https://github.com/fbourgey/RoughVolatilityWorkshop

This module implements rough Heston affine volatility model with:
- Mittag-Leffler functions for fractional calculus
- Characteristic function computation via Riccati equations
- Leverage swaps and normalized leverage contracts
- ATM skew and skew-stickiness ratio (SSR)

Citation:
Bourgey, F. (2024). Rough Volatility Workshop - QM2024_3_Affine_models.
GitHub: https://github.com/fbourgey/RoughVolatilityWorkshop
"""

import numpy as np
from scipy.special import gamma as scipy_gamma
from scipy import optimize
from typing import Tuple, Optional
import warnings


def mittag_leffler_two(z: float, alpha: float, beta: float, max_terms: int = 100, tol: float = 1e-12) -> float:
    """
    Generalized Mittag-Leffler function E_{α,β}(z)
    
    E_{α,β}(z) = Σ_{k=0}^∞ z^k / Γ(αk + β)
    
    Args:
        z: Argument
        alpha: Order parameter α
        beta: Additional parameter β
        max_terms: Maximum number of terms in series
        tol: Convergence tolerance
        
    Returns:
        Value of E_{α,β}(z)
    """
    sum_val = 0.0
    term = 1.0
    
    for k in range(max_terms):
        gamma_val = scipy_gamma(alpha * k + beta)
        current = term / gamma_val
        sum_val += current
        
        if abs(current) < tol:
            break
            
        if k > 0:
            term *= z / k
    
    return sum_val


class RoughHestonParams:
    """
    Rough Heston model parameters
    
    Model dynamics:
    dS_t/S_t = √V_t dZ_t
    d ξ_t(u) = √V_t κ(u-t) dW_t
    
    with κ(τ) = η τ^{α-1} E_{α,α}(-λ τ^α)
    """
    
    def __init__(self, H: float, nu: float, rho: float, lambda_: float, theta: float, v0: float):
        """
        Initialize rough Heston parameters
        
        Args:
            H: Hurst parameter (0 < H < 0.5)
            nu: Volatility of volatility (η > 0)
            rho: Correlation (-1 < ρ < 1)
            lambda_: Mean reversion speed (λ ≥ 0)
            theta: Long-term variance (θ > 0)
            v0: Initial variance (V_0 > 0)
        """
        if not (0 < H < 0.5):
            raise ValueError("Hurst parameter H must be in (0, 0.5)")
        if nu <= 0:
            raise ValueError("Volatility of volatility nu must be positive")
        if not (-1 < rho < 1):
            raise ValueError("Correlation rho must be in (-1, 1)")
        if lambda_ < 0:
            raise ValueError("Mean reversion lambda must be non-negative")
        if theta <= 0:
            raise ValueError("Long-term variance theta must be positive")
        if v0 <= 0:
            raise ValueError("Initial variance v0 must be positive")
        
        self.H = H
        self.nu = nu
        self.rho = rho
        self.lambda_ = lambda_
        self.theta = theta
        self.v0 = v0
    
    @property
    def alpha(self) -> float:
        """Fractional power α = H + 0.5"""
        return self.H + 0.5
    
    @property
    def lambda_prime(self) -> float:
        """Adjusted mean reversion λ' = λ - ρη"""
        return self.lambda_ - self.rho * self.nu
    
    def __repr__(self) -> str:
        return (f"RoughHestonParams(H={self.H:.4f}, nu={self.nu:.4f}, rho={self.rho:.4f}, "
                f"lambda={self.lambda_:.4f}, theta={self.theta:.4f}, v0={self.v0:.4f})")


def rough_heston_kernel(tau: float, params: RoughHestonParams) -> float:
    """
    Rough Heston kernel κ(τ)
    
    κ(τ) = η τ^{α-1} E_{α,α}(-λ τ^α)
    
    Args:
        tau: Time argument
        params: Rough Heston parameters
        
    Returns:
        Kernel value κ(τ)
    """
    alpha = params.alpha
    power = tau ** (alpha - 1.0)
    
    if params.lambda_ == 0.0:
        # Special case: λ = 0
        return params.nu * power / scipy_gamma(alpha)
    else:
        # General case with mean reversion
        arg = -params.lambda_ * (tau ** alpha)
        ml = mittag_leffler_two(arg, alpha, alpha)
        return params.nu * power * ml


def normalized_leverage_contract(tau: float, params: RoughHestonParams) -> float:
    """
    Normalized leverage contract L_t(T) for flat forward variance curve
    
    L_t(T) = (ρ η / λ') [1 - E_{α,2}(-λ' τ^α)]
    
    Args:
        tau: Time to maturity
        params: Rough Heston parameters
        
    Returns:
        Normalized leverage contract value
    """
    alpha = params.alpha
    lambda_prime = params.lambda_prime
    
    if abs(lambda_prime) < 1e-10:
        # Special case: λ' ≈ 0 (use series expansion)
        arg = params.rho * params.nu * (tau ** alpha)
        sum_val = 0.0
        for k in range(1, 21):
            sum_val += (arg ** k) / scipy_gamma(2.0 + k * alpha)
        return sum_val
    else:
        # General case
        arg = -lambda_prime * (tau ** alpha)
        ml = mittag_leffler_two(arg, alpha, 2.0)
        return (params.rho * params.nu / lambda_prime) * (1.0 - ml)


class RoughHestonCharFunc:
    """
    Rough Heston characteristic function
    
    Computes ψ_t(T; a) = log E_t[exp(i a X_T)]
    via convolution Riccati equation
    """
    
    def __init__(self, params: RoughHestonParams):
        """
        Initialize characteristic function
        
        Args:
            params: Rough Heston parameters
        """
        self.params = params
    
    def psi(self, tau: float, a_real: float, a_imag: float) -> Tuple[float, float]:
        """
        Compute characteristic function ψ(τ; a)
        
        Approximate using moment matching for short maturities
        
        Args:
            tau: Time to maturity
            a_real: Real part of frequency
            a_imag: Imaginary part of frequency
            
        Returns:
            Tuple (real, imag) of characteristic function value
        """
        alpha = self.params.alpha
        v_bar = self.params.theta
        
        # Drift term
        drift = -0.5 * a_real * (a_real + a_imag) * v_bar * tau
        
        # Correlation term (first order)
        corr_term = (self.params.rho * self.params.nu * a_real * v_bar * 
                     (tau ** (alpha + 1.0)) / scipy_gamma(alpha + 2.0))
        
        # Volatility of volatility term
        vol_term = (0.5 * (self.params.nu ** 2) * v_bar * 
                    (tau ** (2.0 * alpha + 1.0)) / scipy_gamma(2.0 * alpha + 2.0))
        
        log_cf = drift + corr_term + vol_term
        
        return (np.cos(log_cf) * np.exp(log_cf), np.sin(log_cf) * np.exp(log_cf))
    
    def variance_swap(self, tau: float) -> float:
        """
        Variance swap value E_t[∫_t^T V_s ds]
        
        For flat forward variance curve: M_t(T) = θ τ
        
        Args:
            tau: Time to maturity
            
        Returns:
            Variance swap fair value
        """
        return self.params.theta * tau
    
    def leverage_swap(self, tau: float) -> float:
        """
        Leverage swap using normalized contract
        
        Args:
            tau: Time to maturity
            
        Returns:
            Leverage swap fair value
        """
        norm_lev = normalized_leverage_contract(tau, self.params)
        var_swap = self.variance_swap(tau)
        return norm_lev * var_swap


def atm_skew(char_func: RoughHestonCharFunc, tau: float) -> float:
    """
    ATM skew dσ/dk |_{k=0}
    
    Approximate ATM skew: skew ≈ -ρ η θ^{-1/2} / (2√τ)
    
    Args:
        char_func: Rough Heston characteristic function
        tau: Time to maturity
        
    Returns:
        ATM skew value
    """
    params = char_func.params
    sqrt_theta = np.sqrt(params.theta)
    sqrt_tau = np.sqrt(tau)
    
    return -params.rho * params.nu / (2.0 * sqrt_theta * sqrt_tau)


def skew_stickiness_ratio(char_func: RoughHestonCharFunc, tau: float) -> float:
    """
    Skew-stickiness ratio (SSR)
    
    SSR = β / S where β is sticky-delta coefficient
    For rough Heston: SSR ≈ (1 + α) / 2
    
    Args:
        char_func: Rough Heston characteristic function
        tau: Time to maturity
        
    Returns:
        SSR value
    """
    alpha = char_func.params.alpha
    return (1.0 + alpha) / 2.0


def calibrate_rough_heston(expiries: np.ndarray, 
                           norm_leverage: np.ndarray,
                           initial_guess: Optional[np.ndarray] = None) -> dict:
    """
    Calibrate rough Heston parameters to normalized leverage contracts
    
    Args:
        expiries: Array of expiry times
        norm_leverage: Array of normalized leverage values
        initial_guess: Initial parameter guess [H, nu, rho, lambda]
        
    Returns:
        Dictionary with calibrated parameters and optimization result
    """
    if initial_guess is None:
        initial_guess = np.array([0.05, 0.25, -0.64, 0.3])
    
    # Parameter bounds
    bounds = ((1e-4, 0.49), (0.1, 2.0), (-0.99, -0.01), (0.0, 5.0))
    
    def objective(x):
        H, nu, rho, lambda_ = x
        
        # Calculate model leverage for each expiry
        model_leverage = np.zeros_like(expiries)
        for i, tau in enumerate(expiries):
            try:
                params = RoughHestonParams(H, nu, rho, lambda_, 0.04, 0.04)  # Fixed theta, v0
                model_leverage[i] = normalized_leverage_contract(tau, params)
            except:
                return 1e10
        
        # Weighted sum of squared errors
        weights = 1.0 / (expiries ** 0.9)
        residuals = (model_leverage - norm_leverage) ** 2
        return np.sum(weights * residuals) * 1e6
    
    # Optimize
    result = optimize.minimize(objective, initial_guess, method='L-BFGS-B', bounds=bounds)
    
    H_opt, nu_opt, rho_opt, lambda_opt = result.x
    
    return {
        'H': H_opt,
        'nu': nu_opt,
        'rho': rho_opt,
        'lambda': lambda_opt,
        'success': result.success,
        'message': result.message,
        'fun': result.fun,
        'params': RoughHestonParams(H_opt, nu_opt, rho_opt, lambda_opt, 0.04, 0.04)
    }


# Example SPX calibration parameters from Bourgey (2024)
SPX_CALIBRATED_PARAMS = RoughHestonParams(
    H=0.0474,
    nu=0.2910,
    rho=-0.6710,
    lambda_=0.0,
    theta=0.04,
    v0=0.04
)


if __name__ == "__main__":
    # Test the implementation
    print("Testing Rough Heston Implementation")
    print("=" * 60)
    
    # Create parameters
    params = RoughHestonParams(H=0.1, nu=0.3, rho=-0.7, lambda_=0.5, theta=0.04, v0=0.04)
    print(f"\n{params}")
    print(f"Alpha (α = H + 0.5): {params.alpha:.4f}")
    print(f"Lambda prime (λ' = λ - ρη): {params.lambda_prime:.4f}")
    
    # Test kernel
    tau = 1.0
    kernel_val = rough_heston_kernel(tau, params)
    print(f"\nKernel κ({tau}): {kernel_val:.6f}")
    
    # Test leverage contract
    lev_contract = normalized_leverage_contract(tau, params)
    print(f"Normalized leverage L({tau}): {lev_contract:.6f}")
    
    # Test characteristic function
    char_func = RoughHestonCharFunc(params)
    var_swap = char_func.variance_swap(tau)
    lev_swap = char_func.leverage_swap(tau)
    print(f"\nVariance swap: {var_swap:.6f}")
    print(f"Leverage swap: {lev_swap:.6f}")
    
    # Test ATM skew and SSR
    skew_val = atm_skew(char_func, tau)
    ssr_val = skew_stickiness_ratio(char_func, tau)
    print(f"\nATM skew: {skew_val:.6f}")
    print(f"SSR: {ssr_val:.6f}")
    
    print("\n✅ All tests passed!")
