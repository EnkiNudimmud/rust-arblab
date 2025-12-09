// rust_core/src/options.rs
//! Options pricing and Greeks calculations
//! Implements Black-Scholes model for vanilla European options

use std::f64::consts::PI;

/// Standard normal cumulative distribution function
pub fn norm_cdf(x: f64) -> f64 {
    0.5 * (1.0 + libm::erf(x / std::f64::consts::SQRT_2))
}

/// Standard normal probability density function
pub fn norm_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
}

/// Black-Scholes option pricing and Greeks
#[derive(Debug, Clone)]
pub struct BlackScholesOption {
    pub spot: f64,          // Current stock price S
    pub strike: f64,        // Strike price K
    pub time_to_expiry: f64, // Time to expiration T-t (years)
    pub risk_free_rate: f64, // Risk-free rate r
    pub dividend_yield: f64, // Continuous dividend yield D
    pub volatility: f64,     // Volatility σ
    pub is_call: bool,       // true for call, false for put
}

impl BlackScholesOption {
    pub fn new(
        spot: f64,
        strike: f64,
        time_to_expiry: f64,
        risk_free_rate: f64,
        dividend_yield: f64,
        volatility: f64,
        is_call: bool,
    ) -> Self {
        Self {
            spot,
            strike,
            time_to_expiry,
            risk_free_rate,
            dividend_yield,
            volatility,
            is_call,
        }
    }

    /// Calculate d1 parameter
    pub fn d1(&self) -> f64 {
        let numerator = (self.spot / self.strike).ln()
            + (self.risk_free_rate - self.dividend_yield + 0.5 * self.volatility.powi(2))
                * self.time_to_expiry;
        let denominator = self.volatility * self.time_to_expiry.sqrt();
        numerator / denominator
    }

    /// Calculate d2 parameter
    pub fn d2(&self) -> f64 {
        self.d1() - self.volatility * self.time_to_expiry.sqrt()
    }

    /// Calculate option price using Black-Scholes formula
    pub fn price(&self) -> f64 {
        if self.time_to_expiry <= 0.0 {
            // At expiry
            return if self.is_call {
                (self.spot - self.strike).max(0.0)
            } else {
                (self.strike - self.spot).max(0.0)
            };
        }

        let d1 = self.d1();
        let d2 = self.d2();
        
        let discount_factor = (-self.risk_free_rate * self.time_to_expiry).exp();
        let dividend_discount = (-self.dividend_yield * self.time_to_expiry).exp();

        if self.is_call {
            self.spot * dividend_discount * norm_cdf(d1)
                - self.strike * discount_factor * norm_cdf(d2)
        } else {
            self.strike * discount_factor * norm_cdf(-d2)
                - self.spot * dividend_discount * norm_cdf(-d1)
        }
    }

    /// Calculate Delta: ∂V/∂S
    pub fn delta(&self) -> f64 {
        if self.time_to_expiry <= 0.0 {
            return if self.is_call {
                if self.spot > self.strike { 1.0 } else { 0.0 }
            } else {
                if self.spot < self.strike { -1.0 } else { 0.0 }
            };
        }

        let d1 = self.d1();
        let dividend_discount = (-self.dividend_yield * self.time_to_expiry).exp();

        if self.is_call {
            dividend_discount * norm_cdf(d1)
        } else {
            -dividend_discount * norm_cdf(-d1)
        }
    }

    /// Calculate Gamma: ∂²V/∂S²
    pub fn gamma(&self) -> f64 {
        if self.time_to_expiry <= 0.0 {
            return 0.0;
        }

        let d1 = self.d1();
        let dividend_discount = (-self.dividend_yield * self.time_to_expiry).exp();
        
        dividend_discount * norm_pdf(d1) / (self.spot * self.volatility * self.time_to_expiry.sqrt())
    }

    /// Calculate Vega: ∂V/∂σ (per 1% volatility change)
    pub fn vega(&self) -> f64 {
        if self.time_to_expiry <= 0.0 {
            return 0.0;
        }

        let d1 = self.d1();
        let dividend_discount = (-self.dividend_yield * self.time_to_expiry).exp();
        
        // Return vega per 1% (divide by 100)
        self.spot * dividend_discount * norm_pdf(d1) * self.time_to_expiry.sqrt() / 100.0
    }

    /// Calculate Theta: -∂V/∂t (per day)
    pub fn theta(&self) -> f64 {
        if self.time_to_expiry <= 0.0 {
            return 0.0;
        }

        let d1 = self.d1();
        let d2 = self.d2();
        let sqrt_t = self.time_to_expiry.sqrt();
        let discount_factor = (-self.risk_free_rate * self.time_to_expiry).exp();
        let dividend_discount = (-self.dividend_yield * self.time_to_expiry).exp();

        let term1 = -(self.spot * dividend_discount * norm_pdf(d1) * self.volatility) 
            / (2.0 * sqrt_t);

        let theta_annual = if self.is_call {
            term1
                + self.dividend_yield * self.spot * dividend_discount * norm_cdf(d1)
                - self.risk_free_rate * self.strike * discount_factor * norm_cdf(d2)
        } else {
            term1
                - self.dividend_yield * self.spot * dividend_discount * norm_cdf(-d1)
                + self.risk_free_rate * self.strike * discount_factor * norm_cdf(-d2)
        };

        // Return theta per day (divide by 365)
        theta_annual / 365.0
    }

    /// Calculate Rho: ∂V/∂r (per 1% rate change)
    pub fn rho(&self) -> f64 {
        if self.time_to_expiry <= 0.0 {
            return 0.0;
        }

        let d2 = self.d2();
        let discount_factor = (-self.risk_free_rate * self.time_to_expiry).exp();

        let rho_annual = if self.is_call {
            self.strike * self.time_to_expiry * discount_factor * norm_cdf(d2)
        } else {
            -self.strike * self.time_to_expiry * discount_factor * norm_cdf(-d2)
        };

        // Return rho per 1% (divide by 100)
        rho_annual / 100.0
    }

    /// Get all Greeks at once
    pub fn greeks(&self) -> Greeks {
        Greeks {
            delta: self.delta(),
            gamma: self.gamma(),
            theta: self.theta(),
            vega: self.vega(),
            rho: self.rho(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Greeks {
    pub delta: f64,
    pub gamma: f64,
    pub theta: f64,
    pub vega: f64,
    pub rho: f64,
}

/// Delta hedging profit calculator
pub struct DeltaHedgingStrategy {
    pub option: BlackScholesOption,
    pub implied_vol: f64,      // Market implied volatility σ̃
    pub actual_vol: f64,       // Actual realized volatility σ
    pub hedging_vol: f64,      // Volatility used for delta hedging
}

impl DeltaHedgingStrategy {
    pub fn new(
        spot: f64,
        strike: f64,
        time_to_expiry: f64,
        risk_free_rate: f64,
        dividend_yield: f64,
        implied_vol: f64,
        actual_vol: f64,
        hedging_vol: f64,
        is_call: bool,
    ) -> Self {
        let option = BlackScholesOption::new(
            spot,
            strike,
            time_to_expiry,
            risk_free_rate,
            dividend_yield,
            actual_vol, // Use actual vol for the "true" option value
            is_call,
        );

        Self {
            option,
            implied_vol,
            actual_vol,
            hedging_vol,
        }
    }

    /// Calculate guaranteed profit when hedging with actual volatility
    /// Profit = V(S,t;σ_actual) - V(S,t;σ_implied)
    pub fn guaranteed_profit_actual_hedge(&self) -> f64 {
        let actual_price = self.option.price();
        
        let mut implied_option = self.option.clone();
        implied_option.volatility = self.implied_vol;
        let implied_price = implied_option.price();

        actual_price - implied_price
    }

    /// Calculate instantaneous mark-to-market P&L when hedging with implied volatility
    /// dP&L = 0.5 * (σ_actual² - σ_implied²) * S² * Γ * dt
    pub fn mtm_pnl_implied_hedge(&self, dt: f64) -> f64 {
        let mut implied_option = self.option.clone();
        implied_option.volatility = self.implied_vol;
        let gamma = implied_option.gamma();

        0.5 * (self.actual_vol.powi(2) - self.implied_vol.powi(2))
            * self.option.spot.powi(2)
            * gamma
            * dt
    }

    /// Calculate instantaneous mark-to-market P&L for custom hedging volatility
    /// dP&L = 0.5 * (σ_actual² - σ_hedge²) * S² * Γ_hedge * dt
    ///     + (Δ_implied - Δ_hedge) * (μ - r + D) * S * dt
    ///     + (Δ_implied - Δ_hedge) * σ_actual * S * dX
    pub fn mtm_pnl_custom_hedge(&self, dt: f64) -> f64 {
        let mut hedge_option = self.option.clone();
        hedge_option.volatility = self.hedging_vol;
        let gamma_hedge = hedge_option.gamma();

        0.5 * (self.actual_vol.powi(2) - self.hedging_vol.powi(2))
            * self.option.spot.powi(2)
            * gamma_hedge
            * dt
    }

    /// Expected total profit when hedging with implied volatility
    /// E[Profit] = 0.5 * (σ² - σ̃²) * ∫₀ᵀ e^(-r(t-t₀)) S² Γ dt
    /// This is path-dependent and requires simulation or integration
    pub fn expected_profit_simulation(
        &self,
        num_simulations: usize,
        num_steps: usize,
        drift: f64, // μ - stock drift
    ) -> (f64, f64) {
        let dt = self.option.time_to_expiry / num_steps as f64;
        let mut profits = Vec::with_capacity(num_simulations);

        for _ in 0..num_simulations {
            let mut s = self.option.spot;
            let mut t = 0.0;
            let mut total_pnl = 0.0;

            for _ in 0..num_steps {
                // Create option at current state
                let mut current_option = self.option.clone();
                current_option.spot = s;
                current_option.time_to_expiry = self.option.time_to_expiry - t;
                current_option.volatility = self.implied_vol;

                // Calculate gamma at current spot
                let gamma = current_option.gamma();

                // Instantaneous P&L
                let pnl = 0.5 * (self.actual_vol.powi(2) - self.implied_vol.powi(2))
                    * s.powi(2)
                    * gamma
                    * dt;

                let discount = (-self.option.risk_free_rate * t).exp();
                total_pnl += pnl * discount;

                // Simulate stock price movement
                let z: f64 = rand::random::<f64>() * 2.0 - 1.0; // Simple random
                let ds = s * (drift * dt + self.actual_vol * dt.sqrt() * z);
                s += ds;
                t += dt;
            }

            profits.push(total_pnl);
        }

        // Calculate mean and std dev
        let mean = profits.iter().sum::<f64>() / profits.len() as f64;
        let variance = profits.iter()
            .map(|p| (p - mean).powi(2))
            .sum::<f64>() / profits.len() as f64;
        
        (mean, variance.sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atm_call_price() {
        let option = BlackScholesOption::new(
            100.0, // spot
            100.0, // strike (ATM)
            1.0,   // 1 year to expiry
            0.05,  // 5% risk-free rate
            0.0,   // no dividend
            0.20,  // 20% volatility
            true,  // call option
        );

        let price = option.price();
        assert!(price > 0.0 && price < 100.0);
        println!("ATM Call Price: {:.4}", price);
    }

    #[test]
    fn test_greeks() {
        let option = BlackScholesOption::new(
            100.0, 100.0, 1.0, 0.05, 0.0, 0.20, true,
        );

        let greeks = option.greeks();
        println!("Delta: {:.4}", greeks.delta);
        println!("Gamma: {:.4}", greeks.gamma);
        println!("Theta: {:.4}", greeks.theta);
        println!("Vega: {:.4}", greeks.vega);
        
        assert!(greeks.delta > 0.0 && greeks.delta < 1.0); // Call delta between 0 and 1
        assert!(greeks.gamma > 0.0); // Gamma always positive
    }

    #[test]
    fn test_put_call_parity() {
        let spot = 100.0;
        let strike = 100.0;
        let t = 1.0;
        let r = 0.05;
        let d = 0.0;
        let vol = 0.20;

        let call = BlackScholesOption::new(spot, strike, t, r, d, vol, true);
        let put = BlackScholesOption::new(spot, strike, t, r, d, vol, false);

        // Put-Call Parity: C - P = S*e^(-Dt) - K*e^(-rt)
        let lhs = call.price() - put.price();
        let rhs = spot * (-d * t).exp() - strike * (-r * t).exp();

        assert!((lhs - rhs).abs() < 1e-10);
    }
}
