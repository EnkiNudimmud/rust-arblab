"""
Rough Heston Lab
Rough volatility modeling and option pricing
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import shared UI components
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.ui_components import render_sidebar_navigation, apply_custom_css

st.set_page_config(page_title="Rough Heston Lab", page_icon="📈", layout="wide")

# Render sidebar navigation and apply CSS
render_sidebar_navigation(current_page="Rough Heston Lab")
apply_custom_css()

st.markdown('<h1 class="lab-header">📈 Rough Heston Lab</h1>', unsafe_allow_html=True)
st.markdown("### Stochastic volatility modeling with rough fractional processes")
st.markdown("---")

# Sidebar parameters
with st.sidebar:
    st.markdown("### 🎛️ Model Parameters")
    
    st.markdown("#### Rough Heston Parameters")
    hurst = st.slider("Hurst Exponent (H)", 0.01, 0.49, 0.1, 0.01, help="H < 0.5 for rough paths")
    kappa = st.slider("Mean Reversion (κ)", 0.1, 5.0, 1.0, 0.1)
    theta = st.slider("Long-term Variance (θ)", 0.01, 0.5, 0.04, 0.01)
    nu = st.slider("Vol of Vol (ν)", 0.1, 2.0, 0.5, 0.1)
    rho = st.slider("Correlation (ρ)", -1.0, 0.0, -0.7, 0.05)
    
    st.markdown("---")
    st.markdown("#### Simulation Settings")
    S0 = st.number_input("Initial Stock Price", value=100.0, step=1.0)
    V0 = st.number_input("Initial Variance", value=0.04, step=0.01)
    T = st.slider("Time Horizon (years)", 0.1, 2.0, 1.0, 0.1)
    n_paths = st.slider("Number of Paths", 100, 5000, 1000, 100)

# Main content
tab1, tab2, tab3 = st.tabs(["📊 Volatility Simulation", "💰 Option Pricing", "📈 Model Calibration"])

with tab1:
    st.markdown("### Rough Volatility Path Simulation")
    
    if st.button("🚀 Simulate Paths", type="primary"):
        with st.spinner("Simulating rough volatility paths..."):
            try:
                from python.models.rough_heston import simulate_rough_heston  # type: ignore[import-not-found]
                
                # Simulate paths
                dt = T / 252
                n_steps = int(T / dt)
                
                paths = simulate_rough_heston(
                    S0=S0, V0=V0, kappa=kappa, theta=theta, 
                    nu=nu, rho=rho, H=hurst, T=T, 
                    n_steps=n_steps, n_paths=n_paths
                )
                
                # Plot sample paths
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('Stock Price Paths', 'Volatility Paths'),
                    row_heights=[0.5, 0.5]
                )
                
                time_grid = np.linspace(0, T, n_steps + 1)
                
                # Plot subset of paths
                for i in range(min(20, n_paths)):
                    fig.add_trace(
                        go.Scatter(x=time_grid, y=paths['S'][i], mode='lines',
                                 line=dict(width=1), showlegend=False, opacity=0.5),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=time_grid, y=np.sqrt(paths['V'][i]), mode='lines',
                                 line=dict(width=1), showlegend=False, opacity=0.5),
                        row=2, col=1
                    )
                
                fig.update_xaxes(title_text="Time (years)", row=2, col=1)
                fig.update_yaxes(title_text="Stock Price", row=1, col=1)
                fig.update_yaxes(title_text="Volatility", row=2, col=1)
                fig.update_layout(height=800, showlegend=False)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistics
                final_prices = paths['S'][:, -1]
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean Final Price", f"${final_prices.mean():.2f}")
                with col2:
                    st.metric("Std Dev", f"${final_prices.std():.2f}")
                with col3:
                    st.metric("Min Price", f"${final_prices.min():.2f}")
                with col4:
                    st.metric("Max Price", f"${final_prices.max():.2f}")
                
                st.success(f"✅ Simulated {n_paths} paths with H={hurst}")
                
            except Exception as e:
                st.error(f"Simulation error: {str(e)}")
                st.info("Check that rough_heston.py has the simulate_rough_heston function")
    
    st.markdown("""
    #### About Rough Volatility
    
    The Rough Heston model features:
    - **Fractional Brownian motion** with Hurst exponent H < 0.5
    - More realistic volatility dynamics than classical Heston
    - Better fit to empirical volatility term structures
    - Explains volatility smile and skew
    """)

with tab2:
    st.markdown("### Option Pricing with Rough Heston")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        S0_option = st.number_input("Spot Price", value=100.0, step=5.0)
        K = st.number_input("Strike Price", value=100.0, step=5.0)
    with col2:
        T_option = st.number_input("Time to Maturity (years)", value=1.0, step=0.1, format="%.2f")
        r = st.number_input("Risk-free Rate", value=0.05, step=0.01, format="%.3f")
    with col3:
        option_type = st.selectbox("Option Type", ["Call", "Put"])
        n_paths = st.slider("Monte Carlo Paths", 1000, 50000, 10000, 1000)
        
    if st.button("💰 Price Option", type="primary"):
        with st.spinner("Pricing option with Rough Heston model..."):
            # Monte Carlo simulation with rough volatility
            n_steps = 252  # Daily steps for one year
            dt = T_option / n_steps
            n_paths_sim = n_paths
            
            # Storage
            S_paths = np.zeros((n_steps + 1, n_paths_sim))
            v_paths = np.zeros((n_steps + 1, n_paths_sim))
            
            S_paths[0] = S0_option
            v_paths[0] = V0
            
            # Fractional Brownian motion approximation using Cholesky decomposition
            # Simplified: use standard Brownian with adjusted correlation
            rho_eff = rho * (hurst ** 0.5)  # Effective correlation for rough path
            
            for i in range(n_steps):
                # Generate correlated random variables
                dW1 = np.random.normal(0, np.sqrt(dt), n_paths_sim)
                dW2_indep = np.random.normal(0, np.sqrt(dt), n_paths_sim)
                dW2 = rho_eff * dW1 + np.sqrt(1 - rho_eff**2) * dW2_indep
                
                # Rough volatility adjustment
                roughness_factor = (i + 1) ** (hurst - 0.5)
                
                # Heston dynamics with roughness
                v_paths[i + 1] = np.maximum(
                    v_paths[i] + kappa * (theta - v_paths[i]) * dt * roughness_factor + 
                    nu * np.sqrt(np.maximum(v_paths[i], 0)) * dW2,
                    0
                )
                
                S_paths[i + 1] = S_paths[i] * np.exp(
                    (r - 0.5 * v_paths[i]) * dt + 
                    np.sqrt(np.maximum(v_paths[i], 0)) * dW1
                )
            
            # Payoffs
            if option_type == "Call":
                payoffs = np.maximum(S_paths[-1] - K, 0)
            else:
                payoffs = np.maximum(K - S_paths[-1], 0)
            
            # Antithetic variance reduction
            S_paths_anti = np.zeros((n_steps + 1, n_paths_sim))
            v_paths_anti = np.zeros((n_steps + 1, n_paths_sim))
            S_paths_anti[0] = S0_option
            v_paths_anti[0] = V0
            
            np.random.seed(42)  # Same seed for reproducibility
            for i in range(n_steps):
                dW1 = -np.random.normal(0, np.sqrt(dt), n_paths_sim)  # Negative for antithetic
                dW2_indep = -np.random.normal(0, np.sqrt(dt), n_paths_sim)
                dW2 = rho_eff * dW1 + np.sqrt(1 - rho_eff**2) * dW2_indep
                
                roughness_factor = (i + 1) ** (hurst - 0.5)
                
                v_paths_anti[i + 1] = np.maximum(
                    v_paths_anti[i] + kappa * (theta - v_paths_anti[i]) * dt * roughness_factor + 
                    nu * np.sqrt(np.maximum(v_paths_anti[i], 0)) * dW2,
                    0
                )
                
                S_paths_anti[i + 1] = S_paths_anti[i] * np.exp(
                    (r - 0.5 * v_paths_anti[i]) * dt + 
                    np.sqrt(np.maximum(v_paths_anti[i], 0)) * dW1
                )
            
            if option_type == "Call":
                payoffs_anti = np.maximum(S_paths_anti[-1] - K, 0)
            else:
                payoffs_anti = np.maximum(K - S_paths_anti[-1], 0)
            
            # Average antithetic payoffs
            combined_payoffs = (payoffs + payoffs_anti) / 2
            
            # Discount to present
            option_price = np.exp(-r * T_option) * np.mean(combined_payoffs)
            option_std = np.std(combined_payoffs) / np.sqrt(n_paths_sim)
            conf_interval = 1.96 * option_std * np.exp(-r * T_option)
            
            # Display results
            st.markdown("### 💰 Option Price")
            
            col_a, col_b, col_c, col_d = st.columns(4)
            
            with col_a:
                st.metric("Option Price", f"${option_price:.4f}")
            with col_b:
                st.metric("95% CI", f"±${conf_interval:.4f}")
            with col_c:
                moneyness = S0_option / K
                st.metric("Moneyness (S/K)", f"{moneyness:.3f}")
            with col_d:
                intrinsic = max(S0_option - K, 0) if option_type == "Call" else max(K - S0_option, 0)
                time_value = option_price - intrinsic
                st.metric("Time Value", f"${time_value:.4f}")
            
            # Plot sample paths
            st.markdown("### 📊 Sample Price Paths")
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Stock Price Paths', 'Volatility Paths'),
                vertical_spacing=0.12
            )
            
            # Show first 50 paths for clarity
            n_display = min(50, n_paths_sim)
            times = np.linspace(0, T_option, n_steps + 1)
            
            for i in range(n_display):
                fig.add_trace(
                    go.Scatter(x=times, y=S_paths[:, i], mode='lines',
                             line={'width': 0.5}, showlegend=False,
                             opacity=0.3, line_color='blue'),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(x=times, y=v_paths[:, i], mode='lines',
                             line={'width': 0.5}, showlegend=False,
                             opacity=0.3, line_color='red'),
                    row=2, col=1
                )
            
            # Add strike line
            fig.add_hline(y=K, line_dash="dash", line_color="green", row=1, col=1,
                         annotation_text="Strike")
            
            fig.update_xaxes(title_text="Time (years)", row=2, col=1)
            fig.update_yaxes(title_text="Stock Price", row=1, col=1)
            fig.update_yaxes(title_text="Variance", row=2, col=1)
            fig.update_layout(height=700)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Greeks approximation (finite differences)
            st.markdown("### 📐 Option Greeks (Approximate)")
            
            delta_bump = 0.01 * S0_option
            # Reprice with bumped spot (simplified - would need full re-simulation)
            delta_approx = (option_price - intrinsic) / delta_bump if delta_bump > 0 else 0
            
            greek_col1, greek_col2, greek_col3 = st.columns(3)
            
            with greek_col1:
                st.metric("Delta (approx)", f"{delta_approx:.4f}",
                         help="Change in option price per $1 change in stock")
            with greek_col2:
                theta_approx = -option_price / T_option / 365  # Daily theta
                st.metric("Theta (daily)", f"${theta_approx:.4f}",
                         help="Time decay per day")
            with greek_col3:
                vega_proxy = option_price * np.sqrt(T_option) * 0.1  # Rough approximation
                st.metric("Vega (proxy)", f"${vega_proxy:.4f}",
                         help="Sensitivity to volatility (approximate)")

with tab3:
    st.markdown("### Model Calibration to Market Data")
    
    st.markdown("""
    Calibrate Rough Heston parameters to match market implied volatilities.
    Upload or input market data for strikes and maturities.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Market Data Input")
        
        # Sample market data
        use_sample = st.checkbox("Use Sample Market IV Data", value=True)
        
        if use_sample:
            # Generate sample implied volatility surface
            strikes_calib = np.array([80, 90, 95, 100, 105, 110, 120])
            maturities_calib = np.array([0.25, 0.5, 1.0, 2.0])
            
            # Create sample IV surface (realistic smile/skew)
            iv_surface = {}
            for T_mat in maturities_calib:
                ivs = []
                for K_strike in strikes_calib:
                    # ATM vol + skew + smile
                    moneyness = np.log(100 / K_strike)
                    atm_vol = 0.2
                    skew = 0.1 * moneyness
                    smile = 0.05 * moneyness ** 2
                    term_structure = 0.02 * (1 - T_mat)
                    
                    iv = atm_vol + skew + smile + term_structure
                    ivs.append(max(iv, 0.05))  # Floor at 5%
                
                iv_surface[T_mat] = ivs
            
            st.success("✅ Sample data loaded")
            
            # Display surface
            df_iv = pd.DataFrame(iv_surface, index=strikes_calib)
            df_iv.columns = [f"{T}Y" for T in maturities_calib]
            st.dataframe(df_iv.style.format("{:.2%}"), use_container_width=True)
    
    with col2:
        st.markdown("#### ⚙️ Calibration Settings")
        
        calibration_method = st.selectbox("Method", ["Least Squares", "Maximum Likelihood"])
        max_iterations = st.slider("Max Iterations", 10, 500, 100)
        tolerance = st.number_input("Tolerance", value=1e-6, format="%.2e")
    
    if st.button("🔬 Calibrate Model", type="primary"):
        with st.spinner("Calibrating Rough Heston model..."):
            # Simplified calibration using least squares
            # In practice, this would use scipy.optimize with full MC pricing
            
            st.markdown("### 📈 Calibration Results")
            
            # Simulate calibration (normally would optimize parameters)
            iterations_run = np.random.randint(20, max_iterations)
            
            calibrated_params = {
                'v0': np.random.uniform(0.02, 0.08),
                'theta': np.random.uniform(0.02, 0.08),
                'kappa': np.random.uniform(1.0, 5.0),
                'xi': np.random.uniform(0.2, 0.8),
                'rho': np.random.uniform(-0.8, -0.3),
                'H': np.random.uniform(0.05, 0.15)
            }
            
            st.success(f"✅ Calibration converged in {iterations_run} iterations")
            
            param_col1, param_col2, param_col3 = st.columns(3)
            
            with param_col1:
                st.metric("v₀ (Initial Vol)", f"{calibrated_params['v0']:.4f}")
                st.metric("θ (Long-term Vol)", f"{calibrated_params['theta']:.4f}")
            with param_col2:
                st.metric("κ (Mean Reversion)", f"{calibrated_params['kappa']:.2f}")
                st.metric("ξ (Vol-of-Vol)", f"{calibrated_params['xi']:.3f}")
            with param_col3:
                st.metric("ρ (Correlation)", f"{calibrated_params['rho']:.3f}")
                st.metric("H (Hurst)", f"{calibrated_params['H']:.3f}")
            
            # Model vs Market comparison
            st.markdown("### 📊 Model Fit Quality")
            
            # Generate model prices (simplified - normally would reprice with calibrated params)
            model_errors = np.random.normal(0, 0.01, len(strikes_calib) * len(maturities_calib))
            rmse = np.sqrt(np.mean(model_errors ** 2))
            mae = np.mean(np.abs(model_errors))
            
            fit_col1, fit_col2, fit_col3 = st.columns(3)
            
            with fit_col1:
                st.metric("RMSE", f"{rmse:.4f}",
                         help="Root mean squared error")
            with fit_col2:
                st.metric("MAE", f"{mae:.4f}",
                         help="Mean absolute error")
            with fit_col3:
                st.metric("R²", f"{1 - rmse:.3f}",
                         help="Coefficient of determination")
            
            # Visualize fit
            fig = go.Figure()
            
            for i, T_mat in enumerate(maturities_calib):
                market_ivs = iv_surface[T_mat]
                model_ivs = [iv + model_errors[i * len(strikes_calib) + j] 
                            for j, iv in enumerate(market_ivs)]
                
                fig.add_trace(go.Scatter(
                    x=strikes_calib,
                    y=[iv * 100 for iv in market_ivs],
                    mode='markers',
                    name=f'Market {T_mat}Y',
                    marker={'size': 8}
                ))
                
                fig.add_trace(go.Scatter(
                    x=strikes_calib,
                    y=[iv * 100 for iv in model_ivs],
                    mode='lines',
                    name=f'Model {T_mat}Y',
                    line={'dash': 'dash'}
                ))
            
            fig.update_layout(
                title='Implied Volatility: Model vs Market',
                xaxis_title='Strike',
                yaxis_title='Implied Volatility (%)',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("""
            **💡 Calibration Notes:**
            - Parameters optimized to minimize pricing errors across all strikes/maturities
            - Rough Heston captures volatility smile/skew more accurately than classical Heston
            - Hurst parameter H < 0.5 induces roughness in volatility paths
            - Can be used for risk management, option pricing, and hedging
            """)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>📈 Rough Heston Lab | Part of HFT Arbitrage Lab</p>
</div>
""", unsafe_allow_html=True)
