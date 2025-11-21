"""
Affine Volatility Models - Rough Heston Dashboard

Based on "Affine Volatility Models" by Bourgey (2024)
Reference: https://github.com/fbourgey/RoughVolatilityWorkshop
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add python directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

from rough_heston import (
    RoughHestonParams,
    RoughHestonCharFunc,
    rough_heston_kernel,
    normalized_leverage_contract,
    atm_skew,
    skew_stickiness_ratio,
    calibrate_rough_heston,
    SPX_CALIBRATED_PARAMS
)

st.set_page_config(page_title="Affine Volatility Models", layout="wide")

# Title and citation
st.title("🌊 Affine Volatility Models: Rough Heston")

st.markdown("""
*Based on "Rough Volatility Workshop - Affine Models" by F. Bourgey (2024)*

**Reference:** [QM2024_3_Affine_models.ipynb](https://github.com/fbourgey/RoughVolatilityWorkshop/blob/main/QM2024_3_Affine_models.ipynb)
""")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Parameters", 
    "📈 Leverage Swaps", 
    "🎲 Characteristic Function",
    "🎯 Calibration",
    "📐 Formulas"
])

# ============================================================================
# TAB 1: PARAMETERS
# ============================================================================
with tab1:
    st.header("Rough Heston Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Parameters")
        
        use_spx = st.checkbox("Use SPX calibrated parameters", value=True)
        
        if use_spx:
            params = SPX_CALIBRATED_PARAMS
            st.info(f"**SPX Calibrated:** H={params.H:.4f}, ν={params.nu:.4f}, ρ={params.rho:.4f}, λ={params.lambda_:.4f}")
        else:
            H = st.slider("Hurst Parameter (H)", 0.01, 0.49, 0.10, 0.01, 
                         help="Roughness: smaller = rougher paths")
            nu = st.slider("Vol of Vol (ν)", 0.1, 2.0, 0.3, 0.05,
                          help="Volatility of volatility")
            rho = st.slider("Correlation (ρ)", -0.99, 0.0, -0.7, 0.01,
                           help="Correlation between returns and volatility")
            lambda_ = st.slider("Mean Reversion (λ)", 0.0, 5.0, 0.5, 0.1,
                               help="Speed of mean reversion")
            theta = st.slider("Long-term Variance (θ)", 0.01, 0.10, 0.04, 0.01,
                             help="Long-run variance level")
            v0 = st.slider("Initial Variance (V₀)", 0.01, 0.10, 0.04, 0.01,
                          help="Current variance level")
            
            try:
                params = RoughHestonParams(H, nu, rho, lambda_, theta, v0)
            except ValueError as e:
                st.error(f"Invalid parameters: {e}")
                st.stop()
        
        st.session_state['rough_heston_params'] = params
    
    with col2:
        st.subheader("Derived Quantities")
        
        st.markdown(f"""
        **Fractional Order:**
        - α = H + 0.5 = {params.alpha:.4f}
        
        **Adjusted Mean Reversion:**
        - λ' = λ - ρν = {params.lambda_prime:.4f}
        
        **Regime:**
        - {"Rough volatility (H < 0.5)" if params.H < 0.5 else "Smooth volatility"}
        - {"Subcritical (α < 1)" if params.alpha < 1 else "Supercritical"}
        
        **Volatility Smile:**
        - Negative skew: ρ < 0 ✓
        - Vol-of-vol effect: ν = {params.nu:.2f}
        """)
    
    # Show kernel plot
    st.subheader("Rough Heston Kernel κ(τ)")
    
    st.latex(r"\kappa(\tau) = \eta \tau^{\alpha-1} E_{\alpha,\alpha}(-\lambda \tau^\alpha)")
    
    taus = np.linspace(0.01, 2.0, 100)
    kernels = [rough_heston_kernel(t, params) for t in taus]
    
    fig_kernel = go.Figure()
    fig_kernel.add_trace(go.Scatter(
        x=taus, y=kernels,
        mode='lines',
        name='Kernel κ(τ)',
        line=dict(color='blue', width=2)
    ))
    fig_kernel.update_layout(
        title="Rough Heston Kernel",
        xaxis_title="Time τ (years)",
        yaxis_title="Kernel Value",
        hovermode='x unified',
        height=400
    )
    st.plotly_chart(fig_kernel, use_container_width=True)

# ============================================================================
# TAB 2: LEVERAGE SWAPS
# ============================================================================
with tab2:
    st.header("Leverage Swaps and Normalized Leverage")
    
    # Get parameters
    params = st.session_state.get('rough_heston_params', SPX_CALIBRATED_PARAMS)
    
    st.markdown("""
    The **normalized leverage contract** measures the realized correlation scaled by variance:
    """)
    
    st.latex(r"L_t(T) = \frac{\rho \eta}{\lambda'} \left[1 - E_{\alpha,2}(-\lambda' \tau^\alpha)\right]")
    
    st.markdown("where λ' = λ - ρη is the adjusted mean reversion.")
    
    # Compute leverage curves
    expiries = np.linspace(0.1, 5.0, 50)
    norm_leverage = [normalized_leverage_contract(t, params) for t in expiries]
    
    # Create characteristic function
    char_func = RoughHestonCharFunc(params)
    var_swaps = [char_func.variance_swap(t) for t in expiries]
    lev_swaps = [char_func.leverage_swap(t) for t in expiries]
    
    # Plot
    col1, col2 = st.columns(2)
    
    with col1:
        fig_norm = go.Figure()
        fig_norm.add_trace(go.Scatter(
            x=expiries, y=norm_leverage,
            mode='lines',
            name='Normalized Leverage',
            line=dict(color='red', width=3)
        ))
        fig_norm.update_layout(
            title="Normalized Leverage Contract",
            xaxis_title="Time to Maturity (years)",
            yaxis_title="L(T)",
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig_norm, use_container_width=True)
    
    with col2:
        fig_swaps = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig_swaps.add_trace(
            go.Scatter(x=expiries, y=var_swaps, name="Variance Swap",
                      line=dict(color='blue', width=2)),
            secondary_y=False
        )
        
        fig_swaps.add_trace(
            go.Scatter(x=expiries, y=lev_swaps, name="Leverage Swap",
                      line=dict(color='green', width=2)),
            secondary_y=True
        )
        
        fig_swaps.update_xaxes(title_text="Time to Maturity (years)")
        fig_swaps.update_yaxes(title_text="Variance Swap", secondary_y=False)
        fig_swaps.update_yaxes(title_text="Leverage Swap", secondary_y=True)
        fig_swaps.update_layout(
            title="Variance & Leverage Swaps",
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig_swaps, use_container_width=True)
    
    # Table
    st.subheader("Leverage Swap Values")
    df_lev = pd.DataFrame({
        'Expiry (Y)': expiries[::5],
        'Norm Leverage': [f"{x:.6f}" for x in norm_leverage[::5]],
        'Var Swap': [f"{x:.6f}" for x in var_swaps[::5]],
        'Lev Swap': [f"{x:.6f}" for x in lev_swaps[::5]]
    })
    st.dataframe(df_lev, use_container_width=True)

# ============================================================================
# TAB 3: CHARACTERISTIC FUNCTION
# ============================================================================
with tab3:
    st.header("Characteristic Function & Market Observables")
    
    params = st.session_state.get('rough_heston_params', SPX_CALIBRATED_PARAMS)
    char_func = RoughHestonCharFunc(params)
    
    st.markdown("""
    The **characteristic function** ψ(τ; a) = log 𝔼[exp(ia X_T)] encodes all derivative prices.
    """)
    
    # ATM Skew
    st.subheader("ATM Skew")
    st.latex(r"\text{Skew} \approx -\frac{\rho \eta}{2\sqrt{\theta \tau}}")
    
    expiries_skew = np.linspace(0.1, 3.0, 50)
    skews = [atm_skew(char_func, t) for t in expiries_skew]
    
    fig_skew = go.Figure()
    fig_skew.add_trace(go.Scatter(
        x=expiries_skew, y=skews,
        mode='lines',
        name='ATM Skew',
        line=dict(color='purple', width=3)
    ))
    fig_skew.update_layout(
        title="ATM Implied Volatility Skew Term Structure",
        xaxis_title="Time to Maturity (years)",
        yaxis_title="dσ/dk at k=0",
        hovermode='x unified',
        height=400
    )
    st.plotly_chart(fig_skew, use_container_width=True)
    
    # SSR
    st.subheader("Skew-Stickiness Ratio (SSR)")
    st.latex(r"\text{SSR} \approx \frac{1 + \alpha}{2}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        ssr_vals = [skew_stickiness_ratio(char_func, t) for t in expiries_skew]
        avg_ssr = np.mean(ssr_vals)
        
        st.metric("Average SSR", f"{avg_ssr:.4f}")
        st.markdown(f"""
        **Interpretation:**
        - SSR = 0: Sticky strike
        - SSR = 0.5: Sticky moneyness
        - SSR = 1: Sticky delta
        
        Current: **{avg_ssr:.4f}** ≈ {"Sticky moneyness" if abs(avg_ssr - 0.5) < 0.1 else "Mixed regime"}
        """)
    
    with col2:
        fig_ssr = go.Figure()
        fig_ssr.add_trace(go.Scatter(
            x=expiries_skew, y=ssr_vals,
            mode='lines',
            name='SSR',
            line=dict(color='orange', width=3)
        ))
        fig_ssr.add_hline(y=0.5, line_dash="dash", line_color="gray",
                         annotation_text="Sticky Moneyness")
        fig_ssr.update_layout(
            title="Skew-Stickiness Ratio",
            xaxis_title="Time to Maturity (years)",
            yaxis_title="SSR",
            height=300
        )
        st.plotly_chart(fig_ssr, use_container_width=True)

# ============================================================================
# TAB 4: CALIBRATION
# ============================================================================
with tab4:
    st.header("Parameter Calibration")
    
    st.markdown("""
    Calibrate rough Heston parameters to market normalized leverage contracts.
    
    **Method:** Minimize weighted squared errors between model and market leverage.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Market Data")
        
        # Example SPX leverage data
        st.markdown("**Example: S&P 500 Normalized Leverage**")
        
        market_expiries = np.array([0.25, 0.5, 1.0, 2.0, 3.0])
        market_leverage = np.array([-0.025, -0.035, -0.045, -0.055, -0.060])
        
        df_market = pd.DataFrame({
            'Expiry (Y)': market_expiries,
            'Norm Leverage': market_leverage
        })
        
        edited_df = st.data_editor(df_market, num_rows="dynamic", use_container_width=True)
        
        if st.button("🎯 Calibrate Parameters", type="primary"):
            with st.spinner("Calibrating..."):
                try:
                    result = calibrate_rough_heston(
                        edited_df['Expiry (Y)'].values,
                        edited_df['Norm Leverage'].values
                    )
                    
                    if result['success']:
                        st.success("✅ Calibration successful!")
                        st.session_state['calibrated_params'] = result['params']
                        st.session_state['calibration_result'] = result
                    else:
                        st.error(f"⚠️ Calibration failed: {result['message']}")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
    
    with col2:
        st.subheader("Calibration Results")
        
        if 'calibration_result' in st.session_state:
            result = st.session_state['calibration_result']
            params_calib = result['params']
            
            st.markdown(f"""
            **Calibrated Parameters:**
            - Hurst (H): **{params_calib.H:.4f}**
            - Vol-of-Vol (ν): **{params_calib.nu:.4f}**
            - Correlation (ρ): **{params_calib.rho:.4f}**
            - Mean Reversion (λ): **{params_calib.lambda_:.4f}**
            
            **Fit Quality:**
            - Objective value: {result['fun']:.6e}
            - Status: {result['message']}
            """)
            
            # Compare fit
            model_leverage = [normalized_leverage_contract(t, params_calib) 
                            for t in edited_df['Expiry (Y)'].values]
            
            fig_fit = go.Figure()
            fig_fit.add_trace(go.Scatter(
                x=edited_df['Expiry (Y)'], 
                y=edited_df['Norm Leverage'],
                mode='markers',
                name='Market',
                marker=dict(size=10, color='red')
            ))
            fig_fit.add_trace(go.Scatter(
                x=edited_df['Expiry (Y)'], 
                y=model_leverage,
                mode='lines+markers',
                name='Model',
                line=dict(color='blue', width=2)
            ))
            fig_fit.update_layout(
                title="Calibration Fit",
                xaxis_title="Expiry (years)",
                yaxis_title="Normalized Leverage",
                hovermode='x unified',
                height=400
            )
            st.plotly_chart(fig_fit, use_container_width=True)
            
            if st.button("Use Calibrated Parameters"):
                st.session_state['rough_heston_params'] = params_calib
                st.success("✅ Parameters updated!")
                st.rerun()
        else:
            st.info("👆 Click 'Calibrate Parameters' to start")

# ============================================================================
# TAB 5: FORMULAS
# ============================================================================
with tab5:
    st.header("📐 Mathematical Formulas")
    
    st.markdown("""
    ## Rough Heston Model
    
    ### Dynamics
    
    Asset price:
    """)
    st.latex(r"\frac{dS_t}{S_t} = \sqrt{V_t} \, dZ_t")
    
    st.markdown("Forward variance curve:")
    st.latex(r"d\xi_t(u) = \sqrt{V_t} \, \kappa(u-t) \, dW_t")
    
    st.markdown("where ⟨Z, W⟩_t = ρt.")
    
    st.markdown("""
    ### Rough Heston Kernel
    
    Power-law decay with exponential cutoff:
    """)
    st.latex(r"\kappa(\tau) = \eta \tau^{\alpha-1} E_{\alpha,\alpha}(-\lambda \tau^\alpha)")
    
    st.markdown("where:")
    st.latex(r"E_{\alpha,\beta}(z) = \sum_{k=0}^\infty \frac{z^k}{\Gamma(\alpha k + \beta)}")
    
    st.markdown("""
    ### Normalized Leverage Contract
    
    Realized correlation scaled by variance:
    """)
    st.latex(r"L_t(T) = \frac{1}{M_t(T)} \int_t^T \langle dS_s/S_s, d\xi_s(T) \rangle")
    
    st.markdown("For flat curve:")
    st.latex(r"L_t(T) = \frac{\rho \eta}{\lambda'} \left[1 - E_{\alpha,2}(-\lambda' \tau^\alpha)\right]")
    
    st.markdown("where λ' = λ - ρη.")
    
    st.markdown("""
    ### ATM Skew
    
    First derivative of implied volatility:
    """)
    st.latex(r"\frac{\partial \sigma}{\partial k}\bigg|_{k=0} \approx -\frac{\rho \eta}{2\sqrt{\theta \tau}}")
    
    st.markdown("""
    ### Skew-Stickiness Ratio (SSR)
    
    Ratio of sticky-delta to sticky-strike behaviors:
    """)
    st.latex(r"\text{SSR} \approx \frac{1 + \alpha}{2}")
    
    st.markdown("""
    where α = H + 0.5 is the fractional order.
    
    ### Parameter Constraints
    
    - **Hurst:** 0 < H < 0.5 (rough regime)
    - **Vol-of-Vol:** η > 0
    - **Correlation:** -1 < ρ < 1 (typically ρ < 0 for equity)
    - **Mean Reversion:** λ ≥ 0
    - **Long-term Variance:** θ > 0
    - **Initial Variance:** V₀ > 0
    
    ---
    
    **Reference:**
    
    Bourgey, F. (2024). *Rough Volatility Workshop - QM2024_3_Affine_models*.  
    GitHub: https://github.com/fbourgey/RoughVolatilityWorkshop
    """)

# Footer
st.markdown("---")
st.markdown("**Rough Heston Affine Volatility Models** | Implementation based on Bourgey (2024)")
