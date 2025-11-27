"""
Delta Hedging & Volatility Arbitrage Strategies
Implementation of strategies from "Which Free Lunch Would You Like Today, Sir?"
by Riaz Ahmad and Paul Wilmott
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
from datetime import datetime, timedelta
import sys
import os

# Add python directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

# Try to import Rust bindings, fall back to pure Python if not available
try:
    import rust_connector
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

st.set_page_config(
    page_title="Options Strategies - Delta Hedging & Volatility Arbitrage",
    page_icon="📊",
    layout="wide"
)

from utils.ui_components import render_sidebar_navigation, apply_custom_css
render_sidebar_navigation(current_page="Options Strategies")
apply_custom_css()

# ==================== Black-Scholes Functions ====================

def norm_cdf(x):
    """Standard normal cumulative distribution function"""
    return norm.cdf(x)

def norm_pdf(x):
    """Standard normal probability density function"""
    return norm.pdf(x)

def bs_d1(S, K, T, r, D, sigma):
    """Calculate d1 parameter"""
    return (np.log(S/K) + (r - D + 0.5*sigma**2)*T) / (sigma * np.sqrt(T))

def bs_d2(S, K, T, r, D, sigma):
    """Calculate d2 parameter"""
    return bs_d1(S, K, T, r, D, sigma) - sigma * np.sqrt(T)

def bs_price(S, K, T, r, D, sigma, is_call=True):
    """Black-Scholes option price"""
    if T <= 0:
        return max(S - K, 0) if is_call else max(K - S, 0)
    
    d1 = bs_d1(S, K, T, r, D, sigma)
    d2 = bs_d2(S, K, T, r, D, sigma)
    
    if is_call:
        return S * np.exp(-D*T) * norm_cdf(d1) - K * np.exp(-r*T) * norm_cdf(d2)
    else:
        return K * np.exp(-r*T) * norm_cdf(-d2) - S * np.exp(-D*T) * norm_cdf(-d1)

def bs_delta(S, K, T, r, D, sigma, is_call=True):
    """Black-Scholes Delta"""
    if T <= 0:
        return 1.0 if (is_call and S > K) else (-1.0 if (not is_call and S < K) else 0.0)
    
    d1 = bs_d1(S, K, T, r, D, sigma)
    
    if is_call:
        return np.exp(-D*T) * norm_cdf(d1)
    else:
        return -np.exp(-D*T) * norm_cdf(-d1)

def bs_gamma(S, K, T, r, D, sigma):
    """Black-Scholes Gamma (same for call and put)"""
    if T <= 0:
        return 0.0
    
    d1 = bs_d1(S, K, T, r, D, sigma)
    return np.exp(-D*T) * norm_pdf(d1) / (S * sigma * np.sqrt(T))

def bs_vega(S, K, T, r, D, sigma):
    """Black-Scholes Vega (per 1% volatility change)"""
    if T <= 0:
        return 0.0
    
    d1 = bs_d1(S, K, T, r, D, sigma)
    return S * np.exp(-D*T) * norm_pdf(d1) * np.sqrt(T) / 100.0

def bs_theta(S, K, T, r, D, sigma, is_call=True):
    """Black-Scholes Theta (per day)"""
    if T <= 0:
        return 0.0
    
    d1 = bs_d1(S, K, T, r, D, sigma)
    d2 = bs_d2(S, K, T, r, D, sigma)
    
    term1 = -(S * np.exp(-D*T) * norm_pdf(d1) * sigma) / (2 * np.sqrt(T))
    
    if is_call:
        theta = term1 + D * S * np.exp(-D*T) * norm_cdf(d1) - r * K * np.exp(-r*T) * norm_cdf(d2)
    else:
        theta = term1 - D * S * np.exp(-D*T) * norm_cdf(-d1) + r * K * np.exp(-r*T) * norm_cdf(-d2)
    
    return theta / 365.0  # Per day

# ==================== Page Header ====================

st.title("📊 Delta Hedging & Volatility Arbitrage")
st.markdown("""
### Implementation of Ahmad & Wilmott's Delta Hedging Strategies

This page implements the strategies from the paper **"Which Free Lunch Would You Like Today, Sir?: 
Delta Hedging, Volatility Arbitrage and Optimal Portfolios"** by Riaz Ahmad and Paul Wilmott.

**Key Concepts:**
- **Three Volatilities**: Implied (σ̃), Actual (σ), and Hedging (σ_h)
- **Case 1**: Hedge with **actual volatility** → Guaranteed profit, but random mark-to-market path
- **Case 2**: Hedge with **implied volatility** → Deterministic P&L accumulation, path-dependent final profit
- **Case 3**: Hedge with **custom volatility** → Trade-off between risk and return
""")

if RUST_AVAILABLE:
    st.success("🚀 **Rust-accelerated calculations enabled**")
else:
    st.info("ℹ️ Using Python calculations (Rust module not available)")

# ==================== Sidebar Configuration ====================

st.sidebar.header("Option Parameters")

spot = st.sidebar.number_input("Spot Price (S)", value=100.0, min_value=1.0, step=1.0)
strike = st.sidebar.number_input("Strike Price (K)", value=100.0, min_value=1.0, step=1.0)
time_to_expiry = st.sidebar.slider("Time to Expiry (years)", 0.1, 2.0, 1.0, 0.1)
risk_free_rate = st.sidebar.slider("Risk-Free Rate (r)", 0.0, 0.20, 0.05, 0.01)
dividend_yield = st.sidebar.slider("Dividend Yield (D)", 0.0, 0.10, 0.0, 0.01)

st.sidebar.header("Volatility Parameters")

implied_vol = st.sidebar.slider("Implied Volatility (σ̃)", 0.05, 1.0, 0.20, 0.01)
actual_vol = st.sidebar.slider("Actual Volatility (σ)", 0.05, 1.0, 0.30, 0.01)
hedging_vol = st.sidebar.slider("Hedging Volatility (σ_h)", 0.05, 1.0, 0.25, 0.01)

st.sidebar.markdown("---")

is_call = st.sidebar.radio("Option Type", ["Call", "Put"]) == "Call"
drift = st.sidebar.slider("Stock Drift (μ)", 0.0, 0.30, 0.10, 0.01)

# ==================== Tab Layout ====================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Greeks & Pricing",
    "💰 P&L Analysis",
    "🎲 Simulation",
    "📊 Portfolio Optimization",
    "📚 Mathematical Formulas"
])

# ==================== TAB 1: Greeks & Pricing ====================

with tab1:
    st.header("Option Valuation & Greeks")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Option Prices")
        
        # Calculate prices with different volatilities
        price_implied = bs_price(spot, strike, time_to_expiry, risk_free_rate, dividend_yield, implied_vol, is_call)
        price_actual = bs_price(spot, strike, time_to_expiry, risk_free_rate, dividend_yield, actual_vol, is_call)
        price_hedging = bs_price(spot, strike, time_to_expiry, risk_free_rate, dividend_yield, hedging_vol, is_call)
        
        pricing_data = pd.DataFrame({
            'Volatility Type': ['Implied (σ̃)', 'Actual (σ)', 'Hedging (σ_h)'],
            'Volatility': [implied_vol, actual_vol, hedging_vol],
            'Option Price': [price_implied, price_actual, price_hedging],
            'Price Difference from Implied': [0.0, price_actual - price_implied, price_hedging - price_implied]
        })
        
        st.dataframe(pricing_data.style.format({
            'Volatility': '{:.2%}',
            'Option Price': '${:.4f}',
            'Price Difference from Implied': '${:.4f}'
        }), use_container_width=True)
        
        # Mispricing indicator
        if price_actual > price_implied:
            st.success(f"✅ **Buy opportunity**: Option underpriced by ${price_actual - price_implied:.4f}")
        elif price_actual < price_implied:
            st.error(f"❌ **Sell opportunity**: Option overpriced by ${implied_vol - price_actual:.4f}")
        else:
            st.info("⚖️ **Fair priced**: No arbitrage opportunity")
    
    with col2:
        st.subheader("The Greeks")
        
        # Calculate Greeks using different volatilities
        greeks_data = []
        for vol_name, vol in [('Implied', implied_vol), ('Actual', actual_vol), ('Hedging', hedging_vol)]:
            greeks_data.append({
                'Volatility': vol_name,
                'Delta (Δ)': bs_delta(spot, strike, time_to_expiry, risk_free_rate, dividend_yield, vol, is_call),
                'Gamma (Γ)': bs_gamma(spot, strike, time_to_expiry, risk_free_rate, dividend_yield, vol),
                'Vega (ν)': bs_vega(spot, strike, time_to_expiry, risk_free_rate, dividend_yield, vol),
                'Theta (Θ)': bs_theta(spot, strike, time_to_expiry, risk_free_rate, dividend_yield, vol, is_call)
            })
        
        greeks_df = pd.DataFrame(greeks_data)
        st.dataframe(greeks_df.style.format({
            'Delta (Δ)': '{:.4f}',
            'Gamma (Γ)': '{:.6f}',
            'Vega (ν)': '{:.4f}',
            'Theta (Θ)': '{:.4f}'
        }), use_container_width=True)
    
    # Greeks visualization
    st.subheader("Greeks Surface: Spot Price vs Time")
    
    spot_range = np.linspace(spot * 0.7, spot * 1.3, 50)
    time_range = np.linspace(0.05, time_to_expiry, 30)
    
    greek_to_plot = st.selectbox("Select Greek to visualize", ["Delta", "Gamma", "Vega", "Theta"])
    
    # Create surface plot
    Z = np.zeros((len(time_range), len(spot_range)))
    
    for i, t in enumerate(time_range):
        for j, s in enumerate(spot_range):
            if greek_to_plot == "Delta":
                Z[i,j] = bs_delta(s, strike, t, risk_free_rate, dividend_yield, actual_vol, is_call)
            elif greek_to_plot == "Gamma":
                Z[i,j] = bs_gamma(s, strike, t, risk_free_rate, dividend_yield, actual_vol)
            elif greek_to_plot == "Vega":
                Z[i,j] = bs_vega(s, strike, t, risk_free_rate, dividend_yield, actual_vol)
            else:  # Theta
                Z[i,j] = bs_theta(s, strike, t, risk_free_rate, dividend_yield, actual_vol, is_call)
    
    fig = go.Figure(data=[go.Surface(x=spot_range, y=time_range, z=Z, colorscale='Viridis')])
    fig.update_layout(
        title=f"{greek_to_plot} Surface",
        scene=dict(
            xaxis_title="Spot Price",
            yaxis_title="Time to Expiry (years)",
            zaxis_title=greek_to_plot
        ),
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)

# ==================== TAB 2: P&L Analysis ====================

with tab2:
    st.header("Profit & Loss Analysis")
    
    st.markdown("""
    ### Three Hedging Strategies:
    
    1. **Hedge with Actual Volatility (σ)**
       - **Guaranteed final profit**: `V(S,t;σ) - V(S,t;σ̃)`
       - Random mark-to-market path (contains dX term)
       - Requires knowing actual volatility accurately
    
    2. **Hedge with Implied Volatility (σ̃)**
       - **Deterministic P&L accumulation**: `½(σ² - σ̃²)S²Γdt`
       - Path-dependent final profit
       - Don't need to know exact actual volatility
    
    3. **Hedge with Custom Volatility (σ_h)**
       - Trade-off between risk and return
       - Can optimize for desired risk/return profile
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Case 1: Hedge with Actual Volatility")
        
        guaranteed_profit = price_actual - price_implied
        
        st.metric(
            "Guaranteed Total Profit",
            f"${guaranteed_profit:.4f}",
            delta=f"{(guaranteed_profit/price_implied)*100:.2f}% of premium"
        )
        
        st.markdown(f"""
        **Analysis:**
        - Buy option at implied price: **${price_implied:.4f}**
        - True value (actual vol): **${price_actual:.4f}**
        - **Guaranteed profit**: **${guaranteed_profit:.4f}**
        
        ⚠️ **Note**: Mark-to-market P&L will fluctuate randomly but final profit is guaranteed!
        
        **Formula**: 
        ```
        Guaranteed Profit = V(S,t;σ) - V(S,t;σ̃)
                         = {guaranteed_profit:.4f}
        ```
        """)
    
    with col2:
        st.subheader("Case 2: Hedge with Implied Volatility")
        
        # Instantaneous P&L rate
        gamma_implied = bs_gamma(spot, strike, time_to_expiry, risk_free_rate, dividend_yield, implied_vol)
        pnl_rate_per_year = 0.5 * (actual_vol**2 - implied_vol**2) * spot**2 * gamma_implied
        pnl_rate_per_day = pnl_rate_per_year / 365.0
        
        st.metric(
            "Instantaneous P&L Rate",
            f"${pnl_rate_per_day:.4f} per day",
            delta=f"${pnl_rate_per_year:.2f} annualized"
        )
        
        st.markdown(f"""
        **Analysis:**
        - **Deterministic P&L accumulation** (no random dX term)
        - Daily P&L rate: **${pnl_rate_per_day:.4f}**
        - Path-dependent final profit
        
        **Formula**:
        ```
        dP&L = ½(σ² - σ̃²)S²Γdt
             = ½({actual_vol:.3f}² - {implied_vol:.3f}²) × {spot}² × {gamma_implied:.6f} × dt
             = {pnl_rate_per_day:.4f} per day
        ```
        
        **Advantage**: Don't need to forecast exact volatility, just need σ > σ̃
        """)
    
    st.markdown("---")
    
    # P&L comparison plot
    st.subheader("Expected P&L: Hedging Volatility Comparison")
    
    vol_range = np.linspace(0.05, 0.60, 100)
    expected_pnls = []
    
    for vol_h in vol_range:
        gamma_h = bs_gamma(spot, strike, time_to_expiry, risk_free_rate, dividend_yield, vol_h)
        pnl = 0.5 * (actual_vol**2 - vol_h**2) * spot**2 * gamma_h * time_to_expiry
        expected_pnls.append(pnl)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=vol_range, y=expected_pnls,
        mode='lines',
        name='Expected P&L',
        line=dict(color='blue', width=3)
    ))
    
    # Mark special points
    fig.add_vline(x=implied_vol, line_dash="dash", line_color="green", 
                  annotation_text=f"Implied σ̃={implied_vol:.2f}")
    fig.add_vline(x=actual_vol, line_dash="dash", line_color="red",
                  annotation_text=f"Actual σ={actual_vol:.2f}")
    fig.add_vline(x=hedging_vol, line_dash="dot", line_color="orange",
                  annotation_text=f"Hedging σ_h={hedging_vol:.2f}")
    
    fig.update_layout(
        title="Expected P&L vs Hedging Volatility",
        xaxis_title="Hedging Volatility (σ_h)",
        yaxis_title="Expected Total P&L ($)",
        height=500,
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

# ==================== TAB 3: Simulation ====================

with tab3:
    st.header("Monte Carlo Simulation: Delta Hedging P&L")
    
    st.markdown("""
    Simulate stock price paths and delta hedging P&L to understand the distribution of outcomes.
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Simulation Parameters")
        
        num_paths = st.selectbox("Number of Paths", [100, 500, 1000, 5000], index=1)
        num_steps = st.selectbox("Steps per Path", [50, 100, 250, 500], index=2)
        
        hedge_mode = st.radio(
            "Hedging Strategy",
            ["Implied Volatility", "Actual Volatility", "Custom Volatility"]
        )
        
        run_simulation = st.button("🎲 Run Simulation", type="primary")
    
    with col2:
        if run_simulation:
            with st.spinner("Running Monte Carlo simulation..."):
                dt = time_to_expiry / num_steps
                sqrt_dt = np.sqrt(dt)
                
                # Select hedging volatility based on mode
                if hedge_mode == "Implied Volatility":
                    sigma_h = implied_vol
                elif hedge_mode == "Actual Volatility":
                    sigma_h = actual_vol
                else:
                    sigma_h = hedging_vol
                
                # Storage for paths
                all_spot_paths = []
                all_pnl_paths = []
                final_pnls = []
                
                for path_idx in range(num_paths):
                    s_path = [spot]
                    pnl_path = [0.0]
                    
                    s = spot
                    t = 0.0
                    total_pnl = 0.0
                    
                    for step in range(num_steps):
                        # Calculate current gamma
                        t_remaining = time_to_expiry - t
                        gamma_current = bs_gamma(s, strike, t_remaining, risk_free_rate, dividend_yield, sigma_h)
                        
                        # Calculate instantaneous P&L
                        instant_pnl = 0.5 * (actual_vol**2 - sigma_h**2) * s**2 * gamma_current * dt
                        discount = np.exp(-risk_free_rate * t)
                        total_pnl += instant_pnl * discount
                        
                        # Evolve stock price
                        z = np.random.randn()
                        ds = s * (drift * dt + actual_vol * sqrt_dt * z)
                        s = max(s + ds, 0.1)  # Prevent negative prices
                        t += dt
                        
                        s_path.append(s)
                        pnl_path.append(total_pnl)
                    
                    all_spot_paths.append(s_path)
                    all_pnl_paths.append(pnl_path)
                    final_pnls.append(total_pnl)
                
                final_pnls = np.array(final_pnls)
                
                # Display statistics
                st.subheader("Simulation Results")
                
                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                
                with metrics_col1:
                    st.metric("Mean P&L", f"${np.mean(final_pnls):.4f}")
                with metrics_col2:
                    st.metric("Std Dev", f"${np.std(final_pnls):.4f}")
                with metrics_col3:
                    sharpe = np.mean(final_pnls) / np.std(final_pnls) if np.std(final_pnls) > 0 else 0
                    st.metric("Sharpe Ratio", f"{sharpe:.2f}")
                
                st.markdown("---")
                
                # Plot results
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=("Stock Price Paths", "P&L Paths", 
                                    "Final P&L Distribution", "P&L Histogram"),
                    specs=[[{"type": "xy"}, {"type": "xy"}],
                           [{"type": "xy"}, {"type": "xy"}]]
                )
                
                # Plot sample paths (max 100 for visibility)
                time_grid = np.linspace(0, time_to_expiry, num_steps + 1)
                sample_size = min(100, num_paths)
                
                for i in range(sample_size):
                    show_legend = (i == 0)
                    fig.add_trace(
                        go.Scatter(x=time_grid, y=all_spot_paths[i], 
                                   mode='lines', line=dict(width=0.5, color='rgba(0,100,200,0.3)'),
                                   showlegend=show_legend, name='Stock Paths' if show_legend else ''),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=time_grid, y=all_pnl_paths[i],
                                   mode='lines', line=dict(width=0.5, color='rgba(0,200,100,0.3)'),
                                   showlegend=show_legend, name='P&L Paths' if show_legend else ''),
                        row=1, col=2
                    )
                
                # Strike line
                fig.add_hline(y=strike, line_dash="dash", line_color="red", row=1, col=1)
                
                # Sorted P&L (shows distribution shape)
                sorted_pnls = np.sort(final_pnls)
                fig.add_trace(
                    go.Scatter(x=np.arange(len(sorted_pnls)), y=sorted_pnls,
                               mode='markers', marker=dict(size=3, color='blue'),
                               name='Sorted P&L'),
                    row=2, col=1
                )
                
                # Histogram
                fig.add_trace(
                    go.Histogram(x=final_pnls, nbinsx=50, name='P&L Distribution',
                                 marker_color='green'),
                    row=2, col=2
                )
                
                fig.update_xaxes(title_text="Time (years)", row=1, col=1)
                fig.update_xaxes(title_text="Time (years)", row=1, col=2)
                fig.update_xaxes(title_text="Path Index", row=2, col=1)
                fig.update_xaxes(title_text="P&L ($)", row=2, col=2)
                
                fig.update_yaxes(title_text="Spot Price ($)", row=1, col=1)
                fig.update_yaxes(title_text="Cumulative P&L ($)", row=1, col=2)
                fig.update_yaxes(title_text="Final P&L ($)", row=2, col=1)
                fig.update_yaxes(title_text="Frequency", row=2, col=2)
                
                fig.update_layout(height=800, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # Percentiles
                st.subheader("P&L Percentiles")
                percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
                percentile_values = np.percentile(final_pnls, percentiles)
                
                percentile_df = pd.DataFrame({
                    'Percentile': [f"{p}%" for p in percentiles],
                    'P&L ($)': percentile_values
                })
                st.dataframe(percentile_df, use_container_width=True)

# ==================== TAB 4: Portfolio Optimization ====================

with tab4:
    st.header("Portfolio Optimization")
    
    st.markdown("""
    ### Multiple Options Portfolio
    
    For a portfolio of options on the same underlying, the total P&L when hedging with implied volatility is:
    
    **Expected Portfolio P&L**:
    ```
    E[Π] = Σᵢ ½(σ² - σ̃ᵢ²) ∫₀ᵀ e⁻ʳᵗ S² Γᵢ(S,t) dt
    ```
    
    **Variance of Portfolio P&L**:
    ```
    Var[Π] = Σᵢ Σⱼ Cov[Πᵢ, Πⱼ]
    ```
    """)
    
    st.info("🚧 Portfolio optimization features coming soon! Will include:\n"
            "- Multi-strike portfolio construction\n"
            "- Utility maximization\n"
            "- Risk-adjusted return optimization\n"
            "- Correlation effects")

# ==================== TAB 5: Mathematical Formulas ====================

with tab5:
    st.header("Mathematical Formulas")
    
    st.markdown(r"""
    ### Black-Scholes Model
    
    **Stock Price Dynamics**:
    $$
    dS = \mu S dt + \sigma S dX
    $$
    
    **Option Price**:
    $$
    V(S,t) = S e^{-D(T-t)} N(d_1) - K e^{-r(T-t)} N(d_2) \quad \text{(Call)}
    $$
    
    where:
    $$
    d_1 = \frac{\ln(S/K) + (r - D + \frac{1}{2}\sigma^2)(T-t)}{\sigma\sqrt{T-t}}
    $$
    $$
    d_2 = d_1 - \sigma\sqrt{T-t}
    $$
    
    ### The Greeks
    
    **Delta**: $\Delta = \frac{\partial V}{\partial S} = e^{-D(T-t)} N(d_1)$ (Call)
    
    **Gamma**: $\Gamma = \frac{\partial^2 V}{\partial S^2} = \frac{e^{-D(T-t)} N'(d_1)}{S \sigma \sqrt{T-t}}$
    
    **Vega**: $\nu = \frac{\partial V}{\partial \sigma} = S e^{-D(T-t)} N'(d_1) \sqrt{T-t}$
    
    **Theta**: $\Theta = -\frac{\partial V}{\partial t} = -\frac{S e^{-D(T-t)} N'(d_1) \sigma}{2\sqrt{T-t}} + D S e^{-D(T-t)} N(d_1) - r K e^{-r(T-t)} N(d_2)$ (Call)
    
    ### Delta Hedging P&L Formulas
    
    #### Case 1: Hedge with Actual Volatility
    
    **Guaranteed Final Profit**:
    $$
    \text{Profit} = V(S,t;\sigma) - V(S,t;\tilde{\sigma})
    $$
    
    **Mark-to-Market P&L** (random path):
    $$
    dP\&L = \frac{1}{2}(\sigma^2 - \tilde{\sigma}^2) S^2 \Gamma^i dt + (\Delta^i - \Delta^a)[(\mu - r + D)S dt + \sigma S dX]
    $$
    
    #### Case 2: Hedge with Implied Volatility
    
    **Instantaneous P&L** (deterministic):
    $$
    dP\&L = \frac{1}{2}(\sigma^2 - \tilde{\sigma}^2) S^2 \Gamma^i dt
    $$
    
    **Total Profit** (path-dependent):
    $$
    \text{Profit} = \frac{1}{2}(\sigma^2 - \tilde{\sigma}^2) \int_0^T e^{-r(t-t_0)} S^2 \Gamma^i(S,t) dt
    $$
    
    #### Case 3: Hedge with Custom Volatility
    
    **Mark-to-Market P&L**:
    $$
    dP\&L = \frac{1}{2}(\sigma^2 - \sigma_h^2) S^2 \Gamma^h dt + (\Delta^i - \Delta^h)[(\mu - r + D)S dt + \sigma S dX]
    $$
    
    **Expected Profit**:
    $$
    E[\text{Profit}] = \frac{1}{2}(\sigma^2 - \sigma_h^2) \int_0^T e^{-rt} E[S^2 \Gamma^h(S,t)] dt
    $$
    
    **Variance of Profit**:
    $$
    \text{Var}[\text{Profit}] = \frac{1}{4}(\sigma^2 - \sigma_h^2)^2 \int_0^T e^{-2rt} E[S^4 (\Gamma^h)^2] dt + \text{other terms}
    $$
    
    ### Key Insights
    
    1. **Volatility Mismatch → Profit**: When $\sigma > \tilde{\sigma}$, buying and delta hedging yields positive expected profit
    
    2. **Hedging Volatility Choice**:
       - $\sigma_h = \sigma$: Guaranteed profit, random path
       - $\sigma_h = \tilde{\sigma}$: Deterministic accumulation, path-dependent
       - $\sigma_h$ between: Trade-off risk vs return
    
    3. **Portfolio Effects**: Combining options with different strikes/maturities can reduce variance through diversification
    
    4. **Gamma Scalping**: The profit comes from "gamma scalping" - rehedging captures the difference between realized and implied volatility
    """)

st.sidebar.markdown("---")
st.sidebar.info("""
**References**:
- Ahmad & Wilmott (2005): "Which Free Lunch Would You Like Today, Sir?"
- Carr (2005): Volatility arbitrage formulas
- Henrard (2003): Path-dependent profit simulations
""")
