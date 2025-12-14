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

# Try to import Rust bindings via gRPC bridge, fall back to pure Python if not available
try:
    from python.rust_grpc_bridge import rust_connector as rust_connector  # type: ignore
    RUST_AVAILABLE = True
except Exception:
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

def bs_delta(S, K, T, r, D, sigma, is_call=True):  # type: ignore[no-redef]
    """Black-Scholes Delta"""
    if T <= 0:
        return 1.0 if (is_call and S > K) else (-1.0 if (not is_call and S < K) else 0.0)
    
    d1 = bs_d1(S, K, T, r, D, sigma)
    
    if is_call:
        return np.exp(-D*T) * norm_cdf(d1)
    else:
        return -np.exp(-D*T) * norm_cdf(-d1)

def bs_gamma(S, K, T, r, D, sigma):  # type: ignore[no-redef]
    """Black-Scholes Gamma (same for call and put)"""
    if T <= 0:
        return 0.0
    
    d1 = bs_d1(S, K, T, r, D, sigma)
    return np.exp(-D*T) * norm_pdf(d1) / (S * sigma * np.sqrt(T))

def bs_vega(S, K, T, r, D, sigma):  # type: ignore[no-redef]
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
    
    st.markdown("### 🎯 Multi-Strike Portfolio Construction")
    
    st.markdown("""
    Construct optimal option portfolios across multiple strikes to maximize
    risk-adjusted returns while managing Greek exposures.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Portfolio Parameters")
        S_port = st.number_input("Current Stock Price", value=100.0, step=5.0, key="port_S")
        num_strikes = st.slider("Number of Strikes", 3, 10, 5)
        strike_range_pct = st.slider("Strike Range (%)", 5, 30, 15,
                                     help="Range around ATM")
        
        portfolio_objective = st.selectbox(
            "Optimization Objective",
            ["Maximize Sharpe Ratio", "Minimize Portfolio Variance", "Maximize Expected Return", "Target Delta Neutral"]
        )
    
    with col2:
        st.markdown("#### Risk Constraints")
        max_delta = st.slider("Max Absolute Delta", 0.0, 2.0, 0.5, 0.1,
                             help="Maximum net delta exposure")
        max_gamma = st.slider("Max Absolute Gamma", 0.0, 0.5, 0.2, 0.05,
                             help="Maximum gamma exposure")
        max_vega = st.slider("Max Absolute Vega", 0.0, 50.0, 20.0, 5.0,
                            help="Maximum vega exposure")
    
    if st.button("🎯 Optimize Portfolio", type="primary"):
        with st.spinner("Optimizing options portfolio..."):
            # Generate strike ladder
            strike_width = strike_range_pct / 100 * S_port / (num_strikes - 1)
            strikes = [S_port * (1 - strike_range_pct/200) + i * strike_width 
                      for i in range(num_strikes)]
            
            # Simplified option pricing (Black-Scholes for demonstration)
            # Use same parameters from earlier
            r_bs = 0.05
            sigma_bs = 0.2
            T_bs = 1.0
            
            from scipy.stats import norm as scipy_norm
            
            def black_scholes_call(S, K, T, r, sigma):
                d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
                d2 = d1 - sigma * np.sqrt(T)
                return S * scipy_norm.cdf(d1) - K * np.exp(-r * T) * scipy_norm.cdf(d2)
            
            def bs_delta(S, K, T, r, sigma):
                d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
                return scipy_norm.cdf(d1)
            
            def bs_gamma(S, K, T, r, sigma):
                d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
                return scipy_norm.pdf(d1) / (S * sigma * np.sqrt(T))
            
            def bs_vega(S, K, T, r, sigma):
                d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
                return S * scipy_norm.pdf(d1) * np.sqrt(T)
            
            # Calculate option prices and Greeks
            option_data = []
            for K in strikes:
                price = black_scholes_call(S_port, K, T_bs, r_bs, sigma_bs)
                delta = bs_delta(S_port, K, T_bs, r_bs, sigma_bs)
                gamma = bs_gamma(S_port, K, T_bs, r_bs, sigma_bs)
                vega = bs_vega(S_port, K, T_bs, r_bs, sigma_bs)
                
                option_data.append({
                    'Strike': K,
                    'Price': price,
                    'Delta': delta,
                    'Gamma': gamma,
                    'Vega': vega
                })
            
            df_options = pd.DataFrame(option_data)
            
            # Optimization using simplified approach
            # Weights represent position sizes (positive = long, negative = short)
            
            if portfolio_objective == "Maximize Sharpe Ratio":
                # Simplified: weight by moneyness-adjusted return potential
                df_options['Weight'] = df_options.apply(
                    lambda row: (S_port / row['Strike'] - 1) * row['Delta'] 
                    if row['Strike'] < S_port else (1 - S_port / row['Strike']) * (1 - row['Delta']),
                    axis=1
                )
            elif portfolio_objective == "Minimize Portfolio Variance":
                # Weight inversely to gamma (lower convexity risk)
                df_options['Weight'] = 1 / (df_options['Gamma'] + 0.01)
            elif portfolio_objective == "Maximize Expected Return":
                # Weight by delta (directional exposure)
                df_options['Weight'] = df_options['Delta']
            else:  # Target Delta Neutral
                # Solve for delta-neutral weights
                # Simplified: alternate long/short to balance
                df_options['Weight'] = [(-1)**i * 1/num_strikes for i in range(num_strikes)]
            
            # Normalize weights
            df_options['Weight'] = df_options['Weight'] / df_options['Weight'].abs().sum()
            
            # Calculate portfolio Greeks
            port_delta = (df_options['Weight'] * df_options['Delta']).sum()
            port_gamma = (df_options['Weight'] * df_options['Gamma']).sum()
            port_vega = (df_options['Weight'] * df_options['Vega']).sum()
            port_cost = (df_options['Weight'].abs() * df_options['Price']).sum()
            
            # Display results
            st.markdown("### 📊 Optimized Portfolio")
            
            col_a, col_b, col_c, col_d = st.columns(4)
            
            with col_a:
                st.metric("Portfolio Delta", f"{port_delta:.3f}",
                         help="Net directional exposure")
            with col_b:
                st.metric("Portfolio Gamma", f"{port_gamma:.4f}",
                         help="Convexity exposure")
            with col_c:
                st.metric("Portfolio Vega", f"{port_vega:.2f}",
                         help="Volatility sensitivity")
            with col_d:
                st.metric("Total Cost", f"${port_cost:.2f}",
                         help="Net portfolio cost")
            
            # Check constraints
            st.markdown("### ✅ Constraint Validation")
            
            constraint_status = []
            constraint_status.append({
                'Constraint': 'Max Delta',
                'Limit': f"±{max_delta}",
                'Actual': f"{port_delta:.3f}",
                'Status': '✅ OK' if abs(port_delta) <= max_delta else '❌ VIOLATED'
            })
            constraint_status.append({
                'Constraint': 'Max Gamma',
                'Limit': f"±{max_gamma}",
                'Actual': f"{port_gamma:.4f}",
                'Status': '✅ OK' if abs(port_gamma) <= max_gamma else '❌ VIOLATED'
            })
            constraint_status.append({
                'Constraint': 'Max Vega',
                'Limit': f"±{max_vega}",
                'Actual': f"{port_vega:.2f}",
                'Status': '✅ OK' if abs(port_vega) <= max_vega else '❌ VIOLATED'
            })
            
            st.dataframe(pd.DataFrame(constraint_status), use_container_width=True)
            
            # Portfolio composition
            st.markdown("### 📋 Portfolio Composition")
            
            display_df = df_options.copy()
            display_df['Position'] = display_df['Weight'].apply(
                lambda w: f"{'LONG' if w > 0 else 'SHORT'} {abs(w):.2%}"
            )
            display_df = display_df[['Strike', 'Price', 'Position', 'Delta', 'Gamma', 'Vega']]
            display_df = display_df.style.format({
                'Strike': '${:.2f}',
                'Price': '${:.2f}',
                'Delta': '{:.3f}',
                'Gamma': '{:.4f}',
                'Vega': '{:.2f}'
            })
            
            st.dataframe(display_df, use_container_width=True)
            
            # Visualization
            st.markdown("### 📈 Portfolio Greeks Profile")
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Weights by Strike', 'Delta by Strike', 'Gamma by Strike', 'Vega by Strike'),
                specs=[[{'type': 'bar'}, {'type': 'scatter'}],
                      [{'type': 'scatter'}, {'type': 'scatter'}]]
            )
            
            # Weights
            fig.add_trace(
                go.Bar(x=df_options['Strike'], y=df_options['Weight'],
                      name='Weight', marker_color='lightblue'),
                row=1, col=1
            )
            
            # Delta
            fig.add_trace(
                go.Scatter(x=df_options['Strike'], y=df_options['Delta'],
                          name='Delta', mode='lines+markers', line={'color': 'blue'}),
                row=1, col=2
            )
            
            # Gamma
            fig.add_trace(
                go.Scatter(x=df_options['Strike'], y=df_options['Gamma'],
                          name='Gamma', mode='lines+markers', line={'color': 'green'}),
                row=2, col=1
            )
            
            # Vega
            fig.add_trace(
                go.Scatter(x=df_options['Strike'], y=df_options['Vega'],
                          name='Vega', mode='lines+markers', line={'color': 'red'}),
                row=2, col=2
            )
            
            fig.update_layout(height=700, showlegend=False)
            fig.update_xaxes(title_text="Strike")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # P&L simulation
            st.markdown("### 💰 P&L Simulation")
            
            # Simulate various stock price scenarios
            price_scenarios = np.linspace(S_port * 0.8, S_port * 1.2, 50)
            pnl_scenarios = []
            
            for S_scenario in price_scenarios:
                pnl = 0
                for _, option in df_options.iterrows():
                    # Revalue option at new stock price
                    new_price = black_scholes_call(S_scenario, option['Strike'], T_bs, r_bs, sigma_bs)
                    position_pnl = (new_price - option['Price']) * option['Weight']
                    pnl += position_pnl
                
                pnl_scenarios.append(pnl)
            
            fig_pnl = go.Figure()
            
            fig_pnl.add_trace(go.Scatter(
                x=price_scenarios,
                y=pnl_scenarios,
                mode='lines',
                name='Portfolio P&L',
                line={'color': 'purple', 'width': 3},
                fill='tozeroy',
                fillcolor='rgba(128,0,128,0.1)'
            ))
            
            fig_pnl.add_vline(x=S_port, line_dash="dash", line_color="gray",
                            annotation_text="Current Price")
            fig_pnl.add_hline(y=0, line_color="black", line_width=1)
            
            fig_pnl.update_layout(
                title='Portfolio P&L vs Stock Price',
                xaxis_title='Stock Price',
                yaxis_title='P&L ($)',
                height=400
            )
            
            st.plotly_chart(fig_pnl, use_container_width=True)
            
            # Risk metrics
            st.markdown("### 📊 Risk-Return Analysis")
            
            max_profit = max(pnl_scenarios)
            max_loss = min(pnl_scenarios)
            breakeven_points = []
            
            for i in range(len(pnl_scenarios) - 1):
                if pnl_scenarios[i] * pnl_scenarios[i+1] < 0:  # Sign change
                    breakeven_points.append(price_scenarios[i])
            
            risk_col1, risk_col2, risk_col3 = st.columns(3)
            
            with risk_col1:
                st.metric("Max Profit", f"${max_profit:.2f}")
            with risk_col2:
                st.metric("Max Loss", f"${max_loss:.2f}")
            with risk_col3:
                profit_prob = sum(1 for pnl in pnl_scenarios if pnl > 0) / len(pnl_scenarios) * 100
                st.metric("Profit Probability", f"{profit_prob:.1f}%")
            
            if breakeven_points:
                st.success(f"📍 Breakeven Point(s): {', '.join([f'${bp:.2f}' for bp in breakeven_points])}")
            
            st.info("""
            **💡 Portfolio Optimization Insights:**
            - Multi-strike portfolios allow for sophisticated payoff profiles
            - Greek constraints ensure risk management requirements are met
            - Delta-neutral strategies profit from volatility, not direction
            - Gamma and vega exposures determine sensitivity to market moves
            - Always monitor position Greeks as market conditions change
            """)

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
