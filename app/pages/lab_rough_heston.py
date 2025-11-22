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
                from python.rough_heston import simulate_rough_heston
                
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
    
    col1, col2 = st.columns(2)
    with col1:
        K = st.number_input("Strike Price", value=100.0, step=5.0)
        option_type = st.selectbox("Option Type", ["Call", "Put"])
    with col2:
        r = st.number_input("Risk-free Rate", value=0.05, step=0.01, format="%.3f")
        
    if st.button("💰 Price Option", type="primary"):
        st.info("🚧 Option pricing with rough Heston model coming soon")
        st.markdown("""
        The pricing will use Monte Carlo simulation with:
        - Rough volatility paths
        - Antithetic variance reduction
        - Control variates for accuracy
        """)

with tab3:
    st.markdown("### Model Calibration to Market Data")
    st.info("🚧 Model calibration to implied volatility surface coming soon")
    
    st.markdown("""
    Calibration features:
    - Fit to market implied volatilities
    - Least-squares or maximum likelihood estimation
    - Validation metrics and diagnostics
    - Parameter stability analysis
    """)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>📈 Rough Heston Lab | Part of HFT Arbitrage Lab</p>
</div>
""", unsafe_allow_html=True)
