"""
Chiarella Model Lab
Agent-based market dynamics with chartist-fundamentalist interaction
"""

import streamlit as st
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

st.set_page_config(page_title="Chiarella Model Lab", page_icon="🌀", layout="wide")

# Render sidebar navigation and apply CSS
render_sidebar_navigation(current_page="Chiarella Model Lab")
apply_custom_css()

st.markdown('<h1 class="lab-header">🌀 Chiarella Model Lab</h1>', unsafe_allow_html=True)
st.markdown("### Agent-based price dynamics with chartist-fundamentalist interaction")
st.markdown("---")

# Sidebar parameters
with st.sidebar:
    st.markdown("### 🎛️ Model Parameters")
    
    st.markdown("#### Agent Parameters")
    beta_f = st.slider("Fundamentalist Strength (β_f)", 0.0, 2.0, 0.5, 0.1)
    beta_c = st.slider("Chartist Strength (β_c)", 0.0, 2.0, 1.0, 0.1)
    gamma = st.slider("Switching Rate (γ)", 0.0, 5.0, 1.0, 0.1)
    
    st.markdown("---")
    st.markdown("#### Market Parameters")
    P_fundamental = st.number_input("Fundamental Price", value=100.0, step=5.0)
    volatility = st.slider("Noise Volatility", 0.0, 0.5, 0.1, 0.01)
    
    st.markdown("---")
    st.markdown("#### Simulation Settings")
    T = st.slider("Time Horizon", 100, 2000, 500, 100)
    P0 = st.number_input("Initial Price", value=95.0, step=1.0)

# Main content
tab1, tab2, tab3 = st.tabs(["📊 Price Dynamics", "🔄 Bifurcation Analysis", "📈 Trading Signals"])

with tab1:
    st.markdown("### Agent-Based Price Simulation")
    
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🚀 Simulate", type="primary", use_container_width=True):
            with st.spinner("Simulating agent dynamics..."):
                # Simulate Chiarella model
                dt = 0.1
                n_steps = int(T / dt)
                
                prices = np.zeros(n_steps)
                fundamentalists = np.zeros(n_steps)
                chartists = np.zeros(n_steps)
                
                prices[0] = P0
                fundamentalists[0] = 0.5
                chartists[0] = 0.5
                
                for t in range(1, n_steps):
                    # Current fractions
                    n_f = fundamentalists[t-1]
                    n_c = chartists[t-1]
                    
                    # Demands
                    D_f = beta_f * (P_fundamental - prices[t-1])
                    D_c = beta_c * (prices[t-1] - prices[max(0, t-10)])
                    
                    # Price update
                    noise = np.random.normal(0, volatility)
                    prices[t] = prices[t-1] + 0.1 * (n_f * D_f + n_c * D_c) + noise
                    
                    # Agent switching (discrete choice)
                    profit_f = D_f * (prices[t] - prices[t-1])
                    profit_c = D_c * (prices[t] - prices[t-1])
                    
                    exp_f = np.exp(gamma * profit_f)
                    exp_c = np.exp(gamma * profit_c)
                    
                    fundamentalists[t] = exp_f / (exp_f + exp_c)
                    chartists[t] = exp_c / (exp_f + exp_c)
                
                time_grid = np.arange(n_steps) * dt
                
                # Plot results
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('Market Price', 'Agent Fractions'),
                    row_heights=[0.6, 0.4],
                    vertical_spacing=0.1
                )
                
                # Price chart
                fig.add_trace(
                    go.Scatter(x=time_grid, y=prices, name='Market Price', 
                             line=dict(color='blue', width=2)),
                    row=1, col=1
                )
                fig.add_hline(y=P_fundamental, line_dash="dash", line_color="red", 
                            row=1, col=1, annotation_text="Fundamental")
                
                # Agent fractions
                fig.add_trace(
                    go.Scatter(x=time_grid, y=fundamentalists, name='Fundamentalists',
                             line=dict(color='green', width=2)),
                    row=2, col=1
                )
                fig.add_trace(
                    go.Scatter(x=time_grid, y=chartists, name='Chartists',
                             line=dict(color='orange', width=2)),
                    row=2, col=1
                )
                
                fig.update_xaxes(title_text="Time", row=2, col=1)
                fig.update_yaxes(title_text="Price", row=1, col=1)
                fig.update_yaxes(title_text="Fraction", row=2, col=1)
                fig.update_layout(height=800, showlegend=True, hovermode='x unified')
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Final Price", f"${prices[-1]:.2f}")
                with col2:
                    st.metric("Price Volatility", f"{np.std(prices):.2f}")
                with col3:
                    st.metric("Mean Deviation", f"${np.mean(np.abs(prices - P_fundamental)):.2f}")
                with col4:
                    st.metric("Final Fundamentalists", f"{fundamentalists[-1]:.2%}")
                
                # Regime detection
                st.markdown("#### 📊 Market Regime")
                if fundamentalists[-1] > 0.6:
                    st.success("🟢 Fundamentalist-dominated: Price converging to fundamental value")
                elif chartists[-1] > 0.6:
                    st.warning("🟡 Chartist-dominated: Trend-following behavior, potential bubbles")
                else:
                    st.info("🔵 Mixed regime: Balanced agent behavior")
    
    with col1:
        st.markdown("""
        #### About the Chiarella Model
        
        The model simulates two types of traders:
        
        **Fundamentalists** (Green):
        - Trade based on deviation from fundamental value
        - Provide stabilizing force
        - Profit from mean reversion
        
        **Chartists** (Orange):
        - Trade based on recent trends
        - Create momentum and overshooting
        - Can amplify price moves
        
        Agents switch between strategies based on recent profitability, 
        creating endogenous regime changes and realistic price dynamics.
        """)

with tab2:
    st.markdown("### Bifurcation Analysis")
    st.info("🚧 Coming soon: Parameter sensitivity and stability analysis")
    
    st.markdown("""
    This section will show:
    - Stability regions in parameter space
    - Bifurcation diagrams
    - Phase portraits
    - Lyapunov exponents
    """)

with tab3:
    st.markdown("### Trading Signals from Agent Dynamics")
    st.info("🚧 Coming soon: Regime-based trading strategies")
    
    st.markdown("""
    Generate trading signals based on:
    - Agent fraction transitions
    - Price-fundamental gaps during fundamentalist dominance
    - Momentum signals during chartist dominance
    - Regime switching indicators
    """)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🌀 Chiarella Model Lab | Part of HFT Arbitrage Lab</p>
</div>
""", unsafe_allow_html=True)
