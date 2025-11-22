"""
Portfolio Analytics Lab
Advanced portfolio optimization, risk metrics, and performance analysis
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

st.set_page_config(page_title="Portfolio Analytics Lab", page_icon="📊", layout="wide")

# Initialize session state
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {
        'positions': {},
        'cash': 100000.0,
        'initial_capital': 100000.0,
        'history': []
    }
if 'historical_data' not in st.session_state:
    st.session_state.historical_data = None

# Render sidebar navigation and apply CSS
render_sidebar_navigation(current_page="Portfolio Analytics Lab")
apply_custom_css()

st.markdown('<h1 class="lab-header">📊 Portfolio Analytics Lab</h1>', unsafe_allow_html=True)
st.markdown("### Advanced portfolio optimization and risk analysis")
st.markdown("---")

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📊 Current Portfolio", "🎯 Optimization", "📉 Risk Analysis", "📈 Performance"])

with tab1:
    st.markdown("### Current Portfolio Status")
    
    portfolio = st.session_state.portfolio
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_value = portfolio['cash']
    for symbol, pos in portfolio['positions'].items():
        total_value += pos['quantity'] * pos['avg_price']
    
    pnl = total_value - portfolio['initial_capital']
    pnl_pct = (pnl / portfolio['initial_capital']) * 100 if portfolio['initial_capital'] > 0 else 0
    
    with col1:
        st.markdown(f'<div class="metric-card"><strong>Total Value</strong><br/><span style="font-size: 1.5rem; color: #667eea;">${total_value:,.2f}</span></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><strong>Cash</strong><br/><span style="font-size: 1.5rem; color: #667eea;">${portfolio["cash"]:,.2f}</span></div>', unsafe_allow_html=True)
    with col3:
        color = "#10b981" if pnl >= 0 else "#ef4444"
        st.markdown(f'<div class="metric-card"><strong>P&L</strong><br/><span style="font-size: 1.5rem; color: {color};">${pnl:,.2f}</span></div>', unsafe_allow_html=True)
    with col4:
        color = "#10b981" if pnl_pct >= 0 else "#ef4444"
        st.markdown(f'<div class="metric-card"><strong>Return</strong><br/><span style="font-size: 1.5rem; color: {color};">{pnl_pct:+.2f}%</span></div>', unsafe_allow_html=True)
    
    st.markdown("#### Holdings")
    if portfolio['positions']:
        holdings_data = []
        for symbol, pos in portfolio['positions'].items():
            holdings_data.append({
                'Symbol': symbol,
                'Quantity': pos['quantity'],
                'Avg Price': f"${pos['avg_price']:.2f}",
                'Current Value': f"${pos['quantity'] * pos['avg_price']:,.2f}"
            })
        st.dataframe(pd.DataFrame(holdings_data), use_container_width=True, hide_index=True)
    else:
        st.info("No positions currently held")

with tab2:
    st.markdown("### Portfolio Optimization")
    
    if st.session_state.historical_data is None:
        st.warning("⚠️ Please load historical data first")
        if st.button("💾 Load Data"):
            st.switch_page("pages/data_loader.py")
    else:
        data = st.session_state.historical_data
        available_symbols = [col for col in data.columns if col not in ['Date', 'date', 'timestamp']]
        
        st.markdown("#### Mean-Variance Optimization")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_assets = st.multiselect(
                "Select Assets for Portfolio",
                available_symbols,
                default=available_symbols[:5] if len(available_symbols) >= 5 else available_symbols
            )
        
        with col2:
            target_return = st.slider("Target Return (%/year)", 0.0, 50.0, 10.0, 1.0)
            risk_free_rate = st.number_input("Risk-free Rate (%)", value=4.0, step=0.5) / 100
        
        if len(selected_assets) >= 2:
            if st.button("🎯 Optimize Portfolio", type="primary"):
                with st.spinner("Computing optimal weights..."):
                    try:
                        # Calculate returns
                        prices = data[selected_assets].dropna()
                        returns = prices.pct_change().dropna()
                        
                        # Calculate statistics
                        mean_returns = returns.mean() * 252  # Annualized
                        cov_matrix = returns.cov() * 252
                        
                        # Try Rust optimization if available
                        try:
                            import rust_connector
                            
                            # CARA utility optimization
                            risk_aversion = 2.0
                            weights = rust_connector.cara_optimal_weights_rust(
                                mean_returns.values.tolist(),
                                cov_matrix.values.tolist(),
                                risk_aversion
                            )
                            
                            st.success("✅ Optimization completed using Rust backend")
                        except:
                            # Fallback to equal weights
                            st.warning("Using equal weights (Rust optimization unavailable)")
                            weights = np.ones(len(selected_assets)) / len(selected_assets)
                        
                        # Display results
                        st.markdown("#### Optimal Weights")
                        
                        weights_df = pd.DataFrame({
                            'Asset': selected_assets,
                            'Weight': [f"{w:.2%}" for w in weights]
                        })
                        st.dataframe(weights_df, use_container_width=True, hide_index=True)
                        
                        # Portfolio metrics
                        port_return = np.dot(weights, mean_returns)
                        port_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
                        sharpe = (port_return - risk_free_rate) / port_vol if port_vol > 0 else 0
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Expected Return", f"{port_return:.2%}")
                        with col2:
                            st.metric("Volatility", f"{port_vol:.2%}")
                        with col3:
                            st.metric("Sharpe Ratio", f"{sharpe:.2f}")
                        
                        # Efficient frontier plot
                        st.markdown("#### Efficient Frontier")
                        
                        n_portfolios = 1000
                        results = np.zeros((3, n_portfolios))
                        
                        for i in range(n_portfolios):
                            w = np.random.random(len(selected_assets))
                            w /= w.sum()
                            
                            results[0, i] = np.dot(w, mean_returns)
                            results[1, i] = np.sqrt(np.dot(w, np.dot(cov_matrix, w)))
                            results[2, i] = (results[0, i] - risk_free_rate) / results[1, i]
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=results[1], y=results[0],
                            mode='markers',
                            marker=dict(
                                color=results[2],
                                colorscale='Viridis',
                                showscale=True,
                                colorbar=dict(title="Sharpe Ratio"),
                                size=5
                            ),
                            name='Random Portfolios'
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=[port_vol], y=[port_return],
                            mode='markers',
                            marker=dict(color='red', size=15, symbol='star'),
                            name='Optimal Portfolio'
                        ))
                        
                        fig.update_layout(
                            title='Efficient Frontier',
                            xaxis_title='Volatility (σ)',
                            yaxis_title='Expected Return',
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Optimization error: {str(e)}")
        else:
            st.info("Please select at least 2 assets for optimization")

with tab3:
    st.markdown("### Risk Analysis")
    
    st.info("🚧 Coming soon: Value at Risk (VaR), Expected Shortfall (ES), and stress testing")
    
    st.markdown("""
    ### Planned Risk Metrics:
    
    - **Value at Risk (VaR)**: Potential loss at confidence level
    - **Expected Shortfall (CVaR)**: Average loss beyond VaR
    - **Maximum Drawdown**: Largest peak-to-trough decline
    - **Beta & Correlation**: Market sensitivity analysis
    - **Stress Testing**: Scenarios and shocks
    """)

with tab4:
    st.markdown("### Performance Attribution")
    
    st.info("🚧 Coming soon: Return attribution, factor exposure, and benchmark comparison")
    
    st.markdown("""
    ### Performance Metrics:
    
    - **Sharpe Ratio**: Risk-adjusted return
    - **Information Ratio**: Alpha generation vs benchmark
    - **Sortino Ratio**: Downside risk-adjusted return
    - **Calmar Ratio**: Return vs maximum drawdown
    - **Win Rate**: Percentage of profitable trades
    """)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>📊 Portfolio Analytics Lab | Part of HFT Arbitrage Lab</p>
</div>
""", unsafe_allow_html=True)
