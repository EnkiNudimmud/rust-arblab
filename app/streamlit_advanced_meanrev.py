"""
Advanced Mean-Reversion Portfolio Dashboard
Complete implementation of 5 advanced features with large-scale real-world data

Features:
1. CARA Utility Maximization
2. Transaction Cost Modeling
3. Multi-Period Optimization
4. Risk-Adjusted Weights (Sharpe)
5. Optimal Stopping Times
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import sys

sys.path.append('/Users/melvinalvarez/Documents/Workspace/rust-hft-arbitrage-lab')

from python import meanrev
from python.data_fetcher import (
    fetch_intraday_data, get_close_prices, get_universe_symbols
)

# Page config
st.set_page_config(
    page_title="Advanced Mean-Reversion Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Check Rust availability
try:
    import rust_connector
    RUST_AVAILABLE = True
    rust_status = "✅ Using Rust"
except ImportError:
    RUST_AVAILABLE = False
    rust_status = "🔧 Using Python"

# Title
st.title("📊 Advanced Mean-Reversion Portfolio Analysis")
st.markdown(f"**Backend**: {rust_status}")
st.markdown("---")

# Sidebar - Configuration
st.sidebar.header("⚙️ Configuration")

# Universe selection
universe_options = {
    "Tech Stocks (30)": "sp500_tech",
    "Finance (15)": "sp500_finance",
    "Test (5)": "test"
}
universe_choice = st.sidebar.selectbox(
    "Stock Universe",
    list(universe_options.keys())
)
universe = universe_options[universe_choice]

# Time period
days_back = st.sidebar.slider("Days of History", 30, 180, 90)
interval = st.sidebar.selectbox(
    "Data Interval",
    ["1h", "1d"],
    index=0
)

# Data source
data_source = st.sidebar.selectbox(
    "Data Source",
    ["Synthetic", "Yahoo Finance", "Finnhub"],
    index=0
)

# Fetch data button
if st.sidebar.button("🔄 Fetch Data", type="primary"):
    with st.spinner("Fetching data..."):
        symbols = get_universe_symbols(universe)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        data = fetch_intraday_data(
            symbols=symbols,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            interval=interval,
            source=data_source.lower().replace(" ", "")
        )
        
        prices = get_close_prices(data).fillna(method='ffill')
        returns = meanrev.compute_log_returns(prices)
        
        st.session_state['prices'] = prices
        st.session_state['returns'] = returns
        st.session_state['symbols'] = symbols
        st.success(f"✅ Fetched {prices.shape[0]} periods × {prices.shape[1]} symbols")

# Check if data is loaded
if 'prices' not in st.session_state:
    st.info("👈 Configure parameters and click 'Fetch Data' to begin analysis")
    st.stop()

prices = st.session_state['prices']
returns = st.session_state['returns']
symbols = st.session_state['symbols']

# Tabs for different analyses
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Data Overview",
    "🎯 CARA Utility",
    "⚖️ Sharpe Optimization",
    "⏱️ Optimal Thresholds",
    "💰 Transaction Costs",
    "🔄 Multi-Period"
])

# Tab 1: Data Overview
with tab1:
    st.header("Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Symbols", len(symbols))
    with col2:
        st.metric("Time Periods", len(prices))
    with col3:
        st.metric("Data Points", f"{prices.size:,}")
    with col4:
        st.metric("Date Range", f"{(prices.index[-1] - prices.index[0]).days}d")
    
    # Price chart
    st.subheader("Price Evolution (Normalized)")
    n_show = min(10, len(symbols))
    normalized = prices.iloc[:, :n_show] / prices.iloc[0, :n_show] * 100
    
    fig = go.Figure()
    for col in normalized.columns:
        fig.add_trace(go.Scatter(
            x=normalized.index,
            y=normalized[col],
            name=col,
            mode='lines'
        ))
    fig.update_layout(
        height=400,
        hovermode='x unified',
        xaxis_title="Time",
        yaxis_title="Normalized Price (Base=100)"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    st.subheader("Returns Correlation Matrix")
    corr = returns.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale='RdBu',
        zmid=0
    ))
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # PCA Analysis
    st.subheader("PCA Analysis")
    n_components = st.slider("Number of Components", 3, 10, 5)
    
    with st.spinner("Computing PCA..."):
        components, pca_info = meanrev.pca_portfolios(returns, n_components=n_components)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Explained variance
        fig = go.Figure(go.Bar(
            x=list(range(1, len(pca_info['explained_variance_ratio_'])+1)),
            y=pca_info['explained_variance_ratio_'],
            marker_color='lightblue'
        ))
        fig.update_layout(
            title="Explained Variance by Component",
            xaxis_title="Component",
            yaxis_title="Variance Explained",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Cumulative variance
        fig = go.Figure(go.Scatter(
            x=list(range(1, len(pca_info['explained_variance_ratio_'])+1)),
            y=np.cumsum(pca_info['explained_variance_ratio_']),
            mode='lines+markers',
            marker=dict(size=8)
        ))
        fig.update_layout(
            title="Cumulative Variance",
            xaxis_title="Component",
            yaxis_title="Cumulative Variance",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # PC1 weights
    st.subheader("Principal Component 1 - Portfolio Weights")
    weights_df = pd.DataFrame({
        'Symbol': prices.columns,
        'Weight': components[0]
    }).sort_values('Weight', key=abs, ascending=False)
    
    fig = go.Figure(go.Bar(
        x=weights_df['Symbol'],
        y=weights_df['Weight'],
        marker_color=['green' if w > 0 else 'red' for w in weights_df['Weight']]
    ))
    fig.update_layout(height=400, xaxis_title="Symbol", yaxis_title="Weight")
    st.plotly_chart(fig, use_container_width=True)

# Tab 2: CARA Utility
with tab2:
    st.header("CARA Utility Maximization (Appendix A)")
    
    st.markdown("""
    ### Theory
    CARA utility function: $U(W) = -\\exp(-\\gamma W)$
    
    Optimal weights: $w^* = \\frac{1}{\\gamma} \\Sigma^{-1} \\mu$
    
    Where $\\gamma$ is the risk aversion parameter.
    """)
    
    # Parameters
    col1, col2 = st.columns(2)
    with col1:
        gamma_values = st.multiselect(
            "Risk Aversion Levels (γ)",
            [0.5, 1.0, 2.0, 5.0, 10.0],
            default=[1.0, 2.0, 5.0]
        )
    
    if not gamma_values:
        st.warning("Please select at least one risk aversion level")
        st.stop()
    
    # Compute
    expected_returns = returns.mean().values
    covariance = returns.cov().values
    
    cara_results = {}
    for gamma in gamma_values:
        with st.spinner(f"Computing CARA optimal weights for γ={gamma}..."):
            result = meanrev.cara_optimal_weights(expected_returns, covariance, gamma=gamma)
            cara_results[gamma] = result
    
    # Results table
    st.subheader("Results Summary")
    results_data = []
    for gamma in gamma_values:
        r = cara_results[gamma]
        results_data.append({
            'γ': gamma,
            'Expected Return': f"{r['expected_return']:.4f}",
            'Expected Std': f"{np.sqrt(r['expected_variance']):.4f}",
            'Weights Sum': f"{sum(r['weights']):.4f}",
            '|Weights|': f"{sum(abs(w) for w in r['weights']):.4f}"
        })
    st.dataframe(pd.DataFrame(results_data), use_container_width=True)
    
    # Weights comparison
    st.subheader("Portfolio Weights Comparison")
    fig = make_subplots(
        rows=len(gamma_values), cols=1,
        subplot_titles=[f"γ = {g}" for g in gamma_values],
        vertical_spacing=0.05
    )
    
    for i, gamma in enumerate(gamma_values, 1):
        weights = cara_results[gamma]['weights']
        colors = ['green' if w > 0 else 'red' for w in weights]
        
        fig.add_trace(
            go.Bar(x=prices.columns, y=weights, marker_color=colors, showlegend=False),
            row=i, col=1
        )
    
    fig.update_layout(height=300*len(gamma_values))
    st.plotly_chart(fig, use_container_width=True)
    
    # Efficient frontier
    st.subheader("Efficient Frontier")
    returns_list = [cara_results[g]['expected_return'] for g in gamma_values]
    stds_list = [np.sqrt(cara_results[g]['expected_variance']) for g in gamma_values]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=stds_list,
        y=returns_list,
        mode='markers+lines',
        marker=dict(size=15, color=gamma_values, colorscale='Viridis',
                   showscale=True, colorbar=dict(title="γ")),
        text=[f"γ={g}" for g in gamma_values],
        hovertemplate='%{text}<br>Std: %{x:.4f}<br>Return: %{y:.4f}<extra></extra>'
    ))
    fig.update_layout(
        xaxis_title="Expected Volatility",
        yaxis_title="Expected Return",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

# Tab 3: Sharpe Optimization
with tab3:
    st.header("Sharpe Ratio Optimization")
    
    st.markdown("""
    ### Theory
    Maximize Sharpe ratio: $SR = \\frac{E[R_p] - r_f}{\\sigma_p}$
    
    Optimal weights: $w^* = \\frac{\\Sigma^{-1}(\\mu - r_f \\mathbf{1})}{\\mathbf{1}^T \\Sigma^{-1}(\\mu - r_f \\mathbf{1})}$
    """)
    
    # Parameters
    risk_free_rates = st.multiselect(
        "Risk-Free Rates",
        [0.0, 0.02, 0.04, 0.05],
        default=[0.0, 0.02]
    )
    
    if not risk_free_rates:
        st.warning("Please select at least one risk-free rate")
        st.stop()
    
    # Compute
    sharpe_results = {}
    for rf in risk_free_rates:
        with st.spinner(f"Computing Sharpe-optimal weights for rf={rf:.1%}..."):
            result = meanrev.sharpe_optimal_weights(expected_returns, covariance, risk_free_rate=rf)
            sharpe_results[rf] = result
    
    # Results
    st.subheader("Results Summary")
    results_data = []
    for rf in risk_free_rates:
        r = sharpe_results[rf]
        results_data.append({
            'Risk-Free Rate': f"{rf:.1%}",
            'Sharpe Ratio': f"{r['sharpe_ratio']:.4f}",
            'Expected Return': f"{r['expected_return']:.4f}",
            'Expected Std': f"{r['expected_std']:.4f}",
            'Weights Sum': f"{sum(r['weights']):.4f}"
        })
    st.dataframe(pd.DataFrame(results_data), use_container_width=True)
    
    # Weights visualization
    st.subheader("Portfolio Weights")
    fig = make_subplots(
        rows=1, cols=len(risk_free_rates),
        subplot_titles=[f"rf={rf:.1%}" for rf in risk_free_rates]
    )
    
    for i, rf in enumerate(risk_free_rates, 1):
        weights = sharpe_results[rf]['weights']
        colors = ['green' if w > 0 else 'red' for w in weights]
        
        fig.add_trace(
            go.Bar(x=prices.columns, y=weights, marker_color=colors, showlegend=False),
            row=1, col=i
        )
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# Tab 4: Optimal Thresholds
with tab4:
    st.header("Optimal Stopping Times")
    
    st.markdown("""
    ### Theory
    Based on OU process parameters, determine optimal entry/exit thresholds:
    
    $z_{\\text{entry}} = 1.5 \\cdot \\sqrt{1 + 100c}$
    
    $z_{\\text{exit}} = 0.3 \\cdot \\sqrt[4]{1 + 100c}$
    """)
    
    # First compute OU parameters for PC1 portfolio
    with st.spinner("Computing OU parameters..."):
        components, _ = meanrev.pca_portfolios(returns, n_components=1)
        portfolio_prices = (prices @ components[0])
        ou_params = meanrev.estimate_ou_params(portfolio_prices)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("θ (Mean Reversion)", f"{ou_params['theta']:.6f}")
    with col2:
        st.metric("μ (Long-term Mean)", f"${ou_params['mu']:.2f}")
    with col3:
        st.metric("σ (Volatility)", f"{ou_params['sigma']:.4f}")
    with col4:
        half_life = ou_params.get('half_life', np.log(2)/ou_params['theta'])
        st.metric("Half-Life", f"{half_life:.2f} periods")
    
    # Transaction costs
    transaction_costs = st.multiselect(
        "Transaction Costs (basis points)",
        [1, 5, 10, 20, 50],
        default=[1, 10, 50]
    )
    
    if not transaction_costs:
        st.warning("Please select at least one transaction cost")
        st.stop()
    
    # Compute thresholds
    threshold_results = {}
    for cost_bps in transaction_costs:
        cost = cost_bps / 10000  # Convert bps to decimal
        result = meanrev.optimal_thresholds(
            theta=ou_params['theta'],
            mu=ou_params['mu'],
            sigma=ou_params['sigma'],
            transaction_cost=cost
        )
        threshold_results[cost_bps] = result
    
    # Results table
    st.subheader("Optimal Thresholds")
    results_data = []
    for cost_bps in transaction_costs:
        r = threshold_results[cost_bps]
        results_data.append({
            'Transaction Cost (bps)': cost_bps,
            'Entry Threshold (σ)': f"{r['optimal_entry']:.2f}",
            'Exit Threshold (σ)': f"{r['optimal_exit']:.2f}",
            'Expected Holding (periods)': f"{r['expected_holding_period']:.1f}"
        })
    st.dataframe(pd.DataFrame(results_data), use_container_width=True)
    
    # Visualization
    st.subheader("Thresholds vs Transaction Costs")
    entry_thresholds = [threshold_results[c]['optimal_entry'] for c in transaction_costs]
    exit_thresholds = [threshold_results[c]['optimal_exit'] for c in transaction_costs]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=transaction_costs, y=entry_thresholds,
        name="Entry Threshold",
        mode='lines+markers',
        line=dict(color='red', width=3)
    ))
    fig.add_trace(go.Scatter(
        x=transaction_costs, y=exit_thresholds,
        name="Exit Threshold",
        mode='lines+markers',
        line=dict(color='green', width=3)
    ))
    fig.update_layout(
        xaxis_title="Transaction Cost (bps)",
        yaxis_title="Threshold (σ)",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

# Tab 5: Transaction Costs
with tab5:
    st.header("Transaction Cost Modeling")
    
    st.markdown("""
    ### Theory
    Net return with proportional transaction costs:
    
    $R_{\\text{net}} = R_{\\text{gross}} - c \\cdot |\\Delta \\text{position}| \\cdot \\text{price}$
    """)
    
    # Parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        entry_z = st.slider("Entry Z-Score", 1.0, 3.0, 2.0, 0.1)
    with col2:
        exit_z = st.slider("Exit Z-Score", 0.1, 1.0, 0.5, 0.1)
    with col3:
        cost_scenarios = st.multiselect(
            "Cost Scenarios (bps)",
            [0, 1, 5, 10, 20, 50],
            default=[0, 10, 50]
        )
    
    if not cost_scenarios:
        st.warning("Please select at least one cost scenario")
        st.stop()
    
    # Compute
    components, _ = meanrev.pca_portfolios(returns, n_components=1)
    portfolio_prices = (prices @ components[0])
    
    backtest_results = {}
    for cost_bps in cost_scenarios:
        cost = cost_bps / 10000
        with st.spinner(f"Backtesting with {cost_bps} bps..."):
            result = meanrev.backtest_with_costs(
                portfolio_prices,
                entry_z=entry_z,
                exit_z=exit_z,
                transaction_cost=cost
            )
            backtest_results[cost_bps] = result
    
    # Results table
    st.subheader("Backtest Results")
    results_data = []
    for cost_bps in cost_scenarios:
        r = backtest_results[cost_bps]
        results_data.append({
            'Cost (bps)': cost_bps,
            'Final PnL': f"${r['pnl'][-1]:,.2f}",
            'Total Costs': f"${r['total_costs']:,.2f}",
            'Sharpe Ratio': f"{r['sharpe']:.3f}",
            'Max Drawdown': f"{r['max_drawdown']:.2%}"
        })
    st.dataframe(pd.DataFrame(results_data), use_container_width=True)
    
    # PnL evolution
    st.subheader("Cumulative PnL Evolution")
    fig = go.Figure()
    for cost_bps in cost_scenarios:
        pnl = backtest_results[cost_bps]['pnl']
        fig.add_trace(go.Scatter(
            x=list(range(len(pnl))),
            y=pnl,
            name=f"{cost_bps} bps",
            mode='lines'
        ))
    fig.update_layout(
        xaxis_title="Time Period",
        yaxis_title="Cumulative PnL ($)",
        height=500,
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Positions
    st.subheader("Trading Positions")
    cost_bps = cost_scenarios[0]
    positions = backtest_results[cost_bps]['positions']
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(positions))),
        y=positions,
        fill='tozeroy',
        name="Position"
    ))
    fig.update_layout(
        xaxis_title="Time Period",
        yaxis_title="Position",
        height=300
    )
    st.plotly_chart(fig, use_container_width=True)

# Tab 6: Multi-Period
with tab6:
    st.header("Multi-Period Portfolio Optimization")
    
    st.markdown("""
    ### Theory
    Dynamic programming for T periods with rebalancing costs:
    
    $\\max_{w_1, ..., w_T} \\sum_{t=1}^T U(R_t | w_t) - \\lambda \\sum_{t=2}^T c \\cdot ||w_t - w_{t-1}||$
    """)
    
    # Parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        n_periods = st.selectbox("Number of Rebalancing Periods", [5, 10, 20, 50], index=1)
    with col2:
        gamma = st.slider("Risk Aversion (γ)", 0.5, 10.0, 2.0, 0.5)
    with col3:
        trans_cost = st.slider("Transaction Cost (%)", 0.0, 0.5, 0.1, 0.05) / 100
    
    # Compute
    with st.spinner("Running multi-period optimization..."):
        result = meanrev.multiperiod_optimize(
            returns,
            covariance,
            gamma=gamma,
            transaction_cost=trans_cost,
            n_periods=n_periods
        )
    
    # Results
    st.subheader("Results Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rebalancing Periods", n_periods)
    with col2:
        avg_holding = len(returns) / n_periods
        st.metric("Avg Holding (periods)", f"{avg_holding:.0f}")
    with col3:
        st.metric("Expected Utility", f"{result['expected_utility']:.6f}")
    
    # Weight evolution
    st.subheader("Portfolio Weight Evolution (Top 5 Assets)")
    weights_seq = result['weights_sequence']
    rebal_times = result['rebalance_times']
    
    fig = go.Figure()
    top_assets = prices.columns[:5]
    
    for i, asset in enumerate(top_assets):
        weights_over_time = [w[i] for w in weights_seq]
        fig.add_trace(go.Scatter(
            x=rebal_times,
            y=weights_over_time,
            name=asset,
            mode='lines+markers'
        ))
    
    fig.update_layout(
        xaxis_title="Time Period",
        yaxis_title="Portfolio Weight",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Weights heatmap
    st.subheader("All Weights Heatmap")
    weights_matrix = np.array(weights_seq).T
    
    fig = go.Figure(data=go.Heatmap(
        z=weights_matrix,
        x=rebal_times,
        y=prices.columns,
        colorscale='RdBu',
        zmid=0
    ))
    fig.update_layout(
        xaxis_title="Rebalancing Period",
        yaxis_title="Symbol",
        height=800
    )
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(f"""
**System Info:**
- Backend: {rust_status}
- Data Points: {prices.size:,}
- Computation Time: Real-time
- Framework: Streamlit + PyO3 + Rust
""")
