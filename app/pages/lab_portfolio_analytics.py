"""
Portfolio Analytics Lab
Advanced portfolio optimization, risk metrics, and performance analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
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
    
    if 'portfolio_weights' not in st.session_state:
        st.info("💡 Optimize a portfolio in the Optimization tab first")
    else:
        weights = st.session_state['portfolio_weights']
        data = st.session_state['historical_data']
        selected_assets = st.session_state['selected_assets']
        
        st.markdown("#### 🎯 Risk Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            confidence_level = st.slider("VaR Confidence Level", 90, 99, 95, 1,
                                        help="Confidence level for VaR calculation")
            time_horizon = st.slider("Time Horizon (days)", 1, 30, 10,
                                    help="Risk measurement period")
        
        with col2:
            num_simulations = st.slider("Monte Carlo Simulations", 1000, 50000, 10000, 1000,
                                       help="Number of scenarios for stress testing")
        
        if st.button("📊 Calculate Risk Metrics", type="primary"):
            with st.spinner("Computing risk analytics..."):
                # Extract returns for selected assets
                returns_dict = {}
                for asset in selected_assets:
                    if isinstance(data, dict):
                        df = data[asset]
                    else:
                        df = data[data['symbol'] == asset].copy() if 'symbol' in data.columns else data.copy()
                    
                    close_col = None
                    for col in df.columns:
                        if col.lower() == 'close':
                            close_col = col
                            break
                    
                    if close_col:
                        prices = df[close_col].values
                        returns_dict[asset] = np.diff(prices) / prices[:-1]
                
                # Align returns to same length
                min_length = min(len(r) for r in returns_dict.values())
                aligned_returns = {k: v[-min_length:] for k, v in returns_dict.items()}
                
                # Portfolio returns
                returns_matrix = np.array([aligned_returns[asset] for asset in selected_assets]).T
                portfolio_returns = returns_matrix @ weights
                
                # 1. Value at Risk (VaR)
                var_percentile = 100 - confidence_level
                var_historical = np.percentile(portfolio_returns, var_percentile) * np.sqrt(time_horizon)
                
                # Parametric VaR (assuming normal distribution)
                var_parametric = norm.ppf(var_percentile / 100) * np.std(portfolio_returns) * np.sqrt(time_horizon)
                
                # 2. Expected Shortfall (CVaR)
                threshold = np.percentile(portfolio_returns, var_percentile)
                cvar = portfolio_returns[portfolio_returns <= threshold].mean() * np.sqrt(time_horizon)
                
                # 3. Maximum Drawdown
                cum_returns = np.cumprod(1 + portfolio_returns)
                running_max = np.maximum.accumulate(cum_returns)
                drawdowns = (cum_returns - running_max) / running_max
                max_drawdown = np.min(drawdowns)
                
                # 4. Beta (if we have market proxy)
                # Using equal-weighted portfolio as market proxy
                market_returns = returns_matrix.mean(axis=1)
                covariance = np.cov(portfolio_returns, market_returns)[0, 1]
                market_variance = np.var(market_returns)
                beta = covariance / market_variance if market_variance > 0 else 1.0
                
                # Display metrics
                st.markdown("### 📊 Risk Metrics")
                
                col_a, col_b, col_c, col_d = st.columns(4)
                
                with col_a:
                    st.metric("VaR (Historical)", f"{var_historical*100:.2f}%",
                             help=f"Maximum loss at {confidence_level}% confidence")
                with col_b:
                    st.metric("VaR (Parametric)", f"{var_parametric*100:.2f}%",
                             help="VaR assuming normal distribution")
                with col_c:
                    st.metric("Expected Shortfall", f"{cvar*100:.2f}%",
                             help="Average loss beyond VaR")
                with col_d:
                    st.metric("Max Drawdown", f"{max_drawdown*100:.2f}%",
                             help="Largest peak-to-trough decline")
                
                col_e, col_f, col_g, col_h = st.columns(4)
                
                with col_e:
                    st.metric("Portfolio Beta", f"{beta:.2f}",
                             help="Sensitivity to market movements")
                with col_f:
                    volatility = np.std(portfolio_returns) * np.sqrt(252)
                    st.metric("Annualized Vol", f"{volatility*100:.2f}%")
                with col_g:
                    sharpe = np.mean(portfolio_returns) / np.std(portfolio_returns) * np.sqrt(252)
                    st.metric("Sharpe Ratio", f"{sharpe:.2f}")
                with col_h:
                    sortino_denominator = np.std(portfolio_returns[portfolio_returns < 0])
                    sortino = np.mean(portfolio_returns) / sortino_denominator * np.sqrt(252) if sortino_denominator > 0 else 0
                    st.metric("Sortino Ratio", f"{sortino:.2f}")
                
                # Monte Carlo Stress Testing
                st.markdown("### 🎲 Monte Carlo Stress Testing")
                
                # Simulate portfolio scenarios
                mean_return = np.mean(portfolio_returns)
                std_return = np.std(portfolio_returns)
                
                simulated_returns = np.random.normal(
                    mean_return * time_horizon,
                    std_return * np.sqrt(time_horizon),
                    num_simulations
                )
                
                # Stress scenarios
                scenarios = {
                    'Base Case': simulated_returns,
                    'Market Crash (-2σ)': np.random.normal(mean_return * time_horizon - 2 * std_return * np.sqrt(time_horizon),
                                                          std_return * np.sqrt(time_horizon), num_simulations),
                    'High Volatility (2x σ)': np.random.normal(mean_return * time_horizon,
                                                               2 * std_return * np.sqrt(time_horizon), num_simulations),
                    'Bull Market (+2σ)': np.random.normal(mean_return * time_horizon + 2 * std_return * np.sqrt(time_horizon),
                                                         std_return * np.sqrt(time_horizon), num_simulations)
                }
                
                # Plot scenarios
                fig = go.Figure()
                
                for scenario_name, scenario_returns in scenarios.items():
                    fig.add_trace(go.Histogram(
                        x=scenario_returns * 100,
                        name=scenario_name,
                        opacity=0.6,
                        nbinsx=50
                    ))
                
                fig.update_layout(
                    title=f'Portfolio Return Distribution ({time_horizon}-day horizon)',
                    xaxis_title='Return (%)',
                    yaxis_title='Frequency',
                    barmode='overlay',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Scenario statistics
                st.markdown("#### 📈 Scenario Analysis")
                
                scenario_stats = []
                for scenario_name, scenario_returns in scenarios.items():
                    scenario_stats.append({
                        'Scenario': scenario_name,
                        'Mean': f"{np.mean(scenario_returns)*100:.2f}%",
                        'Median': f"{np.median(scenario_returns)*100:.2f}%",
                        'Std Dev': f"{np.std(scenario_returns)*100:.2f}%",
                        f'VaR {confidence_level}%': f"{np.percentile(scenario_returns, var_percentile)*100:.2f}%",
                        'Min': f"{np.min(scenario_returns)*100:.2f}%",
                        'Max': f"{np.max(scenario_returns)*100:.2f}%"
                    })
                
                st.dataframe(pd.DataFrame(scenario_stats), use_container_width=True)
                
                # Risk decomposition
                st.markdown("### 🔍 Risk Decomposition")
                
                # Component VaR (marginal contribution to risk)
                component_vars = []
                for i, asset in enumerate(selected_assets):
                    # Perturb weight slightly
                    perturbed_weights = weights.copy()
                    epsilon = 0.01
                    perturbed_weights[i] += epsilon
                    perturbed_weights = perturbed_weights / perturbed_weights.sum()  # Renormalize
                    
                    perturbed_returns = returns_matrix @ perturbed_weights
                    perturbed_var = np.percentile(perturbed_returns, var_percentile) * np.sqrt(time_horizon)
                    
                    marginal_var = (perturbed_var - var_historical) / epsilon
                    component_var = marginal_var * weights[i]
                    
                    component_vars.append({
                        'Asset': asset,
                        'Weight': f"{weights[i]*100:.1f}%",
                        'Marginal VaR': f"{marginal_var*100:.3f}%",
                        'Component VaR': f"{component_var*100:.3f}%",
                        'Contribution': f"{(component_var/var_historical)*100:.1f}%" if var_historical != 0 else "N/A"
                    })
                
                st.dataframe(pd.DataFrame(component_vars), use_container_width=True)

with tab4:
    st.markdown("### Performance Attribution")
    
    if 'portfolio_weights' not in st.session_state:
        st.info("💡 Optimize a portfolio in the Optimization tab first")
    else:
        weights = st.session_state['portfolio_weights']
        data = st.session_state['historical_data']
        selected_assets = st.session_state['selected_assets']
        
        st.markdown("#### 📊 Return Attribution Analysis")
        
        if st.button("🔍 Analyze Attribution", type="primary"):
            with st.spinner("Computing attribution..."):
                # Extract returns
                returns_dict = {}
                for asset in selected_assets:
                    if isinstance(data, dict):
                        df = data[asset]
                    else:
                        df = data[data['symbol'] == asset].copy() if 'symbol' in data.columns else data.copy()
                    
                    close_col = None
                    for col in df.columns:
                        if col.lower() == 'close':
                            close_col = col
                            break
                    
                    if close_col:
                        prices = df[close_col].values
                        returns_dict[asset] = np.diff(prices) / prices[:-1]
                
                # Align returns
                min_length = min(len(r) for r in returns_dict.values())
                aligned_returns = {k: v[-min_length:] for k, v in returns_dict.items()}
                
                # Portfolio returns
                returns_matrix = np.array([aligned_returns[asset] for asset in selected_assets]).T
                portfolio_returns = returns_matrix @ weights
                
                # Attribution metrics
                st.markdown("### 📈 Asset Contribution")
                
                attribution_data = []
                total_return = np.sum(portfolio_returns)
                
                for i, asset in enumerate(selected_assets):
                    asset_returns = aligned_returns[asset]
                    asset_contribution = np.sum(asset_returns) * weights[i]
                    contribution_pct = (asset_contribution / total_return * 100) if total_return != 0 else 0
                    
                    attribution_data.append({
                        'Asset': asset,
                        'Weight': f"{weights[i]*100:.1f}%",
                        'Asset Return': f"{np.sum(asset_returns)*100:.2f}%",
                        'Contribution': f"{asset_contribution*100:.2f}%",
                        '% of Total': f"{contribution_pct:.1f}%",
                        'Sharpe Ratio': f"{(np.mean(asset_returns) / np.std(asset_returns) * np.sqrt(252)):.2f}"
                    })
                
                st.dataframe(pd.DataFrame(attribution_data), use_container_width=True)
                
                # Contribution breakdown chart
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=[a['Asset'] for a in attribution_data],
                    y=[float(a['Contribution'].rstrip('%')) for a in attribution_data],
                    name='Return Contribution',
                    marker_color='lightblue'
                ))
                
                fig.update_layout(
                    title='Asset Return Contribution (%)',
                    xaxis_title='Asset',
                    yaxis_title='Contribution (%)',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Factor exposure (simplified: momentum and volatility)
                st.markdown("### 🔬 Factor Exposure")
                
                factor_data = []
                
                for i, asset in enumerate(selected_assets):
                    asset_returns = aligned_returns[asset]
                    
                    # Momentum factor (trailing 20-day return)
                    momentum = np.mean(asset_returns[-20:]) if len(asset_returns) >= 20 else np.mean(asset_returns)
                    
                    # Volatility factor
                    volatility = np.std(asset_returns)
                    
                    # Size factor (weight)
                    size_exposure = weights[i]
                    
                    factor_data.append({
                        'Asset': asset,
                        'Momentum': f"{momentum*100:.3f}%",
                        'Volatility': f"{volatility*100:.2f}%",
                        'Size Exposure': f"{size_exposure*100:.1f}%"
                    })
                
                st.dataframe(pd.DataFrame(factor_data), use_container_width=True)
                
                # Portfolio-level metrics
                st.markdown("### 📊 Portfolio Summary")
                
                col_a, col_b, col_c, col_d = st.columns(4)
                
                with col_a:
                    st.metric("Total Return", f"{total_return*100:.2f}%")
                with col_b:
                    ann_return = np.mean(portfolio_returns) * 252 * 100
                    st.metric("Annualized Return", f"{ann_return:.2f}%")
                with col_c:
                    ann_vol = np.std(portfolio_returns) * np.sqrt(252) * 100
                    st.metric("Annualized Vol", f"{ann_vol:.2f}%")
                with col_d:
                    sharpe = (np.mean(portfolio_returns) / np.std(portfolio_returns) * np.sqrt(252))
                    st.metric("Sharpe Ratio", f"{sharpe:.2f}")
    
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
