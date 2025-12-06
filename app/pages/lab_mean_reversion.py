"""
Mean Reversion Lab
Advanced statistical arbitrage analysis and strategy development
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import shared UI components
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.ui_components import render_sidebar_navigation, apply_custom_css, ensure_data_loaded

st.set_page_config(page_title="Mean Reversion Lab", page_icon="📉", layout="wide")

# Initialize session state
if 'historical_data' not in st.session_state:
    st.session_state.historical_data = None
if 'meanrev_results' not in st.session_state:
    st.session_state.meanrev_results = None

# Auto-load most recent dataset if no data is loaded
data_available = ensure_data_loaded()

# Render sidebar navigation and apply CSS
render_sidebar_navigation(current_page="Mean Reversion Lab")
apply_custom_css()

st.markdown('<h1 class="lab-header">📉 Mean Reversion Lab</h1>', unsafe_allow_html=True)
st.markdown("### Statistical arbitrage with Z-score analysis and cointegration testing")
st.markdown("---")

# Check if data is loaded
if not data_available or st.session_state.historical_data is None or st.session_state.historical_data.empty:
    st.markdown("""
    <div class="info-card">
        <h3>📊 No Data Loaded</h3>
        <p>Please load historical data first to use the Mean Reversion Lab.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚀 Go to Data Loader", type="primary", use_container_width=True):
        st.switch_page("pages/data_loader.py")
    st.stop()

# Main content
data = st.session_state.historical_data

# Handle multi-index data structure from data loader
if data is not None and isinstance(data.index, pd.MultiIndex):
    # Reset index to get symbol and timestamp as columns
    data = data.reset_index()

# Get list of unique symbols
if data is not None and 'symbol' in data.columns:
    available_symbols = data['symbol'].unique().tolist()
    st.success(f"✅ Data loaded: {len(data)} records, {len(available_symbols)} symbols")
else:
    available_symbols = [col for col in data.columns if col not in ['Date', 'date', 'timestamp', 'open', 'high', 'low', 'close', 'volume']] if data is not None else []
    st.success(f"✅ Data loaded: {len(data) if data is not None else 0} records, {len(available_symbols)} assets")

# Sidebar parameters
with st.sidebar:
    st.markdown("### 🎛️ Analysis Parameters")
    
    # Symbol selection
    if not available_symbols:
        st.error("No symbols found in data")
        st.stop()
    
    selected_symbol = st.selectbox("Select Asset", available_symbols)
    
    st.markdown("---")
    st.markdown("#### Z-Score Parameters")
    window = st.slider("Rolling Window", 10, 200, 50, 5)
    entry_threshold = st.slider("Entry Threshold (Z-score)", 1.0, 4.0, 2.0, 0.1)
    exit_threshold = st.slider("Exit Threshold (Z-score)", 0.0, 2.0, 0.5, 0.1)
    
    st.markdown("---")
    st.markdown("#### Cointegration Test")
    test_pairs = st.checkbox("Test Cointegration", value=False)
    
    if test_pairs and len(available_symbols) >= 2:
        symbol2 = st.selectbox("Pair Asset", [s for s in available_symbols if s != selected_symbol])

# Main analysis
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Z-Score Analysis", "🔄 Pairs Trading", "🎯 Multi-Asset Cointegration", "📈 Strategy Backtest", "📉 Performance", "✨ Sparse Mean-Reversion"])

with tab1:
    st.markdown("### Z-Score Mean Reversion Analysis")
    
    # Extract price data for selected symbol
    if data is not None and 'symbol' in data.columns:
        # Multi-symbol data structure
        symbol_data = data[data['symbol'] == selected_symbol].copy()
        if 'timestamp' in symbol_data.columns:
            symbol_data = symbol_data.set_index('timestamp').sort_index()
        prices = symbol_data['close'].dropna()
    elif data is not None and selected_symbol in data.columns:
        # Single-column data structure
        prices = data[selected_symbol].dropna()
    else:
        st.error(f"Symbol {selected_symbol} not found in data")
        st.stop()
    
    if len(prices) == 0:
        st.error(f"No price data available for {selected_symbol}")
        st.stop()
    
    # Ensure prices are numeric
    prices = pd.to_numeric(prices, errors='coerce').dropna()  # type: ignore[union-attr]
    
    if len(prices) < window:
        st.error(f"Not enough data points. Need at least {window} points, got {len(prices)}")
        st.stop()
    
    # Calculate statistics
    rolling_mean = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()
    z_score = (prices - rolling_mean) / rolling_std
    
    # Create visualization
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Price & Moving Average', 'Z-Score'),
        row_heights=[0.6, 0.4],
        vertical_spacing=0.1
    )
    
    # Price chart
    fig.add_trace(
        go.Scatter(x=prices.index, y=prices, name='Price', line={'color': 'blue', 'width': 2}),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=rolling_mean.index, y=rolling_mean, name=f'MA({window})', line={'color': 'orange', 'width': 2, 'dash': 'dash'}),
        row=1, col=1
    )
    
    # Z-score chart
    fig.add_trace(
        go.Scatter(x=z_score.index, y=z_score, name='Z-Score', line={'color': 'purple', 'width': 2}),
        row=2, col=1
    )
    
    # Add threshold lines
    fig.add_hline(y=entry_threshold, line_dash="dash", line_color="red", row=2, col=1, annotation_text="Entry Long")
    fig.add_hline(y=-entry_threshold, line_dash="dash", line_color="green", row=2, col=1, annotation_text="Entry Short")
    fig.add_hline(y=exit_threshold, line_dash="dot", line_color="gray", row=2, col=1, annotation_text="Exit")
    fig.add_hline(y=-exit_threshold, line_dash="dot", line_color="gray", row=2, col=1)
    
    fig.update_layout(height=800, showlegend=True, hovermode='x unified')
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Z-Score", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True, key="zscore_chart")
    
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Z-Score", f"{z_score.iloc[-1]:.2f}")
    with col2:
        st.metric("Mean", f"{prices.mean():.2f}")
    with col3:
        st.metric("Std Dev", f"{prices.std():.2f}")
    with col4:
        st.metric("Current Price", f"{prices.iloc[-1]:.2f}")
    
    # Trading signals
    st.markdown("#### 📊 Recent Trading Signals")
    signals = []
    for i in range(len(z_score)-1, max(0, len(z_score)-20), -1):
        if not np.isnan(z_score.iloc[i]):
            if z_score.iloc[i] > entry_threshold:
                signals.append({'Date': prices.index[i], 'Signal': 'SELL (Overbought)', 'Z-Score': f"{z_score.iloc[i]:.2f}", 'Price': f"{prices.iloc[i]:.2f}"})
            elif z_score.iloc[i] < -entry_threshold:
                signals.append({'Date': prices.index[i], 'Signal': 'BUY (Oversold)', 'Z-Score': f"{z_score.iloc[i]:.2f}", 'Price': f"{prices.iloc[i]:.2f}"})
    
    if signals:
        st.dataframe(pd.DataFrame(signals), use_container_width=True, hide_index=True)
    else:
        st.info("No recent signals in the last 20 periods")

with tab2:
    st.markdown("### Pairs Trading & Cointegration")
    
    if test_pairs and len(available_symbols) >= 2:
        try:
            from python.strategies.meanrev import engle_granger_test
            
            # Extract price data for both symbols
            if data is not None and 'symbol' in data.columns:
                # Long format data
                symbol1_data = data[data['symbol'] == selected_symbol].copy()
                symbol2_data = data[data['symbol'] == symbol2].copy()
                if 'timestamp' in symbol1_data.columns:
                    symbol1_data = symbol1_data.set_index('timestamp').sort_index()
                    symbol2_data = symbol2_data.set_index('timestamp').sort_index()
                prices1 = symbol1_data['close'].dropna()
                prices2 = symbol2_data['close'].dropna()
            elif data is not None and selected_symbol in data.columns:
                # Wide format data
                prices1 = data[selected_symbol].dropna()
                prices2 = data[symbol2].dropna()
            else:
                prices1 = pd.Series()
                prices2 = pd.Series()
            
            # Align data
            common_idx = prices1.index.intersection(prices2.index)
            prices1 = prices1.loc[common_idx]
            prices2 = prices2.loc[common_idx]
            
            # Cointegration test
            st.markdown(f"#### Testing {selected_symbol} vs {symbol2}")
            
            try:
                result = engle_granger_test(prices1, prices2)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Test Statistic", f"{result['statistic']:.4f}")
                    st.metric("Beta (Hedge Ratio)", f"{result['beta']:.4f}")
                with col2:
                    st.metric("P-Value", f"{result['pvalue']:.4f}")
                    cointegrated = result['pvalue'] < 0.05
                    st.metric("Cointegrated?", "✅ Yes" if cointegrated else "❌ No")
                
                if cointegrated:
                    st.success("These pairs are cointegrated and suitable for pairs trading!")
                else:
                    st.warning("These pairs are not significantly cointegrated.")
                    
            except Exception as e:
                st.error(f"Cointegration test requires statsmodels: {str(e)}")
                st.info("Install statsmodels: `pip install statsmodels`")
            
            # Spread analysis
            spread = prices1 - result.get('beta', 1.0) * prices2
            spread_zscore = (spread - spread.mean()) / spread.std()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=common_idx, y=spread_zscore, name='Spread Z-Score', line=dict(color='purple', width=2)))
            fig.add_hline(y=2, line_dash="dash", line_color="red", annotation_text="Sell")
            fig.add_hline(y=-2, line_dash="dash", line_color="green", annotation_text="Buy")
            fig.add_hline(y=0, line_dash="dot", line_color="gray")
            fig.update_layout(title="Pairs Trading Spread", height=400)
            st.plotly_chart(fig, use_container_width=True, key="spread_chart")
            
        except ImportError as e:
            st.error(f"Missing required library: {str(e)}")
            st.info("Some statistical tests require additional packages")
    else:
        st.info("Enable cointegration testing in the sidebar to analyze pairs trading opportunities")

with tab3:
    st.markdown("### 🎯 Multi-Asset Cointegration & Optimal Switching")
    st.markdown("Test multiple assets for cointegration and analyze mean-reversion with Hurst exponent")
    
    if len(available_symbols) < 2:
        st.warning("Need at least 2 assets for cointegration analysis")
    else:
        try:
            from python.strategies.optimal_switching import (
                engle_granger_cointegration, johansen_cointegration,
                estimate_ou_parameters, solve_hjb_pde, 
                backtest_optimal_switching, compute_strategy_metrics
            )
            from python.strategies.sparse_meanrev import hurst_exponent
            
            st.markdown("#### Select Assets for Analysis")
            
            # Method selection
            test_method = st.radio(
                "Cointegration Method",
                ["Pairwise (Engle-Granger)", "Multi-asset (Johansen)"],
                horizontal=True
            )
            
            if test_method == "Pairwise (Engle-Granger)":
                # Pairwise testing with all available symbols
                st.markdown("#### Pairwise Cointegration Matrix")
                
                if st.button("🔍 Run Pairwise Analysis", type="primary"):
                    with st.spinner("Testing all pairs for cointegration..."):
                        # Build cointegration matrix
                        n_assets = min(len(available_symbols), 20)  # Limit to 20 for performance
                        symbols_subset = available_symbols[:n_assets]
                        
                        results_matrix = np.zeros((n_assets, n_assets))
                        hedge_ratios = {}
                        hurst_values = {}
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        total_pairs = n_assets * (n_assets - 1) // 2
                        completed = 0
                        
                        for i in range(n_assets):
                            for j in range(i + 1, n_assets):
                                sym1 = symbols_subset[i]
                                sym2 = symbols_subset[j]
                                
                                # Extract prices
                                if 'symbol' in data.columns:
                                    p1 = data[data['symbol'] == sym1].set_index('timestamp')['close']
                                    p2 = data[data['symbol'] == sym2].set_index('timestamp')['close']
                                else:
                                    continue
                                
                                # Test cointegration
                                try:
                                    coint_result = engle_granger_cointegration(p1, p2)
                                    
                                    # Store p-value in matrix
                                    results_matrix[i, j] = coint_result.p_value
                                    results_matrix[j, i] = coint_result.p_value
                                    
                                    if coint_result.is_cointegrated:
                                        hedge_ratios[f"{sym1}/{sym2}"] = coint_result.hedge_ratio
                                        
                                        # Compute spread and test for mean-reversion
                                        common_idx = p1.index.intersection(p2.index)
                                        spread = p1.loc[common_idx] - coint_result.hedge_ratio * p2.loc[common_idx]
                                        
                                        try:
                                            hurst_result = hurst_exponent(spread)
                                            hurst_values[f"{sym1}/{sym2}"] = {
                                                'hurst': hurst_result.hurst_exponent,
                                                'is_mean_reverting': hurst_result.is_mean_reverting,
                                                'interpretation': hurst_result.interpretation
                                            }
                                        except:
                                            pass
                                except:
                                    results_matrix[i, j] = 1.0
                                    results_matrix[j, i] = 1.0
                                
                                completed += 1
                                progress_bar.progress(completed / total_pairs)
                                status_text.text(f"Testing {sym1} vs {sym2} ({completed}/{total_pairs})")
                        
                        progress_bar.empty()
                        status_text.empty()
                        
                        # Display results
                        st.markdown("#### Cointegration P-Value Matrix")
                        st.markdown("*Green = cointegrated (p < 0.05), Red = not cointegrated*")
                        
                        # Create heatmap
                        import plotly.graph_objects as go
                        
                        fig = go.Figure(data=go.Heatmap(
                            z=results_matrix,
                            x=symbols_subset,
                            y=symbols_subset,
                            colorscale='RdYlGn_r',
                            zmin=0,
                            zmax=0.1,
                            text=results_matrix,
                            texttemplate='%{text:.3f}',
                            textfont={"size": 10},
                            colorbar=dict(title="P-value")
                        ))
                        
                        fig.update_layout(
                            title="Cointegration P-Values (lower = more cointegrated)",
                            xaxis_title="Asset 2",
                            yaxis_title="Asset 1",
                            height=600
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show cointegrated pairs
                        if len(hedge_ratios) > 0:
                            st.markdown("#### ✅ Cointegrated Pairs")
                            
                            pairs_data = []
                            for pair, ratio in hedge_ratios.items():
                                hurst_info = hurst_values.get(pair, {})
                                pairs_data.append({
                                    'Pair': pair,
                                    'Hedge Ratio': f"{ratio:.4f}",
                                    'Hurst': f"{hurst_info.get('hurst', 0.0):.4f}",
                                    'Mean-Reverting': '✅' if hurst_info.get('is_mean_reverting', False) else '❌',
                                    'Interpretation': hurst_info.get('interpretation', 'N/A')
                                })
                            
                            pairs_df = pd.DataFrame(pairs_data)
                            st.dataframe(pairs_df, use_container_width=True)
                            
                            # Allow user to select a pair for optimal switching analysis
                            st.markdown("#### 🎯 Optimal Switching Analysis")
                            selected_pair = st.selectbox(
                                "Select a cointegrated pair for optimal switching strategy",
                                list(hedge_ratios.keys())
                            )
                            
                            if st.button("📊 Analyze Optimal Switching Strategy", type="primary"):
                                with st.spinner("Computing optimal switching boundaries..."):
                                    sym1, sym2 = selected_pair.split('/')
                                    
                                    # Extract prices
                                    if 'symbol' in data.columns:
                                        p1 = data[data['symbol'] == sym1].set_index('timestamp')['close']
                                        p2 = data[data['symbol'] == sym2].set_index('timestamp')['close']
                                    
                                    common_idx = p1.index.intersection(p2.index)
                                    p1_aligned = p1.loc[common_idx]
                                    p2_aligned = p2.loc[common_idx]
                                    
                                    hedge_ratio = hedge_ratios[selected_pair]
                                    spread = p1_aligned - hedge_ratio * p2_aligned
                                    
                                    # Estimate OU parameters
                                    ou_params = estimate_ou_parameters(spread)
                                    
                                    st.markdown("##### OU Process Parameters")
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.metric("κ (mean-reversion)", f"{ou_params.kappa:.4f}")
                                    with col2:
                                        st.metric("θ (long-term mean)", f"{ou_params.theta:.4f}")
                                    with col3:
                                        st.metric("σ (volatility)", f"{ou_params.sigma:.4f}")
                                    with col4:
                                        st.metric("Half-life (days)", f"{ou_params.half_life:.2f}")
                                    
                                    # Solve HJB equations
                                    st.markdown("##### Solving HJB Equations...")
                                    
                                    transaction_cost = st.slider("Transaction Cost (%)", 0.0, 1.0, 0.1, 0.05) / 100
                                    discount_rate = st.slider("Discount Rate (%/year)", 0.0, 10.0, 5.0, 0.5) / 100
                                    
                                    spread_std = spread.std()
                                    spread_mean = spread.mean()
                                    
                                    boundaries = solve_hjb_pde(
                                        ou_params,
                                        transaction_cost,
                                        discount_rate,
                                        spread_mean - 3*spread_std,
                                        spread_mean + 3*spread_std,
                                        n_points=500,
                                        max_iterations=5000
                                    )
                                    
                                    st.success("✅ Optimal boundaries computed!")
                                    st.code(str(boundaries))
                                    
                                    # Plot value functions
                                    st.markdown("##### Value Functions")
                                    
                                    fig_value = go.Figure()
                                    fig_value.add_trace(go.Scatter(
                                        x=boundaries.spread_grid,
                                        y=boundaries.V_open,
                                        name='V_open',
                                        line=dict(color='blue', width=2)
                                    ))
                                    fig_value.add_trace(go.Scatter(
                                        x=boundaries.spread_grid,
                                        y=boundaries.V_buy,
                                        name='V_buy (long spread)',
                                        line=dict(color='green', width=2)
                                    ))
                                    fig_value.add_trace(go.Scatter(
                                        x=boundaries.spread_grid,
                                        y=boundaries.V_sell,
                                        name='V_sell (short spread)',
                                        line=dict(color='red', width=2)
                                    ))
                                    
                                    # Add boundary lines
                                    fig_value.add_vline(x=boundaries.open_to_buy, line_dash="dash",
                                                       line_color="green", annotation_text="Open→Buy")
                                    fig_value.add_vline(x=boundaries.open_to_sell, line_dash="dash",
                                                       line_color="red", annotation_text="Open→Sell")
                                    fig_value.add_vline(x=boundaries.buy_to_close, line_dash="dot",
                                                       line_color="orange", annotation_text="Buy→Close")
                                    fig_value.add_vline(x=boundaries.sell_to_close, line_dash="dot",
                                                       line_color="purple", annotation_text="Sell→Close")
                                    
                                    fig_value.update_layout(
                                        title="Value Functions and Optimal Switching Boundaries",
                                        xaxis_title="Spread Value",
                                        yaxis_title="Value Function",
                                        height=500,
                                        showlegend=True
                                    )
                                    
                                    st.plotly_chart(fig_value, use_container_width=True)
                                    
                                    # Backtest strategy
                                    st.markdown("##### Backtest Results")
                                    
                                    equity_curve, trades_df = backtest_optimal_switching(
                                        p1_aligned,
                                        p2_aligned,
                                        hedge_ratio,
                                        boundaries,
                                        transaction_cost_bps=transaction_cost * 10000
                                    )
                                    
                                    metrics = compute_strategy_metrics(equity_curve, trades_df)
                                    
                                    # Display metrics
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.metric("Total Return", f"{metrics['Total Return']:.2%}")
                                        st.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")
                                    with col2:
                                        st.metric("Max Drawdown", f"{metrics['Max Drawdown']:.2%}")
                                        st.metric("Num Trades", f"{int(metrics['Num Trades'])}")
                                    with col3:
                                        st.metric("Win Rate", f"{metrics['Win Rate']:.2%}")
                                        st.metric("Avg Win", f"${metrics['Avg Win']:.2f}")
                                    with col4:
                                        st.metric("Avg Loss", f"${metrics['Avg Loss']:.2f}")
                                        st.metric("Profit Factor", f"{metrics['Profit Factor']:.2f}")
                                    
                                    # Plot equity curve
                                    fig_eq = go.Figure()
                                    fig_eq.add_trace(go.Scatter(
                                        x=equity_curve['timestamp'],
                                        y=equity_curve['total_equity'],
                                        name='Portfolio Value',
                                        line=dict(color='blue', width=2)
                                    ))
                                    
                                    fig_eq.update_layout(
                                        title="Portfolio Equity Curve",
                                        xaxis_title="Date",
                                        yaxis_title="Portfolio Value ($)",
                                        height=400
                                    )
                                    
                                    st.plotly_chart(fig_eq, use_container_width=True)
                                    
                                    # Plot spread with trading signals
                                    fig_spread = go.Figure()
                                    fig_spread.add_trace(go.Scatter(
                                        x=equity_curve['timestamp'],
                                        y=equity_curve['spread'],
                                        name='Spread',
                                        line=dict(color='black', width=1)
                                    ))
                                    
                                    # Color regions by state
                                    colors = {'open': 'gray', 'buy': 'lightgreen', 'sell': 'lightcoral'}
                                    for state, color in colors.items():
                                        state_data = equity_curve[equity_curve['state'] == state]
                                        if len(state_data) > 0:
                                            fig_spread.add_trace(go.Scatter(
                                                x=state_data['timestamp'],
                                                y=state_data['spread'],
                                                mode='markers',
                                                marker=dict(color=color, size=3),
                                                name=state.capitalize(),
                                                showlegend=True
                                            ))
                                    
                                    fig_spread.add_hline(y=boundaries.open_to_buy, line_dash="dash",
                                                        line_color="green", annotation_text="Open→Buy")
                                    fig_spread.add_hline(y=boundaries.open_to_sell, line_dash="dash",
                                                        line_color="red", annotation_text="Open→Sell")
                                    
                                    fig_spread.update_layout(
                                        title="Spread with Optimal Trading Signals",
                                        xaxis_title="Date",
                                        yaxis_title="Spread Value",
                                        height=400
                                    )
                                    
                                    st.plotly_chart(fig_spread, use_container_width=True)
                                    
                                    # Show trades table
                                    if len(trades_df) > 0:
                                        st.markdown("##### Trade Log")
                                        st.dataframe(trades_df, use_container_width=True)
                            
                            # Bulk Optimal Switching Analysis
                            st.markdown("---")
                            st.markdown("#### 🚀 Bulk Optimal Switching Analysis")
                            st.markdown("Automatically run optimal switching on all cointegrated pairs and rank by expected profit")
                            
                            if st.button("🎯 Run Bulk Optimal Switching", type="primary", key="bulk_optimal"):
                                with st.spinner("Analyzing all cointegrated pairs..."):
                                    bulk_results = []
                                    
                                    progress_bar = st.progress(0)
                                    status_text = st.empty()
                                    
                                    # Transaction cost and discount rate
                                    transaction_cost = 0.001  # 0.1%
                                    discount_rate = 0.05  # 5%
                                    
                                    for idx, pair in enumerate(hedge_ratios.keys()):
                                        try:
                                            status_text.text(f"Processing {pair}...")
                                            sym1, sym2 = pair.split('/')
                                            
                                            # Extract prices
                                            if 'symbol' in data.columns:
                                                p1 = data[data['symbol'] == sym1].set_index('timestamp')['close']
                                                p2 = data[data['symbol'] == sym2].set_index('timestamp')['close']
                                            
                                            common_idx = p1.index.intersection(p2.index)
                                            p1_aligned = p1.loc[common_idx]
                                            p2_aligned = p2.loc[common_idx]
                                            
                                            hedge_ratio = hedge_ratios[pair]
                                            spread = p1_aligned - hedge_ratio * p2_aligned
                                            
                                            # Estimate OU parameters
                                            ou_params = estimate_ou_parameters(spread)
                                            
                                            # Skip if parameters are invalid
                                            if ou_params.kappa <= 0 or ou_params.sigma <= 0:
                                                continue
                                            
                                            # Solve HJB equations
                                            spread_std = spread.std()
                                            spread_mean = spread.mean()
                                            
                                            boundaries = solve_hjb_pde(
                                                ou_params,
                                                transaction_cost,
                                                discount_rate,
                                                spread_mean - 3*spread_std,
                                                spread_mean + 3*spread_std,
                                                n_points=300,  # Fewer points for speed
                                                max_iterations=3000
                                            )
                                            
                                            # Backtest strategy
                                            equity_curve, trades_df = backtest_optimal_switching(
                                                p1_aligned,
                                                p2_aligned,
                                                hedge_ratio,
                                                boundaries,
                                                transaction_cost_bps=transaction_cost * 10000
                                            )
                                            
                                            metrics = compute_strategy_metrics(equity_curve, trades_df)
                                            
                                            # Store results
                                            bulk_results.append({
                                                'Pair': pair,
                                                'Hedge Ratio': hedge_ratio,
                                                'Kappa': ou_params.kappa,
                                                'Theta': ou_params.theta,
                                                'Sigma': ou_params.sigma,
                                                'Half-Life': ou_params.half_life,
                                                'Hurst': hurst_values.get(pair, {}).get('hurst', 0.0),
                                                'Total Return': metrics['Total Return'],
                                                'Sharpe Ratio': metrics['Sharpe Ratio'],
                                                'Max Drawdown': metrics['Max Drawdown'],
                                                'Num Trades': int(metrics['Num Trades']),
                                                'Win Rate': metrics['Win Rate'],
                                                'Profit Factor': metrics['Profit Factor'],
                                                'Expected Profit': metrics['Total Return'] * (1 - abs(metrics['Max Drawdown']))  # Risk-adjusted return
                                            })
                                            
                                        except Exception as e:
                                            st.warning(f"Failed to process {pair}: {str(e)}")
                                        
                                        progress_bar.progress((idx + 1) / len(hedge_ratios))
                                    
                                    progress_bar.empty()
                                    status_text.empty()
                                    
                                    if len(bulk_results) > 0:
                                        # Create DataFrame and sort by expected profit
                                        bulk_df = pd.DataFrame(bulk_results)
                                        bulk_df = bulk_df.sort_values('Expected Profit', ascending=False)
                                        
                                        st.success(f"✅ Analyzed {len(bulk_df)} pairs successfully!")
                                        
                                        # Display summary metrics
                                        st.markdown("##### 📊 Performance Summary")
                                        
                                        col1, col2, col3, col4 = st.columns(4)
                                        with col1:
                                            st.metric("Best Return", f"{bulk_df['Total Return'].max():.2%}")
                                            st.metric("Avg Return", f"{bulk_df['Total Return'].mean():.2%}")
                                        with col2:
                                            st.metric("Best Sharpe", f"{bulk_df['Sharpe Ratio'].max():.2f}")
                                            st.metric("Avg Sharpe", f"{bulk_df['Sharpe Ratio'].mean():.2f}")
                                        with col3:
                                            st.metric("Best Win Rate", f"{bulk_df['Win Rate'].max():.2%}")
                                            st.metric("Avg Win Rate", f"{bulk_df['Win Rate'].mean():.2%}")
                                        with col4:
                                            st.metric("Total Pairs", len(bulk_df))
                                            profitable = (bulk_df['Total Return'] > 0).sum()
                                            st.metric("Profitable", f"{profitable}/{len(bulk_df)}")
                                        
                                        # Display top performers
                                        st.markdown("##### 🏆 Top Performing Pairs (by Risk-Adjusted Return)")
                                        
                                        # Format display DataFrame
                                        display_df = bulk_df.copy()
                                        display_df['Total Return'] = display_df['Total Return'].apply(lambda x: f"{x:.2%}")
                                        display_df['Sharpe Ratio'] = display_df['Sharpe Ratio'].apply(lambda x: f"{x:.2f}")
                                        display_df['Max Drawdown'] = display_df['Max Drawdown'].apply(lambda x: f"{x:.2%}")
                                        display_df['Win Rate'] = display_df['Win Rate'].apply(lambda x: f"{x:.2%}")
                                        display_df['Profit Factor'] = display_df['Profit Factor'].apply(lambda x: f"{x:.2f}")
                                        display_df['Hedge Ratio'] = display_df['Hedge Ratio'].apply(lambda x: f"{x:.4f}")
                                        display_df['Kappa'] = display_df['Kappa'].apply(lambda x: f"{x:.4f}")
                                        display_df['Half-Life'] = display_df['Half-Life'].apply(lambda x: f"{x:.2f}")
                                        display_df['Hurst'] = display_df['Hurst'].apply(lambda x: f"{x:.4f}")
                                        display_df['Expected Profit'] = display_df['Expected Profit'].apply(lambda x: f"{x:.4f}")
                                        
                                        st.dataframe(
                                            display_df.head(20),  # Top 20
                                            use_container_width=True,
                                            height=600
                                        )
                                        
                                        # Visualize performance distribution
                                        st.markdown("##### 📈 Performance Distribution")
                                        
                                        fig_dist = go.Figure()
                                        
                                        # Scatter plot: Return vs Sharpe
                                        fig_dist.add_trace(go.Scatter(
                                            x=bulk_df['Sharpe Ratio'],
                                            y=bulk_df['Total Return'],
                                            mode='markers+text',
                                            text=bulk_df['Pair'],
                                            textposition="top center",
                                            textfont=dict(size=8),
                                            marker=dict(
                                                size=bulk_df['Num Trades'] / 2,
                                                color=bulk_df['Expected Profit'],
                                                colorscale='Viridis',
                                                showscale=True,
                                                colorbar=dict(title="Expected Profit")
                                            ),
                                            showlegend=False
                                        ))
                                        
                                        fig_dist.update_layout(
                                            title="Risk-Return Profile (size = num trades, color = expected profit)",
                                            xaxis_title="Sharpe Ratio",
                                            yaxis_title="Total Return",
                                            height=600,
                                            showlegend=False
                                        )
                                        
                                        st.plotly_chart(fig_dist, use_container_width=True)
                                        
                                        # Portfolio construction suggestion
                                        st.markdown("##### 💼 Portfolio Construction Recommendation")
                                        
                                        # Select top N pairs
                                        n_pairs = st.slider("Number of pairs for portfolio", 1, min(10, len(bulk_df)), 5)
                                        
                                        top_pairs = bulk_df.head(n_pairs)
                                        
                                        st.markdown(f"**Top {n_pairs} Pairs:**")
                                        for _, row in top_pairs.iterrows():
                                            st.markdown(f"- **{row['Pair']}**: Return {row['Total Return']}, Sharpe {row['Sharpe Ratio']:.2f}, Trades {int(row['Num Trades'])}")
                                        
                                        # Calculate portfolio metrics
                                        portfolio_return = top_pairs['Total Return'].mean()
                                        portfolio_sharpe = top_pairs['Sharpe Ratio'].mean()
                                        portfolio_max_dd = top_pairs['Max Drawdown'].mean()
                                        
                                        st.info(f"""
                                        **Expected Portfolio Performance:**
                                        - Average Return: {portfolio_return:.2%}
                                        - Average Sharpe: {portfolio_sharpe:.2f}
                                        - Average Max Drawdown: {portfolio_max_dd:.2%}
                                        - Diversification: {n_pairs} independent pairs
                                        """)
                                        
                                        # Download results
                                        csv = bulk_df.to_csv(index=False)
                                        st.download_button(
                                            label="📥 Download Full Results (CSV)",
                                            data=csv,
                                            file_name="optimal_switching_bulk_results.csv",
                                            mime="text/csv"
                                        )
                                    else:
                                        st.error("❌ No pairs could be successfully analyzed")
                        else:
                            st.warning("No cointegrated pairs found. Try different assets or adjust significance level.")
            
            else:  # Johansen method
                st.markdown("#### Multi-Asset Cointegration (Johansen Test)")
                st.info("Select 3-10 assets for multi-asset cointegration testing")
                
                max_assets = min(10, len(available_symbols))
                selected_assets = st.multiselect(
                    "Select assets for Johansen test",
                    available_symbols,
                    default=available_symbols[:min(5, len(available_symbols))]
                )
                
                if len(selected_assets) >= 2 and st.button("🔍 Run Johansen Test", type="primary"):
                    with st.spinner("Running Johansen cointegration test..."):
                        # Build price matrix
                        price_dict = {}
                        for sym in selected_assets:
                            if 'symbol' in data.columns:
                                prices = data[data['symbol'] == sym].set_index('timestamp')['close']
                                price_dict[sym] = prices
                        
                        # Align all series
                        prices_df = pd.DataFrame(price_dict).dropna()
                        
                        if len(prices_df) < 50:
                            st.error("Insufficient data points. Need at least 50 periods.")
                        else:
                            try:
                                results = johansen_cointegration(prices_df)
                                
                                st.markdown("#### Johansen Test Results")
                                
                                for i, result in enumerate(results):
                                    with st.expander(f"Cointegrating Relationship {i+1}"):
                                        st.code(result.summary())
                                        
                                        if result.is_cointegrated:
                                            st.success("✅ Significant cointegrating relationship found!")
                                        else:
                                            st.info("ℹ️ No significant cointegration at this level")
                                
                            except Exception as e:
                                st.error(f"Johansen test failed: {str(e)}")
                                st.info("Make sure you have statsmodels installed: `pip install statsmodels`")
        
        except ImportError as e:
            st.error(f"Required module not available: {str(e)}")
            st.info("Install required packages: `pip install statsmodels scipy`")

with tab4:
    st.markdown("### Strategy Backtest")
    
    if 'historical_data' not in st.session_state or st.session_state.historical_data is None:
        st.warning("⚠️ Please load data first from the Data Loader page")
    else:
        data = st.session_state.historical_data
        
        # Get symbols
        if isinstance(data, dict):
            symbols = list(data.keys())
        elif isinstance(data, pd.DataFrame):
            if 'symbol' in data.columns:
                symbols = data['symbol'].unique().tolist()
            else:
                symbols = ['Data']
        else:
            symbols = []
        
        if len(symbols) > 0:
            st.markdown("#### 🎯 Strategy Parameters")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                selected_symbol = st.selectbox("Symbol", symbols, key="backtest_symbol")
                lookback = st.slider("Lookback Window", 10, 200, 50, help="Window for mean calculation")
            
            with col2:
                entry_threshold = st.slider("Entry Threshold (σ)", 0.5, 3.0, 2.0, 0.1,
                                           help="Standard deviations from mean to enter")
                exit_threshold = st.slider("Exit Threshold (σ)", 0.0, 2.0, 0.5, 0.1,
                                          help="Standard deviations from mean to exit")
            
            with col3:
                transaction_cost_bps = st.slider("Transaction Cost (bps)", 0, 50, 10,
                                                 help="Basis points per trade")
                slippage_bps = st.slider("Slippage (bps)", 0, 50, 5,
                                        help="Average slippage per trade")
            
            if st.button("🚀 Run Backtest", type="primary"):
                with st.spinner("Running backtest..."):
                    # Extract data
                    if isinstance(data, dict):
                        df = data[selected_symbol]
                    elif isinstance(data, pd.DataFrame):
                        if 'symbol' in data.columns:
                            df = data[data['symbol'] == selected_symbol].copy()
                        else:
                            df = data.copy()
                    else:
                        st.error("Unsupported data format")
                        st.stop()
                    
                    # Find close column
                    close_col = None
                    for col in df.columns:
                        if col.lower() == 'close':
                            close_col = col
                            break
                    
                    if close_col is None:
                        st.error(f"Close price column not found")
                        st.stop()
                    
                    prices = df[close_col].values
                    
                    # Compute mean reversion signals
                    rolling_mean = pd.Series(prices).rolling(lookback).mean().values
                    rolling_std = pd.Series(prices).rolling(lookback).std().values
                    
                    # Z-score
                    z_scores = (prices - rolling_mean) / (rolling_std + 1e-8)  # type: ignore[operator]
                    
                    # Generate signals
                    positions = np.zeros(len(prices))
                    trades = []
                    entry_prices = []
                    
                    position = 0
                    entry_price = 0
                    
                    for i in range(lookback, len(prices)):
                        if position == 0:
                            # Enter long when oversold
                            if z_scores[i] < -entry_threshold:
                                position = 1
                                entry_price = prices[i]
                                trades.append({'idx': i, 'type': 'LONG', 'price': prices[i]})
                            # Enter short when overbought
                            elif z_scores[i] > entry_threshold:
                                position = -1
                                entry_price = prices[i]
                                trades.append({'idx': i, 'type': 'SHORT', 'price': prices[i]})
                        else:
                            # Exit conditions
                            exit_signal = False
                            
                            if position == 1:  # Long position
                                if z_scores[i] > -exit_threshold:  # Mean reversion
                                    exit_signal = True
                            elif position == -1:  # Short position
                                if z_scores[i] < exit_threshold:  # Mean reversion
                                    exit_signal = True
                            
                            if exit_signal:
                                pnl_pct = (prices[i] - entry_price) / entry_price * position
                                trades.append({
                                    'idx': i,
                                    'type': 'EXIT',
                                    'price': prices[i],
                                    'pnl_pct': pnl_pct,
                                    'position': position
                                })
                                position = 0
                        
                        positions[i] = position
                    
                    # Calculate P&L with costs
                    total_cost_bps = transaction_cost_bps + slippage_bps
                    cost_per_trade = total_cost_bps / 10000.0  # Convert to decimal
                    
                    trade_pnls = []
                    for i, trade in enumerate(trades):
                        if trade['type'] == 'EXIT':
                            gross_pnl = trade['pnl_pct']
                            # Deduct costs (2x for entry and exit)
                            net_pnl = gross_pnl - (2 * cost_per_trade)
                            trade_pnls.append(net_pnl)
                    
                    # Store in session state for performance metrics tab
                    st.session_state['backtest_results'] = {
                        'trades': trades,
                        'positions': positions,
                        'trade_pnls': trade_pnls,
                        'prices': prices,
                        'z_scores': z_scores,
                        'df_index': df.index,
                        'symbol': selected_symbol
                    }
                    
                    # Display results
                    st.markdown("### 📊 Backtest Results")
                    
                    col_a, col_b, col_c, col_d = st.columns(4)
                    
                    with col_a:
                        st.metric("Total Trades", len(trade_pnls))
                    with col_b:
                        total_return = sum(trade_pnls) * 100
                        st.metric("Total Return", f"{total_return:.2f}%")
                    with col_c:
                        win_rate = sum(1 for pnl in trade_pnls if pnl > 0) / len(trade_pnls) * 100 if trade_pnls else 0
                        st.metric("Win Rate", f"{win_rate:.1f}%")
                    with col_d:
                        avg_pnl = np.mean(trade_pnls) * 100 if trade_pnls else 0
                        st.metric("Avg Trade", f"{avg_pnl:.2f}%")
                    
                    # Plot equity curve and signals
                    fig = make_subplots(
                        rows=3, cols=1,
                        subplot_titles=('Price & Signals', 'Z-Score', 'Cumulative P&L'),
                        vertical_spacing=0.1,
                        row_heights=[0.4, 0.3, 0.3]
                    )
                    
                    # Price chart
                    fig.add_trace(
                        go.Scatter(x=df.index, y=prices, name='Price', line={'color': 'blue'}),
                        row=1, col=1
                    )
                    
                    # Add trade markers
                    long_entries = [t for t in trades if t['type'] == 'LONG']
                    short_entries = [t for t in trades if t['type'] == 'SHORT']
                    exits = [t for t in trades if t['type'] == 'EXIT']
                    
                    if long_entries:
                        fig.add_trace(
                            go.Scatter(
                                x=[df.index[t['idx']] for t in long_entries],
                                y=[t['price'] for t in long_entries],
                                mode='markers',
                                marker={'symbol': 'triangle-up', 'size': 10, 'color': 'green'},
                                name='Long Entry'
                            ),
                            row=1, col=1
                        )
                    
                    if short_entries:
                        fig.add_trace(
                            go.Scatter(
                                x=[df.index[t['idx']] for t in short_entries],
                                y=[t['price'] for t in short_entries],
                                mode='markers',
                                marker={'symbol': 'triangle-down', 'size': 10, 'color': 'red'},
                                name='Short Entry'
                            ),
                            row=1, col=1
                        )
                    
                    if exits:
                        fig.add_trace(
                            go.Scatter(
                                x=[df.index[t['idx']] for t in exits],
                                y=[t['price'] for t in exits],
                                mode='markers',
                                marker={'symbol': 'x', 'size': 10, 'color': 'orange'},
                                name='Exit'
                            ),
                            row=1, col=1
                        )
                    
                    # Z-score
                    fig.add_trace(
                        go.Scatter(x=df.index, y=z_scores, name='Z-Score', line={'color': 'purple'}),
                        row=2, col=1
                    )
                    fig.add_hline(y=entry_threshold, line_dash="dash", line_color="red", row=2, col=1)
                    fig.add_hline(y=-entry_threshold, line_dash="dash", line_color="green", row=2, col=1)
                    fig.add_hline(y=0, line_color="gray", row=2, col=1)
                    
                    # Cumulative P&L
                    cum_pnl = np.cumsum(trade_pnls) * 100 if trade_pnls else [0]
                    trade_indices = [t['idx'] for t in trades if t['type'] == 'EXIT']
                    
                    fig.add_trace(
                        go.Scatter(
                            x=[df.index[idx] for idx in trade_indices] if trade_indices else [df.index[0]],
                            y=cum_pnl if trade_indices else [0],
                            name='Cumulative P&L',
                            line={'color': 'green', 'width': 2},
                            fill='tozeroy'
                        ),
                        row=3, col=1
                    )
                    
                    fig.update_xaxes(title_text="Time", row=3, col=1)
                    fig.update_yaxes(title_text="Price", row=1, col=1)
                    fig.update_yaxes(title_text="Z-Score (σ)", row=2, col=1)
                    fig.update_yaxes(title_text="Cumulative Return (%)", row=3, col=1)
                    fig.update_layout(height=900, showlegend=True, hovermode='x unified')
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.success("✅ Backtest complete! View detailed metrics in the Performance Metrics tab.")

with tab5:
    st.markdown("### Performance Metrics")
    
    if 'backtest_results' not in st.session_state:
        st.info("💡 Run a backtest in the Strategy Backtest tab first")
    else:
        results = st.session_state['backtest_results']
        trade_pnls = results['trade_pnls']
        
        if not trade_pnls:
            st.warning("No completed trades to analyze")
        else:
            st.markdown("#### 📈 Risk-Adjusted Performance")
            
            # Calculate metrics
            returns = np.array(trade_pnls)
            cum_returns = np.cumsum(returns)
            
            # Sharpe Ratio (annualized, assuming ~250 trading days)
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe_ratio = (mean_return / std_return) * np.sqrt(250) if std_return > 0 else 0
            
            # Maximum Drawdown
            running_max = np.maximum.accumulate(cum_returns)
            drawdowns = cum_returns - running_max
            max_drawdown = np.min(drawdowns) * 100 if len(drawdowns) > 0 else 0
            
            # Win/Loss metrics
            wins = [r for r in returns if r > 0]
            losses = [r for r in returns if r < 0]
            win_rate = len(wins) / len(returns) * 100
            
            avg_win = np.mean(wins) * 100 if wins else 0
            avg_loss = np.mean(losses) * 100 if losses else 0
            profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float('inf')
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}",
                         help="Risk-adjusted return (annualized)")
            with col2:
                st.metric("Max Drawdown", f"{max_drawdown:.2f}%",
                         help="Largest peak-to-trough decline")
            with col3:
                st.metric("Win Rate", f"{win_rate:.1f}%",
                         help="Percentage of profitable trades")
            with col4:
                st.metric("Profit Factor", f"{profit_factor:.2f}" if profit_factor != float('inf') else "∞",
                         help="Gross profit / Gross loss")
            
            col5, col6, col7, col8 = st.columns(4)
            
            with col5:
                st.metric("Avg Win", f"{avg_win:.2f}%")
            with col6:
                st.metric("Avg Loss", f"{avg_loss:.2f}%")
            with col7:
                st.metric("Total Trades", len(returns))
            with col8:
                total_pnl = sum(returns) * 100
                st.metric("Total P&L", f"{total_pnl:.2f}%")
            
            # Detailed analysis
            st.markdown("#### 📊 Trade Distribution")
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                # P&L histogram
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=returns * 100,
                    nbinsx=30,
                    name='Trade P&L',
                    marker_color='lightblue'
                ))
                fig_hist.update_layout(
                    title='Distribution of Trade Returns',
                    xaxis_title='Return (%)',
                    yaxis_title='Frequency',
                    height=400
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col_b:
                # Drawdown chart
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(
                    y=drawdowns * 100,
                    name='Drawdown',
                    line={'color': 'red', 'width': 2},
                    fill='tozeroy',
                    fillcolor='rgba(255,0,0,0.1)'
                ))
                fig_dd.update_layout(
                    title='Underwater Plot (Drawdowns)',
                    xaxis_title='Trade Number',
                    yaxis_title='Drawdown (%)',
                    height=400
                )
                st.plotly_chart(fig_dd, use_container_width=True)
            
            # Trade log
            st.markdown("#### 📋 Trade Log")
            
            trade_log = []
            for i, pnl in enumerate(returns):
                trade_log.append({
                    'Trade #': i + 1,
                    'P&L (%)': f"{pnl * 100:.2f}",
                    'Cumulative (%)': f"{cum_returns[i] * 100:.2f}",
                    'Result': '✅ Win' if pnl > 0 else '❌ Loss'
                })
            
            st.dataframe(pd.DataFrame(trade_log), use_container_width=True, height=300)

with tab6:
    st.markdown("### ✨ Sparse Mean-Reverting Portfolios")
    st.markdown("Advanced sparse decomposition algorithms for identifying small, mean-reverting portfolios")
    
    try:
        from python.strategies.sparse_meanrev import (
            sparse_pca, box_tao_decomposition, hurst_exponent, 
            sparse_cointegration, RUST_AVAILABLE
        )
        
        if RUST_AVAILABLE:
            st.success("⚡ Rust acceleration enabled - High performance mode")
        else:
            st.info("ℹ️ Using Python fallback implementations")
        
        # Method selection
        st.markdown("#### Select Method")
        method = st.selectbox(
            "Algorithm",
            ["Sparse PCA", "Box & Tao Decomposition", "Sparse Cointegration"],
            help="Choose sparse decomposition method"
        )
        
        # Get multi-asset data
        if 'symbol' in data.columns and len(available_symbols) >= 5:
            # Build price matrix
            st.markdown(f"**Using {len(available_symbols)} assets for analysis**")
            
            # Pivot to get price matrix
            if 'close' in data.columns:
                prices_pivot = data.pivot(columns='symbol', values='close')
            else:
                st.error("No 'close' price column found")
                st.stop()
            
            # Remove NaN
            prices_pivot = prices_pivot.dropna()
            
            if len(prices_pivot) < 100:
                st.warning("Insufficient data points. Need at least 100 periods.")
                st.stop()
            
            # Parameters
            col1, col2 = st.columns(2)
            with col1:
                if method == "Sparse PCA":
                    lambda_param = st.slider("Sparsity (λ)", 0.01, 1.0, 0.2, 0.01,
                                            help="Higher = sparser portfolios")
                    n_components = st.slider("Components", 1, 5, 3)
                elif method == "Box & Tao Decomposition":
                    lambda_param = st.slider("Sparsity (λ)", 0.01, 1.0, 0.1, 0.01)
                    mu_param = st.slider("Augmented Lagrangian (μ)", 0.001, 0.1, 0.01, 0.001)
                else:  # Sparse Cointegration
                    lambda_l1 = st.slider("L1 Penalty", 0.01, 1.0, 0.1, 0.01)
                    lambda_l2 = st.slider("L2 Penalty", 0.001, 0.1, 0.01, 0.001)
                    target_idx = st.selectbox("Target Asset", range(len(available_symbols)),
                                             format_func=lambda x: available_symbols[x])
            
            with col2:
                lookback = st.slider("Lookback Period", 100, len(prices_pivot), 
                                   min(252, len(prices_pivot)))
            
            # Run analysis button
            if st.button("🚀 Run Sparse Analysis", type="primary"):
                with st.spinner("Computing sparse decomposition..."):
                    try:
                        # Use last lookback periods
                        recent_prices = prices_pivot.iloc[-lookback:]
                        returns = recent_prices.pct_change().dropna()
                        
                        if method == "Sparse PCA":
                            result = sparse_pca(returns, n_components=n_components, 
                                              lambda_=lambda_param)
                            
                            st.markdown("#### 📊 Sparse PCA Results")
                            st.code(result.summary())
                            
                            # Display weights
                            st.markdown("**Portfolio Weights**")
                            for i in range(n_components):
                                weights = result.get_portfolio(i)
                                weights.index = available_symbols[:len(weights)]
                                
                                # Only show non-zero weights
                                non_zero = weights[weights.abs() > 0.001]
                                
                                st.markdown(f"**Component {i+1}** ({len(non_zero)} assets)")
                                fig = go.Figure(data=[
                                    go.Bar(x=non_zero.index, y=non_zero.values,
                                          marker_color=['green' if x > 0 else 'red' for x in non_zero.values])
                                ])
                                fig.update_layout(title=f'Component {i+1} Weights',
                                                xaxis_title='Asset',
                                                yaxis_title='Weight',
                                                height=300)
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Compute portfolio value
                                portfolio_returns = (returns * weights).sum(axis=1)
                                portfolio_value = (1 + portfolio_returns).cumprod()
                                
                                # Test for mean-reversion
                                hurst_result = hurst_exponent(portfolio_value)
                                st.info(f"**Hurst Exponent:** {hurst_result.hurst_exponent:.4f} "
                                       f"({hurst_result.interpretation})")
                                
                                if hurst_result.is_mean_reverting:
                                    st.success("✅ Portfolio is mean-reverting - Suitable for trading!")
                                else:
                                    st.warning("⚠️ Portfolio not significantly mean-reverting")
                        
                        elif method == "Box & Tao Decomposition":
                            result = box_tao_decomposition(recent_prices, 
                                                          lambda_=lambda_param,
                                                          mu=mu_param)
                            
                            st.markdown("#### 🔄 Box & Tao Decomposition")
                            st.code(result.convergence_info())
                            
                            # Visualize decomposition
                            fig = make_subplots(rows=3, cols=1,
                                              subplot_titles=('Low-Rank (Common Factors)',
                                                            'Sparse (Mean-Reverting Opportunities)',
                                                            'Noise'))
                            
                            for i, asset in enumerate(available_symbols[:5]):  # Show first 5 assets
                                if i < result.low_rank.shape[1]:
                                    fig.add_trace(go.Scatter(y=result.low_rank[:, i], 
                                                           name=f'{asset} L',
                                                           line=dict(width=1)),
                                                row=1, col=1)
                                    fig.add_trace(go.Scatter(y=result.sparse[:, i],
                                                           name=f'{asset} S',
                                                           line=dict(width=1)),
                                                row=2, col=1)
                                    fig.add_trace(go.Scatter(y=result.noise[:, i],
                                                           name=f'{asset} N',
                                                           line=dict(width=1)),
                                                row=3, col=1)
                            
                            fig.update_layout(height=900, showlegend=False)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Use sparse component for trading
                            st.markdown("**Trading Opportunities from Sparse Component**")
                            sparse_df = pd.DataFrame(result.sparse, 
                                                    columns=available_symbols[:result.sparse.shape[1]])
                            
                            # Find most sparse assets (largest absolute values)
                            sparse_magnitude = sparse_df.abs().mean()
                            top_sparse = sparse_magnitude.nlargest(5)
                            
                            st.write("Top 5 assets with sparse (idiosyncratic) behavior:")
                            st.dataframe(top_sparse.to_frame('Magnitude'))
                        
                        else:  # Sparse Cointegration
                            result = sparse_cointegration(recent_prices.values,
                                                         target_asset=target_idx,
                                                         lambda_l1=lambda_l1,
                                                         lambda_l2=lambda_l2)
                            
                            st.markdown("#### 🔗 Sparse Cointegration")
                            st.code(result.summary())
                            
                            # Display weights
                            weights_series = pd.Series(result.weights, 
                                                      index=available_symbols[:len(result.weights)])
                            non_zero = weights_series[weights_series.abs() > 0.001]
                            
                            st.markdown(f"**Portfolio ({len(non_zero)} assets)**")
                            fig = go.Figure(data=[
                                go.Bar(x=non_zero.index, y=non_zero.values,
                                      marker_color=['green' if x > 0 else 'red' for x in non_zero.values])
                            ])
                            fig.update_layout(title='Cointegrating Portfolio Weights',
                                            height=400)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Test residuals for mean-reversion
                            hurst_result = hurst_exponent(result.residuals)
                            
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("Hurst Exponent", f"{hurst_result.hurst_exponent:.4f}")
                                if hurst_result.is_mean_reverting:
                                    st.success("✅ Residuals are mean-reverting")
                                else:
                                    st.warning("⚠️ Residuals not mean-reverting")
                            
                            with col_b:
                                st.metric("95% CI", 
                                        f"[{hurst_result.confidence_interval[0]:.4f}, "
                                        f"{hurst_result.confidence_interval[1]:.4f}]")
                            
                            # Plot residuals
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(y=result.residuals, 
                                                    name='Cointegration Residuals',
                                                    line=dict(color='blue', width=1)))
                            fig.add_hline(y=0, line_dash="dash", line_color="gray")
                            fig.update_layout(title='Cointegrating Residuals (Should be Stationary)',
                                            xaxis_title='Time',
                                            yaxis_title='Residual',
                                            height=400)
                            st.plotly_chart(fig, use_container_width=True)
                    
                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
        else:
            st.warning("⚠️ Need at least 5 assets for sparse portfolio analysis")
            st.info("Load more symbols in the Data Loader page")
    
    except ImportError as e:
        st.error("❌ Sparse mean-reversion module not available")
        st.code(f"Import error: {str(e)}")
        st.info("Make sure `python/sparse_meanrev.py` is in your path and Rust connector is built")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>📉 Mean Reversion Lab | Part of HFT Arbitrage Lab</p>
</div>
""", unsafe_allow_html=True)
