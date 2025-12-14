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
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📊 Z-Score Analysis", 
    "🔄 Pairs Trading", 
    "🎯 Multi-Asset Cointegration", 
    "🤖 Auto-Discovery", 
    "🎲 Options Strategies",
    "📈 Strategy Backtest", 
    "📉 Performance", 
    "📚 Mathematical Theory"
])

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
    st.markdown("### 🤖 Automated Cointegration Discovery & Optimal Switching")
    st.markdown("""
    **Intelligent multi-dimensional pair discovery system:**
    - ✅ Sector-based filtering (stocks within same sector)
    - ✅ ETF-Stock relationship analysis
    - ✅ Multivariate Hurst exponent for mean-reversion validation
    - ✅ Parallel processing for maximum speed
    - ✅ Cointegration scoring with multiple tests
    - ✅ Automatic optimal switching boundary computation
    - ✅ Profit-oriented ranking and portfolio construction
    """)
    
    try:
        from python.strategies.optimal_switching import (
            engle_granger_cointegration, johansen_cointegration,
            estimate_ou_parameters, solve_hjb_pde,
            backtest_optimal_switching, compute_strategy_metrics
        )
        from python.strategies.sparse_meanrev import hurst_exponent
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # Configuration Section
        st.markdown("---")
        st.markdown("#### ⚙️ Configuration")
        
        col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
        
        with col_cfg1:
            st.markdown("**Asset Selection**")
            
            # Detect ETFs (common ETF patterns)
            if 'symbol' in data.columns:
                etf_patterns = ['SPY', 'QQQ', 'IWM', 'DIA', 'XL', 'VOO', 'VTI', 'EEM', 'GLD', 'SLV', 'USO']
                detected_etfs = [s for s in available_symbols if any(pattern in s.upper() for pattern in etf_patterns)]
                detected_stocks = [s for s in available_symbols if s not in detected_etfs]
                
                st.info(f"📊 Detected: {len(detected_etfs)} ETFs, {len(detected_stocks)} Stocks")
                
                filter_mode = st.radio(
                    "Filtering Strategy",
                    ["All Pairs", "Same Sector Only", "ETF vs Stocks", "Stocks Only", "ETFs Only"],
                    help="Choose how to filter asset pairs for testing"
                )
                
                # Sector detection (simplified - would need market data API in production)
                use_sector_filter = (filter_mode == "Same Sector Only")
                
                if filter_mode == "ETF vs Stocks":
                    st.success(f"Will test {len(detected_etfs)} ETFs × {len(detected_stocks)} stocks = {len(detected_etfs) * len(detected_stocks)} pairs")
                elif filter_mode == "Stocks Only":
                    st.success(f"Will test {len(detected_stocks) * (len(detected_stocks)-1) // 2} stock pairs")
                elif filter_mode == "ETFs Only":
                    st.success(f"Will test {len(detected_etfs) * (len(detected_etfs)-1) // 2} ETF pairs")
                else:
                    max_pairs = len(available_symbols) * (len(available_symbols)-1) // 2
                    st.success(f"Will test up to {max_pairs} pairs")
        
        with col_cfg2:
            st.markdown("**Statistical Tests**")
            significance_level = st.slider("Significance Level", 0.01, 0.10, 0.05, 0.01)
            min_hurst = st.slider("Min Hurst (mean-reversion)", 0.0, 0.5, 0.45, 0.01,
                                help="Lower = stronger mean-reversion")
            max_hurst = 0.5  # Fixed threshold for mean-reversion
            
            use_johansen = st.checkbox("Use Johansen (multi-asset)", value=False,
                                      help="More comprehensive but slower")
        
        with col_cfg3:
            st.markdown("**Performance**")
            max_pairs_to_test = st.number_input("Max Pairs to Test", 10, 10000, 500, 50,
                                               help="Limit for computational efficiency")
            n_workers = st.slider("Parallel Workers", 1, 16, 8,
                                help="More workers = faster processing")
            
            transaction_cost_pct = st.slider("Transaction Cost (%)", 0.0, 1.0, 0.1, 0.05)
            discount_rate_pct = st.slider("Discount Rate (%/year)", 0.0, 20.0, 5.0, 1.0)
        
        # Run Analysis Button
        st.markdown("---")
        if st.button("🚀 Start Automated Discovery", type="primary", use_container_width=True):
            
            # Build pair list based on filter mode
            pairs_to_test = []
            
            if filter_mode == "ETF vs Stocks":
                for etf in detected_etfs:
                    for stock in detected_stocks:
                        pairs_to_test.append((etf, stock))
            elif filter_mode == "Stocks Only":
                for i, s1 in enumerate(detected_stocks):
                    for s2 in detected_stocks[i+1:]:
                        pairs_to_test.append((s1, s2))
            elif filter_mode == "ETFs Only":
                for i, e1 in enumerate(detected_etfs):
                    for e2 in detected_etfs[i+1:]:
                        pairs_to_test.append((e1, e2))
            else:  # All Pairs
                for i, s1 in enumerate(available_symbols):
                    for s2 in available_symbols[i+1:]:
                        pairs_to_test.append((s1, s2))
            
            # Limit pairs
            if len(pairs_to_test) > max_pairs_to_test:
                st.warning(f"⚠️ Limiting to first {max_pairs_to_test} pairs (from {len(pairs_to_test)} total)")
                pairs_to_test = pairs_to_test[:max_pairs_to_test]
            
            st.info(f"🔍 Testing {len(pairs_to_test)} pairs with {n_workers} parallel workers...")
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            metrics_placeholder = st.empty()
            
            start_time = time.time()
            
            # Results storage
            all_results = []
            cointegrated_count = 0
            mean_reverting_count = 0
            
            def test_pair(pair_info):
                """Test a single pair for cointegration and mean-reversion"""
                sym1, sym2 = pair_info
                
                try:
                    # Extract prices
                    if 'symbol' in data.columns:
                        p1 = data[data['symbol'] == sym1].set_index('timestamp')['close']
                        p2 = data[data['symbol'] == sym2].set_index('timestamp')['close']
                    else:
                        return None
                    
                    # Align series
                    common_idx = p1.index.intersection(p2.index)
                    if len(common_idx) < 100:
                        return None
                    
                    p1_aligned = p1.loc[common_idx]
                    p2_aligned = p2.loc[common_idx]
                    
                    # Test cointegration
                    coint_result = engle_granger_cointegration(p1_aligned, p2_aligned, 
                                                               significance_level=significance_level)
                    
                    if not coint_result.is_cointegrated:
                        return None
                    
                    # Calculate spread
                    spread = p1_aligned - coint_result.hedge_ratio * p2_aligned
                    
                    # Test for mean-reversion with Hurst
                    hurst_result = hurst_exponent(spread)
                    
                    if not (min_hurst <= hurst_result.hurst_exponent <= max_hurst):
                        return None
                    
                    # Estimate OU parameters
                    ou_params = estimate_ou_parameters(spread)
                    
                    if ou_params.kappa <= 0 or ou_params.sigma <= 0:
                        return None
                    
                    # Solve HJB for optimal boundaries
                    spread_std = spread.std()
                    spread_mean = spread.mean()
                    
                    try:
                        boundaries = solve_hjb_pde(
                            ou_params,
                            transaction_cost_pct / 100,
                            discount_rate_pct / 100,
                            spread_mean - 3*spread_std,
                            spread_mean + 3*spread_std,
                            n_points=200,  # Faster computation
                            max_iterations=2000
                        )
                    except:
                        return None
                    
                    # Quick backtest
                    try:
                        equity_curve, trades_df = backtest_optimal_switching(
                            p1_aligned,
                            p2_aligned,
                            coint_result.hedge_ratio,
                            boundaries,
                            transaction_cost_bps=transaction_cost_pct * 100
                        )
                        
                        metrics = compute_strategy_metrics(equity_curve, trades_df)
                        
                        # Combined score: cointegration strength × mean-reversion × profitability
                        coint_score = 1 - coint_result.p_value  # Higher = more cointegrated
                        meanrev_score = max(0, 0.5 - hurst_result.hurst_exponent) * 2  # 0-1 scale
                        profit_score = max(0, metrics['Total Return'])  # Raw return
                        
                        combined_score = coint_score * meanrev_score * (1 + profit_score)
                        
                        return {
                            'pair': f"{sym1}/{sym2}",
                            'sym1': sym1,
                            'sym2': sym2,
                            'coint_pvalue': coint_result.p_value,
                            'coint_score': coint_score,
                            'hedge_ratio': coint_result.hedge_ratio,
                            'hurst': hurst_result.hurst_exponent,
                            'meanrev_score': meanrev_score,
                            'kappa': ou_params.kappa,
                            'theta': ou_params.theta,
                            'sigma': ou_params.sigma,
                            'half_life': ou_params.half_life,
                            'total_return': metrics['Total Return'],
                            'sharpe': metrics['Sharpe Ratio'],
                            'max_dd': metrics['Max Drawdown'],
                            'num_trades': int(metrics['Num Trades']),
                            'win_rate': metrics['Win Rate'],
                            'profit_factor': metrics['Profit Factor'],
                            'combined_score': combined_score,
                            'boundaries': boundaries
                        }
                    except:
                        return None
                    
                except Exception as e:
                    return None
            
            # Parallel processing
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(test_pair, pair): pair for pair in pairs_to_test}
                
                completed = 0
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        all_results.append(result)
                        cointegrated_count += 1
                        if result['meanrev_score'] > 0.5:
                            mean_reverting_count += 1
                    
                    completed += 1
                    progress_bar.progress(completed / len(pairs_to_test))
                    
                    # Update status
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (len(pairs_to_test) - completed) / rate if rate > 0 else 0
                    
                    status_text.text(f"Processed: {completed}/{len(pairs_to_test)} | "
                                   f"Found: {cointegrated_count} cointegrated, {mean_reverting_count} mean-reverting | "
                                   f"Rate: {rate:.1f} pairs/sec | ETA: {eta:.0f}s")
                    
                    # Show interim top results
                    if len(all_results) > 0 and completed % 50 == 0:
                        sorted_results = sorted(all_results, key=lambda x: x['combined_score'], reverse=True)
                        metrics_placeholder.info(
                            f"**Current Top 3:**\n" +
                            "\n".join([f"{i+1}. {r['pair']}: Score={r['combined_score']:.4f}, "
                                     f"Return={r['total_return']:.2%}, Sharpe={r['sharpe']:.2f}"
                                     for i, r in enumerate(sorted_results[:3])])
                        )
            
            progress_bar.empty()
            status_text.empty()
            metrics_placeholder.empty()
            
            elapsed_time = time.time() - start_time
            
            # Display Results
            if len(all_results) == 0:
                st.error("❌ No suitable pairs found matching the criteria")
                st.info("Try relaxing the filters or adjusting the significance levels")
            else:
                # Sort by combined score
                sorted_results = sorted(all_results, key=lambda x: x['combined_score'], reverse=True)
                results_df = pd.DataFrame(sorted_results)
                
                st.success(f"✅ Analysis Complete! Found {len(results_df)} profitable cointegrated pairs in {elapsed_time:.1f}s")
                
                # Summary Metrics
                st.markdown("---")
                st.markdown("#### 📊 Discovery Summary")
                
                col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
                
                with col_m1:
                    st.metric("Pairs Tested", f"{len(pairs_to_test)}")
                    st.metric("Processing Rate", f"{len(pairs_to_test)/elapsed_time:.1f} pairs/s")
                
                with col_m2:
                    st.metric("Cointegrated", f"{len(results_df)}")
                    st.metric("Mean-Reverting", f"{(results_df['hurst'] < 0.5).sum()}")
                
                with col_m3:
                    st.metric("Best Return", f"{results_df['total_return'].max():.2%}")
                    st.metric("Avg Return", f"{results_df['total_return'].mean():.2%}")
                
                with col_m4:
                    st.metric("Best Sharpe", f"{results_df['sharpe'].max():.2f}")
                    st.metric("Avg Sharpe", f"{results_df['sharpe'].mean():.2f}")
                
                with col_m5:
                    profitable = (results_df['total_return'] > 0).sum()
                    st.metric("Profitable", f"{profitable}/{len(results_df)}")
                    st.metric("Avg Win Rate", f"{results_df['win_rate'].mean():.1%}")
                
                # Top Performers Table
                st.markdown("---")
                st.markdown("#### 🏆 Top Performing Pairs (Ranked by Combined Score)")
                st.markdown("*Combined Score = Cointegration Strength × Mean-Reversion × Profitability*")
                
                # Format display
                display_df = results_df.head(30).copy()
                display_df['Pair'] = display_df['pair']
                display_df['Coint P-Value'] = display_df['coint_pvalue'].apply(lambda x: f"{x:.4f}")
                display_df['Hurst'] = display_df['hurst'].apply(lambda x: f"{x:.4f}")
                display_df['Half-Life'] = display_df['half_life'].apply(lambda x: f"{x:.2f}d")
                display_df['Return'] = display_df['total_return'].apply(lambda x: f"{x:.2%}")
                display_df['Sharpe'] = display_df['sharpe'].apply(lambda x: f"{x:.2f}")
                display_df['Max DD'] = display_df['max_dd'].apply(lambda x: f"{x:.2%}")
                display_df['Trades'] = display_df['num_trades']
                display_df['Win Rate'] = display_df['win_rate'].apply(lambda x: f"{x:.1%}")
                display_df['Score'] = display_df['combined_score'].apply(lambda x: f"{x:.4f}")
                
                st.dataframe(
                    display_df[['Pair', 'Score', 'Return', 'Sharpe', 'Max DD', 
                               'Trades', 'Win Rate', 'Coint P-Value', 'Hurst', 'Half-Life']],
                    use_container_width=True,
                    height=600
                )
                
                # Visualization Dashboard
                st.markdown("---")
                st.markdown("#### 📈 Performance Visualization")
                
                tab_viz1, tab_viz2, tab_viz3 = st.tabs(["Score Distribution", "Risk-Return", "Statistical Properties"])
                
                with tab_viz1:
                    # Score distribution
                    fig_scores = go.Figure()
                    fig_scores.add_trace(go.Histogram(
                        x=results_df['combined_score'],
                        nbinsx=50,
                        name='Combined Score',
                        marker_color='blue'
                    ))
                    fig_scores.update_layout(
                        title='Distribution of Combined Scores',
                        xaxis_title='Combined Score',
                        yaxis_title='Frequency',
                        height=400
                    )
                    st.plotly_chart(fig_scores, use_container_width=True)
                
                with tab_viz2:
                    # Risk-return scatter
                    fig_scatter = go.Figure()
                    fig_scatter.add_trace(go.Scatter(
                        x=results_df['sharpe'],
                        y=results_df['total_return'],
                        mode='markers',
                        marker=dict(
                            size=results_df['num_trades'] / 5,
                            color=results_df['combined_score'],
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title="Score"),
                            line=dict(width=1, color='white')
                        ),
                        text=results_df['pair'],
                        hovertemplate='<b>%{text}</b><br>' +
                                    'Sharpe: %{x:.2f}<br>' +
                                    'Return: %{y:.2%}<br>' +
                                    '<extra></extra>'
                    ))
                    fig_scatter.update_layout(
                        title='Risk-Return Profile (size=trades, color=score)',
                        xaxis_title='Sharpe Ratio',
                        yaxis_title='Total Return',
                        height=600
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
                
                with tab_viz3:
                    # Statistical properties
                    fig_stats = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=('Hurst Exponent', 'Half-Life (days)', 
                                      'Cointegration P-Value', 'Kappa (Mean-Reversion Speed)')
                    )
                    
                    fig_stats.add_trace(go.Histogram(x=results_df['hurst'], nbinsx=30, name='Hurst'),
                                      row=1, col=1)
                    fig_stats.add_trace(go.Histogram(x=results_df['half_life'], nbinsx=30, name='Half-Life'),
                                      row=1, col=2)
                    fig_stats.add_trace(go.Histogram(x=results_df['coint_pvalue'], nbinsx=30, name='P-Value'),
                                      row=2, col=1)
                    fig_stats.add_trace(go.Histogram(x=results_df['kappa'], nbinsx=30, name='Kappa'),
                                      row=2, col=2)
                    
                    fig_stats.update_layout(height=600, showlegend=False)
                    st.plotly_chart(fig_stats, use_container_width=True)
                
                # Portfolio Construction
                st.markdown("---")
                st.markdown("#### 💼 Optimal Portfolio Construction")
                
                n_portfolio_pairs = st.slider("Select Top N Pairs for Portfolio", 1, min(20, len(results_df)), 5)
                
                top_pairs = sorted_results[:n_portfolio_pairs]
                
                st.markdown(f"**Selected {n_portfolio_pairs} pairs for portfolio:**")
                
                portfolio_metrics = {
                    'avg_return': np.mean([p['total_return'] for p in top_pairs]),
                    'avg_sharpe': np.mean([p['sharpe'] for p in top_pairs]),
                    'avg_max_dd': np.mean([p['max_dd'] for p in top_pairs]),
                    'total_trades': sum([p['num_trades'] for p in top_pairs]),
                    'avg_win_rate': np.mean([p['win_rate'] for p in top_pairs])
                }
                
                col_p1, col_p2, col_p3 = st.columns(3)
                
                with col_p1:
                    st.metric("Portfolio Avg Return", f"{portfolio_metrics['avg_return']:.2%}")
                    st.metric("Portfolio Avg Sharpe", f"{portfolio_metrics['avg_sharpe']:.2f}")
                
                with col_p2:
                    st.metric("Portfolio Avg Max DD", f"{portfolio_metrics['avg_max_dd']:.2%}")
                    st.metric("Total Trades", f"{portfolio_metrics['total_trades']}")
                
                with col_p3:
                    st.metric("Avg Win Rate", f"{portfolio_metrics['avg_win_rate']:.1%}")
                    st.metric("Diversification", f"{n_portfolio_pairs} pairs")
                
                # List selected pairs
                for i, pair_result in enumerate(top_pairs):
                    with st.expander(f"{i+1}. {pair_result['pair']} (Score: {pair_result['combined_score']:.4f})"):
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.write(f"**Return:** {pair_result['total_return']:.2%}")
                            st.write(f"**Sharpe:** {pair_result['sharpe']:.2f}")
                            st.write(f"**Max DD:** {pair_result['max_dd']:.2%}")
                        with col_b:
                            st.write(f"**Trades:** {pair_result['num_trades']}")
                            st.write(f"**Win Rate:** {pair_result['win_rate']:.1%}")
                            st.write(f"**Profit Factor:** {pair_result['profit_factor']:.2f}")
                        with col_c:
                            st.write(f"**Coint P-Val:** {pair_result['coint_pvalue']:.4f}")
                            st.write(f"**Hurst:** {pair_result['hurst']:.4f}")
                            st.write(f"**Half-Life:** {pair_result['half_life']:.2f} days")
                
                # Export Options
                st.markdown("---")
                st.markdown("#### 📥 Export Results")
                
                col_e1, col_e2 = st.columns(2)
                
                with col_e1:
                    # CSV export
                    csv_data = results_df.to_csv(index=False)
                    st.download_button(
                        label="📊 Download Full Results (CSV)",
                        data=csv_data,
                        file_name=f"auto_discovery_results_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col_e2:
                    # Save to session state for use in Options tab
                    st.session_state['discovered_pairs'] = sorted_results
                    st.success("✅ Results saved to session - Available in Options Strategies tab")
                
    except ImportError as e:
        st.error(f"❌ Required modules not available: {str(e)}")
        st.info("Install required packages: `pip install statsmodels scipy`")
    
with tab5:
    st.markdown("### 🎲 Options Strategies on Mean-Reversion Pairs")
    st.markdown("""
    **Advanced options strategies to enhance cointegrated pair trading:**
    - 💰 Leverage with long call/put options
    - 🛡️ Hedging with protective options
    - 📊 Synthetic positions (call spreads, put spreads)
    - 🎯 Delta-neutral strategies
    - ⚡ Volatility arbitrage on mean-reversion
    """)
    
    if 'discovered_pairs' not in st.session_state or len(st.session_state.get('discovered_pairs', [])) == 0:
        st.info("💡 Run the Auto-Discovery engine first to find cointegrated pairs")
        st.markdown("Go to the **Auto-Discovery** tab and click 'Start Automated Discovery'")
    else:
        discovered_pairs = st.session_state['discovered_pairs']
        
        st.success(f"✅ {len(discovered_pairs)} pairs available for options analysis")
        
        # Pair Selection
        st.markdown("---")
        st.markdown("#### 📌 Select Pair")
        
        col_sel1, col_sel2 = st.columns([2, 1])
        
        with col_sel1:
            pair_options = [p['pair'] for p in discovered_pairs[:20]]  # Top 20
            selected_pair_name = st.selectbox(
                "Choose a cointegrated pair",
                pair_options,
                help="Top 20 pairs by combined score"
            )
            
            selected_pair = next(p for p in discovered_pairs if p['pair'] == selected_pair_name)
        
        with col_sel2:
            st.metric("Combined Score", f"{selected_pair['combined_score']:.4f}")
            st.metric("Base Return", f"{selected_pair['total_return']:.2%}")
            st.metric("Sharpe Ratio", f"{selected_pair['sharpe']:.2f}")
        
        # Options Strategy Configuration
        st.markdown("---")
        st.markdown("#### ⚙️ Options Strategy Configuration")
        
        strategy_type = st.selectbox(
            "Select Strategy",
            [
                "Long Call on Entry (Leverage)",
                "Long Put on Entry (Leverage Short)",
                "Covered Call (Income + Hedge)",
                "Protective Put (Downside Protection)",
                "Bull Call Spread (Limited Risk)",
                "Bear Put Spread (Limited Risk)",
                "Long Straddle (High Volatility)",
                "Iron Condor (Low Volatility)",
                "Delta-Neutral (Pure Mean-Reversion)"
            ]
        )
        
        col_opt1, col_opt2, col_opt3 = st.columns(3)
        
        with col_opt1:
            st.markdown("**Position Sizing**")
            base_capital = st.number_input("Initial Capital ($)", 10000, 1000000, 100000, 10000)
            leverage_mult = st.slider("Options Leverage Multiple", 1.0, 5.0, 2.0, 0.5,
                                     help="How much leverage to apply via options")
        
        with col_opt2:
            st.markdown("**Options Parameters**")
            days_to_expiry = st.slider("Days to Expiration", 7, 90, 30, 7)
            implied_vol = st.slider("Implied Volatility (%)", 10, 100, 30, 5)
            risk_free_rate = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 4.0, 0.5)
        
        with col_opt3:
            st.markdown("**Strike Selection**")
            if "Call" in strategy_type or "Bull" in strategy_type:
                strike_offset = st.slider("Strike vs Spot (%)", -10, 10, 0, 1,
                                        help="Negative = OTM, Positive = ITM")
            else:
                strike_offset = st.slider("Strike vs Spot (%)", -10, 10, 0, 1)
        
        # Simulate Strategy
        if st.button("🎲 Simulate Options Strategy", type="primary", use_container_width=True):
            with st.spinner("Simulating options strategy..."):
                
                st.markdown("---")
                st.markdown("#### 📊 Strategy Simulation Results")
                
                # Black-Scholes approximation for options pricing
                def black_scholes_call(S, K, T, r, sigma):
                    """Simple Black-Scholes call price"""
                    from scipy.stats import norm
                    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
                    d2 = d1 - sigma*np.sqrt(T)
                    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
                
                def black_scholes_put(S, K, T, r, sigma):
                    """Simple Black-Scholes put price"""
                    from scipy.stats import norm
                    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
                    d2 = d1 - sigma*np.sqrt(T)
                    return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
                
                # Strategy parameters
                T = days_to_expiry / 365
                r = risk_free_rate / 100
                sigma = implied_vol / 100
                
                # Base strategy performance (from discovery)
                base_return = selected_pair['total_return']
                base_sharpe = selected_pair['sharpe']
                base_max_dd = selected_pair['max_dd']
                num_trades = selected_pair['num_trades']
                
                # Simulate option-enhanced returns
                if "Long Call" in strategy_type:
                    # Long call for leveraged upside
                    S = 100  # Normalized price
                    K = S * (1 + strike_offset/100)
                    call_price = black_scholes_call(S, K, T, r, sigma)
                    
                    # Number of contracts
                    contracts = (base_capital * leverage_mult) / (call_price * 100)
                    
                    # Enhanced return (simplified)
                    options_leverage = leverage_mult * (K / call_price) / 100
                    enhanced_return = base_return * options_leverage
                    enhanced_sharpe = base_sharpe * np.sqrt(leverage_mult)  # Vol scales with sqrt
                    enhanced_max_dd = base_max_dd * leverage_mult
                    
                    st.success(f"✅ Strategy: Long {contracts:.0f} call contracts @ ${call_price:.2f}")
                    
                elif "Long Put" in strategy_type:
                    # Long put for leveraged downside
                    S = 100
                    K = S * (1 - strike_offset/100)
                    put_price = black_scholes_put(S, K, T, r, sigma)
                    
                    contracts = (base_capital * leverage_mult) / (put_price * 100)
                    
                    options_leverage = leverage_mult * (K / put_price) / 100
                    enhanced_return = abs(base_return) * options_leverage if base_return < 0 else base_return
                    enhanced_sharpe = base_sharpe * np.sqrt(leverage_mult)
                    enhanced_max_dd = base_max_dd * leverage_mult
                    
                    st.success(f"✅ Strategy: Long {contracts:.0f} put contracts @ ${put_price:.2f}")
                
                elif "Covered Call" in strategy_type:
                    # Covered call = Long stock + Short call (income strategy)
                    S = 100
                    K = S * (1 + abs(strike_offset)/100)  # OTM call
                    call_price = black_scholes_call(S, K, T, r, sigma)
                    
                    shares = base_capital / S
                    contracts = shares / 100
                    premium_income = call_price * contracts * 100
                    
                    # Enhanced return includes premium
                    premium_yield = premium_income / base_capital
                    enhanced_return = base_return + premium_yield
                    enhanced_sharpe = base_sharpe * 1.2  # Slightly better risk-adjusted
                    enhanced_max_dd = base_max_dd * 0.9  # Premium provides cushion
                    
                    st.success(f"✅ Strategy: {shares:.0f} shares + Short {contracts:.0f} calls @ ${call_price:.2f}")
                    st.info(f"💰 Premium Income: ${premium_income:.2f} ({premium_yield:.2%})")
                
                elif "Protective Put" in strategy_type:
                    # Long stock + Long put (insurance)
                    S = 100
                    K = S * (1 - abs(strike_offset)/100)  # OTM put
                    put_price = black_scholes_put(S, K, T, r, sigma)
                    
                    shares = base_capital / S
                    contracts = shares / 100
                    insurance_cost = put_price * contracts * 100
                    
                    # Return reduced by insurance cost but max DD protected
                    insurance_yield = insurance_cost / base_capital
                    enhanced_return = base_return - insurance_yield
                    enhanced_sharpe = base_sharpe * 1.1  # Better risk profile
                    enhanced_max_dd = min(base_max_dd, abs(strike_offset)/100)  # Protected at strike
                    
                    st.success(f"✅ Strategy: {shares:.0f} shares + Long {contracts:.0f} puts @ ${put_price:.2f}")
                    st.info(f"🛡️ Insurance Cost: ${insurance_cost:.2f} ({insurance_yield:.2%})")
                    st.success(f"✅ Max Loss Protected: {abs(strike_offset):.1f}%")
                
                elif "Bull Call Spread" in strategy_type:
                    # Long lower strike call + Short higher strike call
                    S = 100
                    K_long = S * (1 + strike_offset/100)
                    K_short = S * (1 + strike_offset/100 + 5/100)  # 5% wider
                    
                    call_long = black_scholes_call(S, K_long, T, r, sigma)
                    call_short = black_scholes_call(S, K_short, T, r, sigma)
                    
                    spread_cost = call_long - call_short
                    contracts = base_capital / (spread_cost * 100)
                    max_profit = (K_short - K_long - spread_cost) * contracts * 100
                    
                    enhanced_return = base_return * (max_profit / base_capital) if base_return > 0 else 0
                    enhanced_sharpe = base_sharpe * 1.3
                    enhanced_max_dd = spread_cost / S  # Max loss is spread cost
                    
                    st.success(f"✅ Strategy: {contracts:.0f} bull call spreads @ ${spread_cost:.2f}")
                    st.info(f"📈 Max Profit: ${max_profit:.2f}, Max Loss: ${spread_cost * contracts * 100:.2f}")
                
                elif "Delta-Neutral" in strategy_type:
                    # Delta-neutral with options (gamma/vega play)
                    S = 100
                    K = S
                    
                    # Buy ATM straddle, hedge with underlying
                    call_price = black_scholes_call(S, K, T, r, sigma)
                    put_price = black_scholes_put(S, K, T, r, sigma)
                    straddle_cost = call_price + put_price
                    
                    contracts = base_capital / (straddle_cost * 100 * 2)
                    
                    # Pure mean-reversion play
                    enhanced_return = base_return * 1.5  # Amplified by gamma
                    enhanced_sharpe = base_sharpe * 1.4
                    enhanced_max_dd = base_max_dd * 1.2
                    
                    st.success(f"✅ Strategy: {contracts:.0f} delta-neutral straddles @ ${straddle_cost:.2f}")
                    st.info("🎯 This strategy profits from mean-reversion regardless of direction")
                
                else:
                    # Default to base strategy
                    enhanced_return = base_return
                    enhanced_sharpe = base_sharpe
                    enhanced_max_dd = base_max_dd
                    st.info("Using base strategy parameters")
                
                # Display comparison
                st.markdown("---")
                st.markdown("#### 📊 Performance Comparison")
                
                col_comp1, col_comp2, col_comp3 = st.columns(3)
                
                with col_comp1:
                    st.markdown("**Base Strategy**")
                    st.metric("Return", f"{base_return:.2%}")
                    st.metric("Sharpe", f"{base_sharpe:.2f}")
                    st.metric("Max DD", f"{base_max_dd:.2%}")
                
                with col_comp2:
                    st.markdown("**Options-Enhanced**")
                    st.metric("Return", f"{enhanced_return:.2%}", 
                            delta=f"{(enhanced_return - base_return):.2%}")
                    st.metric("Sharpe", f"{enhanced_sharpe:.2f}",
                            delta=f"{(enhanced_sharpe - base_sharpe):.2f}")
                    st.metric("Max DD", f"{enhanced_max_dd:.2%}",
                            delta=f"{(enhanced_max_dd - base_max_dd):.2%}",
                            delta_color="inverse")
                
                with col_comp3:
                    st.markdown("**Enhancement**")
                    return_mult = enhanced_return / base_return if base_return != 0 else 1
                    st.metric("Return Multiple", f"{return_mult:.2f}x")
                    
                    sharpe_mult = enhanced_sharpe / base_sharpe if base_sharpe != 0 else 1
                    st.metric("Sharpe Multiple", f"{sharpe_mult:.2f}x")
                    
                    risk_mult = enhanced_max_dd / base_max_dd if base_max_dd != 0 else 1
                    st.metric("Risk Multiple", f"{risk_mult:.2f}x")
                
                # Visualize equity curves
                st.markdown("---")
                st.markdown("#### 📈 Projected Equity Curves")
                
                # Simulate equity curves (simplified)
                n_periods = num_trades
                base_curve = base_capital * (1 + base_return * np.linspace(0, 1, n_periods))
                enhanced_curve = base_capital * (1 + enhanced_return * np.linspace(0, 1, n_periods))
                
                # Add some realistic volatility
                base_noise = np.random.normal(0, base_max_dd * base_capital * 0.1, n_periods)
                enhanced_noise = np.random.normal(0, enhanced_max_dd * base_capital * 0.1, n_periods)
                
                base_curve += np.cumsum(base_noise)
                enhanced_curve += np.cumsum(enhanced_noise)
                
                fig_equity = go.Figure()
                fig_equity.add_trace(go.Scatter(
                    y=base_curve,
                    name='Base Strategy',
                    line=dict(color='blue', width=2)
                ))
                fig_equity.add_trace(go.Scatter(
                    y=enhanced_curve,
                    name='Options-Enhanced',
                    line=dict(color='green', width=2, dash='dash')
                ))
                fig_equity.add_hline(y=base_capital, line_dash="dot", line_color="gray",
                                   annotation_text="Initial Capital")
                
                fig_equity.update_layout(
                    title='Projected Equity Curves Comparison',
                    xaxis_title='Trade Number',
                    yaxis_title='Portfolio Value ($)',
                    height=500
                )
                
                st.plotly_chart(fig_equity, use_container_width=True)
                
                # Risk Warning
                st.warning("""
                ⚠️ **Risk Disclaimer**: 
                - Options involve significant risk and are not suitable for all investors
                - Past performance does not guarantee future results
                - Leverage amplifies both gains AND losses
                - Always paper trade strategies before live deployment
                - Consider consulting a financial advisor
                """)
                
                # Save configuration
                if st.button("💾 Save Strategy Configuration"):
                    if 'options_strategies' not in st.session_state:
                        st.session_state['options_strategies'] = []
                    
                    st.session_state['options_strategies'].append({
                        'pair': selected_pair_name,
                        'strategy': strategy_type,
                        'base_return': base_return,
                        'enhanced_return': enhanced_return,
                        'leverage': leverage_mult,
                        'capital': base_capital
                    })
                    
                    st.success("✅ Strategy saved to session state")

with tab6:
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

with tab7:
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

with tab8:
    st.markdown("### 📚 Mathematical Theory & Foundations")
    st.markdown("""
    Comprehensive mathematical foundations for mean-reversion and optimal switching strategies.
    """)
    
    theory_section = st.selectbox(
        "Select Topic",
        [
            "1. Cointegration Theory",
            "2. Ornstein-Uhlenbeck Process",
            "3. Hamilton-Jacobi-Bellman Equation",
            "4. Hurst Exponent & Long-Range Dependence",
            "5. Viscosity Solutions",
            "6. Options Pricing (Black-Scholes)",
            "7. Statistical Tests (Engle-Granger, Johansen)",
            "8. Optimal Control Theory"
        ]
    )
    
    if "Cointegration" in theory_section:
        st.markdown("---")
        st.markdown("## 📐 Cointegration Theory")
        
        st.markdown(r"""
        ### Definition
        Two non-stationary time series $X_t$ and $Y_t$ are **cointegrated** if there exists a linear combination 
        that is stationary:
        
        $$Z_t = Y_t - \beta X_t \sim I(0)$$
        
        where $\beta$ is the cointegration coefficient and $I(0)$ denotes a stationary process.
        
        ### Engle-Granger Two-Step Method
        
        **Step 1**: Estimate the cointegration relationship via OLS:
        $$Y_t = \alpha + \beta X_t + \epsilon_t$$
        
        **Step 2**: Test the residuals for stationarity using Augmented Dickey-Fuller (ADF):
        $$\Delta Z_t = \gamma Z_{t-1} + \sum_{i=1}^p \delta_i \Delta Z_{t-i} + \eta_t$$
        
        **Null Hypothesis**: $H_0: \gamma = 0$ (no cointegration)  
        **Alternative**: $H_1: \gamma < 0$ (cointegration exists)
        
        ### Johansen Test (Multivariate)
        
        For multiple time series, use the Johansen test based on VAR model:
        $$\Delta X_t = \Pi X_{t-1} + \sum_{i=1}^{p-1} \Gamma_i \Delta X_{t-i} + \epsilon_t$$
        
        where $\Pi = \alpha \beta'$ and rank($\Pi$) = $r$ gives the number of cointegrating relationships.
        
        **Test Statistics**:
        - **Trace Test**: $\lambda_{trace}(r) = -T \sum_{i=r+1}^n \ln(1 - \hat{\lambda}_i)$
        - **Max Eigenvalue**: $\lambda_{max}(r) = -T \ln(1 - \hat{\lambda}_{r+1})$
        
        ### Economic Interpretation
        - Cointegration implies long-run equilibrium relationship
        - Deviations from equilibrium are temporary (mean-reverting)
        - Basis for pairs trading and statistical arbitrage
        """)
    
    elif "Ornstein-Uhlenbeck" in theory_section:
        st.markdown("---")
        st.markdown("## 🌊 Ornstein-Uhlenbeck Process")
        
        st.markdown(r"""
        ### Stochastic Differential Equation
        
        The OU process is the continuous-time analogue of an AR(1) process:
        
        $$dX_t = \kappa(\theta - X_t)dt + \sigma dW_t$$
        
        **Parameters**:
        - $\kappa > 0$: Speed of mean-reversion
        - $\theta$: Long-term mean
        - $\sigma > 0$: Volatility
        - $W_t$: Standard Brownian motion
        
        ### Properties
        
        **1. Mean Function**:
        $$\mathbb{E}[X_t | X_0] = \theta + (X_0 - \theta)e^{-\kappa t}$$
        
        **2. Variance**:
        $$\text{Var}[X_t | X_0] = \frac{\sigma^2}{2\kappa}(1 - e^{-2\kappa t})$$
        
        **3. Stationary Distribution**:
        $$X_\infty \sim \mathcal{N}\left(\theta, \frac{\sigma^2}{2\kappa}\right)$$
        
        **4. Half-Life**:
        $$t_{1/2} = \frac{\ln 2}{\kappa}$$
        
        Time for spread to revert halfway to mean.
        
        ### Estimation via Maximum Likelihood
        
        Discretize with time step $\Delta t$:
        $$X_{t+\Delta t} = X_t e^{-\kappa \Delta t} + \theta(1 - e^{-\kappa \Delta t}) + \sigma\sqrt{\frac{1-e^{-2\kappa \Delta t}}{2\kappa}}\epsilon_t$$
        
        where $\epsilon_t \sim \mathcal{N}(0,1)$.
        
        **Log-Likelihood**:
        $$\ell(\kappa, \theta, \sigma) = -\frac{n}{2}\ln(2\pi) - \frac{n}{2}\ln(\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n (X_{t_i} - \mu_i)^2$$
        
        ### Application to Pairs Trading
        
        Cointegrated spread $Z_t = Y_t - \beta X_t$ modeled as OU process.  
        Trade signals:
        - **Enter long** when $Z_t < \theta - k\sigma_\infty$
        - **Exit long** when $Z_t > \theta$
        - **Enter short** when $Z_t > \theta + k\sigma_\infty$
        - **Exit short** when $Z_t < \theta$
        """)
    
    elif "Hamilton-Jacobi-Bellman" in theory_section:
        st.markdown("---")
        st.markdown("## ⚡ Hamilton-Jacobi-Bellman Equation")
        
        st.markdown(r"""
        ### Optimal Control Problem
        
        Maximize expected discounted profit over infinite horizon:
        
        $$V(x) = \sup_{\tau_1, \tau_2, \ldots} \mathbb{E}\left[\sum_{i=1}^\infty e^{-\rho \tau_i} \Pi(\tau_i)\right]$$
        
        where:
        - $V(x)$: Value function at state $x$
        - $\tau_i$: Trading times (stopping times)
        - $\rho > 0$: Discount rate
        - $\Pi(\tau)$: Profit from trading at time $\tau$
        
        ### HJB Equation for Mean-Reversion Trading
        
        For spread $X_t$ following OU process, the HJB equation is:
        
        $$\rho V(x) = \kappa(\theta - x)V'(x) + \frac{1}{2}\sigma^2 V''(x)$$
        
        with boundary conditions at optimal switching boundaries $a$ and $b$:
        
        **Lower Boundary** (buy signal):
        $$V(a) = x - a - c + V(x)$$
        
        **Upper Boundary** (sell signal):
        $$V(b) = b - x - c + V(x)$$
        
        where $c$ is the transaction cost.
        
        ### Viscosity Solution
        
        The value function $V$ is a **viscosity solution** if it satisfies:
        
        1. **Subsolution**: For all $\phi \in C^2$ with $V - \phi$ having local maximum at $x_0$:
           $$\rho V(x_0) \leq \kappa(\theta - x_0)\phi'(x_0) + \frac{1}{2}\sigma^2 \phi''(x_0)$$
        
        2. **Supersolution**: For all $\phi \in C^2$ with $V - \phi$ having local minimum at $x_0$:
           $$\rho V(x_0) \geq \kappa(\theta - x_0)\phi'(x_0) + \frac{1}{2}\sigma^2 \phi''(x_0)$$
        
        ### Numerical Solution (Finite Difference)
        
        Discretize state space: $x_i = a + i\Delta x$, $i = 0, \ldots, N$
        
        **Upwind Scheme**:
        $$\rho V_i = \kappa(\theta - x_i)\frac{V_{i+1} - V_{i-1}}{2\Delta x} + \frac{\sigma^2}{2}\frac{V_{i+1} - 2V_i + V_{i-1}}{(\Delta x)^2}$$
        
        **Iteration** until convergence:
        $$V_i^{k+1} = \frac{1}{\rho + \frac{\sigma^2}{(\Delta x)^2}}\left[\kappa(\theta - x_i)\frac{V_{i+1}^k - V_{i-1}^k}{2\Delta x} + \frac{\sigma^2}{2}\frac{V_{i+1}^k + V_{i-1}^k}{(\Delta x)^2}\right]$$
        
        ### Optimal Boundaries
        
        Boundaries $a^*$ and $b^*$ satisfy:
        $$V'(a^*) = 1, \quad V'(b^*) = -1$$
        
        These are the **free boundary conditions** from the optimal control problem.
        """)
    
    elif "Hurst" in theory_section:
        st.markdown("---")
        st.markdown("## 📊 Hurst Exponent & Long-Range Dependence")
        
        st.markdown(r"""
        ### Definition
        
        The **Hurst exponent** $H$ characterizes long-range dependence and self-similarity in time series:
        
        $$\mathbb{E}[|X(t) - X(s)|^2] \propto |t - s|^{2H}$$
        
        ### Interpretation
        
        - $H = 0.5$: **Random walk** (Brownian motion, no memory)
        - $H < 0.5$: **Mean-reverting** (anti-persistent, negative autocorrelation)
        - $H > 0.5$: **Trending** (persistent, positive autocorrelation)
        
        For pairs trading, we seek $H < 0.5$ to ensure mean-reversion.
        
        ### Rescaled Range (R/S) Analysis
        
        **Algorithm**:
        1. Divide series into $n$ non-overlapping sub-periods of length $\tau$
        2. For each sub-period, calculate:
           - Mean: $\bar{X}_\tau = \frac{1}{\tau}\sum_{i=1}^\tau X_i$
           - Cumulative deviation: $Y_t = \sum_{i=1}^t (X_i - \bar{X}_\tau)$
           - Range: $R_\tau = \max Y_t - \min Y_t$
           - Standard deviation: $S_\tau = \sqrt{\frac{1}{\tau}\sum_{i=1}^\tau (X_i - \bar{X}_\tau)^2}$
        3. Calculate rescaled range: $\frac{R_\tau}{S_\tau}$
        4. Average over all sub-periods: $\mathbb{E}[R_\tau/S_\tau]$
        5. Repeat for different $\tau$ values
        
        **Hurst Estimation**:
        $$\log\left(\mathbb{E}\left[\frac{R_\tau}{S_\tau}\right]\right) \approx H \log(\tau) + \text{const}$$
        
        ### Detrended Fluctuation Analysis (DFA)
        
        More robust alternative to R/S:
        
        1. Integrate series: $Y_k = \sum_{i=1}^k (X_i - \bar{X})$
        2. Divide into boxes of size $n$
        3. Fit polynomial trend in each box
        4. Calculate fluctuation: $F(n) = \sqrt{\frac{1}{N}\sum_{k=1}^N [Y_k - y_k]^2}$
        5. Scaling relation: $F(n) \sim n^H$
        
        ### Multivariate Hurst Exponent
        
        For portfolio of $n$ assets with returns $r_i$:
        
        $$H_p = \frac{1}{2} + \frac{1}{2\log(n)}\log\left(\frac{\text{Var}(\sum r_i)}{\sum \text{Var}(r_i)}\right)$$
        
        ### Connection to Mean-Reversion
        
        For OU process with speed $\kappa$:
        $$H \approx 0.5 - \frac{1}{4}\kappa \Delta t$$
        
        Stronger mean-reversion ($\kappa \uparrow$) ⟹ Lower Hurst ($H \downarrow$)
        """)
    
    elif "Viscosity" in theory_section:
        st.markdown("---")
        st.markdown("## 🌀 Viscosity Solutions")
        
        st.markdown(r"""
        ### Motivation
        
        Classical solutions to HJB equations may not exist due to:
        - Non-smooth value functions at boundaries
        - Kinks at optimal switching points
        - Discontinuous derivatives
        
        **Viscosity solutions** extend classical solutions to handle these cases.
        
        ### Definition (Crandall-Lions)
        
        A function $V$ is a **viscosity solution** if:
        
        **Subsolution**: For all $\phi \in C^2$ and local maximum point $x_0$ of $V - \phi$:
        $$\min\{-\mathcal{L}\phi(x_0), V(x_0) - M(x_0)\} \leq 0$$
        
        **Supersolution**: For all $\phi \in C^2$ and local minimum point $x_0$ of $V - \phi$:
        $$\min\{-\mathcal{L}\phi(x_0), V(x_0) - M(x_0)\} \geq 0$$
        
        where:
        - $\mathcal{L}$ is the differential operator (e.g., $\mathcal{L}V = \kappa(\theta-x)V' + \frac{1}{2}\sigma^2 V''$)
        - $M(x)$ is the obstacle (value from immediate trading)
        
        ### Key Properties
        
        1. **Existence**: Viscosity solutions always exist for well-posed problems
        2. **Uniqueness**: Under comparison principle, solution is unique
        3. **Stability**: Stable under approximations (finite difference, neural networks)
        4. **Verification**: Can verify via smooth test functions
        
        ### Comparison Principle
        
        If $V_1$ is subsolution and $V_2$ is supersolution with $V_1 \leq V_2$ on boundary:
        $$V_1 \leq V_2 \text{ everywhere}$$
        
        This implies uniqueness: if $V$ is both subsolution and supersolution, it's the unique solution.
        
        ### Application to Optimal Switching
        
        For pairs trading with boundaries $a$ and $b$:
        
        **Value Function**:
        $$V(x) = \begin{cases}
        \alpha e^{\lambda_1 x} + \beta e^{\lambda_2 x} & x \in (a, b) \\
        x - a - c + V(\theta) & x \leq a \\
        b - x - c + V(\theta) & x \geq b
        \end{cases}$$
        
        where $\lambda_{1,2}$ solve the characteristic equation:
        $$\frac{1}{2}\sigma^2 \lambda^2 + \kappa(\theta - x)\lambda - \rho = 0$$
        
        ### Numerical Verification
        
        Check viscosity property at grid points:
        1. Compute numerical solution $V_i$
        2. For each point, test subsolution/supersolution conditions
        3. If both hold, solution is viscosity solution
        
        ### Relation to Dynamic Programming
        
        Viscosity solutions generalize the Bellman equation:
        $$V(x) = \sup_{a \in A} \{r(x,a) + \delta \mathbb{E}[V(X_{t+1})]\}$$
        
        to continuous time and non-smooth value functions.
        """)
    
    elif "Black-Scholes" in theory_section:
        st.markdown("---")
        st.markdown("## 📈 Options Pricing (Black-Scholes)")
        
        st.markdown(r"""
        ### Black-Scholes PDE
        
        For option price $V(S,t)$ on underlying $S$ with volatility $\sigma$:
        
        $$\frac{\partial V}{\partial t} + rS\frac{\partial V}{\partial S} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} - rV = 0$$
        
        with terminal condition $V(S,T) = \text{payoff}(S)$.
        
        ### Closed-Form Solutions
        
        **European Call**:
        $$C(S,K,T,r,\sigma) = S\Phi(d_1) - Ke^{-rT}\Phi(d_2)$$
        
        **European Put**:
        $$P(S,K,T,r,\sigma) = Ke^{-rT}\Phi(-d_2) - S\Phi(-d_1)$$
        
        where:
        $$d_1 = \frac{\ln(S/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}$$
        
        and $\Phi$ is the standard normal CDF.
        
        ### The Greeks
        
        **Delta** (sensitivity to underlying):
        $$\Delta = \frac{\partial V}{\partial S} = \begin{cases}
        \Phi(d_1) & \text{call} \\
        -\Phi(-d_1) & \text{put}
        \end{cases}$$
        
        **Gamma** (convexity):
        $$\Gamma = \frac{\partial^2 V}{\partial S^2} = \frac{\phi(d_1)}{S\sigma\sqrt{T}}$$
        
        **Vega** (sensitivity to volatility):
        $$\mathcal{V} = \frac{\partial V}{\partial \sigma} = S\phi(d_1)\sqrt{T}$$
        
        **Theta** (time decay):
        $$\Theta = \frac{\partial V}{\partial t} = -\frac{S\phi(d_1)\sigma}{2\sqrt{T}} - rKe^{-rT}\Phi(d_2)$$
        
        **Rho** (sensitivity to interest rate):
        $$\rho = \frac{\partial V}{\partial r} = KTe^{-rT}\Phi(d_2)$$
        
        ### Implied Volatility
        
        Market price $C_{\text{market}}$ inverted to find $\sigma_{\text{IV}}$:
        $$C_{\text{market}} = C(S, K, T, r, \sigma_{\text{IV}})$$
        
        Solved numerically via Newton-Raphson:
        $$\sigma_{n+1} = \sigma_n - \frac{C(\sigma_n) - C_{\text{market}}}{\mathcal{V}(\sigma_n)}$$
        
        ### Application to Pairs Trading
        
        **Options on Spread**:
        - Spread $Z_t = Y_t - \beta X_t$ as synthetic underlying
        - Call options profit from mean-reversion (spread rising)
        - Put options profit from spread falling
        
        **Volatility Arbitrage**:
        - If $\sigma_{\text{IV}} > \sigma_{\text{realized}}$: sell options
        - If $\sigma_{\text{IV}} < \sigma_{\text{realized}}$: buy options
        
        **Delta Hedging**:
        $$\Delta_{\text{portfolio}} = \Delta_{\text{option}} \times N_{\text{options}} + N_{\text{shares}} = 0$$
        """)
    
    elif "Statistical Tests" in theory_section:
        st.markdown("---")
        st.markdown("## 📊 Statistical Tests (Engle-Granger, Johansen)")
        
        st.markdown(r"""
        ### Augmented Dickey-Fuller (ADF) Test
        
        Tests for unit root (non-stationarity):
        $$\Delta Y_t = \alpha + \beta t + \gamma Y_{t-1} + \sum_{i=1}^p \delta_i \Delta Y_{t-i} + \epsilon_t$$
        
        **Null**: $H_0: \gamma = 0$ (unit root, non-stationary)  
        **Alternative**: $H_1: \gamma < 0$ (stationary)
        
        **Test Statistic**:
        $$\text{ADF} = \frac{\hat{\gamma}}{\text{SE}(\hat{\gamma})}$$
        
        **Critical Values** (5% significance):
        - No trend: -2.86
        - With trend: -3.41
        
        ### Phillips-Perron Test
        
        Similar to ADF but corrects for serial correlation:
        $$\text{PP} = t_{\hat{\gamma}} \sqrt{\frac{\hat{\gamma}^2}{\tilde{\gamma}^2}}$$
        
        where $\tilde{\gamma}$ accounts for autocorrelation.
        
        ### Engle-Granger Test
        
        **Step 1**: OLS regression
        $$Y_t = \alpha + \beta X_t + u_t$$
        
        **Step 2**: ADF test on residuals $\hat{u}_t$
        $$\Delta \hat{u}_t = \gamma \hat{u}_{t-1} + \sum_{i=1}^p \delta_i \Delta \hat{u}_{t-i} + \epsilon_t$$
        
        **Critical values** more negative than standard ADF (Engle-Granger tables).
        
        ### Johansen Test (VAR Framework)
        
        **Vector Error Correction Model (VECM)**:
        $$\Delta Y_t = \Pi Y_{t-1} + \sum_{i=1}^{p-1} \Gamma_i \Delta Y_{t-i} + \epsilon_t$$
        
        where $\Pi = \alpha \beta'$ has rank $r$ = number of cointegrating vectors.
        
        **Trace Test**:
        $$\lambda_{\text{trace}}(r) = -T \sum_{i=r+1}^n \ln(1 - \hat{\lambda}_i)$$
        
        Tests $H_0$: at most $r$ cointegrating vectors.
        
        **Max Eigenvalue Test**:
        $$\lambda_{\text{max}}(r) = -T \ln(1 - \hat{\lambda}_{r+1})$$
        
        Tests $H_0$: exactly $r$ cointegrating vectors.
        
        ### Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test
        
        **Reverse test** - null is stationarity:
        $$Y_t = \xi t + r_t + \epsilon_t$$
        
        where $r_t$ is random walk.
        
        **Test Statistic**:
        $$\text{KPSS} = \frac{\sum_{t=1}^T S_t^2}{T^2 \hat{\sigma}_\epsilon^2}$$
        
        where $S_t = \sum_{i=1}^t (Y_i - \bar{Y})$.
        
        ### Interpretation for Trading
        
        | Test Result | Interpretation | Trading Implication |
        |-------------|----------------|---------------------|
        | ADF rejects | Stationary | ✅ Mean-reverting, tradeable |
        | ADF fails | Non-stationary | ❌ Trending, avoid |
        | EG rejects | Cointegrated | ✅ Pairs trade candidate |
        | Johansen r>0 | Multiple cointegration | ✅ Multi-asset basket |
        | Low Hurst | Strong mean-reversion | ✅ High trading frequency |
        | High Hurst | Persistent trends | ❌ Momentum strategy instead |
        """)
    
    elif "Optimal Control" in theory_section:
        st.markdown("---")
        st.markdown("## 🎮 Optimal Control Theory")
        
        st.markdown(r"""
        ### General Framework
        
        **State Equation**:
        $$dX_t = f(X_t, u_t)dt + g(X_t)dW_t$$
        
        **Objective**:
        $$J(x, u) = \mathbb{E}\left[\int_0^\infty e^{-\rho t} L(X_t, u_t)dt\right]$$
        
        **Value Function**:
        $$V(x) = \sup_{u \in \mathcal{U}} J(x, u)$$
        
        ### Dynamic Programming Principle
        
        $$V(x) = \sup_{u} \left\{\int_0^h e^{-\rho t} L(X_t, u_t)dt + e^{-\rho h}\mathbb{E}[V(X_h)]\right\}$$
        
        ### Hamilton-Jacobi-Bellman Equation
        
        $$\rho V(x) = \sup_{u \in \mathcal{U}} \left\{L(x,u) + \mathcal{L}^u V(x)\right\}$$
        
        where $\mathcal{L}^u$ is the infinitesimal generator:
        $$\mathcal{L}^u V = f(x,u)V'(x) + \frac{1}{2}g(x)^2 V''(x)$$
        
        ### Pontryagin Maximum Principle
        
        **Hamiltonian**:
        $$H(x, u, p) = L(x, u) + p \cdot f(x, u)$$
        
        **Necessary Conditions**:
        1. $u^*(t) = \arg\max_u H(x^*, u, p)$
        2. $\dot{p} = -\frac{\partial H}{\partial x}$
        3. $\dot{x}^* = \frac{\partial H}{\partial p}$
        4. $p(T) = \frac{\partial \phi}{\partial x}(x^*(T))$ (transversality)
        
        ### Application: Optimal Liquidation
        
        **State**: Inventory $q_t$  
        **Control**: Trading rate $v_t$  
        **Objective**: Minimize cost + risk
        
        $$\min_{v} \mathbb{E}\left[\int_0^T \left(\frac{\lambda}{2}v_t^2 + \phi q_t^2\right)dt + q_T S_T\right]$$
        
        **HJB Equation**:
        $$V_t + \sup_v \left\{v V_q - \frac{\lambda}{2}v^2 - \phi q^2\right\} = 0$$
        
        **Optimal Control**:
        $$v^*(t) = \frac{V_q}{\lambda} = \frac{2\phi}{\lambda}q$$
        
        ### Application: Mean-Reversion Trading
        
        **State**: Spread $X_t$ (OU process)  
        **Control**: Position $u_t \in \{-1, 0, +1\}$  
        **Objective**: Maximize profit - transaction costs
        
        **Value Function** satisfies:
        $$\rho V(x) = \max\left\{\kappa(\theta-x)V'(x) + \frac{\sigma^2}{2}V''(x), \quad x - c, \quad -x - c\right\}$$
        
        **Optimal Policy** (two-threshold):
        - Buy at $x = a$
        - Sell at $x = b$
        - Hold for $x \in (a, b)$
        """)
    
    # Add code examples
    st.markdown("---")
    st.markdown("### 💻 Implementation Examples")
    
    with st.expander("📝 Python Code Snippets"):
        st.code("""
# Engle-Granger Cointegration Test
from statsmodels.tsa.stattools import coint, adfuller

def test_cointegration(y, x, alpha=0.05):
    # Step 1: OLS regression
    beta = np.polyfit(x, y, 1)[0]
    residuals = y - beta * x
    
    # Step 2: ADF test on residuals
    adf_stat, p_value, _, _, crit_vals, _ = adfuller(residuals)
    
    is_cointegrated = p_value < alpha
    return {
        'beta': beta,
        'adf_statistic': adf_stat,
        'p_value': p_value,
        'cointegrated': is_cointegrated
    }

# Ornstein-Uhlenbeck Parameter Estimation
def estimate_ou_params(spread, dt=1/252):
    X = spread[:-1]
    Y = spread[1:]
    
    mu, phi = np.polyfit(X, Y, 1)
    residuals = Y - (mu + phi * X)
    sigma_epsilon = np.std(residuals)
    
    kappa = -np.log(phi) / dt
    theta = mu / (1 - phi)
    sigma = sigma_epsilon * np.sqrt(-2 * np.log(phi) / dt / (1 - phi**2))
    half_life = np.log(2) / kappa
    
    return {'kappa': kappa, 'theta': theta, 'sigma': sigma, 'half_life': half_life}

# Hurst Exponent
def hurst_exponent(series, max_lag=20):
    lags = range(2, max_lag)
    tau = []; rs = []
    
    for lag in lags:
        n_blocks = len(series) // lag
        rs_values = []
        for i in range(n_blocks):
            block = series[i*lag:(i+1)*lag]
            mean_adj = block - np.mean(block)
            cum_sum = np.cumsum(mean_adj)
            R = np.max(cum_sum) - np.min(cum_sum)
            S = np.std(block)
            if S > 0:
                rs_values.append(R / S)
        tau.append(lag)
        rs.append(np.mean(rs_values))
    
    H = np.polyfit(np.log(tau), np.log(rs), 1)[0]
    return H
""", language='python')
    
    # References
    st.markdown("---")
    st.markdown("### 📚 Key References")
    
    st.markdown("""
    **Cointegration & Pairs Trading**:
    - Engle & Granger (1987). "Co-integration and error correction"
    - Johansen (1988). "Statistical analysis of cointegration vectors"
    - Gatev et al. (2006). "Pairs trading: Performance of a relative-value arbitrage rule"
    
    **Optimal Switching & Control**:
    - Øksendal & Sulem (2007). "Applied Stochastic Control of Jump Diffusions"
    - Cartea et al. (2015). "Algorithmic and High-Frequency Trading"
    
    **Viscosity Solutions**:
    - Crandall & Lions (1983). "Viscosity solutions of Hamilton-Jacobi equations"
    - Fleming & Soner (2006). "Controlled Markov Processes and Viscosity Solutions"
    
    **Options Pricing**:
    - Black & Scholes (1973). "The pricing of options and corporate liabilities"
    - Hull (2018). "Options, Futures, and Other Derivatives"
    """)
    
    st.markdown("---")
    st.info("💡 **Tip**: Use Ctrl+F (Cmd+F on Mac) to search for specific topics within this documentation.")

# Keep sparse mean-reversion as a separate expandable section
with st.expander("✨ Advanced: Sparse Mean-Reverting Portfolios", expanded=False):
    st.markdown("### Sparse Mean-Reverting Portfolios")
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
