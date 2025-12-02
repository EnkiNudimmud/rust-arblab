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
from utils.ui_components import render_sidebar_navigation, apply_custom_css

st.set_page_config(page_title="Mean Reversion Lab", page_icon="📉", layout="wide")

# Initialize session state
if 'historical_data' not in st.session_state:
    st.session_state.historical_data = None
if 'meanrev_results' not in st.session_state:
    st.session_state.meanrev_results = None

# Render sidebar navigation and apply CSS
render_sidebar_navigation(current_page="Mean Reversion Lab")
apply_custom_css()

st.markdown('<h1 class="lab-header">📉 Mean Reversion Lab</h1>', unsafe_allow_html=True)
st.markdown("### Statistical arbitrage with Z-score analysis and cointegration testing")
st.markdown("---")

# Check if data is loaded
if st.session_state.historical_data is None or st.session_state.historical_data.empty:
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
tab1, tab2, tab3, tab4 = st.tabs(["📊 Z-Score Analysis", "🔄 Pairs Trading", "📈 Strategy Backtest", "📉 Performance"])

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
            from python.meanrev import engle_granger_test
            
            prices1 = data[selected_symbol].dropna() if data is not None else pd.Series()
            prices2 = data[symbol2].dropna() if data is not None else pd.Series()
            
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

with tab4:
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

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>📉 Mean Reversion Lab | Part of HFT Arbitrage Lab</p>
</div>
""", unsafe_allow_html=True)
