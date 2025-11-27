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
if isinstance(data.index, pd.MultiIndex):
    # Reset index to get symbol and timestamp as columns
    data = data.reset_index()

# Get list of unique symbols
if 'symbol' in data.columns:
    available_symbols = data['symbol'].unique().tolist()
    st.success(f"✅ Data loaded: {len(data)} records, {len(available_symbols)} symbols")
else:
    available_symbols = [col for col in data.columns if col not in ['Date', 'date', 'timestamp', 'open', 'high', 'low', 'close', 'volume']]
    st.success(f"✅ Data loaded: {len(data)} records, {len(available_symbols)} assets")

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
    if 'symbol' in data.columns:
        # Multi-symbol data structure
        symbol_data = data[data['symbol'] == selected_symbol].copy()
        if 'timestamp' in symbol_data.columns:
            symbol_data = symbol_data.set_index('timestamp').sort_index()
        prices = symbol_data['close'].dropna()
    elif selected_symbol in data.columns:
        # Single-column data structure
        prices = data[selected_symbol].dropna()
    else:
        st.error(f"Symbol {selected_symbol} not found in data")
        st.stop()
    
    if len(prices) == 0:
        st.error(f"No price data available for {selected_symbol}")
        st.stop()
    
    # Ensure prices are numeric
    prices = pd.to_numeric(prices, errors='coerce').dropna()
    
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
            
            prices1 = data[selected_symbol].dropna()
            prices2 = data[symbol2].dropna()
            
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
    st.info("🚧 Coming soon: Full strategy backtesting with transaction costs, slippage, and performance metrics")
    
    if st.button("🚀 Go to Strategy Backtest Page", type="primary"):
        st.switch_page("pages/strategy_backtest.py")

with tab4:
    st.markdown("### Performance Metrics")
    st.info("🚧 Coming soon: Sharpe ratio, max drawdown, win rate, and other performance metrics")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>📉 Mean Reversion Lab | Part of HFT Arbitrage Lab</p>
</div>
""", unsafe_allow_html=True)
