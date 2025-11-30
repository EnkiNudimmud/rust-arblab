"""
Momentum Trading Lab - Multi-Timeframe Trend Following Strategies
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

from utils.ui_components import render_sidebar_navigation, apply_custom_css

# Page configuration
st.set_page_config(
    page_title="Momentum Trading Lab",
    page_icon="📈",
    layout="wide"
)

# Apply custom styling and navigation
apply_custom_css()
render_sidebar_navigation(current_page="Momentum Trading Lab")

# Initialize session state
if 'historical_data' not in st.session_state:
    st.session_state.historical_data = None
if 'theme_mode' not in st.session_state:
    st.session_state.theme_mode = 'dark'
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {'positions': {}, 'cash': 100000.0}

# Header
st.markdown('<h1 class="lab-header">📈 Momentum Trading Lab</h1>', unsafe_allow_html=True)
st.markdown("**Multi-timeframe trend following with adaptive position sizing**")

# Check if data is loaded
if st.session_state.historical_data is None:
    st.warning("⚠️ No historical data loaded. Please load data first.")
    if st.button("💾 Go to Data Loader"):
        st.switch_page("pages/data_loader.py")
    st.stop()

data = st.session_state.historical_data

# Ensure data is a DataFrame
if not isinstance(data, pd.DataFrame):
    st.error("❌ Invalid data format. Please reload data from Data Loader.")
    if st.button("💾 Go to Data Loader"):
        st.switch_page("pages/data_loader.py")
    st.stop()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Trend Analysis",
    "⚡ Breakout Detection",
    "🎯 Strategy Builder",
    "📈 Performance"
])

with tab1:
    st.markdown("### Multi-Timeframe Trend Analysis")
    
    # Prepare data
    if 'symbol' in data.columns:
        available_symbols = data['symbol'].unique().tolist()
        selected_symbol = st.selectbox("Select Asset", available_symbols)
        symbol_data = data[data['symbol'] == selected_symbol].copy()
        symbol_data = symbol_data.set_index('timestamp')
        prices = symbol_data['close']
    else:
        available_symbols = [col for col in data.columns if col not in ['timestamp', 'date', 'Date']]
        selected_symbol = st.selectbox("Select Asset", available_symbols)
        prices = data[selected_symbol]
    
    st.info(f"📊 Analyzing {selected_symbol} with {len(prices)} data points")
    
    # Timeframe settings
    col1, col2, col3 = st.columns(3)
    
    with col1:
        short_period = st.slider("Short MA Period", 5, 50, 10)
    
    with col2:
        medium_period = st.slider("Medium MA Period", 20, 100, 50)
    
    with col3:
        long_period = st.slider("Long MA Period", 50, 200, 100)
    
    # Calculate moving averages
    prices_clean = prices.apply(pd.to_numeric, errors='coerce').dropna()
    
    ma_short = prices_clean.rolling(short_period).mean()
    ma_medium = prices_clean.rolling(medium_period).mean()
    ma_long = prices_clean.rolling(long_period).mean()
    
    # Trend strength calculation
    trend_strength = ((ma_short - ma_long) / ma_long * 100).fillna(0)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_price = prices_clean.iloc[-1]
        st.metric("Current Price", f"${current_price:.2f}")
    
    with col2:
        current_trend = trend_strength.iloc[-1]
        trend_emoji = "📈" if current_trend > 2 else "📉" if current_trend < -2 else "➡️"
        st.metric("Trend Strength", f"{trend_emoji} {current_trend:.2f}%")
    
    with col3:
        price_vs_ma = ((current_price - ma_long.iloc[-1]) / ma_long.iloc[-1] * 100)
        st.metric("vs Long MA", f"{price_vs_ma:.2f}%")
    
    with col4:
        volatility = prices_clean.pct_change().std() * np.sqrt(252) * 100
        st.metric("Volatility (Annual)", f"{volatility:.1f}%")
    
    # Price chart with MAs
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        subplot_titles=('Price & Moving Averages', 'Trend Strength'),
        vertical_spacing=0.1
    )
    
    # Prices and MAs
    fig.add_trace(go.Scatter(
        x=prices_clean.index, y=prices_clean,
        name='Price', line=dict(color='white', width=2)
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=ma_short.index, y=ma_short,
        name=f'MA{short_period}', line=dict(color='#00ff00', width=1)
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=ma_medium.index, y=ma_medium,
        name=f'MA{medium_period}', line=dict(color='#ffff00', width=1)
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=ma_long.index, y=ma_long,
        name=f'MA{long_period}', line=dict(color='#ff0000', width=1)
    ), row=1, col=1)
    
    # Trend strength
    colors = ['green' if x > 0 else 'red' for x in trend_strength]
    fig.add_trace(go.Bar(
        x=trend_strength.index, y=trend_strength,
        name='Trend Strength', marker_color=colors,
        showlegend=False
    ), row=2, col=1)
    
    fig.update_layout(
        height=600,
        hovermode='x unified',
        xaxis2_title='Date'
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("### Breakout Detection")
    
    # Prepare data
    if 'symbol' in data.columns:
        symbol_data = data[data['symbol'] == selected_symbol].copy()
        symbol_data = symbol_data.set_index('timestamp')
        prices = symbol_data['close']
        high = symbol_data['high'] if 'high' in symbol_data.columns else prices
        low = symbol_data['low'] if 'low' in symbol_data.columns else prices
    else:
        prices = data[selected_symbol]
        high = prices
        low = prices
    
    prices_clean = prices.apply(pd.to_numeric, errors='coerce').dropna()
    
    # Breakout settings
    col1, col2 = st.columns(2)
    
    with col1:
        lookback = st.slider("Lookback Period", 10, 100, 20)
        
    with col2:
        breakout_threshold = st.slider("Breakout Threshold (%)", 0.5, 5.0, 2.0, 0.5)
    
    # Calculate support/resistance
    rolling_high = prices_clean.rolling(lookback).max()
    rolling_low = prices_clean.rolling(lookback).min()
    rolling_range = rolling_high - rolling_low
    
    # Detect breakouts
    upper_breakout = prices_clean > rolling_high.shift(1) * (1 + breakout_threshold/100)
    lower_breakout = prices_clean < rolling_low.shift(1) * (1 - breakout_threshold/100)
    
    # Display recent breakouts
    st.markdown("#### Recent Breakouts")
    
    recent_upper = upper_breakout.tail(100).sum()
    recent_lower = lower_breakout.tail(100).sum()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Upper Breakouts (Last 100)", recent_upper)
    
    with col2:
        st.metric("Lower Breakouts (Last 100)", recent_lower)
    
    with col3:
        current_level = "RESISTANCE" if prices_clean.iloc[-1] > rolling_high.iloc[-2] else "SUPPORT" if prices_clean.iloc[-1] < rolling_low.iloc[-2] else "NEUTRAL"
        st.metric("Current Level", current_level)
    
    # Chart with support/resistance
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=prices_clean.index, y=prices_clean,
        name='Price', line=dict(color='white', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=rolling_high.index, y=rolling_high,
        name='Resistance', line=dict(color='red', width=1, dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=rolling_low.index, y=rolling_low,
        name='Support', line=dict(color='green', width=1, dash='dash')
    ))
    
    # Mark breakouts
    upper_breaks = prices_clean[upper_breakout]
    lower_breaks = prices_clean[lower_breakout]
    
    if len(upper_breaks) > 0:
        fig.add_trace(go.Scatter(
            x=upper_breaks.index, y=upper_breaks,
            mode='markers', name='Upper Breakout',
            marker=dict(color='lime', size=10, symbol='triangle-up')
        ))
    
    if len(lower_breaks) > 0:
        fig.add_trace(go.Scatter(
            x=lower_breaks.index, y=lower_breaks,
            mode='markers', name='Lower Breakout',
            marker=dict(color='red', size=10, symbol='triangle-down')
        ))
    
    fig.update_layout(
        title='Support/Resistance & Breakouts',
        xaxis_title='Date',
        yaxis_title='Price',
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("### Momentum Strategy Builder")
    
    strategy_type = st.radio(
        "Strategy Type",
        ["Trend Following", "Breakout", "Crossover"],
        horizontal=True
    )
    
    if strategy_type == "Trend Following":
        st.markdown("#### Trend Following Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            entry_threshold = st.slider("Entry Threshold (%)", 1.0, 10.0, 3.0)
        
        with col2:
            exit_threshold = st.slider("Exit Threshold (%)", 0.5, 5.0, 1.0)
        
        # Generate signals
        trend = ((ma_short - ma_long) / ma_long * 100).fillna(0)
        
        long_signal = trend > entry_threshold
        short_signal = trend < -entry_threshold
        exit_signal = np.abs(trend) < exit_threshold
        
        signals = pd.Series(0, index=trend.index)
        signals[long_signal] = 1
        signals[short_signal] = -1
        signals[exit_signal] = 0
        
    elif strategy_type == "Breakout":
        signals = pd.Series(0, index=prices_clean.index)
        signals[upper_breakout] = 1
        signals[lower_breakout] = -1
        
    else:  # Crossover
        short_ma = prices_clean.rolling(10).mean()
        long_ma = prices_clean.rolling(50).mean()
        
        signals = pd.Series(0, index=prices_clean.index)
        signals[short_ma > long_ma] = 1
        signals[short_ma < long_ma] = -1
    
    # Position sizing
    st.markdown("#### Position Sizing")
    
    sizing_method = st.selectbox(
        "Method",
        ["Fixed", "Volatility Adjusted", "Kelly Criterion"]
    )
    
    if sizing_method == "Fixed":
        position_size = st.slider("Position Size (%)", 10, 100, 50) / 100
        positions = signals * position_size
        
    elif sizing_method == "Volatility Adjusted":
        target_vol = st.slider("Target Volatility (%)", 5, 30, 15) / 100
        realized_vol = prices_clean.pct_change().rolling(20).std() * np.sqrt(252)
        vol_scalar = target_vol / realized_vol.clip(lower=0.01)
        positions = signals * vol_scalar.clip(upper=1.0)
        
    else:  # Kelly
        returns = prices_clean.pct_change()
        win_rate = (returns[signals.shift(1) != 0] > 0).sum() / (signals != 0).sum()
        avg_win = returns[returns > 0].mean()
        avg_loss = abs(returns[returns < 0].mean())
        
        kelly = win_rate - (1 - win_rate) / (avg_win / avg_loss) if avg_loss > 0 else 0.5
        kelly = np.clip(kelly, 0, 0.5)  # Cap at 50%
        
        st.info(f"Kelly Criterion suggests {kelly:.1%} position size")
        positions = signals * kelly
    
    # Calculate returns
    strategy_returns = positions.shift(1) * prices_clean.pct_change()
    cumulative_returns = (1 + strategy_returns).cumprod()
    
    # Display metrics
    st.markdown("#### Strategy Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_return = (cumulative_returns.iloc[-1] - 1) * 100
        st.metric("Total Return", f"{total_return:.2f}%")
    
    with col2:
        annual_return = strategy_returns.mean() * 252 * 100
        st.metric("Annual Return", f"{annual_return:.2f}%")
    
    with col3:
        annual_vol = strategy_returns.std() * np.sqrt(252) * 100
        st.metric("Annual Volatility", f"{annual_vol:.2f}%")
    
    with col4:
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
    
    # Equity curve
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=cumulative_returns.index,
        y=cumulative_returns,
        name='Strategy',
        line=dict(color='#667eea', width=2)
    ))
    
    buy_hold = (1 + prices_clean.pct_change()).cumprod()
    fig.add_trace(go.Scatter(
        x=buy_hold.index,
        y=buy_hold,
        name='Buy & Hold',
        line=dict(color='#764ba2', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title='Equity Curve',
        xaxis_title='Date',
        yaxis_title='Cumulative Return',
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.markdown("### Performance Analytics")
    
    if 'strategy_returns' in locals():
        # Drawdown analysis
        cumulative = (1 + strategy_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            max_dd = drawdown.min() * 100
            st.metric("Max Drawdown", f"{max_dd:.2f}%")
        
        with col2:
            n_trades = (signals.diff() != 0).sum()
            st.metric("Number of Trades", n_trades)
        
        with col3:
            win_rate = (strategy_returns[signals.shift(1) != 0] > 0).sum() / n_trades * 100 if n_trades > 0 else 0
            st.metric("Win Rate", f"{win_rate:.1f}%")
        
        # Drawdown plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown * 100,
            fill='tozeroy',
            name='Drawdown',
            line=dict(color='red')
        ))
        
        fig.update_layout(
            title='Drawdown Analysis',
            xaxis_title='Date',
            yaxis_title='Drawdown (%)',
            hovermode='x unified',
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Monthly returns
        st.markdown("#### Monthly Returns Heatmap")
        
        monthly_returns = strategy_returns.resample('M').sum() * 100
        
        if len(monthly_returns) > 12:
            monthly_df = pd.DataFrame({
                'Year': monthly_returns.index.year,
                'Month': monthly_returns.index.month,
                'Return': monthly_returns.values
            })
            
            pivot_table = monthly_df.pivot(index='Year', columns='Month', values='Return')
            
            fig = go.Figure(data=go.Heatmap(
                z=pivot_table.values,
                x=pivot_table.columns,
                y=pivot_table.index,
                colorscale='RdYlGn',
                zmid=0
            ))
            
            fig.update_layout(
                title='Monthly Returns (%)',
                xaxis_title='Month',
                yaxis_title='Year',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888;'>
    <p>📈 Momentum strategies profit from continuation of price trends across multiple timeframes</p>
</div>
""", unsafe_allow_html=True)
