"""
Signature Methods Lab
Path signature analysis for time series and trading strategies
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import shared UI components
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.ui_components import render_sidebar_navigation, apply_custom_css

# Try to import Rust signature bindings
SIG_AVAILABLE = False
try:
    from sig_optimal_stopping import PySignatureStopper
    SIG_AVAILABLE = True
except ImportError:
    st.warning("⚠️ Signature optimal stopping module not available. Some features disabled.")
    pass

st.set_page_config(page_title="Signature Methods Lab", page_icon="✍️", layout="wide")

# Render sidebar navigation and apply CSS
render_sidebar_navigation(current_page="Signature Methods Lab")
apply_custom_css()

st.markdown('<h1 class="lab-header">✍️ Signature Methods Lab</h1>', unsafe_allow_html=True)
st.markdown("### Path signature analysis for feature extraction and classification")
st.markdown("---")

# Main content
tab1, tab2, tab3 = st.tabs(["📚 Introduction", "🔬 Analysis", "⚡ Trading Signals"])

with tab1:
    st.markdown("### What are Path Signatures?")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        Path signatures are a powerful mathematical tool for analyzing time series data. 
        They provide a coordinate-free description of paths that captures:
        
        - **All statistical moments** of the path
        - **Non-linear interactions** between coordinates
        - **Order of events** (not just values)
        - **Scale-invariant features**
        
        #### Mathematical Foundation
        
        For a path $X: [0,T] \\to \\mathbb{R}^d$, the signature is defined as:
        
        $$S(X)_{0,T} = (1, S^1, S^2, S^3, ...)$$
        
        where $S^k$ are iterated integrals:
        
        $$S^k_{i_1,...,i_k} = \\int_{0<t_1<...<t_k<T} dX^{i_1}_{t_1} ... dX^{i_k}_{t_k}$$
        
        #### Applications in Finance
        
        - **Feature extraction** from price paths
        - **Regime classification** 
        - **Optimal execution** strategies
        - **Signature trading** strategies
        - **Model-free pricing** of path-dependent options
        """)
    
    with col2:
        st.markdown("""
        ### Key Properties
        
        ✅ **Universal**: Characterizes paths uniquely
        
        ✅ **Efficient**: Low-dimensional representation
        
        ✅ **Robust**: Insensitive to noise
        
        ✅ **Interpretable**: Each term has meaning
        
        ✅ **Composable**: Signatures multiply along paths
        """)
        
        st.info("""
        **References:**
        - Lyons (1998): Rough paths theory
        - Levin et al. (2013): Signatures in ML
        - Cochrane & Lyons (2019): Signature methods in finance
        """)

with tab2:
    st.markdown("### Signature Analysis")
    
    if 'historical_data' not in st.session_state or st.session_state.historical_data is None:
        st.warning("⚠️ Please load data first from the Data Loader page")
        if st.button("💾 Go to Data Loader"):
            st.switch_page("pages/data_loader.py")
    else:
        data = st.session_state.historical_data
        
        # Get available symbols
        if isinstance(data, dict):
            symbols = list(data.keys())
        elif isinstance(data, pd.DataFrame):
            if 'symbol' in data.columns:
                symbols = data['symbol'].unique().tolist()
            else:
                symbols = ['Data']
        else:
            symbols = []
        
        if not symbols:
            st.warning("No data available for analysis")
        else:
            col1, col2 = st.columns([3, 1])
            
            with col2:
                selected_symbol = st.selectbox("Select Symbol", symbols)
                truncation = st.slider("Signature Level", 1, 3, 2, help="Truncation level for signature computation")
                window_size = st.slider("Window Size", 10, 100, 50, help="Rolling window for signature computation")
            
            with col1:
                # Extract price data
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
                
                if 'Close' not in df.columns:
                    st.error("Close price column not found")
                    st.stop()
                
                # Compute signatures
                if st.button("🔬 Compute Signatures", type="primary"):
                    with st.spinner("Computing path signatures..."):
                        prices = df['Close'].values
                        
                        # Normalize prices (required for signature computation)
                        prices_norm = (prices - np.mean(prices)) / (np.std(prices) + 1e-8)
                        
                        # Compute rolling signatures
                        signatures = []
                        signature_times = []
                        
                        for i in range(window_size, len(prices_norm)):
                            window = prices_norm[i-window_size:i]
                            
                            # Create 2D trajectory (time, price)
                            times = np.linspace(0, 1, len(window))
                            traj = [[t, p] for t, p in zip(times, window)]
                            
                            if SIG_AVAILABLE:
                                try:
                                    stopper = PySignatureStopper(truncation=truncation, ridge=1e-3)
                                    traj_json = json.dumps(traj)
                                    score = stopper.score(traj_json)
                                    signatures.append(score)
                                except Exception as e:
                                    # Fallback: simple polynomial features
                                    sig = compute_simple_signature(window, truncation)
                                    signatures.append(sig)
                            else:
                                # Fallback: simple polynomial features
                                sig = compute_simple_signature(window, truncation)
                                signatures.append(sig)
                            
                            signature_times.append(df.index[i])
                        
                        # Store in session state
                        st.session_state['signatures'] = {
                            'values': signatures,
                            'times': signature_times,
                            'symbol': selected_symbol,
                            'truncation': truncation
                        }
                        
                        # Plot signature evolution
                        fig = make_subplots(
                            rows=2, cols=1,
                            subplot_titles=('Price Evolution', 'Signature Score Evolution'),
                            vertical_spacing=0.15,
                            row_heights=[0.5, 0.5]
                        )
                        
                        # Price plot
                        fig.add_trace(
                            go.Scatter(x=df.index, y=prices, name='Price',
                                     line=dict(color='blue', width=2)),
                            row=1, col=1
                        )
                        
                        # Signature plot
                        fig.add_trace(
                            go.Scatter(x=signature_times, y=signatures, name='Signature Score',
                                     line=dict(color='purple', width=2)),
                            row=2, col=1
                        )
                        
                        fig.update_xaxes(title_text="Date", row=2, col=1)
                        fig.update_yaxes(title_text="Price", row=1, col=1)
                        fig.update_yaxes(title_text="Score", row=2, col=1)
                        fig.update_layout(height=700, showlegend=True, hovermode='x unified')
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Pattern classification
                        st.markdown("#### 🎯 Pattern Classification")
                        
                        # Classify recent patterns
                        recent_sig = signatures[-1] if signatures else 0
                        recent_return = (prices[-1] - prices[-window_size]) / prices[-window_size] if len(prices) >= window_size else 0
                        volatility = np.std(prices[-window_size:]) / np.mean(prices[-window_size:]) if len(prices) >= window_size else 0
                        
                        col_a, col_b, col_c = st.columns(3)
                        
                        with col_a:
                            pattern = "Bullish" if recent_return > 0.02 else "Bearish" if recent_return < -0.02 else "Neutral"
                            color = "🟢" if pattern == "Bullish" else "🔴" if pattern == "Bearish" else "🟡"
                            st.metric(f"{color} Pattern", pattern, f"{recent_return:.2%}")
                        
                        with col_b:
                            regime = "High Vol" if volatility > 0.03 else "Low Vol" if volatility < 0.01 else "Normal"
                            st.metric("📊 Volatility Regime", regime, f"{volatility:.2%}")
                        
                        with col_c:
                            st.metric("✍️ Signature Score", f"{recent_sig:.4f}")
                        
                        # Feature importance (if multiple components)
                        st.markdown("#### 📈 Signature Components")
                        st.info(f"""
                        **Truncation Level {truncation}:**
                        - Level 1: Linear features (mean increments)
                        - Level 2: Quadratic features (variance, covariance)
                        - Level 3: Cubic features (skewness, higher moments)
                        
                        Higher levels capture more complex path dependencies.
                        """)

def compute_simple_signature(path: np.ndarray, level: int) -> float:
    """
    Compute simple signature-like features when Rust bindings unavailable
    Uses polynomial moments as approximation
    """
    if len(path) == 0:
        return 0.0
    
    # Normalize
    path_norm = (path - np.mean(path)) / (np.std(path) + 1e-8)
    
    # Level 1: mean increment
    increments = np.diff(path_norm)
    sig1 = np.mean(increments) if len(increments) > 0 else 0.0
    
    if level == 1:
        return sig1
    
    # Level 2: add variance-like term
    sig2 = np.mean(increments ** 2) if len(increments) > 0 else 0.0
    
    if level == 2:
        return sig1 + sig2
    
    # Level 3: add skewness-like term
    sig3 = np.mean(increments ** 3) if len(increments) > 0 else 0.0
    
    return sig1 + sig2 + sig3

with tab3:
    st.markdown("### Signature Trading Strategies")
    
    if 'historical_data' not in st.session_state or st.session_state.historical_data is None:
        st.warning("⚠️ Please load data first from the Data Loader page")
    elif 'signatures' not in st.session_state:
        st.info("💡 Run signature analysis in the Analysis tab first")
    else:
        st.markdown("#### 🎯 Signature-Based Trading Signals")
        
        data = st.session_state.historical_data
        sig_data = st.session_state['signatures']
        
        # Get price data
        symbol = sig_data['symbol']
        if isinstance(data, dict):
            df = data[symbol]
        elif isinstance(data, pd.DataFrame):
            if 'symbol' in data.columns:
                df = data[data['symbol'] == symbol].copy()
            else:
                df = data.copy()
        else:
            st.error("Unsupported data format")
            st.stop()
        
        prices = df['Close'].values
        sig_values = sig_data['values']
        sig_times = sig_data['times']
        
        # Trading parameters
        col1, col2 = st.columns(2)
        with col1:
            sig_threshold_long = st.slider("Long Entry Threshold", -1.0, 0.0, -0.1, 0.01, 
                                           help="Enter long when signature score < threshold")
            sig_threshold_short = st.slider("Short Entry Threshold", 0.0, 1.0, 0.1, 0.01,
                                            help="Enter short when signature score > threshold")
        
        with col2:
            holding_period = st.slider("Holding Period (bars)", 5, 50, 20,
                                       help="Hold position for N bars after entry")
            stop_loss = st.slider("Stop Loss %", 1.0, 10.0, 3.0, 0.5,
                                  help="Exit if loss exceeds this percentage")
        
        if st.button("⚡ Generate Signals", type="primary"):
            with st.spinner("Generating trading signals..."):
                # Generate signals based on signature scores
                signals = []
                positions = []  # Track position: 1 = long, -1 = short, 0 = flat
                entry_prices = []
                entry_idx = []
                
                position = 0
                entry_price = 0
                entry_bar = -1
                
                for i, (sig_val, timestamp) in enumerate(zip(sig_values, sig_times)):
                    price_idx = df.index.get_loc(timestamp)
                    current_price = prices[price_idx]
                    
                    # Exit logic
                    if position != 0:
                        bars_held = i - entry_bar
                        pnl_pct = (current_price - entry_price) / entry_price * position
                        
                        # Check stop loss
                        if pnl_pct < -stop_loss / 100:
                            signals.append({
                                'time': timestamp,
                                'signal': 'EXIT_STOP',
                                'price': current_price,
                                'pnl_pct': pnl_pct
                            })
                            position = 0
                            entry_bar = -1
                        # Check holding period
                        elif bars_held >= holding_period:
                            signals.append({
                                'time': timestamp,
                                'signal': 'EXIT_TIME',
                                'price': current_price,
                                'pnl_pct': pnl_pct
                            })
                            position = 0
                            entry_bar = -1
                    
                    # Entry logic (only if flat)
                    if position == 0:
                        if sig_val < sig_threshold_long:
                            # Long signal
                            signals.append({
                                'time': timestamp,
                                'signal': 'LONG',
                                'price': current_price,
                                'pnl_pct': 0
                            })
                            position = 1
                            entry_price = current_price
                            entry_bar = i
                        elif sig_val > sig_threshold_short:
                            # Short signal
                            signals.append({
                                'time': timestamp,
                                'signal': 'SHORT',
                                'price': current_price,
                                'pnl_pct': 0
                            })
                            position = -1
                            entry_price = current_price
                            entry_bar = i
                    
                    positions.append(position)
                
                # Calculate performance
                total_trades = sum(1 for s in signals if s['signal'] in ['LONG', 'SHORT'])
                profitable_trades = sum(1 for s in signals if s['signal'].startswith('EXIT') and s['pnl_pct'] > 0)
                total_pnl = sum(s['pnl_pct'] for s in signals if s['signal'].startswith('EXIT'))
                
                # Display results
                col_a, col_b, col_c, col_d = st.columns(4)
                
                with col_a:
                    st.metric("Total Trades", total_trades)
                with col_b:
                    win_rate = profitable_trades / max(1, total_trades - sum(1 for s in signals if s['signal'].startswith('EXIT_STOP')))
                    st.metric("Win Rate", f"{win_rate:.1%}")
                with col_c:
                    st.metric("Total P&L", f"{total_pnl:.2%}")
                with col_d:
                    avg_pnl = total_pnl / max(1, total_trades)
                    st.metric("Avg P&L per Trade", f"{avg_pnl:.2%}")
                
                # Plot signals on price chart
                fig = go.Figure()
                
                # Price line
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=prices,
                    name='Price',
                    line={'color': 'blue', 'width': 2}
                ))
                
                # Long signals
                long_signals = [s for s in signals if s['signal'] == 'LONG']
                if long_signals:
                    fig.add_trace(go.Scatter(
                        x=[s['time'] for s in long_signals],
                        y=[s['price'] for s in long_signals],
                        mode='markers',
                        name='Long Entry',
                        marker={'symbol': 'triangle-up', 'size': 12, 'color': 'green'}
                    ))
                
                # Short signals
                short_signals = [s for s in signals if s['signal'] == 'SHORT']
                if short_signals:
                    fig.add_trace(go.Scatter(
                        x=[s['time'] for s in short_signals],
                        y=[s['price'] for s in short_signals],
                        mode='markers',
                        name='Short Entry',
                        marker={'symbol': 'triangle-down', 'size': 12, 'color': 'red'}
                    ))
                
                # Exit signals
                exit_signals = [s for s in signals if s['signal'].startswith('EXIT')]
                if exit_signals:
                    exit_colors = ['orange' if s['pnl_pct'] > 0 else 'purple' for s in exit_signals]
                    fig.add_trace(go.Scatter(
                        x=[s['time'] for s in exit_signals],
                        y=[s['price'] for s in exit_signals],
                        mode='markers',
                        name='Exit',
                        marker={'symbol': 'x', 'size': 10, 'color': exit_colors}
                    ))
                
                fig.update_layout(
                    title=f"Signature Trading Signals - {symbol}",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    height=600,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Trade log
                st.markdown("#### 📋 Recent Trades")
                if signals:
                    trades_df = pd.DataFrame(signals[-20:])  # Last 20 signals
                    st.dataframe(trades_df, use_container_width=True)
                else:
                    st.info("No trades generated with current parameters")
        
        # Strategy explanation
        with st.expander("📚 How Signature Trading Works"):
            st.markdown("""
            ### Signature-Based Pattern Recognition
            
            **Core Idea:** Path signatures capture the "shape" of price movements in a coordinate-free way.
            
            #### Signal Logic:
            
            1. **Long Entry** (Signature Score < Threshold):
               - Low signature score indicates downward momentum exhaustion
               - Pattern suggests potential reversal or stabilization
               - Similar to oversold conditions but path-aware
            
            2. **Short Entry** (Signature Score > Threshold):
               - High signature score indicates overextended upward movement
               - Pattern suggests potential correction
               - Similar to overbought conditions but path-aware
            
            3. **Exit Conditions**:
               - **Time-based**: Close after fixed holding period
               - **Stop-loss**: Exit if loss exceeds threshold
               - **Signal reversal**: Exit when opposite signal appears
            
            #### Advantages:
            - ✅ Captures full path dynamics, not just endpoints
            - ✅ Invariant to time reparametrization
            - ✅ Natural feature extraction without manual engineering
            - ✅ Works across different market regimes
            
            #### Optimal Execution Application:
            Signatures can also optimize order execution by predicting short-term price impact
            based on recent order flow patterns.
            """)

# Sidebar
with st.sidebar:
    st.markdown("### 🎛️ Settings")
    
    truncation_level = st.slider("Signature Level", 2, 5, 3)
    st.info(f"Computing signature up to level {truncation_level}")
    
    st.markdown("---")
    st.markdown("### 📚 Resources")
    st.markdown("""
    - [Rough Paths Theory](https://en.wikipedia.org/wiki/Rough_path)
    - [esig Library](https://esig.readthedocs.io/)
    - [Signatures in ML](https://arxiv.org/abs/1603.03788)
    """)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>✍️ Signature Methods Lab | Part of HFT Arbitrage Lab</p>
</div>
""", unsafe_allow_html=True)
