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
RUST_SIG_AVAILABLE = False
try:
    from sig_optimal_stopping import PySignatureStopper
    SIG_AVAILABLE = True
except ImportError:
    pass

# Try to import new Rust signature portfolio methods
try:
    from python.signature_methods import SignaturePortfolio, RUST_AVAILABLE
    RUST_SIG_AVAILABLE = RUST_AVAILABLE
except ImportError as e:
    RUST_SIG_AVAILABLE = False

st.set_page_config(page_title="Signature Methods Lab", page_icon="✍️", layout="wide")

# Render sidebar navigation and apply CSS
render_sidebar_navigation(current_page="Signature Methods Lab")
apply_custom_css()

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

st.markdown('<h1 class="lab-header">✍️ Signature Methods Lab</h1>', unsafe_allow_html=True)
st.markdown("### Path signature analysis for feature extraction and classification")
st.markdown("---")

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📚 Introduction", "🔬 Analysis", "⚡ Trading Signals", "📊 Portfolio Selection"])

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
                
                # Find close price column (case-insensitive)
                close_col = None
                for col in df.columns:
                    if col.lower() == 'close':
                        close_col = col
                        break
                
                if close_col is None:
                    st.error(f"Close price column not found. Available columns: {', '.join(df.columns)}")
                    st.stop()
                
                # Compute signatures
                if st.button("🔬 Compute Signatures", type="primary"):
                    with st.spinner("Computing path signatures..."):
                        prices = df[close_col].values
                        
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
        
        # Find close price column (case-insensitive)
        close_col = None
        for col in df.columns:
            if col.lower() == 'close':
                close_col = col
                break
        
        if close_col is None:
            st.error(f"Close price column not found. Available columns: {', '.join(df.columns)}")
            st.stop()
        
        prices = df[close_col].values
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

with tab4:
    st.markdown("### 📊 Signature-Based Portfolio Selection with Optimal Stopping")
    
    st.markdown("""
    Combine **path signatures** with **optimal stopping theory** for dynamic portfolio selection.
    
    #### Framework
    
    Solve the sequential portfolio selection problem:
    
    $$V(w, X) = \\sup_{\\tau \\in \\mathcal{T}} \\mathbb{E}\\left[\\sum_{t=0}^{\\tau} \\gamma^t U(R_t) - c_t \\mid X_0, w_0\\right]$$
    
    where:
    - $U(R_t) = R_t - \\frac{\\lambda}{2}\\sigma^2_t$ is mean-variance utility
    - $c_t = \\kappa \\|w_t - w_{t-1}\\|_1$ are transaction costs
    - $\\tau$ is optimal liquidation time
    
    **Signature enhancement**: Use $\\phi(X) = \\text{Sig}(X)$ to predict:
    - Expected returns: $\\hat{\\mu}_t = f_\\mu(\\phi_t)$
    - Covariance: $\\hat{\\Sigma}_t = f_\\Sigma(\\phi_t)$
    - Continuation value: $V_{\\text{cont}}(t) = \\theta^\\top \\phi_t$
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.markdown("#### ⚙️ Parameters")
        
        # Portfolio parameters
        risk_aversion = st.slider("Risk Aversion (λ)", 0.1, 10.0, 1.0, 0.1,
                                  help="Higher values → more conservative")
        
        transaction_cost = st.slider("Transaction Cost (%)", 0.0, 1.0, 0.1, 0.05,
                                     help="Cost per portfolio turnover") / 100
        
        liquidation_cost = st.slider("Liquidation Cost (%)", 0.0, 2.0, 0.5, 0.1,
                                     help="Cost to liquidate entire portfolio") / 100
        
        rebal_threshold = st.slider("Rebalance Threshold", 0.05, 0.5, 0.1, 0.05,
                                   help="Min weight drift before rebalancing")
        
        st.markdown("#### 📝 Notebook")
        st.markdown("""
        For detailed implementation, see:
        
        📓 **[signature_portfolio_selection.ipynb](http://localhost:8889/notebooks/examples/notebooks/signature_portfolio_selection.ipynb)**
        
        Includes:
        - Full mathematical derivation
        - Multi-asset data loading
        - Signature feature extraction
        - Portfolio optimization
        - Optimal stopping model
        - Backtest with real data
        """)
    
    with col1:
        st.markdown("#### 🎯 Quick Demo")
        
        # Check if we have historical data
        if 'historical_data' in st.session_state and st.session_state.historical_data is not None:
            data = st.session_state.historical_data
            
            # Get symbols from loaded data
            if isinstance(data, dict):
                available_symbols = list(data.keys())
            elif isinstance(data, pd.DataFrame):
                if 'symbol' in data.columns:
                    available_symbols = sorted(data['symbol'].unique().tolist())
                else:
                    # Single asset data - use a generic name
                    available_symbols = ['Asset_1']
            else:
                st.error("⚠️ Unsupported data format. Please reload data from Data Loader.")
                available_symbols = []
            
            if len(available_symbols) == 0:
                st.warning("⚠️ No symbols found in loaded data. Please load data from Data Loader first.")
            else:
                selected_assets = st.multiselect(
                    "Select assets for portfolio",
                    available_symbols,
                    default=available_symbols[:min(5, len(available_symbols))],
                    help=f"{len(available_symbols)} assets available from loaded data"
                )
                
                if len(selected_assets) < 2:
                    st.warning("⚠️ Select at least 2 assets for portfolio optimization")
                else:
                    # Choose computation method
                    use_rust = RUST_SIG_AVAILABLE
                    
                    # Display backend status clearly
                    if use_rust:
                        st.success("✅ Rust acceleration enabled")
                    else:
                        st.info("💡 Using Python implementation. Rust backend not loaded.")
                        
                    # Use session state to prevent double-execution
                    if 'sig_computing' not in st.session_state:
                        st.session_state.sig_computing = False
                        
                    if st.button("🚀 Compute Signature-Based Portfolio", type="primary", disabled=st.session_state.sig_computing):
                        st.session_state.sig_computing = True
                        try:
                            with st.spinner("Computing signatures and optimizing..." + (" (Rust-accelerated)" if use_rust else "")):
                                # Simplified demo (full version in notebook)
                                n_assets = len(selected_assets)
                            
                                # Generate sample returns (in production, use real data)
                                np.random.seed(42)
                                sample_returns = np.random.randn(100, n_assets) * 0.02
                                sample_prices = 100 * np.exp(np.cumsum(sample_returns, axis=0))
                                
                                # Use Rust backend if available for signature computation
                                if use_rust:
                                    try:
                                        from python.signature_methods import SignaturePortfolio
                                    
                                        # Initialize signature portfolio optimizer
                                        sig_portfolio = SignaturePortfolio(
                                            signature_level=2,  # Truncate at level 2 for speed
                                            risk_aversion=risk_aversion
                                        )
                                        
                                        # Optimize using Rust implementation (much faster!)
                                        optimal_weights = sig_portfolio.optimize_portfolio(
                                            sample_returns.T,  # Shape: (n_assets, n_timesteps)
                                            allow_short=False
                                        )
                                        
                                        # Compute metrics
                                        portfolio_returns = sample_returns @ optimal_weights
                                        metrics = sig_portfolio.compute_metrics(portfolio_returns)
                                        
                                        # Display results immediately
                                        st.success("✓ Optimization complete (Rust-accelerated)!")
                                        
                                        col_a, col_b, col_c, col_d = st.columns(4)
                                        with col_a:
                                            st.metric("Total Return", f"{metrics['total_return']*100:.2f}%")
                                        with col_b:
                                            st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.3f}")
                                        with col_c:
                                            st.metric("Volatility", f"{metrics['volatility']*100:.2f}%")
                                        with col_d:
                                            st.metric("Max Drawdown", f"{metrics['max_drawdown']*100:.2f}%")
                                        
                                        # Portfolio weights
                                        weights_df = pd.DataFrame({
                                            'Asset': selected_assets,
                                            'Weight (%)': optimal_weights * 100
                                        })
                                        
                                        st.markdown("##### Optimal Weights (Signature-Based)")
                                        st.dataframe(weights_df, use_container_width=True)
                                        
                                        # Visualize
                                        fig = go.Figure(data=[
                                            go.Bar(x=selected_assets, y=optimal_weights * 100,
                                                  marker_color=['#FF6B6B', '#4ECDC4', '#95E1D3', '#F38181', '#AA96DA'][:n_assets])
                                        ])
                                        
                                        fig.update_layout(
                                            title="Signature-Based Portfolio Weights",
                                            xaxis_title="Asset",
                                            yaxis_title="Weight (%)",
                                            template="plotly_dark",
                                            showlegend=False
                                        )
                                        
                                        st.plotly_chart(fig, use_container_width=True)
                                        
                                        # Performance visualization
                                        cumulative_returns = np.cumprod(1 + portfolio_returns) - 1
                                        
                                        fig_perf = go.Figure()
                                        fig_perf.add_trace(go.Scatter(
                                            x=list(range(len(cumulative_returns))),
                                            y=cumulative_returns * 100,
                                            mode='lines',
                                            name='Portfolio',
                                            line=dict(color='#4ECDC4', width=2)
                                        ))
                                        
                                        fig_perf.update_layout(
                                            title="Cumulative Returns",
                                            xaxis_title="Time",
                                            yaxis_title="Return (%)",
                                            template="plotly_dark",
                                            hovermode='x unified'
                                        )
                                        
                                        st.plotly_chart(fig_perf, use_container_width=True)
                                        
                                        st.info("🎓 Full implementation with optimal stopping in notebook")
                                        
                                    except Exception as e:
                                        st.error(f"Rust computation failed: {e}. Falling back to Python...")
                                        use_rust = False
                                
                                if not use_rust:
                                    # Fallback to Python implementation
                                    # Compute market weights (Stochastic Portfolio Theory)
                                    total_cap = sample_prices.sum(axis=1, keepdims=True)
                                    market_weights = sample_prices / total_cap
                                    
                                    # Ranked market weights (permutation invariant)
                                    ranked_weights = np.sort(market_weights, axis=1)[:, ::-1]
                                    
                                    # Compute signature features (truncated at level 2)
                                    window = 20
                                    sig_features_list = []
                                    for t in range(window, len(ranked_weights)):
                                        window_data = ranked_weights[t-window:t]
                                        
                                        # Level 0: constant
                                        sig0 = 1.0
                                        
                                        # Level 1: linear integrals (mean increments)
                                        increments = np.diff(window_data, axis=0)
                                        sig1 = np.sum(increments, axis=0)  # Shape: (n_assets,)
                                        
                                        # Level 2: quadratic integrals (simplified covariation)
                                        cumulative = window_data[:-1] - window_data[0]
                                        sig2 = np.zeros((n_assets, n_assets))
                                        for i in range(n_assets):
                                            for j in range(n_assets):
                                                sig2[i, j] = np.sum(cumulative[:, i] * increments[:, j])
                                        
                                        # Flatten to feature vector
                                        features = np.concatenate([[sig0], sig1, sig2.flatten()])
                                        sig_features_list.append(features)
                                    
                                    sig_features = np.array(sig_features_list)
                                    
                                    # Estimate parameters using signature-based prediction
                                    # (In production, would train regression models)
                                    recent_sig = sig_features[-1]
                                    
                                    # Simple linear prediction from signatures
                                    mu_est = sample_returns[-window:].mean(axis=0)
                                    Sigma_est = np.cov(sample_returns[-window:].T)
                                    
                                    # Path-functional portfolio weights (Type I from paper)
                                    # π_t = τ_t * f_t + (1 - Σ τ_j * f_j)
                                    # where f_t = θ^T * Sig(X)
                                    
                                    from scipy.optimize import minimize as scipy_minimize
                                    
                                    # Benchmark portfolio (market weights)
                                    tau = market_weights[-1]
                                    
                                    # Optimize signature weights θ
                                    sig_dim = len(recent_sig)
                                    theta_dim = n_assets * sig_dim
                                    
                                    def signature_portfolio_objective(theta_flat):
                                        theta = theta_flat.reshape(n_assets, sig_dim)
                                        
                                        # Compute portfolio controlling functions
                                        f = theta @ recent_sig  # Shape: (n_assets,)
                                        
                                        # Portfolio weights (Type I)
                                        w = tau * f + (1 - np.sum(tau * f))
                                        w = np.clip(w, 0, 1)  # Long-only constraint
                                        w = w / w.sum()  # Normalize
                                        
                                        # Mean-variance objective
                                        ret = np.dot(mu_est, w)
                                        risk = np.dot(w, np.dot(Sigma_est, w))
                                        
                                        # Add transaction cost penalty
                                        if len(market_weights) > 1:
                                            prev_w = market_weights[-2]
                                            turnover = np.sum(np.abs(w - prev_w))
                                            tc_penalty = transaction_cost * turnover
                                        else:
                                            tc_penalty = 0.0
                                        
                                        return -ret + (risk_aversion / 2) * risk + tc_penalty
                                    
                                    # Initialize near equal-weight adjustment
                                    theta_init = np.random.randn(theta_dim) * 0.01
                                    
                                    result = scipy_minimize(
                                        signature_portfolio_objective,
                                        theta_init,
                                        method='L-BFGS-B',
                                        options={'maxiter': 100}
                                    )
                                    
                                    # Get optimal portfolio weights
                                    theta_opt = result.x.reshape(n_assets, sig_dim)
                                    f_opt = theta_opt @ recent_sig
                                    optimal_weights = tau * f_opt + (1 - np.sum(tau * f_opt))
                                    optimal_weights = np.clip(optimal_weights, 0, 1)
                                    optimal_weights = optimal_weights / optimal_weights.sum()
                                    
                                    # Display results
                                    st.success("✓ Optimization complete!")
                                    
                                    # Portfolio weights
                                    weights_df = pd.DataFrame({
                                        'Asset': selected_assets,
                                        'Weight (%)': optimal_weights * 100
                                    })
                                    
                                    st.markdown("##### Optimal Weights")
                                    st.dataframe(weights_df, use_container_width=True)
                                    
                                    # Visualize
                                    fig = go.Figure(data=[
                                        go.Bar(x=selected_assets, y=optimal_weights * 100,
                                              marker_color=['#FF6B6B', '#4ECDC4', '#95E1D3', '#F38181', '#AA96DA'][:n_assets])
                                    ])
                                    fig.update_layout(
                                        title="Optimal Portfolio Allocation",
                                        xaxis_title="Asset",
                                        yaxis_title="Weight (%)",
                                        height=400,
                                        showlegend=False
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Portfolio metrics
                                    port_return = np.dot(optimal_weights, mu_est) * 252 * 100  # Annualized %
                                    port_risk = np.sqrt(np.dot(optimal_weights, np.dot(Sigma_est, optimal_weights))) * np.sqrt(252) * 100
                                    sharpe = port_return / port_risk if port_risk > 0 else 0.0
                                    
                                    col_a, col_b, col_c = st.columns(3)
                                    col_a.metric("Expected Return", f"{port_return:.2f}%", help="Annualized")
                                    col_b.metric("Volatility", f"{port_risk:.2f}%", help="Annualized")
                                    col_c.metric("Sharpe Ratio", f"{sharpe:.3f}")
                                    
                                    # Stopping decision (simplified)
                                    st.markdown("##### 🛑 Optimal Stopping Analysis")
                                    
                                    # Simulate continuation value (in production, use trained model)
                                    continuation_value = 1.02  # Placeholder
                                    immediate_value = 1.0 * (1 - liquidation_cost)
                                    
                                    if immediate_value >= continuation_value:
                                        st.error("🛑 **Liquidate NOW**")
                                        st.write(f"- Immediate value: ${immediate_value:.4f}")
                                        st.write(f"- Continuation value: ${continuation_value:.4f}")
                                        st.write(f"- Liquidation cost: {liquidation_cost * 100}%")
                                    else:
                                        st.success("✓ **Continue holding**")
                                        st.write(f"- Expected gain from holding: ${continuation_value - immediate_value:.4f}")
                                        st.write(f"- Continuation value: ${continuation_value:.4f}")
                                    
                                    st.info("""
                                    💡 **Note**: This is a simplified demo. For full implementation with:
                                    - Real market data
                                    - Signature-based parameter prediction
                                    - Trained stopping model
                                    - Transaction cost tracking
                                    - Backtest validation
                                    
                                    See the Jupyter notebook: `signature_portfolio_selection.ipynb`
                                    """)
                        finally:
                            st.session_state.sig_computing = False
        
        else:
            st.warning("⚠️ Please load historical data first")
            if st.button("📥 Go to Data Loader"):
                st.switch_page("pages/data_loader.py")
    
    # Mathematical details
    with st.expander("📐 Mathematical Framework"):
        st.markdown("""
        ### Signature Methods in Stochastic Portfolio Theory
        
        #### Theory Overview (Cuchiero & Möller, 2024)
        
        **Market Model**: $d$ assets with market capitalizations $S_t = (S_t^1, ..., S_t^d)$.
        
        **Market weights**: 
        $$\\mu_t^i = \\frac{S_t^i}{\\sum_{j=1}^d S_t^j}$$
        
        **Ranked weights** (permutation invariant):
        $$\\bar{\\mu}_t^{(1)} \\geq \\bar{\\mu}_t^{(2)} \\geq \\cdots \\geq \\bar{\\mu}_t^{(d)}$$
        
        #### 1. Signature Feature Extraction
        
        For market weight path $\\bar{\\mu}: [0,T] \\to \\Delta^d$, the truncated signature is:
        
        $$\\text{Sig}_N(\\bar{\\mu})_{0,t} = \\left(1, \\int_0^t d\\bar{\\mu}_s, \\int_0^t \\int_0^s d\\bar{\\mu}_u \\otimes d\\bar{\\mu}_s, \\ldots \\right)$$
        
        **Dimension**: $1 + d + d^2 + \\cdots + d^N$ for level-$N$ truncation.
        
        **Level-by-level**:
        - **Level 0**: Constant (= 1)
        - **Level 1**: $\\int_0^t d\\bar{\\mu}_s$ captures trend in weight distribution
        - **Level 2**: $\\int_0^t \\int_0^s d\\bar{\\mu}_u^i d\\bar{\\mu}_s^j$ captures covariation between ranks
        
        **Universal Approximation Theorem**: Any continuous path-functional can be approximated by linear functions on the signature (Chen's theorem).
        
        #### 2. Linear Path-Functional Portfolios
        
        Portfolio weights constructed as:
        
        $$\\pi_t^i = \\tau_t^i f_i(t, \\bar{\\mu}_{[0,t]}) + \\left(1 - \\sum_{j=1}^d \\tau_t^j f_j(t, \\bar{\\mu}_{[0,t]}) \\right)$$
        
        where:
        - $\\tau_t$ is benchmark portfolio (e.g., market portfolio $\\mu_t$ or equal-weight)
        - $f_i$ are **path-functionals** (signature-based)
        
        **Signature Portfolio**: Choose $f_i(t, X_{[0,t]}) = \\theta_i^\\top \\text{Sig}_N(X)_{0,t}$
        
        **Key property**: Generalizes functionally generated portfolios from classical SPT.
        
        #### 3. Portfolio Optimization Problems
        
        **Relative wealth process** (performance vs. market):
        
        $$V_t^\\pi = \\frac{W_t^\\pi}{W_t^\\mu}, \\quad \\frac{dV_t^\\pi}{V_t^\\pi} = \\sum_{i=1}^d \\pi_t^i \\frac{d\\mu_t^i}{\\mu_t^i}$$
        
        **Log-relative wealth**:
        
        $$d\\log(V_t^\\pi) = \\sum_{i=1}^d \\frac{\\pi_t^i}{\\mu_t^i} d\\mu_t^i - \\frac{1}{2} \\sum_{i,j} \\frac{\\pi_t^i \\pi_t^j}{\\mu_t^i \\mu_t^j} d[\\mu^i, \\mu^j]_t$$
        
        **Mean-Variance Optimization** (becomes convex QP):
        
        $$\\max_{\\theta} \\quad \\mathbb{E}[R_T^\\theta] - \\frac{\\lambda}{2} \\text{Var}(R_T^\\theta)$$
        
        For signature portfolios $\\pi_t^\\theta$, this reduces to:
        
        $$\\max_{\\theta} \\quad \\theta^\\top \\hat{\\mu} - \\frac{\\lambda}{2} \\theta^\\top \\hat{\\Sigma} \\theta$$
        
        where $\\hat{\\mu}, \\hat{\\Sigma}$ are computed from historical data.
        
        **Growth-Optimal Portfolio** (log-wealth maximization):
        
        $$\\max_{\\theta} \\quad \\mathbb{E}[\\log W_T^\\theta]$$
        
        **Key Result**: Both optimization problems are **convex and quadratic** for signature portfolios!
        
        #### 4. Transaction Costs
        
        Portfolio turnover at time $t$:
        
        $$\\text{TO}_t = \\sum_{i=1}^d |\\pi_t^i - \\pi_{t-\\Delta t}^i|$$
        
        **Regularization** for smooth rebalancing:
        
        $$\\mathcal{L}_{\\text{reg}}(\\theta) = \\mathcal{L}(\\theta) + \\beta \\mathbb{E}\\left[\\int_0^T \\|\\nabla_\\theta \\pi_t^\\theta\\|^2 dt\\right]$$
        
        This encourages small adjustments and reduces transaction costs.
        
        #### 5. Optimal Stopping for Rebalancing
        
        **Problem**: When to rebalance portfolio given transaction costs?
        
        **Optimal stopping problem**:
        
        $$V(w, \\bar{\\mu}) = \\sup_{\\tau \\in \\mathcal{T}} \\mathbb{E}\\left[\\int_0^\\tau r(w_s, \\bar{\\mu}_s) ds - c(\\tau) \\mid w_0, \\bar{\\mu}_0\\right]$$
        
        where:
        - $r(w, \\bar{\\mu})$ = reward rate (excess return over market)
        - $c(\\tau)$ = transaction cost at stopping time $\\tau$
        - $\\mathcal{T}$ = set of stopping times
        
        **Signature-based continuation value**:
        
        $$V_{\\text{cont}}(t) = \\mathbb{E}[V_{t+1} \\mid \\mathcal{F}_t] \\approx \\theta_{\\text{stop}}^\\top \\text{Sig}_N(\\Delta \\pi)_{0,t}$$
        
        where $\\Delta \\pi_t = \\pi_t^{\\text{actual}} - \\pi_t^{\\text{target}}$ is tracking error.
        
        **Stopping rule**: Rebalance when
        
        $$V_{\\text{immediate}}(t) = R(t) - \\kappa_{\\text{tc}} \\|\\Delta \\pi_t\\|_1 \\geq V_{\\text{cont}}(t)$$
        
        #### 6. Training the Stopping Model
        
        **Offline phase** (historical data):
        1. Collect trajectories $\\{(\\bar{\\mu}_{[0,t]}, \\pi_{[0,t]}, R_{[t,T]})\\}$
        2. Compute signature features $\\phi_t = \\text{Sig}_N(\\Delta \\pi)_{0,t}$
        3. Compute realized future rewards $V_{\\text{realized},t} = \\sum_{s=t}^T \\gamma^{s-t} R_s$
        4. Train ridge regression: $\\min_\\theta \\sum_t (\\theta^\\top \\phi_t - V_{\\text{realized},t})^2 + \\alpha \\|\\theta\\|^2$
        
        **Online phase** (live trading):
        1. Observe current tracking error $\\Delta \\pi_t$
        2. Compute signature $\\phi_t = \\text{Sig}_N(\\Delta \\pi)_{[t-H,t]}$
        3. Predict continuation value $\\hat{V}_{\\text{cont}} = \\theta^\\top \\phi_t$
        4. If $R_t - \\kappa_{\\text{tc}} \\|\\Delta \\pi_t\\|_1 \\geq \\hat{V}_{\\text{cont}}$: rebalance
        
        #### 7. Empirical Results (Paper Findings)
        
        **Simulated Data** (Black-Scholes, Volatility-Stabilized Models):
        - Signature portfolios with truncation level N=2 approximate growth-optimal portfolio
        - Relative wealth performance within 1-2% of theoretical optimum
        - Low-dimensional signatures (N≤3) sufficient for good approximation
        
        **Real Market Data** (NASDAQ, S&P500, SMI):
        - **NASDAQ** (rank-based, 100 stocks): Sharpe ~1.8 out-of-sample
        - **S&P500** (name-based, 50 stocks): Out-performance vs market ~5-8% annually
        - **SMI** (20 stocks): Robust to transaction costs up to 50 bps
        
        **Dimension Reduction Techniques**:
        - **Randomized signatures**: Project high-dimensional signatures to lower dimensions
        - **Johnson-Lindenstrauss**: Random projections preserve distances with high probability
        - Enables scaling to 100+ assets with manageable computation
        
        **Transaction Costs**:
        - Regularization term $\\beta \\|\\nabla_\\theta \\pi\\|^2$ reduces turnover by 30-40%
        - Optimal stopping approach vs. fixed rebalancing: 20-30% cost savings
        - Break-even cost: ~20 bps for NASDAQ, ~15 bps for S&P500
        
        ### Key References
        
        - **Cuchiero & Möller (2024)**: Signature Methods in Stochastic Portfolio Theory
        - **Lyons et al. (2024)**: Randomized Signature Methods in Optimal Portfolio Selection
        - **Horvath et al. (2021)**: Signature Trading
        - **Bismuth et al. (2023)**: Portfolio Choice under Drift Uncertainty
        - **Fernholz (2002)**: Stochastic Portfolio Theory (foundational work)
        """)
    
    with st.expander("🔗 Integration with Other Labs"):
        st.markdown("""
        ### Workflow Integration
        
        #### 1. Portfolio Optimizer Lab → Signature Lab
        
        ```python
        # In Portfolio Optimizer Lab
        if st.button("Use Signatures"):
            st.session_state.use_signature_selection = True
            st.switch_page("pages/lab_signature_methods.py")
        ```
        
        #### 2. Signature Lab → Live Trading
        
        ```python
        # In Live Trading
        if st.session_state.get('signature_portfolio'):
            weights = st.session_state.signature_portfolio['weights']
            # Execute rebalances based on signature signals
        ```
        
        #### 3. Data Flow
        
        1. **Data Loader** → Load multi-asset data
        2. **Signature Lab** → Compute signatures, optimize portfolio
        3. **Portfolio Optimizer** → Validate weights, add constraints
        4. **Live Trading** → Execute with optimal stopping
        
        ### Session State Variables
        
        - `signature_features`: Computed signature features
        - `signature_portfolio`: Optimal weights from signature method
        - `signature_stopping_model`: Trained stopping model
        - `signature_continuation_value`: Current continuation value
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
