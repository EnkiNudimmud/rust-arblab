"""
PCA Arbitrage Lab - Principal Component Analysis for Multi-Asset Trading
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.ui_components import render_sidebar_navigation, apply_custom_css, ensure_data_loaded

# Page configuration
st.set_page_config(
    page_title="PCA Arbitrage Lab",
    page_icon="🎯",
    layout="wide"
)

# Initialize session state
if 'historical_data' not in st.session_state:
    st.session_state.historical_data = None

# Auto-load most recent dataset if no data is loaded
data_available = ensure_data_loaded()

# Apply custom styling and navigation
apply_custom_css()
render_sidebar_navigation(current_page="PCA Arbitrage Lab")
if 'theme_mode' not in st.session_state:
    st.session_state.theme_mode = 'light'
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {'positions': {}, 'cash': 100000.0}

# Header
st.markdown('<h1 class="lab-header">🎯 PCA Arbitrage Lab</h1>', unsafe_allow_html=True)
st.markdown("**Principal Component Analysis for dimensionality reduction and factor-based trading**")

# Check if data is loaded
if not data_available or st.session_state.historical_data is None:
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
    "📊 PCA Analysis",
    "🎯 Factor Trading",
    "📈 Portfolio Construction",
    "🔬 Backtesting"
])

with tab1:
    st.markdown("### Principal Component Analysis")
    
    # Prepare data
    if data is not None and 'symbol' in data.columns:
        # Long format - pivot to wide
        price_data = data.pivot(index='timestamp', columns='symbol', values='close')
    elif data is not None:
        # Already wide format
        price_data = data.select_dtypes(include=[np.number]) if data is not None else pd.DataFrame()
    
    # Calculate returns
    returns = price_data.pct_change().dropna()
    
    st.info(f"📊 Analyzing {len(returns.columns)} assets with {len(returns)} time periods")
    
    # PCA settings
    col1, col2 = st.columns([2, 1])
    
    with col1:
        n_components = st.slider(
            "Number of Components",
            min_value=2,
            max_value=min(10, len(returns.columns)),
            value=min(5, len(returns.columns)),
            help="Number of principal components to extract"
        )
    
    with col2:
        standardize = st.checkbox("Standardize Data", value=True, help="Scale data to unit variance")
    
    if st.button("🔬 Run PCA Analysis", type="primary"):
        with st.spinner("Computing principal components..."):
            # Standardize if requested
            if standardize:
                scaler = StandardScaler()
                returns_scaled = scaler.fit_transform(returns)
            else:
                returns_scaled = returns.values
            
            # Perform PCA
            pca = PCA(n_components=n_components)
            components = pca.fit_transform(returns_scaled)
            
            # Store results
            st.session_state.pca_model = pca
            st.session_state.pca_components = components
            st.session_state.pca_returns = returns
            
            # Display results
            st.success("✅ PCA analysis completed!")
            
            # Explained variance
            st.markdown("#### Explained Variance")
            
            variance_df = pd.DataFrame({
                'Component': [f'PC{i+1}' for i in range(n_components)],
                'Variance Explained (%)': pca.explained_variance_ratio_ * 100,
                'Cumulative (%)': np.cumsum(pca.explained_variance_ratio_) * 100
            })
            
            st.dataframe(variance_df, use_container_width=True, hide_index=True)
            
            # Scree plot
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=variance_df['Component'],
                y=variance_df['Variance Explained (%)'],
                name='Individual',
                marker_color='#667eea'
            ))
            fig.add_trace(go.Scatter(
                x=variance_df['Component'],
                y=variance_df['Cumulative (%)'],
                name='Cumulative',
                mode='lines+markers',
                marker=dict(color='#764ba2', size=10),
                yaxis='y2'
            ))
            
            fig.update_layout(
                title='Scree Plot - Explained Variance',
                xaxis_title='Principal Component',
                yaxis_title='Variance Explained (%)',
                yaxis2=dict(title='Cumulative (%)', overlaying='y', side='right'),
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Component loadings
            st.markdown("#### Component Loadings (Top Assets)")
            
            loadings = pd.DataFrame(
                pca.components_.T,
                columns=[f'PC{i+1}' for i in range(n_components)],
                index=returns.columns
            )
            
            # Show top loadings for each component
            for i in range(min(3, n_components)):
                pc_name = f'PC{i+1}'
                st.markdown(f"**{pc_name} - Top 5 Positive & Negative Loadings**")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("*Positive:*")
                    top_positive = loadings[pc_name].nlargest(5)
                    st.dataframe(
                        pd.DataFrame({
                            'Asset': top_positive.index,
                            'Loading': top_positive.values
                        }),
                        hide_index=True
                    )
                
                with col2:
                    st.markdown("*Negative:*")
                    top_negative = loadings[pc_name].nsmallest(5)
                    st.dataframe(
                        pd.DataFrame({
                            'Asset': top_negative.index,
                            'Loading': top_negative.values
                        }),
                        hide_index=True
                    )

with tab2:
    st.markdown("### Factor-Based Trading Signals")
    
    if 'pca_model' not in st.session_state:
        st.info("👈 Run PCA analysis first in the 'PCA Analysis' tab")
    else:
        pca = st.session_state.pca_model
        components = st.session_state.pca_components
        returns = st.session_state.pca_returns
        
        st.markdown("#### Component Momentum Signals")
        
        # Calculate component returns
        component_df = pd.DataFrame(
            components,
            columns=[f'PC{i+1}' for i in range(pca.n_components_)],
            index=returns.index
        )
        
        component_returns = component_df.pct_change().dropna()
        
        # Signal settings
        lookback = st.slider("Signal Lookback Period", 5, 50, 20)
        
        # Calculate signals (momentum)
        signals = component_returns.rolling(lookback).mean()
        signals = signals.apply(lambda x: np.tanh(x * 10))  # Normalize to [-1, 1]
        
        # Display latest signals
        st.markdown("#### Latest Component Signals")
        
        latest_signals = signals.iloc[-1]
        
        cols = st.columns(min(5, len(latest_signals)))
        for i, (component, signal) in enumerate(latest_signals.items()):
            with cols[i % len(cols)]:
                color = "🟢" if signal > 0.3 else "🔴" if signal < -0.3 else "🟡"
                st.metric(
                    component,
                    f"{signal:.3f}",
                    delta=f"{color} {'Long' if signal > 0 else 'Short'}"
                )
        
        # Plot component time series
        st.markdown("#### Principal Component Time Series")
        
        selected_pc = st.multiselect(
            "Select Components to Plot",
            options=component_df.columns.tolist(),
            default=component_df.columns[:3].tolist()
        )
        
        if selected_pc:
            fig = go.Figure()
            
            for pc in selected_pc:
                fig.add_trace(go.Scatter(
                    x=component_df.index,
                    y=component_df[pc],
                    name=pc,
                    mode='lines'
                ))
            
            fig.update_layout(
                title='Principal Component Evolution',
                xaxis_title='Date',
                yaxis_title='Component Value',
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("### PCA-Based Portfolio Construction")
    
    if 'pca_model' not in st.session_state:
        st.info("👈 Run PCA analysis first in the 'PCA Analysis' tab")
    else:
        pca = st.session_state.pca_model
        returns = st.session_state.pca_returns
        
        st.markdown("#### Portfolio Strategy")
        
        strategy = st.radio(
            "Select Strategy",
            ["Minimum Variance (PC1)", "Risk Parity", "Factor Tilted"],
            help="Choose portfolio construction approach"
        )
        
        if st.button("🎯 Construct Portfolio", type="primary"):
            with st.spinner("Optimizing portfolio..."):
                loadings = pca.components_.T
                
                if strategy == "Minimum Variance (PC1)":
                    # Weight inversely to PC1 exposure (low systematic risk)
                    pc1_loadings = np.abs(loadings[:, 0])
                    weights = 1 / (pc1_loadings + 0.01)
                    weights = weights / weights.sum()
                    
                elif strategy == "Risk Parity":
                    # Equal risk contribution from each component
                    component_vols = np.std(components, axis=0)
                    inv_vols = 1 / (component_vols + 0.01)
                    
                    # Allocate based on inverse volatility
                    factor_weights = inv_vols / inv_vols.sum()
                    
                    # Map back to assets
                    weights = np.abs(loadings @ factor_weights)
                    weights = weights / weights.sum()
                    
                else:  # Factor Tilted
                    # Tilt toward PC1 (market factor)
                    pc1_loadings = loadings[:, 0]
                    weights = np.abs(pc1_loadings)
                    weights = weights / weights.sum()
                
                # Display portfolio
                portfolio_df = pd.DataFrame({
                    'Asset': returns.columns,
                    'Weight': weights,
                    'Weight (%)': weights * 100
                }).sort_values('Weight', ascending=False)
                
                st.markdown("#### Optimized Portfolio Weights")
                st.dataframe(portfolio_df.head(20), use_container_width=True, hide_index=True)
                
                # Portfolio statistics
                portfolio_returns = (returns * weights).sum(axis=1)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    ann_return = portfolio_returns.mean() * 252
                    st.metric("Annual Return", f"{ann_return:.2%}")
                
                with col2:
                    ann_vol = portfolio_returns.std() * np.sqrt(252)
                    st.metric("Annual Volatility", f"{ann_vol:.2%}")
                
                with col3:
                    sharpe = ann_return / ann_vol if ann_vol > 0 else 0
                    st.metric("Sharpe Ratio", f"{sharpe:.2f}")
                
                with col4:
                    max_dd = (portfolio_returns.cumsum().expanding().max() - portfolio_returns.cumsum()).max()
                    st.metric("Max Drawdown", f"{max_dd:.2%}")
                
                # Cumulative returns
                fig = go.Figure()
                
                cumulative = (1 + portfolio_returns).cumprod()
                fig.add_trace(go.Scatter(
                    x=portfolio_returns.index,
                    y=cumulative,
                    name='PCA Portfolio',
                    line=dict(color='#667eea', width=2)
                ))
                
                # Benchmark (equal weight)
                equal_weight_returns = returns.mean(axis=1)
                equal_cumulative = (1 + equal_weight_returns).cumprod()
                fig.add_trace(go.Scatter(
                    x=equal_weight_returns.index,
                    y=equal_cumulative,
                    name='Equal Weight Benchmark',
                    line=dict(color='#764ba2', width=2, dash='dash')
                ))
                
                fig.update_layout(
                    title='Cumulative Returns - PCA Portfolio vs Benchmark',
                    xaxis_title='Date',
                    yaxis_title='Cumulative Return',
                    hovermode='x unified',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.markdown("### Strategy Backtesting")
    
    if 'pca_model' not in st.session_state:
        st.info("👈 Run PCA analysis first in the 'PCA Analysis' tab")
    else:
        st.markdown("#### Factor Timing Strategy")
        
        col1, col2 = st.columns(2)
        
        with col1:
            trading_component = st.selectbox(
                "Component to Trade",
                [f'PC{i+1}' for i in range(st.session_state.pca_model.n_components_)]
            )
        
        with col2:
            signal_threshold = st.slider("Signal Threshold", 0.1, 0.9, 0.3, 0.1)
        
        if st.button("📊 Run Backtest", type="primary"):
            with st.spinner("Running backtest..."):
                returns = st.session_state.pca_returns
                components = st.session_state.pca_components
                pca = st.session_state.pca_model
                
                # Get component index
                comp_idx = int(trading_component.replace('PC', '')) - 1
                
                # Calculate signals
                component_returns = pd.Series(
                    np.diff(components[:, comp_idx], prepend=0),
                    index=returns.index
                )
                
                signals = component_returns.rolling(20).mean()
                signals = signals.apply(lambda x: 1 if x > signal_threshold else -1 if x < -signal_threshold else 0)
                
                # Get loadings for this component
                loadings = pca.components_[comp_idx]
                
                # Construct factor portfolio (top 5 positive, top 5 negative)
                top_assets = np.argsort(np.abs(loadings))[-5:]
                factor_portfolio = returns.iloc[:, top_assets].mean(axis=1)  # type: ignore[call-overload]
                
                # Apply signals
                strategy_returns = signals.shift(1) * factor_portfolio
                
                # Calculate metrics
                cumulative_strategy = (1 + strategy_returns).cumprod()
                cumulative_buy_hold = (1 + factor_portfolio).cumprod()  # type: ignore[attr-defined]
                
                total_return = cumulative_strategy.iloc[-1] - 1
                buy_hold_return = cumulative_buy_hold.iloc[-1] - 1
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Strategy Return", f"{total_return:.2%}")
                
                with col2:
                    st.metric("Buy & Hold Return", f"{buy_hold_return:.2%}")
                
                with col3:
                    outperformance = total_return - buy_hold_return
                    st.metric("Outperformance", f"{outperformance:.2%}")
                
                # Plot equity curves
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=cumulative_strategy.index,
                    y=cumulative_strategy,
                    name='PCA Factor Strategy',
                    line=dict(color='#667eea', width=2)
                ))
                
                fig.add_trace(go.Scatter(
                    x=cumulative_buy_hold.index,
                    y=cumulative_buy_hold,
                    name='Buy & Hold',
                    line=dict(color='#764ba2', width=2, dash='dash')
                ))
                
                fig.update_layout(
                    title=f'Backtest Results - {trading_component} Factor Timing',
                    xaxis_title='Date',
                    yaxis_title='Cumulative Return',
                    hovermode='x unified',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Trade statistics
                st.markdown("#### Trade Statistics")
                
                trades = signals[signals != 0]
                n_trades = len(trades)
                long_trades = (trades == 1).sum()
                short_trades = (trades == -1).sum()
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Trades", n_trades)
                
                with col2:
                    st.metric("Long Trades", long_trades)
                
                with col3:
                    st.metric("Short Trades", short_trades)
                
                with col4:
                    win_rate = (strategy_returns[signals != 0] > 0).sum() / n_trades if n_trades > 0 else 0
                    st.metric("Win Rate", f"{win_rate:.1%}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888;'>
    <p>💡 PCA reduces dimensionality by extracting orthogonal factors that explain maximum variance</p>
</div>
""", unsafe_allow_html=True)
