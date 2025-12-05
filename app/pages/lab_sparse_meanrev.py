"""
Sparse Mean-Reversion Portfolio Lab
Advanced sparse decomposition for mean-reverting portfolio construction
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import shared UI components
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.ui_components import render_sidebar_navigation, apply_custom_css

# Import sparse meanrev module
try:
    from python.sparse_meanrev import (
        sparse_pca, box_tao_decomposition, hurst_exponent,
        sparse_cointegration, generate_sparse_meanrev_signals
    )
    SPARSE_MEANREV_AVAILABLE = True
except ImportError:
    SPARSE_MEANREV_AVAILABLE = False

st.set_page_config(page_title="Sparse Mean-Reversion Lab", page_icon="🎯", layout="wide")

# Initialize session state
if 'historical_data' not in st.session_state:
    st.session_state.historical_data = None
if 'sparse_portfolios' not in st.session_state:
    st.session_state.sparse_portfolios = {}
if 'sparse_backtest_results' not in st.session_state:
    st.session_state.sparse_backtest_results = None

# Render sidebar and CSS
render_sidebar_navigation(current_page="Sparse Mean-Reversion Lab")
apply_custom_css()

st.markdown('<h1 class="lab-header">🎯 Sparse Mean-Reversion Portfolio Lab</h1>', unsafe_allow_html=True)
st.markdown("### Advanced sparse decomposition for identifying small, mean-reverting portfolios")
st.markdown("---")

# Check data availability
if st.session_state.historical_data is None or st.session_state.historical_data.empty:
    st.markdown("""
    <div class="info-card">
        <h3>📊 No Data Loaded</h3>
        <p>Please load historical data first to use the Sparse Mean-Reversion Lab.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚀 Go to Data Loader", type="primary", use_container_width=True):
        st.switch_page("pages/data_loader.py")
    st.stop()

if not SPARSE_MEANREV_AVAILABLE:
    st.error("⚠️ Sparse mean-reversion module not available. Please check installation.")
    st.stop()

# Get data
df = st.session_state.historical_data
prices_df = df.pivot_table(index='timestamp', columns='symbol', values='close')
returns_df = prices_df.pct_change().dropna()

# Sidebar configuration
st.sidebar.markdown("## 🎯 Portfolio Configuration")

# Method selection
method = st.sidebar.selectbox(
    "Portfolio Construction Method",
    ["Sparse PCA", "Box & Tao Decomposition", "Sparse Cointegration", "Hurst-Based Selection", "All Methods"]
)

st.sidebar.markdown("### Parameters")

# Common parameters
n_components = st.sidebar.slider("Number of Components", 1, 5, 1)
max_assets = st.sidebar.slider("Max Assets in Portfolio", 3, 20, 8)

# Method-specific parameters
if method in ["Sparse PCA", "All Methods"]:
    lambda_spca = st.sidebar.slider("Sparse PCA λ (sparsity)", 0.01, 1.0, 0.1, 0.01)

if method in ["Box & Tao Decomposition", "All Methods"]:
    lambda_bt = st.sidebar.slider("Box-Tao λ (sparsity)", 0.01, 0.5, 0.1, 0.01)

if method in ["Sparse Cointegration", "All Methods"]:
    l1_ratio = st.sidebar.slider("Elastic Net L1 Ratio", 0.1, 1.0, 0.7, 0.05)
    alpha_en = st.sidebar.slider("Elastic Net α", 0.01, 1.0, 0.1, 0.01)

# Risk parameters
st.sidebar.markdown("### Risk Management")
risk_free_rate = st.sidebar.number_input("Risk-Free Rate (%)", 0.0, 10.0, 2.0, 0.1) / 100
target_volatility = st.sidebar.number_input("Target Volatility (%)", 5.0, 50.0, 15.0, 1.0) / 100

# Main content tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Portfolio Construction",
    "📈 Backtest Results", 
    "🔄 Live Monitoring",
    "⚙️ Advanced Analysis",
    "📚 Documentation"
])

with tab1:
    st.markdown("## Portfolio Construction")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🚀 Construct Portfolios", type="primary", use_container_width=True):
            with st.spinner("Constructing sparse mean-reverting portfolios..."):
                portfolios = {}
                
                # Compute covariance matrix
                cov_matrix = returns_df.cov().values
                
                try:
                    # Method 1: Sparse PCA
                    if method in ["Sparse PCA", "All Methods"]:
                        with st.status("Running Sparse PCA...", expanded=True) as status:
                            st.write("Computing sparse principal components...")
                            result = sparse_pca(
                                cov_matrix,
                                n_components=n_components,
                                lambda_=lambda_spca,
                                max_iter=1000,
                                tol=1e-6
                            )
                            
                            for i in range(n_components):
                                weights = result.weights[i]
                                # Keep only top max_assets
                                top_idx = np.argsort(np.abs(weights))[-max_assets:]
                                sparse_weights = np.zeros_like(weights)
                                sparse_weights[top_idx] = weights[top_idx]
                                # Normalize
                                if np.abs(sparse_weights).sum() > 0:
                                    sparse_weights /= np.abs(sparse_weights).sum()
                                
                                portfolios[f"SparsePCA_λ{lambda_spca}_C{i}"] = {
                                    'weights': sparse_weights,
                                    'method': 'Sparse PCA',
                                    'variance_explained': result.variance_explained[i],
                                    'sparsity': result.sparsity[i]
                                }
                            
                            status.update(label="Sparse PCA complete!", state="complete")
                    
                    # Method 2: Box & Tao
                    if method in ["Box & Tao Decomposition", "All Methods"]:
                        with st.status("Running Box & Tao Decomposition...", expanded=True) as status:
                            st.write("Decomposing into low-rank + sparse + noise...")
                            result = box_tao_decomposition(
                                prices_df.values,
                                lambda_=lambda_bt,
                                max_iter=500,
                                tol=1e-5
                            )
                            
                            # Find assets with highest idiosyncratic component
                            sparse_component = result.sparse
                            asset_importance = np.var(sparse_component, axis=0)
                            top_assets = np.argsort(asset_importance)[-max_assets:]
                            
                            weights = np.zeros(len(returns_df.columns))
                            weights[top_assets] = 1.0 / len(top_assets)
                            
                            # Calculate rank and sparsity
                            rank = np.linalg.matrix_rank(result.low_rank)
                            sparsity = (result.sparse != 0).sum() / result.sparse.size
                            
                            portfolios[f"BoxTao_λ{lambda_bt}"] = {
                                'weights': weights,
                                'method': 'Box & Tao',
                                'rank': rank,
                                'sparsity': sparsity
                            }
                            
                            status.update(label="Box & Tao complete!", state="complete")
                    
                    # Method 3: Sparse Cointegration
                    if method in ["Sparse Cointegration", "All Methods"]:
                        with st.status("Running Sparse Cointegration...", expanded=True) as status:
                            st.write("Finding cointegrated portfolios...")
                            result = sparse_cointegration(
                                prices_df.values,
                                target_asset=0,
                                lambda_l1=float(alpha_en) * float(l1_ratio),
                                lambda_l2=float(alpha_en) * (1 - float(l1_ratio)),
                                max_iter=1000,
                                tol=1e-6
                            )
                            
                            portfolios[f"Cointegration_L1{l1_ratio}"] = {
                                'weights': result.weights,
                                'method': 'Cointegration',
                                'sparsity': result.sparsity,
                                'non_zero_count': result.non_zero_count
                            }
                            
                            status.update(label="Cointegration complete!", state="complete")
                    
                    # Method 4: Hurst-based
                    if method in ["Hurst-Based Selection", "All Methods"]:
                        with st.status("Analyzing Hurst exponents...", expanded=True) as status:
                            st.write("Computing mean-reversion properties...")
                            
                            hurst_values = []
                            for col in prices_df.columns:
                                try:
                                    result = hurst_exponent(
                                        prices_df[col].values,
                                        min_window=8,
                                        max_window=64
                                    )
                                    hurst_values.append((col, result.hurst_exponent, result.is_mean_reverting))
                                except:
                                    pass
                            
                            # Select most mean-reverting
                            hurst_values.sort(key=lambda x: x[1])
                            selected = [h[0] for h in hurst_values[:max_assets] if h[2]]
                            
                            if selected:
                                weights = np.zeros(len(returns_df.columns))
                                for symbol in selected:
                                    idx = returns_df.columns.get_loc(symbol)
                                    weights[idx] = 1.0 / len(selected)
                                
                                portfolios[f"Hurst_Top{len(selected)}"] = {
                                    'weights': weights,
                                    'method': 'Hurst',
                                    'assets': selected,
                                    'mean_hurst': np.mean([h[1] for h in hurst_values[:len(selected)]])
                                }
                            
                            status.update(label="Hurst analysis complete!", state="complete")
                    
                    # Store portfolios
                    st.session_state.sparse_portfolios = portfolios
                    st.success(f"✅ Constructed {len(portfolios)} portfolios successfully!")
                    
                except Exception as e:
                    st.error(f"Error during portfolio construction: {str(e)}")
                    st.exception(e)
    
    with col2:
        st.markdown("### Quick Stats")
        st.metric("Assets Available", len(returns_df.columns))
        st.metric("Time Periods", len(returns_df))
        st.metric("Avg Daily Return", f"{returns_df.mean().mean()*100:.3f}%")
        st.metric("Avg Volatility", f"{returns_df.std().mean()*np.sqrt(252)*100:.1f}%")
    
    # Display constructed portfolios
    if st.session_state.sparse_portfolios:
        st.markdown("### Constructed Portfolios")
        
        for name, pf in st.session_state.sparse_portfolios.items():
            with st.expander(f"📊 {name}", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                weights = pf['weights']
                active_idx = np.where(np.abs(weights) > 1e-6)[0]
                
                with col1:
                    st.markdown("**Portfolio Composition**")
                    for idx in active_idx:
                        symbol = returns_df.columns[idx]
                        w = weights[idx]
                        st.write(f"{symbol}: {w*100:.2f}%")
                
                with col2:
                    st.markdown("**Method Details**")
                    st.write(f"Method: {pf['method']}")
                    st.write(f"Assets: {len(active_idx)}")
                    
                    if 'variance_explained' in pf:
                        st.write(f"Var Explained: {pf['variance_explained']:.2%}")
                    if 'adf_pvalue' in pf:
                        st.write(f"ADF p-value: {pf['adf_pvalue']:.4f}")
                    if 'half_life' in pf:
                        st.write(f"Half-life: {pf['half_life']:.1f} days")
                
                with col3:
                    st.markdown("**Performance Metrics**")
                    # Compute quick metrics
                    pf_returns = returns_df.values @ weights
                    pf_sharpe = np.mean(pf_returns) / np.std(pf_returns) * np.sqrt(252)
                    pf_vol = np.std(pf_returns) * np.sqrt(252)
                    
                    st.metric("Sharpe Ratio", f"{pf_sharpe:.2f}")
                    st.metric("Volatility", f"{pf_vol*100:.1f}%")

with tab2:
    st.markdown("## Backtest Results")
    
    if not st.session_state.sparse_portfolios:
        st.info("👆 Please construct portfolios first")
    else:
        # Portfolio selection
        selected_pf = st.selectbox(
            "Select Portfolio to Backtest",
            list(st.session_state.sparse_portfolios.keys())
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            entry_z = st.number_input("Entry Z-Score", 1.0, 4.0, 2.0, 0.1)
        with col2:
            exit_z = st.number_input("Exit Z-Score", 0.1, 2.0, 0.5, 0.1)
        with col3:
            transaction_cost = st.number_input("Transaction Cost (bps)", 0.0, 50.0, 10.0, 1.0) / 10000
        
        if st.button("🔄 Run Backtest", type="primary"):
            with st.spinner("Running backtest..."):
                weights = st.session_state.sparse_portfolios[selected_pf]['weights']
                
                # Compute portfolio value
                pf_value = prices_df.values @ weights
                pf_returns = np.diff(pf_value) / pf_value[:-1]
                
                # Z-score
                pf_mean = np.mean(pf_value)
                pf_std = np.std(pf_value)
                z_score = (pf_value - pf_mean) / pf_std
                
                # Simple mean-reversion strategy
                position = 0
                pnl = []
                trades = []
                
                for t in range(len(pf_returns)):
                    z = z_score[t+1]  # Align with returns
                    
                    # Entry
                    if position == 0:
                        if z < -entry_z:
                            position = 1
                            pnl.append(-transaction_cost)
                            trades.append((t, 'BUY', z))
                        elif z > entry_z:
                            position = -1
                            pnl.append(-transaction_cost)
                            trades.append((t, 'SELL', z))
                        else:
                            pnl.append(0)
                    # Exit
                    elif abs(z) < exit_z:
                        pnl.append(position * pf_returns[t] - transaction_cost)
                        trades.append((t, 'CLOSE', z))
                        position = 0
                    else:
                        pnl.append(position * pf_returns[t])
                
                cum_pnl = np.cumsum(pnl)
                
                # Store results
                st.session_state.sparse_backtest_results = {
                    'pnl': pnl,
                    'cum_pnl': cum_pnl,
                    'trades': trades,
                    'z_score': z_score,
                    'pf_value': pf_value
                }
                
                # Metrics
                total_return = cum_pnl[-1]
                sharpe = np.mean(pnl) / np.std(pnl) * np.sqrt(252) if np.std(pnl) > 0 else 0
                max_dd = np.min(cum_pnl - np.maximum.accumulate(cum_pnl))
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Return", f"{total_return*100:.2f}%")
                col2.metric("Sharpe Ratio", f"{sharpe:.2f}")
                col3.metric("Max Drawdown", f"{max_dd*100:.2f}%")
                col4.metric("Number of Trades", len([t for t in trades if t[1] in ['BUY', 'SELL']]))
        
        # Display backtest results
        if st.session_state.sparse_backtest_results:
            result = st.session_state.sparse_backtest_results
            
            # Plot performance
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Cumulative P&L', 'Z-Score with Trades', 
                               'P&L Distribution', 'Portfolio Value'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"type": "histogram"}, {"secondary_y": False}]]
            )
            
            # Cumulative P&L
            fig.add_trace(
                go.Scatter(y=result['cum_pnl']*100, mode='lines', name='Cum P&L',
                          line=dict(color='green', width=2)),
                row=1, col=1
            )
            
            # Z-score with trades
            fig.add_trace(
                go.Scatter(y=result['z_score'], mode='lines', name='Z-Score',
                          line=dict(color='blue')),
                row=1, col=2
            )
            
            # Add trading bands (note: add_hline doesn't support row/col for non-subplot figures)
            # These lines would need to be added to a subplot figure if you want them on specific subplots
            
            # Trade markers
            buy_trades = [t for t in result['trades'] if t[1] == 'BUY']
            sell_trades = [t for t in result['trades'] if t[1] == 'SELL']
            
            if buy_trades:
                fig.add_trace(
                    go.Scatter(x=[t[0] for t in buy_trades],
                              y=[result['z_score'][t[0]] for t in buy_trades],
                              mode='markers', marker=dict(symbol='triangle-up', size=10, color='green'),
                              name='Buy', showlegend=True),
                    row=1, col=2
                )
            
            if sell_trades:
                fig.add_trace(
                    go.Scatter(x=[t[0] for t in sell_trades],
                              y=[result['z_score'][t[0]] for t in sell_trades],
                              mode='markers', marker=dict(symbol='triangle-down', size=10, color='red'),
                              name='Sell', showlegend=True),
                    row=1, col=2
                )
            
            # P&L distribution
            fig.add_trace(
                go.Histogram(x=np.array(result['pnl'])*100, nbinsx=50, name='P&L', showlegend=False),
                row=2, col=1
            )
            
            # Portfolio value
            fig.add_trace(
                go.Scatter(y=result['pf_value'], mode='lines', name='Portfolio Value',
                          line=dict(color='purple')),
                row=2, col=2
            )
            
            fig.update_yaxes(title_text="P&L (%)", row=1, col=1)
            fig.update_yaxes(title_text="Z-Score", row=1, col=2)
            fig.update_yaxes(title_text="Frequency", row=2, col=1)
            fig.update_yaxes(title_text="Value", row=2, col=2)
            
            fig.update_layout(height=800, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("## Live Monitoring")
    st.info("🔄 Live monitoring will be available when connected to live data feed")
    
    # Placeholder for live monitoring features

with tab4:
    st.markdown("## Advanced Analysis")
    
    if not st.session_state.sparse_portfolios:
        st.info("👆 Please construct portfolios first")
    else:
        st.markdown("### Portfolio Comparison")
        
        # Compare all portfolios
        comparison_data = []
        
        for name, pf in st.session_state.sparse_portfolios.items():
            weights = pf['weights']
            pf_returns = returns_df.values @ weights
            
            # Compute comprehensive metrics
            metrics = {
                'Portfolio': name,
                'Method': pf['method'],
                'Assets': np.sum(np.abs(weights) > 1e-6),
                'Sharpe': np.mean(pf_returns) / np.std(pf_returns) * np.sqrt(252) if np.std(pf_returns) > 0 else 0,
                'Volatility': np.std(pf_returns) * np.sqrt(252) * 100,
                'Return': np.mean(pf_returns) * 252 * 100
            }
            
            # Hurst exponent
            pf_value = (1 + pf_returns).cumprod()
            try:
                hurst_result = hurst_exponent(pf_value, min_window=8, max_window=64)
                metrics['Hurst'] = hurst_result.hurst_exponent
                metrics['Mean-Reverting'] = hurst_result.is_mean_reverting
            except:
                metrics['Hurst'] = np.nan
                metrics['Mean-Reverting'] = False
            
            comparison_data.append(metrics)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Display table
        st.dataframe(
            comparison_df.style.background_gradient(subset=['Sharpe'], cmap='RdYlGn')
                              .format({'Sharpe': '{:.2f}', 'Volatility': '{:.1f}%',
                                      'Return': '{:.2f}%', 'Hurst': '{:.3f}'}),
            use_container_width=True
        )
        
        # Scatter plot: Risk vs Return
        fig = go.Figure()
        
        for _, row in comparison_df.iterrows():
            fig.add_trace(go.Scatter(
                x=[row['Volatility']],
                y=[row['Return']],
                mode='markers+text',
                marker=dict(size=15, color='green' if row['Mean-Reverting'] else 'red'),
                text=[row['Portfolio']],
                textposition='top center',
                name=row['Portfolio']
            ))
        
        fig.update_layout(
            title='Risk-Return Profile',
            xaxis_title='Volatility (%)',
            yaxis_title='Annual Return (%)',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.markdown("## Documentation")
    
    st.markdown("""
    ### Sparse Mean-Reversion Methods
    
    This lab implements advanced sparse decomposition algorithms for identifying small, 
    mean-reverting portfolios in high-dimensional asset universes.
    
    #### Methods
    
    1. **Sparse PCA**
       - Finds sparse principal components with L1 regularization
       - Maximizes variance while enforcing sparsity
       - Formula: max w^T Σ w - λ ||w||₁ s.t. ||w||₂ = 1
    
    2. **Box & Tao Decomposition (Robust PCA)**
       - Decomposes price matrix: X = L + S + N
       - L: low-rank (common factors)
       - S: sparse (idiosyncratic mean-reversion)
       - N: noise
    
    3. **Sparse Cointegration**
       - Elastic Net regression for cointegrated portfolios
       - Combines L1 (sparsity) and L2 (smoothness) regularization
       - Tests stationarity via Augmented Dickey-Fuller test
    
    4. **Hurst-Based Selection**
       - Selects assets with Hurst exponent < 0.5
       - R/S analysis for mean-reversion detection
    
    #### Key Parameters
    
    - **λ (lambda)**: Sparsity parameter - higher values → sparser portfolios
    - **Entry/Exit Z-scores**: Trading signals based on standardized deviations
    - **Transaction costs**: Impact on realized returns
    - **Max assets**: Portfolio size constraint
    
    #### References
    
    - d'Aspremont (2011): "Identifying Small Mean Reverting Portfolios"
    - Candès et al. (2011): "Robust Principal Component Analysis?"
    - Zou & Hastie (2005): "Regularization and Variable Selection via Elastic Net"
    """)
