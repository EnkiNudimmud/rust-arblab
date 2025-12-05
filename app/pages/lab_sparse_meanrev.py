"""
Sparse Mean-Reversion Portfolio Lab
Advanced sparse decomposition for mean-reverting portfolio construction with multi-period analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from scipy import stats
from scipy.optimize import minimize

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

# Import optimizr for advanced metrics
try:
    import optimizr
    OPTIMIZR_AVAILABLE = True
except ImportError:
    OPTIMIZR_AVAILABLE = False

st.set_page_config(page_title="Sparse Mean-Reversion Lab", page_icon="🎯", layout="wide")

# Initialize session state
if 'historical_data' not in st.session_state:
    st.session_state.historical_data = None
if 'sparse_portfolios' not in st.session_state:
    st.session_state.sparse_portfolios = {}
if 'sparse_backtest_results' not in st.session_state:
    st.session_state.sparse_backtest_results = None
if 'hurst_analysis' not in st.session_state:
    st.session_state.hurst_analysis = None
if 'rolling_analysis' not in st.session_state:
    st.session_state.rolling_analysis = None

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

# Helper functions
def safe_sharpe_ratio(returns, risk_free_rate=0.02):
    """Calculate Sharpe ratio with NaN handling"""
    if len(returns) == 0:
        return np.nan
    mean_return = np.nanmean(returns)
    std_return = np.nanstd(returns)
    if std_return == 0 or np.isnan(std_return):
        return np.nan
    return (mean_return - risk_free_rate/252) / std_return * np.sqrt(252)

def calculate_half_life(series):
    """Calculate half-life of mean reversion"""
    series = series[~np.isnan(series)]
    if len(series) < 2:
        return np.nan
    lag = series[:-1]
    diff = series[1:] - series[:-1]
    lag = lag - np.mean(lag)
    try:
        beta = np.polyfit(lag, diff, 1)[0]
        if beta >= 0:
            return np.nan
        half_life = -np.log(2) / beta
        return half_life if half_life > 0 else np.nan
    except:
        return np.nan

def analyze_rolling_hurst(prices, window=60, step=10):
    """Compute rolling Hurst exponent"""
    results = []
    for i in range(window, len(prices), step):
        window_data = prices[i-window:i]
        try:
            h_result = hurst_exponent(window_data, min_window=8, max_window=min(32, window//2))
            results.append({
                'index': i,
                'hurst': h_result.hurst_exponent,
                'is_mean_reverting': h_result.is_mean_reverting
            })
        except:
            pass
    return pd.DataFrame(results)

def optimize_portfolio_weights(returns, method='sharpe', constraints=None):
    """Optimize portfolio weights using various objectives"""
    n_assets = returns.shape[1]
    
    def portfolio_stats(weights):
        pf_return = np.nansum(returns @ weights) / len(returns) * 252
        pf_vol = np.sqrt(np.dot(weights.T, np.dot(np.cov(returns.T, ddof=0), weights))) * np.sqrt(252)
        return pf_return, pf_vol
    
    def sharpe_objective(weights):
        pf_return, pf_vol = portfolio_stats(weights)
        if pf_vol == 0:
            return -np.inf
        return -(pf_return / pf_vol)
    
    def min_variance_objective(weights):
        _, pf_vol = portfolio_stats(weights)
        return pf_vol
    
    # Constraints
    cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # weights sum to 1
    bounds = tuple((-0.5, 0.5) for _ in range(n_assets))  # Allow short selling
    
    if constraints:
        cons.extend(constraints)
    
    # Initial guess - equal weight
    x0 = np.array([1/n_assets] * n_assets)
    
    objective = sharpe_objective if method == 'sharpe' else min_variance_objective
    
    try:
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons)
        if result.success:
            return result.x
    except:
        pass
    
    return x0

def compute_stability_metrics(portfolio_returns, benchmark_returns=None):
    """Compute comprehensive stability and risk metrics"""
    metrics = {}
    
    # Clean data
    portfolio_returns = portfolio_returns[~np.isnan(portfolio_returns)]
    
    if len(portfolio_returns) < 2:
        return {k: np.nan for k in ['max_dd', 'calmar', 'sortino', 'tail_ratio', 'beta', 'alpha']}
    
    # Maximum drawdown
    cum_returns = (1 + portfolio_returns).cumprod()
    running_max = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns - running_max) / running_max
    metrics['max_dd'] = np.min(drawdown)
    
    # Calmar ratio
    ann_return = np.mean(portfolio_returns) * 252
    metrics['calmar'] = ann_return / abs(metrics['max_dd']) if metrics['max_dd'] != 0 else np.nan
    
    # Sortino ratio
    downside_returns = portfolio_returns[portfolio_returns < 0]
    downside_std = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else np.nan
    metrics['sortino'] = ann_return / downside_std if downside_std and downside_std != 0 else np.nan
    
    # Tail ratio
    returns_5pct = np.percentile(portfolio_returns, 95)
    returns_95pct = np.percentile(portfolio_returns, 5)
    metrics['tail_ratio'] = abs(returns_5pct / returns_95pct) if returns_95pct != 0 else np.nan
    
    # Beta and Alpha
    if benchmark_returns is not None and len(benchmark_returns) == len(portfolio_returns):
        benchmark_returns = benchmark_returns[~np.isnan(benchmark_returns)]
        try:
            covariance = np.cov(portfolio_returns, benchmark_returns)[0][1]
            benchmark_var = np.var(benchmark_returns)
            metrics['beta'] = covariance / benchmark_var if benchmark_var != 0 else np.nan
            metrics['alpha'] = (ann_return - metrics['beta'] * np.mean(benchmark_returns) * 252) if not np.isnan(metrics['beta']) else np.nan
        except:
            metrics['beta'] = np.nan
            metrics['alpha'] = np.nan
    else:
        metrics['beta'] = np.nan
        metrics['alpha'] = np.nan
    
    return metrics

# Sidebar configuration
st.sidebar.markdown("## 🎯 Portfolio Configuration")

# Method selection
method = st.sidebar.selectbox(
    "Portfolio Construction Method",
    ["Sparse PCA", "Box & Tao Decomposition", "Sparse Cointegration", "Hurst-Based Selection", "All Methods"]
)

st.sidebar.markdown("### Analysis Parameters")

# Multi-period analysis
enable_multiperiod = st.sidebar.checkbox("Enable Multi-Period Analysis", value=True)

if enable_multiperiod:
    lookback_days = st.sidebar.slider("Lookback Window (days)", 30, 365, 90)
    rolling_window = st.sidebar.slider("Rolling Window (days)", 20, 120, 60)

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

# Advanced optimization
st.sidebar.markdown("### Advanced Optimization")
use_advanced_opt = st.sidebar.checkbox("Use Advanced Weight Optimization", value=True)
opt_method = st.sidebar.selectbox("Optimization Objective", ["Sharpe Ratio", "Min Variance", "Max Return"]) if use_advanced_opt else None

# Main content tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Portfolio Construction",
    "📈 Multi-Period Analysis",
    "🎯 Hurst Exponent Analysis",
    "📉 Backtest Results", 
    "⚙️ Advanced Metrics",
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
                    # Compute quick metrics with proper NaN handling
                    pf_returns = returns_df.values @ weights
                    pf_returns = pf_returns[~np.isnan(pf_returns)]
                    
                    if len(pf_returns) > 0:
                        pf_sharpe = safe_sharpe_ratio(pf_returns, risk_free_rate)
                        pf_vol = np.nanstd(pf_returns) * np.sqrt(252)
                        pf_return = np.nanmean(pf_returns) * 252
                        
                        st.metric("Sharpe Ratio", f"{pf_sharpe:.2f}" if not np.isnan(pf_sharpe) else "N/A")
                        st.metric("Volatility", f"{pf_vol*100:.1f}%" if not np.isnan(pf_vol) else "N/A")
                        st.metric("Annual Return", f"{pf_return*100:.1f}%" if not np.isnan(pf_return) else "N/A")
                    else:
                        st.warning("Insufficient data for metrics")

with tab2:
    st.markdown("## Multi-Period Analysis")
    st.markdown("*Analyze portfolio stability and mean-reversion properties across different time periods*")
    
    if not st.session_state.sparse_portfolios:
        st.info("👆 Please construct portfolios first")
    else:
        selected_pf = st.selectbox(
            "Select Portfolio for Multi-Period Analysis",
            list(st.session_state.sparse_portfolios.keys()),
            key="multiperiod_select"
        )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            n_periods = st.number_input("Number of Periods", 3, 10, 5)
        with col2:
            period_length = st.selectbox("Period Length", ["1 Month", "2 Months", "3 Months", "6 Months"])
        with col3:
            overlap = st.slider("Period Overlap (%)", 0, 50, 20)
        
        if st.button("🔄 Run Multi-Period Analysis", type="primary"):
            with st.spinner("Analyzing across multiple periods..."):
                weights = st.session_state.sparse_portfolios[selected_pf]['weights']
                
                # Convert period length to days
                period_days = {"1 Month": 21, "2 Months": 42, "3 Months": 63, "6 Months": 126}[period_length]
                step_days = int(period_days * (1 - overlap/100))
                
                period_results = []
                
                for period_idx in range(n_periods):
                    start_idx = period_idx * step_days
                    end_idx = start_idx + period_days
                    
                    if end_idx > len(returns_df):
                        break
                    
                    period_returns = returns_df.iloc[start_idx:end_idx]
                    period_prices = prices_df.iloc[start_idx:end_idx]
                    
                    # Portfolio returns for this period
                    pf_returns = (period_returns.values @ weights)
                    pf_returns = pf_returns[~np.isnan(pf_returns)]
                    
                    if len(pf_returns) < 10:
                        continue
                    
                    # Compute metrics
                    pf_value = (1 + pf_returns).cumprod()
                    
                    metrics = {
                        'Period': f"P{period_idx+1}",
                        'Start': period_returns.index[0].strftime('%Y-%m-%d'),
                        'End': period_returns.index[-1].strftime('%Y-%m-%d'),
                        'Sharpe': safe_sharpe_ratio(pf_returns, risk_free_rate),
                        'Volatility': np.nanstd(pf_returns) * np.sqrt(252) * 100,
                        'Return': np.nanmean(pf_returns) * 252 * 100,
                    }
                    
                    # Hurst exponent
                    try:
                        h_result = hurst_exponent(pf_value, min_window=8, max_window=min(32, len(pf_value)//3))
                        metrics['Hurst'] = h_result.hurst_exponent
                        metrics['Mean-Reverting'] = h_result.is_mean_reverting
                    except:
                        metrics['Hurst'] = np.nan
                        metrics['Mean-Reverting'] = False
                    
                    # Half-life
                    metrics['Half-Life'] = calculate_half_life(pf_value)
                    
                    # Stability metrics
                    stability = compute_stability_metrics(pf_returns)
                    metrics.update(stability)
                    
                    period_results.append(metrics)
                
                if period_results:
                    results_df = pd.DataFrame(period_results)
                    st.session_state.rolling_analysis = results_df
                    
                    # Display summary statistics
                    st.markdown("### Period-by-Period Performance")
                    
                    # Format the dataframe
                    display_df = results_df.copy()
                    for col in ['Sharpe', 'Hurst', 'Half-Life', 'calmar', 'sortino']:
                        if col in display_df.columns:
                            display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}" if not np.isnan(x) else "N/A")
                    
                    for col in ['Volatility', 'Return', 'max_dd']:
                        if col in display_df.columns:
                            display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}%" if not np.isnan(x) else "N/A")
                    
                    st.dataframe(display_df, use_container_width=True)
                    
                    # Visualizations
                    st.markdown("### Performance Stability Across Periods")
                    
                    fig = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=('Sharpe Ratio Evolution', 'Hurst Exponent Over Time',
                                       'Return vs Volatility', 'Half-Life Stability'),
                        specs=[[{"secondary_y": False}, {"secondary_y": False}],
                               [{"secondary_y": False}, {"secondary_y": False}]]
                    )
                    
                    # Sharpe ratio evolution
                    sharpe_values = results_df['Sharpe'].values
                    sharpe_clean = sharpe_values[~np.isnan(sharpe_values)]
                    if len(sharpe_clean) > 0:
                        fig.add_trace(
                            go.Scatter(x=results_df['Period'], y=results_df['Sharpe'],
                                      mode='lines+markers', name='Sharpe',
                                      line=dict(color='blue', width=2),
                                      marker=dict(size=8)),
                            row=1, col=1
                        )
                        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
                    
                    # Hurst exponent
                    hurst_values = results_df['Hurst'].values
                    hurst_clean = hurst_values[~np.isnan(hurst_values)]
                    if len(hurst_clean) > 0:
                        colors = ['green' if mr else 'red' for mr in results_df['Mean-Reverting']]
                        fig.add_trace(
                            go.Scatter(x=results_df['Period'], y=results_df['Hurst'],
                                      mode='lines+markers', name='Hurst',
                                      line=dict(color='purple', width=2),
                                      marker=dict(size=10, color=colors)),
                            row=1, col=2
                        )
                        fig.add_hline(y=0.5, line_dash="dash", line_color="orange", 
                                     annotation_text="H=0.5 (Random Walk)", row=1, col=2)
                    
                    # Risk-return scatter
                    return_values = results_df['Return'].values
                    vol_values = results_df['Volatility'].values
                    valid_idx = ~(np.isnan(return_values) | np.isnan(vol_values))
                    if valid_idx.any():
                        fig.add_trace(
                            go.Scatter(x=results_df.loc[valid_idx, 'Volatility'],
                                      y=results_df.loc[valid_idx, 'Return'],
                                      mode='markers+text',
                                      text=results_df.loc[valid_idx, 'Period'],
                                      textposition='top center',
                                      marker=dict(size=12, color=results_df.loc[valid_idx, 'Sharpe'],
                                                colorscale='RdYlGn', showscale=True,
                                                colorbar=dict(title="Sharpe")),
                                      name='Periods'),
                            row=2, col=1
                        )
                    
                    # Half-life stability
                    hl_values = results_df['Half-Life'].values
                    hl_clean = hl_values[~np.isnan(hl_values)]
                    if len(hl_clean) > 0:
                        fig.add_trace(
                            go.Bar(x=results_df['Period'], y=results_df['Half-Life'],
                                  marker_color='lightblue', name='Half-Life'),
                            row=2, col=2
                        )
                    
                    fig.update_xaxes(title_text="Period", row=1, col=1)
                    fig.update_xaxes(title_text="Period", row=1, col=2)
                    fig.update_xaxes(title_text="Volatility (%)", row=2, col=1)
                    fig.update_xaxes(title_text="Period", row=2, col=2)
                    
                    fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=1)
                    fig.update_yaxes(title_text="Hurst Exponent", row=1, col=2)
                    fig.update_yaxes(title_text="Return (%)", row=2, col=1)
                    fig.update_yaxes(title_text="Days", row=2, col=2)
                    
                    fig.update_layout(height=800, showlegend=False, title_text="Multi-Period Analysis Dashboard")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Summary statistics
                    st.markdown("### Stability Summary")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        avg_sharpe = np.nanmean(results_df['Sharpe'])
                        std_sharpe = np.nanstd(results_df['Sharpe'])
                        st.metric("Avg Sharpe", f"{avg_sharpe:.2f}", 
                                 delta=f"±{std_sharpe:.2f}" if not np.isnan(std_sharpe) else "N/A")
                    
                    with col2:
                        avg_hurst = np.nanmean(results_df['Hurst'])
                        mr_pct = results_df['Mean-Reverting'].sum() / len(results_df) * 100
                        st.metric("Avg Hurst", f"{avg_hurst:.3f}",
                                 delta=f"{mr_pct:.0f}% mean-rev")
                    
                    with col3:
                        avg_hl = np.nanmean(results_df['Half-Life'])
                        st.metric("Avg Half-Life", f"{avg_hl:.1f} days" if not np.isnan(avg_hl) else "N/A")
                    
                    with col4:
                        consistency = (results_df['Sharpe'] > 0).sum() / len(results_df) * 100
                        st.metric("Positive Periods", f"{consistency:.0f}%")
                
                else:
                    st.warning("Insufficient data for multi-period analysis")
        
        # Display stored results
        if st.session_state.rolling_analysis is not None:
            with st.expander("📊 View Stored Multi-Period Results", expanded=False):
                st.dataframe(st.session_state.rolling_analysis, use_container_width=True)

with tab3:
    st.markdown("## Hurst Exponent Analysis")
    st.markdown("*Deep dive into mean-reversion properties using Hurst exponent*")
    
    st.markdown("""
    ### Understanding Hurst Exponent
    
    The Hurst exponent (H) characterizes the long-term memory of time series:
    - **H < 0.5**: Mean-reverting (anti-persistent) - ideal for mean-reversion strategies
    - **H = 0.5**: Random walk (no memory) - unpredictable
    - **H > 0.5**: Trending (persistent) - momentum strategies work better
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🔍 Analyze All Assets", type="primary", use_container_width=True):
            with st.spinner("Computing Hurst exponents for all assets..."):
                hurst_results = []
                
                for symbol in prices_df.columns:
                    prices = prices_df[symbol].dropna().values
                    
                    if len(prices) < 50:
                        continue
                    
                    try:
                        h_result = hurst_exponent(prices, min_window=10, max_window=min(64, len(prices)//3))
                        
                        # Also compute rolling Hurst
                        rolling = analyze_rolling_hurst(prices, window=60, step=10)
                        
                        hurst_results.append({
                            'Symbol': symbol,
                            'Hurst': h_result.hurst_exponent,
                            'Std Error': h_result.standard_error,
                            'Mean-Reverting': h_result.is_mean_reverting,
                            'Trending': h_result.is_trending,
                            'Confidence': h_result.confidence_level,
                            'Rolling Mean': np.mean(rolling['hurst']) if len(rolling) > 0 else np.nan,
                            'Rolling Std': np.std(rolling['hurst']) if len(rolling) > 0 else np.nan,
                            'Half-Life': calculate_half_life(prices)
                        })
                    except Exception as e:
                        st.warning(f"Could not compute Hurst for {symbol}: {str(e)}")
                
                if hurst_results:
                    hurst_df = pd.DataFrame(hurst_results)
                    hurst_df = hurst_df.sort_values('Hurst')
                    st.session_state.hurst_analysis = hurst_df
                    
                    st.success(f"✅ Analyzed {len(hurst_df)} assets")
    
    with col2:
        st.markdown("### Filter Settings")
        show_all = st.checkbox("Show All Assets", value=False)
        if not show_all:
            hurst_filter = st.selectbox("Filter by", 
                                       ["Mean-Reverting Only (H<0.5)", 
                                        "Trending Only (H>0.5)",
                                        "All"])
    
    # Display results
    if st.session_state.hurst_analysis is not None:
        hurst_df = st.session_state.hurst_analysis.copy()
        
        # Apply filters
        if not show_all and hurst_filter == "Mean-Reverting Only (H<0.5)":
            hurst_df = hurst_df[hurst_df['Mean-Reverting']]
        elif not show_all and hurst_filter == "Trending Only (H>0.5)":
            hurst_df = hurst_df[hurst_df['Trending']]
        
        st.markdown("### Hurst Exponent Results")
        
        # Color-coded table
        def color_hurst(val):
            if pd.isna(val):
                return ''
            if val < 0.5:
                return 'background-color: #90EE90'  # Light green
            elif val > 0.5:
                return 'background-color: #FFB6C1'  # Light red
            else:
                return 'background-color: #FFFFE0'  # Light yellow
        
        styled_df = hurst_df.style.applymap(color_hurst, subset=['Hurst'])
        styled_df = styled_df.format({
            'Hurst': '{:.4f}',
            'Std Error': '{:.4f}',
            'Confidence': '{:.2%}',
            'Rolling Mean': '{:.4f}',
            'Rolling Std': '{:.4f}',
            'Half-Life': '{:.1f}'
        })
        
        st.dataframe(styled_df, use_container_width=True)
        
        # Visualizations
        st.markdown("### Visual Analysis")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Hurst Distribution', 'Hurst vs Half-Life',
                           'Confidence Levels', 'Rolling Stability'),
            specs=[[{"type": "histogram"}, {"secondary_y": False}],
                   [{"type": "bar"}, {"secondary_y": False}]]
        )
        
        # Histogram of Hurst values
        mean_rev = hurst_df[hurst_df['Hurst'] < 0.5]['Hurst']
        trending = hurst_df[hurst_df['Hurst'] > 0.5]['Hurst']
        
        if len(mean_rev) > 0:
            fig.add_trace(
                go.Histogram(x=mean_rev, name='Mean-Reverting', marker_color='green', opacity=0.7, nbinsx=20),
                row=1, col=1
            )
        if len(trending) > 0:
            fig.add_trace(
                go.Histogram(x=trending, name='Trending', marker_color='red', opacity=0.7, nbinsx=20),
                row=1, col=1
            )
        
        # Hurst vs Half-Life scatter
        valid_hl = hurst_df[~hurst_df['Half-Life'].isna()]
        if len(valid_hl) > 0:
            colors = ['green' if mr else 'red' for mr in valid_hl['Mean-Reverting']]
            fig.add_trace(
                go.Scatter(x=valid_hl['Hurst'], y=valid_hl['Half-Life'],
                          mode='markers+text', text=valid_hl['Symbol'],
                          textposition='top center', marker=dict(size=10, color=colors),
                          name='Assets', showlegend=False),
                row=1, col=2
            )
        
        # Confidence levels
        top_10 = hurst_df.nsmallest(10, 'Hurst')  # Most mean-reverting
        fig.add_trace(
            go.Bar(x=top_10['Symbol'], y=top_10['Confidence']*100,
                  marker_color='lightblue', name='Confidence', showlegend=False),
            row=2, col=1
        )
        
        # Rolling stability (std of rolling Hurst)
        stable_assets = hurst_df[~hurst_df['Rolling Std'].isna()].nsmallest(10, 'Rolling Std')
        if len(stable_assets) > 0:
            fig.add_trace(
                go.Bar(x=stable_assets['Symbol'], y=stable_assets['Rolling Std'],
                      marker_color='orange', name='Stability', showlegend=False),
                row=2, col=2
            )
        
        fig.update_xaxes(title_text="Hurst Exponent", row=1, col=1)
        fig.update_xaxes(title_text="Hurst Exponent", row=1, col=2)
        fig.update_xaxes(title_text="Symbol", row=2, col=1, tickangle=45)
        fig.update_xaxes(title_text="Symbol", row=2, col=2, tickangle=45)
        
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_yaxes(title_text="Half-Life (days)", row=1, col=2)
        fig.update_yaxes(title_text="Confidence (%)", row=2, col=1)
        fig.update_yaxes(title_text="Rolling Std", row=2, col=2)
        
        fig.update_layout(height=800, showlegend=True, title_text="Comprehensive Hurst Analysis")
        st.plotly_chart(fig, use_container_width=True)
        
        # Key insights
        st.markdown("### Key Insights")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            mr_count = hurst_df['Mean-Reverting'].sum()
            mr_pct = mr_count / len(hurst_df) * 100
            st.metric("Mean-Reverting Assets", f"{mr_count}", delta=f"{mr_pct:.1f}% of total")
        
        with col2:
            avg_hurst = hurst_df['Hurst'].mean()
            st.metric("Average Hurst", f"{avg_hurst:.4f}",
                     delta="Mean-Rev" if avg_hurst < 0.5 else "Trending")
        
        with col3:
            best_mr = hurst_df.loc[hurst_df['Hurst'].idxmin(), 'Symbol']
            best_mr_h = hurst_df['Hurst'].min()
            st.metric("Best Mean-Reverter", best_mr, delta=f"H={best_mr_h:.3f}")
        
        with col4:
            avg_confidence = hurst_df['Confidence'].mean()
            st.metric("Avg Confidence", f"{avg_confidence:.1%}")
        
        # Export option
        st.markdown("### Export Results")
        csv = hurst_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Hurst Analysis CSV",
            data=csv,
            file_name=f"hurst_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

with tab4:
    st.markdown("## Backtest Results")
    st.markdown("*Test mean-reversion trading strategy with advanced execution logic*")
    
    if not st.session_state.sparse_portfolios:
        st.info("👆 Please construct portfolios first")
    else:
        # Portfolio selection
        selected_pf = st.selectbox(
            "Select Portfolio to Backtest",
            list(st.session_state.sparse_portfolios.keys()),
            key="backtest_select"
        )
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            entry_z = st.number_input("Entry Z-Score", 1.0, 4.0, 2.0, 0.1)
        with col2:
            exit_z = st.number_input("Exit Z-Score", 0.1, 2.0, 0.5, 0.1)
        with col3:
            transaction_cost = st.number_input("Transaction Cost (bps)", 0.0, 50.0, 10.0, 1.0) / 10000
        with col4:
            max_holding = st.number_input("Max Holding Period (days)", 5, 60, 20)
        
        if st.button("🔄 Run Advanced Backtest", type="primary"):
            with st.spinner("Running advanced backtest..."):
                weights = st.session_state.sparse_portfolios[selected_pf]['weights']
                
                # Compute portfolio value
                pf_value = prices_df.values @ weights
                pf_returns = np.diff(pf_value) / pf_value[:-1]
                
                # Clean data
                valid_mask = ~np.isnan(pf_returns)
                pf_returns_clean = pf_returns[valid_mask]
                
                # Rolling z-score (more robust)
                lookback = min(60, len(pf_value) // 3)
                z_score = np.zeros(len(pf_value))
                for i in range(lookback, len(pf_value)):
                    window = pf_value[i-lookback:i]
                    z_score[i] = (pf_value[i] - np.mean(window)) / np.std(window)
                
                # Advanced mean-reversion strategy
                position = 0
                pnl = []
                trades = []
                position_duration = 0
                entry_price = 0
                
                for t in range(len(pf_returns)):
                    if t >= lookback and valid_mask[t]:
                        z = z_score[t+1]
                        
                        # Entry logic
                        if position == 0:
                            if z < -entry_z:  # Oversold - buy
                                position = 1
                                entry_price = pf_value[t+1]
                                pnl.append(-transaction_cost)
                                trades.append((t, 'BUY', z, pf_value[t+1]))
                                position_duration = 0
                            elif z > entry_z:  # Overbought - sell
                                position = -1
                                entry_price = pf_value[t+1]
                                pnl.append(-transaction_cost)
                                trades.append((t, 'SELL', z, pf_value[t+1]))
                                position_duration = 0
                            else:
                                pnl.append(0)
                        # Exit logic
                        else:
                            position_duration += 1
                            # Exit conditions: mean reversion or max holding period
                            if abs(z) < exit_z or position_duration >= max_holding:
                                exit_reason = "REVERSION" if abs(z) < exit_z else "MAX_HOLD"
                                pnl.append(position * pf_returns[t] - transaction_cost)
                                trades.append((t, f'CLOSE_{exit_reason}', z, pf_value[t+1]))
                                position = 0
                                position_duration = 0
                            else:
                                pnl.append(position * pf_returns[t])
                    else:
                        pnl.append(0)
                
                cum_pnl = np.cumsum(pnl)
                
                # Comprehensive metrics
                pnl_array = np.array(pnl)
                sharpe = safe_sharpe_ratio(pnl_array, risk_free_rate)
                
                # Max drawdown
                cum_max = np.maximum.accumulate(cum_pnl)
                drawdowns = cum_pnl - cum_max
                max_dd = np.min(drawdowns)
                
                # Win rate
                trade_pnls = []
                for i in range(len(trades)-1):
                    if 'CLOSE' in trades[i+1][1]:
                        trade_pnl = cum_pnl[trades[i+1][0]] - cum_pnl[trades[i][0]]
                        trade_pnls.append(trade_pnl)
                
                win_rate = len([p for p in trade_pnls if p > 0]) / len(trade_pnls) if trade_pnls else 0
                avg_win = np.mean([p for p in trade_pnls if p > 0]) if trade_pnls else 0
                avg_loss = np.mean([p for p in trade_pnls if p < 0]) if trade_pnls else 0
                profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf
                
                # Store results
                st.session_state.sparse_backtest_results = {
                    'pnl': pnl,
                    'cum_pnl': cum_pnl,
                    'trades': trades,
                    'z_score': z_score,
                    'pf_value': pf_value,
                    'drawdowns': drawdowns,
                    'trade_pnls': trade_pnls
                }
                
                # Display metrics
                st.markdown("### Performance Metrics")
                col1, col2, col3, col4, col5 = st.columns(5)
                
                total_return = cum_pnl[-1]
                n_trades = len([t for t in trades if t[1] in ['BUY', 'SELL']])
                
                col1.metric("Total Return", f"{total_return*100:.2f}%")
                col2.metric("Sharpe Ratio", f"{sharpe:.2f}" if not np.isnan(sharpe) else "N/A")
                col3.metric("Max Drawdown", f"{max_dd*100:.2f}%")
                col4.metric("Win Rate", f"{win_rate*100:.1f}%")
                col5.metric("Profit Factor", f"{profit_factor:.2f}" if not np.isinf(profit_factor) else "∞")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Number of Trades", n_trades)
                col2.metric("Avg Win", f"{avg_win*100:.2f}%")
                col3.metric("Avg Loss", f"{avg_loss*100:.2f}%")
        
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
            
            fig.update_layout(height=800, showlegend=True, title_text="Backtest Performance Dashboard")
            st.plotly_chart(fig, use_container_width=True)
            
            # Trade analysis
            if len(trade_pnls) > 0:
                st.markdown("### Trade Analysis")
                
                trade_df = pd.DataFrame({
                    'Trade #': range(1, len(trade_pnls) + 1),
                    'P&L (%)': [p*100 for p in trade_pnls],
                    'Cumulative (%)': np.cumsum(trade_pnls) * 100
                })
                
                fig_trades = go.Figure()
                fig_trades.add_trace(go.Bar(
                    x=trade_df['Trade #'],
                    y=trade_df['P&L (%)'],
                    marker_color=['green' if p > 0 else 'red' for p in trade_df['P&L (%)']],
                    name='Trade P&L'
                ))
                fig_trades.add_trace(go.Scatter(
                    x=trade_df['Trade #'],
                    y=trade_df['Cumulative (%)'],
                    mode='lines+markers',
                    line=dict(color='blue', width=2),
                    name='Cumulative P&L'
                ))
                fig_trades.update_layout(
                    title="Individual Trade Performance",
                    xaxis_title="Trade Number",
                    yaxis_title="P&L (%)",
                    height=400
                )
                st.plotly_chart(fig_trades, use_container_width=True)

with tab5:
    st.markdown("## Advanced Metrics & Portfolio Analytics")
    
    if not st.session_state.sparse_portfolios:
        st.info("👆 Please construct portfolios first")
    else:
        st.markdown("### Comprehensive Portfolio Comparison")
        
        # Compare all portfolios with advanced metrics
        comparison_data = []
        
        for name, pf in st.session_state.sparse_portfolios.items():
            weights = pf['weights']
            pf_returns = returns_df.values @ weights
            pf_returns = pf_returns[~np.isnan(pf_returns)]
            
            if len(pf_returns) < 10:
                continue
            
            # Basic metrics
            metrics = {
                'Portfolio': name,
                'Method': pf['method'],
                'Assets': int(np.sum(np.abs(weights) > 1e-6)),
                'Sharpe': safe_sharpe_ratio(pf_returns, risk_free_rate),
                'Volatility': np.nanstd(pf_returns) * np.sqrt(252) * 100,
                'Return': np.nanmean(pf_returns) * 252 * 100
            }
            
            # Hurst exponent and half-life
            pf_value = (1 + pf_returns).cumprod()
            try:
                hurst_result = hurst_exponent(pf_value, min_window=8, max_window=min(64, len(pf_value)//3))
                metrics['Hurst'] = hurst_result.hurst_exponent
                metrics['Mean-Reverting'] = hurst_result.is_mean_reverting
                metrics['Hurst Confidence'] = hurst_result.confidence_level
            except:
                metrics['Hurst'] = np.nan
                metrics['Mean-Reverting'] = False
                metrics['Hurst Confidence'] = 0.0
            
            metrics['Half-Life'] = calculate_half_life(pf_value)
            
            # Advanced risk metrics
            stability = compute_stability_metrics(pf_returns)
            metrics.update(stability)
            
            # Information Ratio (vs equal-weight benchmark)
            benchmark_returns = returns_df.mean(axis=1).values
            benchmark_returns = benchmark_returns[~np.isnan(benchmark_returns)]
            active_returns = pf_returns[:len(benchmark_returns)] - benchmark_returns[:len(pf_returns)]
            metrics['Info Ratio'] = np.mean(active_returns) / np.std(active_returns) * np.sqrt(252) if np.std(active_returns) > 0 else np.nan
            
            # Skewness and Kurtosis
            metrics['Skewness'] = stats.skew(pf_returns)
            metrics['Kurtosis'] = stats.kurtosis(pf_returns)
            
            comparison_data.append(metrics)
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            
            # Display comprehensive table
            st.markdown("#### Performance & Risk Metrics")
            
            display_df = comparison_df[['Portfolio', 'Method', 'Assets', 'Sharpe', 'Volatility', 
                                       'Return', 'Hurst', 'Half-Life', 'max_dd', 'calmar', 
                                       'sortino', 'Info Ratio']].copy()
            
            # Format display
            format_dict = {
                'Sharpe': '{:.2f}',
                'Volatility': '{:.1f}%',
                'Return': '{:.2f}%',
                'Hurst': '{:.4f}',
                'Half-Life': '{:.1f}',
                'max_dd': '{:.2f}%',
                'calmar': '{:.2f}',
                'sortino': '{:.2f}',
                'Info Ratio': '{:.2f}'
            }
            
            styled_df = display_df.style.background_gradient(subset=['Sharpe'], cmap='RdYlGn')
            styled_df = styled_df.background_gradient(subset=['Hurst'], cmap='RdYlGn_r')
            styled_df = styled_df.format(format_dict, na_rep='N/A')
            
            st.dataframe(styled_df, use_container_width=True)
            
            # Advanced visualizations
            st.markdown("### Visual Analytics")
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Risk-Return Efficient Frontier', 'Hurst vs Sharpe',
                               'Stability Analysis (Sortino vs Calmar)', 'Return Distribution Characteristics'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # 1. Efficient Frontier
            valid_vol = comparison_df[~comparison_df['Volatility'].isna()]
            if len(valid_vol) > 0:
                colors = ['green' if mr else 'red' for mr in comparison_df['Mean-Reverting']]
                fig.add_trace(
                    go.Scatter(x=comparison_df['Volatility'], y=comparison_df['Return'],
                              mode='markers+text', text=comparison_df['Portfolio'],
                              textposition='top center',
                              marker=dict(size=12, color=colors, line=dict(width=1, color='black')),
                              name='Portfolios', showlegend=False),
                    row=1, col=1
                )
            
            # 2. Hurst vs Sharpe
            valid_hurst = comparison_df[~comparison_df['Hurst'].isna() & ~comparison_df['Sharpe'].isna()]
            if len(valid_hurst) > 0:
                fig.add_trace(
                    go.Scatter(x=valid_hurst['Hurst'], y=valid_hurst['Sharpe'],
                              mode='markers+text', text=valid_hurst['Portfolio'],
                              textposition='top center',
                              marker=dict(size=10, color=valid_hurst['Mean-Reverting'].map({True: 'green', False: 'red'})),
                              name='Hurst-Sharpe', showlegend=False),
                    row=1, col=2
                )
                fig.add_vline(x=0.5, line_dash="dash", line_color="orange", row=1, col=2)
                fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=2)
            
            # 3. Sortino vs Calmar
            valid_stability = comparison_df[~comparison_df['sortino'].isna() & ~comparison_df['calmar'].isna()]
            if len(valid_stability) > 0:
                fig.add_trace(
                    go.Scatter(x=valid_stability['sortino'], y=valid_stability['calmar'],
                              mode='markers+text', text=valid_stability['Portfolio'],
                              textposition='top center',
                              marker=dict(size=10, color='purple'),
                              name='Stability', showlegend=False),
                    row=2, col=1
                )
            
            # 4. Skewness vs Kurtosis
            valid_dist = comparison_df[~comparison_df['Skewness'].isna() & ~comparison_df['Kurtosis'].isna()]
            if len(valid_dist) > 0:
                fig.add_trace(
                    go.Scatter(x=valid_dist['Skewness'], y=valid_dist['Kurtosis'],
                              mode='markers+text', text=valid_dist['Portfolio'],
                              textposition='top center',
                              marker=dict(size=10, color='orange'),
                              name='Distribution', showlegend=False),
                    row=2, col=2
                )
                fig.add_vline(x=0, line_dash="dash", line_color="gray", row=2, col=2)
                fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=2)
            
            fig.update_xaxes(title_text="Volatility (%)", row=1, col=1)
            fig.update_xaxes(title_text="Hurst Exponent", row=1, col=2)
            fig.update_xaxes(title_text="Sortino Ratio", row=2, col=1)
            fig.update_xaxes(title_text="Skewness", row=2, col=2)
            
            fig.update_yaxes(title_text="Return (%)", row=1, col=1)
            fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=2)
            fig.update_yaxes(title_text="Calmar Ratio", row=2, col=1)
            fig.update_yaxes(title_text="Kurtosis", row=2, col=2)
            
            fig.update_layout(height=800, showlegend=False, title_text="Advanced Portfolio Analytics")
            st.plotly_chart(fig, use_container_width=True)
            
            # Best portfolio recommendations
            st.markdown("### Portfolio Rankings")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### Best Sharpe Ratio")
                best_sharpe = comparison_df.loc[comparison_df['Sharpe'].idxmax()]
                st.success(f"**{best_sharpe['Portfolio']}**")
                st.write(f"Sharpe: {best_sharpe['Sharpe']:.2f}")
                st.write(f"Hurst: {best_sharpe['Hurst']:.4f}")
            
            with col2:
                st.markdown("#### Most Mean-Reverting")
                mr_portfolios = comparison_df[comparison_df['Mean-Reverting']]
                if len(mr_portfolios) > 0:
                    best_mr = mr_portfolios.loc[mr_portfolios['Hurst'].idxmin()]
                    st.success(f"**{best_mr['Portfolio']}**")
                    st.write(f"Hurst: {best_mr['Hurst']:.4f}")
                    st.write(f"Half-Life: {best_mr['Half-Life']:.1f} days")
                else:
                    st.info("No mean-reverting portfolios found")
            
            with col3:
                st.markdown("#### Best Risk-Adjusted")
                best_sortino = comparison_df.loc[comparison_df['sortino'].idxmax()]
                st.success(f"**{best_sortino['Portfolio']}**")
                st.write(f"Sortino: {best_sortino['sortino']:.2f}")
                st.write(f"Max DD: {best_sortino['max_dd']:.2f}%")

with tab6:
    st.markdown("## Documentation")
    
    st.markdown("""
    ### Sparse Mean-Reversion Lab - Advanced Guide
    
    This lab implements state-of-the-art sparse decomposition algorithms for identifying small, 
    mean-reverting portfolios in high-dimensional asset universes with comprehensive multi-period analysis.
    
    ---
    
    #### 🎯 Methods Overview
    
    ##### 1. **Sparse PCA (Principal Component Analysis)**
    - Finds sparse principal components with L1 regularization
    - Maximizes variance while enforcing sparsity
    - **Formula**: max w^T Σ w - λ ||w||₁ s.t. ||w||₂ = 1
    - **Use case**: Identify dominant market factors with interpretable components
    - **Parameter λ**: Controls sparsity (higher = fewer assets)
    
    ##### 2. **Box & Tao Decomposition (Robust PCA)**
    - Decomposes price matrix into three components:
      - **L** (Low-rank): Common market factors
      - **S** (Sparse): Idiosyncratic mean-reversion signals
      - **N** (Noise): Random fluctuations
    - **Formula**: X = L + S + N
    - **Use case**: Separate systematic vs idiosyncratic components
    - **Parameter λ**: Sparsity penalty on S matrix
    
    ##### 3. **Sparse Cointegration**
    - Elastic Net regression for cointegrated portfolios
    - Combines L1 (sparsity) and L2 (smoothness) regularization
    - Tests stationarity via Augmented Dickey-Fuller test
    - **Formula**: min ||y - Xβ||² + λ₁||β||₁ + λ₂||β||²
    - **Use case**: Find stationary linear combinations (pairs/triplets)
    - **Parameters**: L1 ratio (sparsity) and α (overall regularization)
    
    ##### 4. **Hurst-Based Selection**
    - Selects assets with Hurst exponent H < 0.5
    - R/S (Rescaled Range) analysis for mean-reversion detection
    - **Interpretation**:
      - H < 0.5: Mean-reverting (anti-persistent)
      - H = 0.5: Random walk
      - H > 0.5: Trending (persistent)
    - **Use case**: Directly identify mean-reverting instruments
    
    ---
    
    #### 📊 Advanced Features
    
    ##### Multi-Period Analysis
    - Evaluates portfolio stability across different time periods
    - Computes rolling Hurst exponents
    - Identifies regime changes
    - **Metrics tracked**:
      - Sharpe ratio evolution
      - Half-life stability
      - Drawdown patterns
      - Win rate consistency
    
    ##### Hurst Exponent Deep Dive
    - Individual asset analysis
    - Rolling window analysis (60-day default)
    - Confidence intervals via bootstrap
    - Visual distribution analysis
    - **Applications**:
      - Asset selection for mean-reversion strategies
      - Regime detection (trending vs mean-reverting markets)
      - Strategy switching signals
    
    ##### Advanced Optimization
    - Sharpe ratio maximization
    - Minimum variance optimization
    - Maximum return targeting
    - Constraints: weights sum to 1, short-selling allowed
    
    ---
    
    #### 📈 Key Metrics Explained
    
    - **Sharpe Ratio**: Risk-adjusted return = (Return - RiskFree) / Volatility
    - **Sortino Ratio**: Like Sharpe but only penalizes downside volatility
    - **Calmar Ratio**: Return / Max Drawdown (risk of large losses)
    - **Information Ratio**: Active return / Tracking error vs benchmark
    - **Half-Life**: Expected time for mean reversion (days)
    - **Profit Factor**: Avg Win / Avg Loss ratio
    - **Tail Ratio**: 95th percentile / 5th percentile returns
    - **Skewness**: Asymmetry of return distribution
    - **Kurtosis**: "Fat tails" measure (extreme events)
    
    ---
    
    #### ⚙️ Parameter Guidelines
    
    | Parameter | Range | Recommendation | Effect |
    |-----------|-------|----------------|--------|
    | **Sparse PCA λ** | 0.01-1.0 | 0.1 | Higher = fewer assets |
    | **Box-Tao λ** | 0.01-0.5 | 0.1 | Higher = sparser signals |
    | **L1 Ratio** | 0.1-1.0 | 0.7 | 1.0 = pure Lasso, 0.0 = pure Ridge |
    | **Entry Z-Score** | 1.0-4.0 | 2.0 | Higher = fewer, stronger signals |
    | **Exit Z-Score** | 0.1-2.0 | 0.5 | Lower = faster exits |
    | **Transaction Cost** | 0-50 bps | 10 bps | Typical for equities |
    | **Max Holding** | 5-60 days | 20 | Prevent stuck positions |
    
    ---
    
    #### 🔬 Research Background
    
    This implementation is based on seminal academic research:
    
    1. **d'Aspremont et al. (2008)**: "Optimal Solutions for Sparse Principal Component Analysis"
       - First practical algorithm for sparse PCA
       - Convex relaxation approach
    
    2. **Candès et al. (2011)**: "Robust Principal Component Analysis?"
       - Robust PCA via principal component pursuit
       - Low-rank + sparse decomposition
    
    3. **Zou & Hastie (2005)**: "Regularization and Variable Selection via Elastic Net"
       - Combines L1 and L2 penalties
       - Better than Lasso for correlated features
    
    4. **Hurst (1951)** & **Mandelbrot & Wallis (1969)**: R/S Analysis
       - Originally for Nile river flow prediction
       - Applied to financial time series
    
    ---
    
    #### 💡 Best Practices
    
    1. **Data Quality**: Ensure sufficient history (>252 days recommended)
    2. **Multiple Methods**: Compare results across different approaches
    3. **Rolling Analysis**: Validate stability across time periods
    4. **Transaction Costs**: Always include realistic cost assumptions
    5. **Risk Management**: Monitor drawdowns and use position limits
    6. **Rebalancing**: Consider portfolio turnover costs
    7. **Regime Awareness**: Mean reversion works better in ranging markets
    
    ---
    
    #### 📚 Further Reading
    
    - Avellaneda, M. & Lee, J.H. (2010): "Statistical Arbitrage in the U.S. Equities Market"
    - Pole, A. (2007): "Statistical Arbitrage: Algorithmic Trading Insights and Techniques"
    - Chan, E. (2013): "Algorithmic Trading: Winning Strategies and Their Rationale"
    
    ---
    
    **⚠️ Disclaimer**: This tool is for research and educational purposes only. 
    Past performance does not guarantee future results. Always conduct thorough backtesting 
    and risk analysis before live trading.
    """)
