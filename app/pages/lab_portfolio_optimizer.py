"""
Portfolio Optimizer Lab
Advanced stock ranking using bubble risk, mean reversion, and Markov regime switching
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize
from scipy.stats import norm
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import shared UI components
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.ui_components import render_sidebar_navigation, apply_custom_css

st.set_page_config(page_title="Portfolio Optimizer Lab", page_icon="🎯", layout="wide")

# Render sidebar navigation and apply CSS
render_sidebar_navigation(current_page="Portfolio Optimizer Lab")
apply_custom_css()

st.markdown('<h1 class="lab-header">🎯 Portfolio Optimizer Lab</h1>', unsafe_allow_html=True)
st.markdown("### Multi-factor stock ranking with regime-switching optimization")
st.markdown("---")

# Sidebar configuration
with st.sidebar:
    st.markdown("### 🎛️ Analysis Configuration")
    
    st.markdown("#### Chiarella Parameters")
    beta_f = st.slider("Fundamentalist Strength (β_f)", 0.1, 2.0, 0.5, 0.1)
    beta_c = st.slider("Chartist Strength (β_c)", 0.1, 2.0, 1.0, 0.1)
    gamma = st.slider("Switching Rate (γ)", 0.1, 5.0, 1.0, 0.1)
    
    st.markdown("---")
    st.markdown("#### Mean Reversion Parameters")
    lookback_window = st.slider("Lookback Window", 20, 200, 50, 10)
    half_life_threshold = st.slider("Half-Life Threshold (days)", 5, 60, 20, 5)
    
    st.markdown("---")
    st.markdown("#### Regime Detection")
    num_regimes = st.selectbox("Number of Regimes", [2, 3], index=0)
    regime_lookback = st.slider("Regime Lookback", 50, 500, 200, 50)

# Helper Functions

def estimate_half_life(prices):
    """Estimate mean reversion half-life using AR(1) model"""
    returns = np.diff(np.log(prices))
    lagged_returns = returns[:-1]
    current_returns = returns[1:]
    
    if len(lagged_returns) < 2:
        return np.inf
    
    # AR(1): r_t = φ * r_{t-1} + ε
    phi = np.corrcoef(lagged_returns, current_returns)[0, 1]
    
    if phi >= 1 or phi <= 0:
        return np.inf
    
    half_life = -np.log(2) / np.log(phi)
    return half_life

def calculate_hurst_exponent(prices, max_lag=20):
    """Calculate Hurst exponent for mean reversion detection"""
    lags = range(2, min(max_lag, len(prices) // 2))
    tau = []
    
    for lag in lags:
        # Calculate variance of differences
        pp = np.subtract(prices[lag:], prices[:-lag])
        tau.append(np.std(pp))
    
    if len(tau) < 2:
        return 0.5
    
    # Linear regression on log-log plot
    lags_log = np.log(list(lags))
    tau_log = np.log(tau)
    
    poly = np.polyfit(lags_log, tau_log, 1)
    hurst = poly[0]
    
    return hurst

def markov_regime_switching(returns, n_regimes=2):
    """
    Simplified Markov regime switching model
    Classifies returns into high/low volatility regimes
    """
    # Sort returns by absolute value
    abs_returns = np.abs(returns)
    
    if n_regimes == 2:
        # Two regimes: low vol and high vol
        threshold = np.median(abs_returns)
        regimes = (abs_returns > threshold).astype(int)
        
        regime_vols = [
            np.std(returns[regimes == 0]) if np.sum(regimes == 0) > 0 else 0,
            np.std(returns[regimes == 1]) if np.sum(regimes == 1) > 0 else 0
        ]
        
        regime_means = [
            np.mean(returns[regimes == 0]) if np.sum(regimes == 0) > 0 else 0,
            np.mean(returns[regimes == 1]) if np.sum(regimes == 1) > 0 else 0
        ]
        
    else:  # 3 regimes
        # Three regimes: low, medium, high volatility
        percentile_33 = np.percentile(abs_returns, 33)
        percentile_66 = np.percentile(abs_returns, 66)
        
        regimes = np.zeros(len(returns), dtype=int)
        regimes[abs_returns > percentile_33] = 1
        regimes[abs_returns > percentile_66] = 2
        
        regime_vols = [
            np.std(returns[regimes == i]) if np.sum(regimes == i) > 0 else 0
            for i in range(3)
        ]
        
        regime_means = [
            np.mean(returns[regimes == i]) if np.sum(regimes == i) > 0 else 0
            for i in range(3)
        ]
    
    # Calculate transition probabilities
    transition_matrix = np.zeros((n_regimes, n_regimes))
    for i in range(len(regimes) - 1):
        transition_matrix[regimes[i], regimes[i + 1]] += 1
    
    # Normalize rows
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    transition_matrix = np.divide(transition_matrix, row_sums, 
                                  where=row_sums > 0, 
                                  out=np.zeros_like(transition_matrix))
    
    return regimes, regime_vols, regime_means, transition_matrix

def calculate_bubble_score(prices, fundamental, volatility, lambda_param, vol_threshold=3.0):
    """Calculate bubble risk score based on Chiarella dynamics"""
    # Overvaluation
    mispricing_pct = (prices[-1] - fundamental[-1]) / fundamental[-1]
    
    # Elevated volatility
    vol_elevated = volatility > vol_threshold
    
    # Unstable regime (Lambda > 1.5)
    is_unstable = lambda_param > 1.5
    
    # Momentum
    returns = np.diff(prices[-20:]) / prices[-20:-1] if len(prices) > 20 else [0]
    momentum_strong = abs(np.mean(returns)) > 0.002
    
    # Bubble score (0-1)
    bubble_score = 0.0
    
    # Overvaluation (40%)
    if mispricing_pct > 0.05:
        bubble_score += 0.4 * min(mispricing_pct / 0.2, 1.0)
    
    # Elevated volatility (25%)
    if vol_elevated:
        bubble_score += 0.25
    
    # Unstable regime (20%)
    if is_unstable:
        bubble_score += 0.20
    
    # Strong momentum (15%)
    if momentum_strong and mispricing_pct > 0:
        bubble_score += 0.15
    
    return bubble_score

def optimize_portfolio_weights(scores_matrix, returns_matrix, cov_matrix, risk_aversion=2.0):
    """
    Optimize portfolio weights based on multi-factor scores
    Maximize: score-weighted returns - risk_aversion * variance
    """
    n_assets = len(scores_matrix)
    
    # Objective: maximize utility
    def objective(weights):
        # Weighted score (higher is better for mean reversion, lower for bubble risk)
        score_weighted = np.dot(weights, scores_matrix)
        
        # Expected return
        expected_return = np.dot(weights, returns_matrix)
        
        # Portfolio variance
        portfolio_var = np.dot(weights, np.dot(cov_matrix, weights))
        
        # Utility: return - risk_aversion * variance + score_bonus
        utility = expected_return - risk_aversion * portfolio_var + 0.1 * score_weighted
        
        return -utility  # Minimize negative utility
    
    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},  # Weights sum to 1
    ]
    
    # Bounds: 0 to 0.3 per asset (no short selling, max 30% per position)
    bounds = [(0, 0.3) for _ in range(n_assets)]
    
    # Initial guess: equal weights
    x0 = np.ones(n_assets) / n_assets
    
    # Optimize
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result.x if result.success else x0

# Main content
tab1, tab2, tab3 = st.tabs(["📊 Stock Analysis", "🎯 Portfolio Optimization", "📈 Results & Backtest"])

with tab1:
    st.markdown("### Multi-Factor Stock Ranking")
    
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
        
        if len(symbols) < 2:
            st.warning("⚠️ Please load at least 2 symbols for portfolio analysis")
        else:
            st.markdown(f"**Available Symbols:** {len(symbols)}")
            
            # Multi-select for analysis
            selected_symbols = st.multiselect(
                "Select Stocks to Analyze",
                symbols,
                default=symbols[:min(5, len(symbols))],
                help="Select 2-20 stocks for analysis"
            )
            
            if len(selected_symbols) < 2:
                st.info("Please select at least 2 stocks")
            elif len(selected_symbols) > 20:
                st.warning("Maximum 20 stocks allowed for analysis")
            else:
                if st.button("🔍 Analyze Stocks", type="primary"):
                    with st.spinner("Analyzing stocks across multiple factors..."):
                        analysis_results = []
                        
                        for symbol in selected_symbols:
                            try:
                                # Extract data
                                if isinstance(data, dict):
                                    df = data[symbol]
                                elif isinstance(data, pd.DataFrame):
                                    if 'symbol' in data.columns:
                                        df = data[data['symbol'] == symbol].copy()
                                    else:
                                        df = data.copy()
                                else:
                                    continue
                                
                                # Find close column
                                close_col = None
                                for col in df.columns:
                                    if col.lower() == 'close':
                                        close_col = col
                                        break
                                
                                if close_col is None:
                                    continue
                                
                                prices = df[close_col].values
                                
                                if len(prices) < 50:
                                    continue
                                
                                # 1. Mean Reversion Metrics
                                half_life = estimate_half_life(prices)
                                hurst = calculate_hurst_exponent(prices)
                                
                                # Z-score
                                rolling_mean = pd.Series(prices).rolling(lookback_window).mean().values
                                rolling_std = pd.Series(prices).rolling(lookback_window).std().values
                                current_zscore = (prices[-1] - rolling_mean[-1]) / (rolling_std[-1] + 1e-8)
                                
                                # Mean reversion score (0-1, higher is better)
                                mr_score = 0.0
                                if half_life < half_life_threshold:
                                    mr_score += 0.4
                                if hurst < 0.5:
                                    mr_score += 0.3 * (0.5 - hurst) / 0.5
                                if abs(current_zscore) > 1.5:
                                    mr_score += 0.3 * min(abs(current_zscore) / 3.0, 1.0)
                                
                                # 2. Bubble Risk (Chiarella)
                                # Estimate fundamental
                                fundamental = pd.Series(prices).ewm(span=100).mean().values
                                
                                # Calculate returns and volatility
                                returns = np.diff(prices) / prices[:-1]
                                volatility = np.std(returns[-20:]) * 100 if len(returns) > 20 else 0
                                
                                # Lambda parameter
                                Lambda = (beta_c * gamma) / (beta_f * 0.2) if beta_f > 0 else 1.0
                                
                                # Bubble score (0-1, lower is better)
                                bubble_score = calculate_bubble_score(prices, fundamental, volatility, Lambda)
                                
                                # 3. Regime Switching Analysis
                                regime_returns = returns[-regime_lookback:] if len(returns) > regime_lookback else returns
                                regimes, regime_vols, regime_means, trans_matrix = markov_regime_switching(
                                    regime_returns, num_regimes
                                )
                                
                                # Current regime
                                current_regime = regimes[-1]
                                current_regime_vol = regime_vols[current_regime]
                                current_regime_mean = regime_means[current_regime]
                                
                                # Regime stability score
                                regime_stability = trans_matrix[current_regime, current_regime]
                                
                                # 4. Combined Score
                                # Prefer: high mean reversion, low bubble risk, stable regime
                                combined_score = (
                                    0.4 * mr_score +
                                    0.3 * (1 - bubble_score) +
                                    0.3 * regime_stability
                                )
                                
                                analysis_results.append({
                                    'Symbol': symbol,
                                    'MR Score': mr_score,
                                    'Half-Life': half_life if half_life != np.inf else 999,
                                    'Hurst': hurst,
                                    'Z-Score': current_zscore,
                                    'Bubble Score': bubble_score,
                                    'Lambda': Lambda,
                                    'Volatility': volatility,
                                    'Regime': current_regime,
                                    'Regime Vol': current_regime_vol * 100,
                                    'Regime Mean': current_regime_mean * 100,
                                    'Regime Stability': regime_stability,
                                    'Combined Score': combined_score,
                                    'prices': prices,
                                    'returns': returns
                                })
                                
                            except Exception as e:
                                st.warning(f"Error analyzing {symbol}: {str(e)}")
                                continue
                        
                        if not analysis_results:
                            st.error("No stocks could be analyzed. Check data quality.")
                        else:
                            # Store in session state
                            st.session_state['stock_analysis'] = analysis_results
                            
                            # Create DataFrame for display
                            df_analysis = pd.DataFrame([
                                {k: v for k, v in r.items() if k not in ['prices', 'returns']}
                                for r in analysis_results
                            ])
                            
                            # Sort by combined score
                            df_analysis = df_analysis.sort_values('Combined Score', ascending=False)
                            
                            st.markdown("### 📊 Stock Rankings")
                            
                            # Add rank
                            df_analysis.insert(0, 'Rank', range(1, len(df_analysis) + 1))
                            
                            # Color coding
                            def color_score(val):
                                if val > 0.7:
                                    return 'background-color: #90EE90'  # Light green
                                elif val > 0.4:
                                    return 'background-color: #FFFFE0'  # Light yellow
                                else:
                                    return 'background-color: #FFB6C1'  # Light red
                            
                            # Display with formatting
                            styled_df = df_analysis.style.format({
                                'MR Score': '{:.3f}',
                                'Half-Life': '{:.1f}',
                                'Hurst': '{:.3f}',
                                'Z-Score': '{:.2f}',
                                'Bubble Score': '{:.3f}',
                                'Lambda': '{:.2f}',
                                'Volatility': '{:.2f}%',
                                'Regime Vol': '{:.2f}%',
                                'Regime Mean': '{:.3f}%',
                                'Regime Stability': '{:.2f}',
                                'Combined Score': '{:.3f}'
                            }).applymap(color_score, subset=['Combined Score'])
                            
                            st.dataframe(styled_df, use_container_width=True, height=400)
                            
                            # Top performers
                            st.markdown("### 🏆 Top Performers")
                            
                            top_3 = df_analysis.head(3)
                            
                            col1, col2, col3 = st.columns(3)
                            
                            for idx, (col, (_, row)) in enumerate(zip([col1, col2, col3], top_3.iterrows())):
                                with col:
                                    st.markdown(f"**#{idx + 1}: {row['Symbol']}**")
                                    st.metric("Combined Score", f"{row['Combined Score']:.3f}")
                                    st.caption(f"MR: {row['MR Score']:.2f} | Bubble: {row['Bubble Score']:.2f}")
                            
                            # Visualizations
                            st.markdown("### 📈 Factor Analysis")
                            
                            fig = make_subplots(
                                rows=2, cols=2,
                                subplot_titles=('Mean Reversion Score', 'Bubble Risk Score', 
                                              'Regime Stability', 'Combined Score'),
                                specs=[[{'type': 'bar'}, {'type': 'bar'}],
                                      [{'type': 'bar'}, {'type': 'bar'}]]
                            )
                            
                            # Mean Reversion
                            fig.add_trace(
                                go.Bar(x=df_analysis['Symbol'], y=df_analysis['MR Score'],
                                      name='MR Score', marker_color='lightblue'),
                                row=1, col=1
                            )
                            
                            # Bubble Risk
                            fig.add_trace(
                                go.Bar(x=df_analysis['Symbol'], y=df_analysis['Bubble Score'],
                                      name='Bubble Score', marker_color='lightcoral'),
                                row=1, col=2
                            )
                            
                            # Regime Stability
                            fig.add_trace(
                                go.Bar(x=df_analysis['Symbol'], y=df_analysis['Regime Stability'],
                                      name='Regime Stability', marker_color='lightgreen'),
                                row=2, col=1
                            )
                            
                            # Combined Score
                            fig.add_trace(
                                go.Bar(x=df_analysis['Symbol'], y=df_analysis['Combined Score'],
                                      name='Combined Score', marker_color='gold'),
                                row=2, col=2
                            )
                            
                            fig.update_layout(height=700, showlegend=False)
                            fig.update_xaxes(tickangle=45)
                            
                            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("### Portfolio Weight Optimization")
    
    if 'stock_analysis' not in st.session_state:
        st.info("💡 Run stock analysis in the Stock Analysis tab first")
    else:
        analysis_results = st.session_state['stock_analysis']
        
        st.markdown(f"**Analyzing {len(analysis_results)} stocks**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            risk_aversion = st.slider("Risk Aversion", 0.5, 5.0, 2.0, 0.5,
                                     help="Higher = more conservative")
            min_score = st.slider("Min Combined Score", 0.0, 0.8, 0.3, 0.1,
                                 help="Filter stocks below this score")
        
        with col2:
            max_positions = st.slider("Max Positions", 3, 10, 5,
                                     help="Maximum stocks in portfolio")
            target_return = st.slider("Target Annual Return %", 5.0, 30.0, 15.0, 1.0)
        
        if st.button("🎯 Optimize Portfolio", type="primary"):
            with st.spinner("Optimizing portfolio weights..."):
                # Filter by minimum score
                filtered_results = [r for r in analysis_results if r['Combined Score'] >= min_score]
                
                if len(filtered_results) < 2:
                    st.error(f"Not enough stocks meet the minimum score threshold ({min_score:.2f})")
                else:
                    # Take top N by combined score
                    sorted_results = sorted(filtered_results, key=lambda x: x['Combined Score'], reverse=True)
                    top_n = sorted_results[:max_positions]
                    
                    # Prepare matrices
                    symbols = [r['Symbol'] for r in top_n]
                    scores = np.array([r['Combined Score'] for r in top_n])
                    
                    # Calculate expected returns (annualized)
                    returns_list = []
                    for r in top_n:
                        mean_return = np.mean(r['returns']) * 252  # Annualize
                        returns_list.append(mean_return)
                    
                    expected_returns = np.array(returns_list)
                    
                    # Calculate covariance matrix
                    returns_matrix = np.array([r['returns'][-min(252, len(r['returns'])):] for r in top_n])
                    
                    # Align lengths
                    min_len = min(len(r) for r in returns_matrix)
                    returns_matrix = np.array([r[-min_len:] for r in returns_matrix])
                    
                    cov_matrix = np.cov(returns_matrix) * 252  # Annualize
                    
                    # Optimize
                    optimal_weights = optimize_portfolio_weights(scores, expected_returns, cov_matrix, risk_aversion)
                    
                    # Store results
                    st.session_state['optimal_portfolio'] = {
                        'symbols': symbols,
                        'weights': optimal_weights,
                        'scores': scores,
                        'expected_returns': expected_returns,
                        'cov_matrix': cov_matrix,
                        'analysis_results': top_n
                    }
                    
                    # Display results
                    st.markdown("### 💼 Optimal Portfolio")
                    
                    # Portfolio metrics
                    port_return = np.dot(optimal_weights, expected_returns)
                    port_vol = np.sqrt(np.dot(optimal_weights, np.dot(cov_matrix, optimal_weights)))
                    port_sharpe = port_return / port_vol if port_vol > 0 else 0
                    
                    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                    
                    with metric_col1:
                        st.metric("Expected Return", f"{port_return*100:.2f}%")
                    with metric_col2:
                        st.metric("Volatility", f"{port_vol*100:.2f}%")
                    with metric_col3:
                        st.metric("Sharpe Ratio", f"{port_sharpe:.2f}")
                    with metric_col4:
                        avg_score = np.dot(optimal_weights, scores)
                        st.metric("Avg Score", f"{avg_score:.3f}")
                    
                    # Weights table
                    st.markdown("### 📊 Portfolio Composition")
                    
                    portfolio_df = pd.DataFrame({
                        'Symbol': symbols,
                        'Weight': optimal_weights,
                        'Weight %': optimal_weights * 100,
                        'Score': scores,
                        'Expected Return': expected_returns * 100,
                        'Contribution': optimal_weights * expected_returns * 100
                    })
                    
                    portfolio_df = portfolio_df.sort_values('Weight', ascending=False)
                    
                    styled_portfolio = portfolio_df.style.format({
                        'Weight': '{:.4f}',
                        'Weight %': '{:.2f}%',
                        'Score': '{:.3f}',
                        'Expected Return': '{:.2f}%',
                        'Contribution': '{:.2f}%'
                    }).background_gradient(subset=['Weight %'], cmap='Blues')
                    
                    st.dataframe(styled_portfolio, use_container_width=True)
                    
                    # Visualization
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        # Pie chart
                        fig_pie = go.Figure(data=[go.Pie(
                            labels=symbols,
                            values=optimal_weights,
                            hole=0.4
                        )])
                        fig_pie.update_layout(title='Portfolio Allocation', height=400)
                        st.plotly_chart(fig_pie, use_container_width=True)
                    
                    with col_b:
                        # Bar chart
                        fig_bar = go.Figure(data=[go.Bar(
                            x=symbols,
                            y=optimal_weights * 100,
                            marker_color='lightblue'
                        )])
                        fig_bar.update_layout(
                            title='Portfolio Weights (%)',
                            yaxis_title='Weight (%)',
                            height=400
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)
                    
                    # Correlation matrix
                    st.markdown("### 🔗 Correlation Matrix")
                    
                    corr_matrix = np.corrcoef(returns_matrix)
                    
                    fig_corr = go.Figure(data=go.Heatmap(
                        z=corr_matrix,
                        x=symbols,
                        y=symbols,
                        colorscale='RdBu',
                        zmid=0,
                        text=np.round(corr_matrix, 2),
                        texttemplate='%{text}',
                        textfont={"size": 10}
                    ))
                    
                    fig_corr.update_layout(
                        title='Return Correlations',
                        height=500
                    )
                    
                    st.plotly_chart(fig_corr, use_container_width=True)

with tab3:
    st.markdown("### Backtest & Results")
    
    if 'optimal_portfolio' not in st.session_state:
        st.info("💡 Optimize a portfolio in the Portfolio Optimization tab first")
    else:
        portfolio = st.session_state['optimal_portfolio']
        
        st.markdown("#### 📈 Historical Performance Simulation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            backtest_period = st.slider("Backtest Period (days)", 30, 252, 126)
            rebalance_freq = st.selectbox("Rebalance Frequency", ["None", "Weekly", "Monthly"], index=0)
        
        with col2:
            initial_capital = st.number_input("Initial Capital ($)", value=100000.0, step=10000.0)
            transaction_cost = st.slider("Transaction Cost (bps)", 0, 50, 10)
        
        if st.button("📊 Run Backtest", type="primary"):
            with st.spinner("Running backtest..."):
                symbols = portfolio['symbols']
                weights = portfolio['weights']
                analysis_results = portfolio['analysis_results']
                
                # Simulate portfolio returns
                returns_matrix = np.array([r['returns'][-backtest_period:] for r in analysis_results])
                
                # Align lengths
                min_len = min(len(r) for r in returns_matrix)
                returns_matrix = np.array([r[-min_len:] for r in returns_matrix])
                
                # Portfolio returns
                portfolio_returns = returns_matrix.T @ weights
                
                # Apply transaction costs (simplified)
                cost_per_trade = transaction_cost / 10000.0
                portfolio_returns -= cost_per_trade  # Assume daily rebalancing approximation
                
                # Cumulative returns
                portfolio_value = initial_capital * np.cumprod(1 + portfolio_returns)
                
                # Benchmark: equal weight
                equal_weight_returns = returns_matrix.mean(axis=0)
                benchmark_value = initial_capital * np.cumprod(1 + equal_weight_returns)
                
                # Metrics
                total_return = (portfolio_value[-1] - initial_capital) / initial_capital
                ann_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
                ann_vol = np.std(portfolio_returns) * np.sqrt(252)
                sharpe = ann_return / ann_vol if ann_vol > 0 else 0
                
                # Drawdown
                running_max = np.maximum.accumulate(portfolio_value)
                drawdowns = (portfolio_value - running_max) / running_max
                max_drawdown = np.min(drawdowns)
                
                # Display metrics
                st.markdown("### 📊 Performance Metrics")
                
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                
                with metric_col1:
                    st.metric("Total Return", f"{total_return*100:.2f}%")
                with metric_col2:
                    st.metric("Annual Return", f"{ann_return*100:.2f}%")
                with metric_col3:
                    st.metric("Sharpe Ratio", f"{sharpe:.2f}")
                with metric_col4:
                    st.metric("Max Drawdown", f"{max_drawdown*100:.2f}%")
                
                # Benchmark comparison
                bench_return = (benchmark_value[-1] - initial_capital) / initial_capital
                outperformance = (total_return - bench_return) * 100
                
                st.info(f"**Outperformance vs Equal Weight:** {outperformance:+.2f}%")
                
                # Plot
                st.markdown("### 📈 Equity Curves")
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    y=portfolio_value,
                    name='Optimized Portfolio',
                    line={'color': 'blue', 'width': 2}
                ))
                
                fig.add_trace(go.Scatter(
                    y=benchmark_value,
                    name='Equal Weight Benchmark',
                    line={'color': 'gray', 'width': 2, 'dash': 'dash'}
                ))
                
                fig.update_layout(
                    title='Portfolio Value Over Time',
                    yaxis_title='Portfolio Value ($)',
                    xaxis_title='Days',
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Drawdown chart
                st.markdown("### 📉 Drawdown Analysis")
                
                fig_dd = go.Figure()
                
                fig_dd.add_trace(go.Scatter(
                    y=drawdowns * 100,
                    fill='tozeroy',
                    fillcolor='rgba(255,0,0,0.1)',
                    line={'color': 'red', 'width': 2},
                    name='Drawdown'
                ))
                
                fig_dd.update_layout(
                    title='Portfolio Drawdown',
                    yaxis_title='Drawdown (%)',
                    xaxis_title='Days',
                    height=400
                )
                
                st.plotly_chart(fig_dd, use_container_width=True)
                
                # Summary
                st.markdown("### 📋 Summary")
                
                st.success(f"""
                **Portfolio Performance Summary:**
                - Optimized portfolio returned **{total_return*100:.2f}%** over {len(portfolio_returns)} days
                - Annualized return: **{ann_return*100:.2f}%**
                - Risk-adjusted return (Sharpe): **{sharpe:.2f}**
                - Maximum drawdown: **{max_drawdown*100:.2f}%**
                - Outperformed equal-weight benchmark by **{outperformance:+.2f}%**
                
                **Strategy Highlights:**
                - Combined mean reversion, bubble risk, and regime switching
                - Dynamic weight optimization based on multi-factor scores
                - Risk-aware position sizing with correlation considerations
                """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🎯 Portfolio Optimizer Lab | Advanced Multi-Factor Selection</p>
</div>
""", unsafe_allow_html=True)
