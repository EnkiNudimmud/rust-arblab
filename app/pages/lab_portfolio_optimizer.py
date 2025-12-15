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
import time
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import shared UI components
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.ui_components import render_sidebar_navigation, apply_custom_css

def estimate_remaining_time(start_time, completed, total):
    """Estimate remaining time for a task"""
    if completed == 0:
        return "Calculating..."
    elapsed = time.time() - start_time
    rate = elapsed / completed
    remaining = rate * (total - completed)
    if remaining < 60:
        return f"{int(remaining)}s"
    elif remaining < 3600:
        minutes = int(remaining // 60)
        seconds = int(remaining % 60)
        return f"{minutes}m {seconds}s"
    else:
        hours = int(remaining // 3600)
        minutes = int((remaining % 3600) // 60)
        return f"{hours}h {minutes}m"

# Try to import drift uncertainty module
import hft_py.portfolio_drift as pdrift
DRIFT_UNCERTAINTY_AVAILABLE = True

st.set_page_config(page_title="Portfolio Optimizer Lab", page_icon="🎯", layout="wide")

# Render sidebar navigation and apply CSS
render_sidebar_navigation(current_page="Portfolio Optimizer Lab")
apply_custom_css()

st.markdown('<h1 class="lab-header">🎯 Portfolio Optimizer Lab</h1>', unsafe_allow_html=True)
st.markdown("### Multi-factor stock ranking with regime-switching optimization")

# Show drift uncertainty feature status
if DRIFT_UNCERTAINTY_AVAILABLE:
    st.success("✅ **New Feature Available**: Drift Uncertainty Portfolio Optimization - See `examples/notebooks/portfolio_drift_uncertainty.ipynb` for examples")
else:
    st.info("💡 **Optional Feature**: Build Rust bindings to enable drift uncertainty portfolio optimization. See the Jupyter notebook in `examples/notebooks/portfolio_drift_uncertainty.ipynb`")

st.markdown("---")

# Ensure data is loaded (will auto-load most recent dataset if needed)
from utils.ui_components import ensure_data_loaded
data_available = ensure_data_loaded()

# Check data availability
if not data_available or st.session_state.historical_data is None or st.session_state.historical_data.empty:
    st.markdown("""
    <div class="info-card">
        <h3>📊 No Data Loaded</h3>
        <p>Please load historical data first to use the Portfolio Optimizer Lab.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚀 Go to Data Loader", type="primary", use_container_width=True):
        st.switch_page("pages/data_loader.py")
    st.stop()

# Try to import regime portfolio module
import hft_py.regime_portfolio as regime_portfolio
REGIME_PORTFOLIO_AVAILABLE = True

# Mode selector at top
modes = ["Multi-Factor Analysis", "Drift Uncertainty (Robust)"]
if REGIME_PORTFOLIO_AVAILABLE:
    modes.append("Regime Switching Jump Diffusion")

optimization_mode = st.radio(
    "Select Optimization Mode:",
    modes,
    horizontal=True,
    help="Multi-Factor: Traditional portfolio optimization. Drift Uncertainty: Robust optimization accounting for uncertain expected returns. MRSJD: Advanced regime-switching jump-diffusion model."
)

st.markdown("---")

# Sidebar configuration
with st.sidebar:
    st.markdown("### 🎛️ Analysis Configuration")
    
    # Drift uncertainty parameters (shown only in that mode)
    if optimization_mode == "Drift Uncertainty (Robust)":
        st.markdown("#### 🛡️ Drift Uncertainty Parameters")
        risk_aversion = st.slider("Risk Aversion (γ)", 0.5, 10.0, 2.0, 0.5,
                                  help="Higher values = more risk-averse")
        drift_uncertainty = st.slider("Drift Uncertainty (δ)", 0.0, 0.15, 0.02, 0.01,
                                     help="Uncertainty range for expected returns")
        time_horizon_days = st.slider("Time Horizon (days)", 1, 30, 10, 1,
                                      help="Investment or rebalancing period")
        transaction_cost_bps = st.slider("Transaction Cost (bps)", 0, 100, 10, 5,
                                         help="Cost per trade in basis points")
        num_steps = st.slider("Time Steps", 20, 200, 100, 10,
                              help="Number of discretization steps")
        st.markdown("---")
    
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

def _get_symbols(data):
    """Extract symbols from various data formats"""
    if isinstance(data, dict):
        return list(data.keys())
    elif isinstance(data, pd.DataFrame):
        if 'symbol' in data.columns:
            return data['symbol'].unique().tolist()
        else:
            return ['Data']
    else:
        return []

def _compute_statistics(data, symbols):
    """Compute mean returns and covariance matrix from historical data"""
    try:
        returns_list = []
        symbols_used = []
        
        for symbol in symbols:
            if isinstance(data, dict):
                df = data[symbol]
            elif isinstance(data, pd.DataFrame):
                if 'symbol' in data.columns:
                    df = data[data['symbol'] == symbol].copy()
                else:
                    df = data.copy()
            else:
                continue
            
            if 'close' in df.columns:
                prices = df['close'].values
                if len(prices) > 1:
                    returns = np.diff(np.log(prices))
                    if len(returns) > 0:
                        returns_list.append(returns)
                        symbols_used.append(symbol)
        
        if len(returns_list) < 2:
            return None, None, None
        
        # Align lengths
        min_len = min(len(r) for r in returns_list)
        returns_matrix = np.array([r[:min_len] for r in returns_list])
        
        # Compute statistics
        mu = returns_matrix.mean(axis=1).tolist()
        cov = np.cov(returns_matrix).tolist()
        
        return mu, cov, symbols_used
        
    except Exception as e:
        st.error(f"Error computing statistics: {e}")
        return None, None, None

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

# Main content - conditional based on mode
if optimization_mode == "Drift Uncertainty (Robust)":
    # Drift Uncertainty Mode
    if not DRIFT_UNCERTAINTY_AVAILABLE:
        st.error("⚠️ **Drift Uncertainty Module Not Available**")
        st.markdown("""
        The drift uncertainty optimization module requires Rust bindings to be built.
        
        **To enable this feature:**
        1. Fix compilation errors in `rust_core/src/orderbook.rs` (see documentation)
        2. Build bindings: `cd rust_python_bindings && maturin develop`
        3. Restart Streamlit
        
        **For now**, you can explore the implementation in the Jupyter notebook:
        `examples/notebooks/portfolio_drift_uncertainty.ipynb`
        """)
    else:
        # Drift uncertainty tabs
        drift_tabs = st.tabs([
            "🛡️ Robust Portfolio", 
            "📉 Liquidation", 
            "🔄 Transition",
            "⚠️ Risk Analysis"
        ])
        
        with drift_tabs[0]:
            st.markdown("### 🛡️ Robust Portfolio Choice under Drift Uncertainty")
            st.markdown("""
            Optimize portfolio weights accounting for **uncertainty in expected returns**.
            
            **Method**: Worst-case optimization with CARA utility  
            **Formula**: Maximize min_{μ ∈ [μ̂-δ, μ̂+δ]} E[U(w^T μ - (γ/2)w^T Σ w)]
            """)
            
            if 'historical_data' not in st.session_state or st.session_state.historical_data is None:
                st.warning("⚠️ Please load data first from the Data Loader page")
                if st.button("💾 Go to Data Loader", key="drift_loader_1"):
                    st.switch_page("pages/data_loader.py")
            else:
                data = st.session_state.historical_data
                symbols = _get_symbols(data)
                
                if len(symbols) < 2:
                    st.warning("⚠️ Please load at least 2 symbols")
                else:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        selected_assets = st.multiselect(
                            "Select Assets for Portfolio",
                            symbols,
                            default=symbols[:min(5, len(symbols))],
                            key="robust_assets"
                        )
                    with col2:
                        initial_wealth = st.number_input(
                            "Initial Wealth ($)", 
                            min_value=1000.0, 
                            value=100000.0, 
                            step=10000.0,
                            format="%.0f"
                        )
                    
                    if len(selected_assets) >= 2:
                        if st.button("🔍 Optimize Robust Portfolio", type="primary", key="opt_robust"):
                            with st.spinner("Computing robust portfolio..."):
                                try:
                                    # Calculate returns and statistics
                                    mu, cov, symbols_used = _compute_statistics(data, selected_assets)
                                    
                                    if mu is None:
                                        st.error("Failed to compute statistics from data")
                                    else:
                                        # Optimize
                                        result = pdrift.portfolio_choice_drift_uncertainty(
                                            mu=mu,
                                            cov=cov,
                                            risk_aversion=risk_aversion,
                                            drift_uncertainty=drift_uncertainty
                                        )
                                        
                                        # Display metrics
                                        col1, col2, col3, col4 = st.columns(4)
                                        with col1:
                                            st.metric("Expected Return", 
                                                     f"{result['expected_return']*100:.2f}%")
                                        with col2:
                                            st.metric("Worst-Case Return", 
                                                     f"{result['worst_case_return']*100:.2f}%",
                                                     delta=f"{(result['worst_case_return']-result['expected_return'])*100:.2f}%")
                                        with col3:
                                            st.metric("Portfolio Variance", 
                                                     f"{result['variance']:.4f}")
                                        with col4:
                                            st.metric("Utility Value", 
                                                     f"{result['utility']:.4f}")
                                        
                                        # Weights chart
                                        weights_array = np.array(result['weights'])
                                        fig_weights = go.Figure(data=[
                                            go.Bar(
                                                x=symbols_used,
                                                y=weights_array * 100,
                                                marker_color='steelblue',
                                                text=[f"{w*100:.1f}%" for w in weights_array],
                                                textposition='auto'
                                            )
                                        ])
                                        fig_weights.update_layout(
                                            title="Optimal Portfolio Weights (Robust)",
                                            xaxis_title="Asset",
                                            yaxis_title="Weight (%)",
                                            height=400,
                                            showlegend=False
                                        )
                                        st.plotly_chart(fig_weights, use_container_width=True)
                                        
                                        # Sensitivity analysis
                                        st.markdown("#### 📊 Sensitivity to Drift Uncertainty")
                                        delta_levels = [0.0, 0.01, 0.02, 0.03, 0.05, 0.10]
                                        sensitivity_data = []
                                        
                                        start_time = time.time()
                                        progress_bar = st.progress(0)
                                        time_text = st.empty()
                                        for i, delta in enumerate(delta_levels):
                                            r = pdrift.portfolio_choice_drift_uncertainty(
                                                mu=mu, cov=cov,
                                                risk_aversion=risk_aversion,
                                                drift_uncertainty=delta
                                            )
                                            sensitivity_data.append({
                                                'Uncertainty (%)': f"{delta*100:.1f}",
                                                'Expected Return (%)': r['expected_return'] * 100,
                                                'Worst-Case Return (%)': r['worst_case_return'] * 100,
                                                'Utility': r['utility']
                                            })
                                            progress_pct = (i + 1) / len(delta_levels)
                                            progress_bar.progress(progress_pct)
                                            time_text.text(f"⏱️ Estimated time remaining: {estimate_remaining_time(start_time, (i + 1), len(delta_levels))}")
                                        progress_bar.empty()
                                        time_text.empty()
                                        
                                        sens_df = pd.DataFrame(sensitivity_data)
                                        
                                        # Plot sensitivity
                                        fig_sens = go.Figure()
                                        fig_sens.add_trace(go.Scatter(
                                            x=sens_df['Uncertainty (%)'],
                                            y=sens_df['Expected Return (%)'],
                                            mode='lines+markers',
                                            name='Expected Return',
                                            line=dict(color='green', width=3)
                                        ))
                                        fig_sens.add_trace(go.Scatter(
                                            x=sens_df['Uncertainty (%)'],
                                            y=sens_df['Worst-Case Return (%)'],
                                            mode='lines+markers',
                                            name='Worst-Case Return',
                                            line=dict(color='red', width=3)
                                        ))
                                        fig_sens.update_layout(
                                            title="Return Sensitivity to Drift Uncertainty",
                                            xaxis_title="Drift Uncertainty (%)",
                                            yaxis_title="Return (%)",
                                            height=400,
                                            hovermode='x unified'
                                        )
                                        st.plotly_chart(fig_sens, use_container_width=True)
                                        
                                        st.dataframe(sens_df, use_container_width=True)
                                        
                                        # Position sizes
                                        st.markdown("#### 💰 Position Sizes")
                                        position_df = pd.DataFrame({
                                            'Asset': symbols_used,
                                            'Weight (%)': weights_array * 100,
                                            'Dollar Amount ($)': weights_array * initial_wealth,
                                            'Shares (if $100/share)': (weights_array * initial_wealth) / 100
                                        })
                                        st.dataframe(position_df.style.format({
                                            'Weight (%)': '{:.2f}',
                                            'Dollar Amount ($)': '${:,.2f}',
                                            'Shares (if $100/share)': '{:.0f}'
                                        }), use_container_width=True)
                                        
                                        st.success(f"""
                                        **Optimization Complete!**
                                        - Optimized {len(symbols_used)} assets
                                        - Risk aversion: γ = {risk_aversion}
                                        - Drift uncertainty: δ = {drift_uncertainty*100:.1f}%
                                        - Worst-case protection: {(result['expected_return']-result['worst_case_return'])*100:.2f}% buffer
                                        """)
                                        
                                except Exception as e:
                                    st.error(f"Optimization failed: {e}")
                                    import traceback
                                    with st.expander("Error Details"):
                                        st.code(traceback.format_exc())
        
        with drift_tabs[1]:
            st.markdown("### 📉 Optimal Liquidation Strategy")
            st.markdown("""
            Minimize trading costs when liquidating a position over time.
            
            **Method**: Exponential decay liquidation with drift uncertainty  
            **Objective**: Minimize expected cost + risk penalty
            """)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                liq_position = st.number_input(
                    "Position to Liquidate (shares)",
                    min_value=1.0,
                    value=10000.0,
                    step=1000.0,
                    format="%.0f"
                )
            with col2:
                liq_time = st.number_input(
                    "Time Horizon (days)",
                    min_value=1,
                    value=time_horizon_days,
                    step=1
                )
            with col3:
                temp_impact = st.number_input(
                    "Temporary Impact (λ)",
                    min_value=0.0001,
                    value=0.01,
                    step=0.001,
                    format="%.4f",
                    help="Market impact parameter"
                )
            
            if st.button("📊 Compute Liquidation Strategy", type="primary", key="compute_liq"):
                with st.spinner("Optimizing liquidation schedule..."):
                    try:
                        result = pdrift.liquidation_drift_uncertainty(
                            initial_position=liq_position,
                            time_horizon=liq_time,
                            drift_uncertainty=drift_uncertainty,
                            risk_aversion=risk_aversion,
                            temporary_impact=temp_impact,
                            num_steps=num_steps
                        )
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Expected Cost", f"${result['expected_cost']:,.2f}")
                        with col2:
                            st.metric("Worst-Case Cost", f"${result['worst_case_cost']:,.2f}",
                                     delta=f"+${result['worst_case_cost']-result['expected_cost']:,.2f}")
                        
                        # Visualizations
                        fig_liq = make_subplots(
                            rows=2, cols=1,
                            subplot_titles=("Remaining Position Over Time", "Trading Velocity"),
                            vertical_spacing=0.12
                        )
                        
                        # Remaining position
                        fig_liq.add_trace(
                            go.Scatter(
                                x=result['times'],
                                y=result['trading_schedule'],
                                mode='lines',
                                name='Remaining Position',
                                line=dict(color='steelblue', width=3),
                                fill='tozeroy',
                                fillcolor='rgba(70, 130, 180, 0.2)'
                            ),
                            row=1, col=1
                        )
                        
                        # Trading rates
                        fig_liq.add_trace(
                            go.Scatter(
                                x=result['times'],
                                y=result['trading_rates'],
                                mode='lines',
                                name='Trading Rate',
                                line=dict(color='orangered', width=2),
                                fill='tozeroy',
                                fillcolor='rgba(255, 69, 0, 0.2)'
                            ),
                            row=2, col=1
                        )
                        
                        fig_liq.update_xaxes(title_text="Time (days)", row=2, col=1)
                        fig_liq.update_yaxes(title_text="Shares", row=1, col=1)
                        fig_liq.update_yaxes(title_text="Shares/day", row=2, col=1)
                        fig_liq.update_layout(height=700, showlegend=False, hovermode='x unified')
                        
                        st.plotly_chart(fig_liq, use_container_width=True)
                        
                        # Summary statistics
                        st.markdown("#### 📋 Liquidation Summary")
                        summary_cols = st.columns(4)
                        with summary_cols[0]:
                            st.metric("Total Shares", f"{liq_position:,.0f}")
                        with summary_cols[1]:
                            avg_daily = liq_position / liq_time
                            st.metric("Avg Daily Volume", f"{avg_daily:,.0f}")
                        with summary_cols[2]:
                            peak_rate = max(result['trading_rates'])
                            st.metric("Peak Trading Rate", f"{peak_rate:,.0f}")
                        with summary_cols[3]:
                            cost_per_share = result['expected_cost'] / liq_position
                            st.metric("Cost per Share", f"${cost_per_share:.4f}")
                        
                        st.success(f"""
                        **Liquidation Strategy Computed!**
                        - Uses exponential decay schedule
                        - Minimizes market impact while managing drift risk
                        - Total expected cost: ${result['expected_cost']:,.2f}
                        - Worst-case buffer: ${result['worst_case_cost']-result['expected_cost']:,.2f}
                        """)
                        
                    except Exception as e:
                        st.error(f"Liquidation computation failed: {e}")
                        import traceback
                        with st.expander("Error Details"):
                            st.code(traceback.format_exc())
        
        with drift_tabs[2]:
            st.markdown("### 🔄 Portfolio Transition Strategy")
            st.markdown("""
            Optimally rebalance from current portfolio to target portfolio over time.
            
            **Method**: Minimize transition costs with transaction cost consideration  
            **Output**: Optimal weight trajectory and trading velocities
            """)
            
            if 'historical_data' not in st.session_state or st.session_state.historical_data is None:
                st.warning("⚠️ Please load data first from the Data Loader page")
            else:
                data = st.session_state.historical_data
                symbols = _get_symbols(data)
                
                if len(symbols) >= 2:
                    transition_assets = st.multiselect(
                        "Select Assets for Transition",
                        symbols,
                        default=symbols[:min(3, len(symbols))],
                        key="transition_assets"
                    )
                    
                    if len(transition_assets) >= 2:
                        col1, col2 = st.columns(2)
                        
                        # Current weights
                        with col1:
                            st.markdown("**Current Weights (%)**")
                            current_weights = []
                            equal_weight = 100.0 / len(transition_assets)
                            for i, asset in enumerate(transition_assets):
                                w = st.number_input(
                                    asset,
                                    key=f"cur_{asset}_{i}",
                                    min_value=0.0,
                                    max_value=100.0,
                                    value=equal_weight,
                                    step=5.0,
                                    format="%.1f"
                                )
                                current_weights.append(w / 100.0)
                        
                        # Target weights
                        with col2:
                            st.markdown("**Target Weights (%)**")
                            target_weights = []
                            for i, asset in enumerate(transition_assets):
                                w = st.number_input(
                                    asset,
                                    key=f"tgt_{asset}_{i}",
                                    min_value=0.0,
                                    max_value=100.0,
                                    value=equal_weight,
                                    step=5.0,
                                    format="%.1f"
                                )
                                target_weights.append(w / 100.0)
                        
                        # Validation
                        cur_sum = sum(current_weights)
                        tgt_sum = sum(target_weights)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Current Sum", f"{cur_sum*100:.1f}%")
                        with col2:
                            st.metric("Target Sum", f"{tgt_sum*100:.1f}%")
                        with col3:
                            max_change = max([abs(t-c) for t, c in zip(target_weights, current_weights)])
                            st.metric("Max Weight Change", f"{max_change*100:.1f}%")
                        
                        if abs(cur_sum - 1.0) > 0.01 or abs(tgt_sum - 1.0) > 0.01:
                            st.warning("⚠️ Weights should sum to 100% (1.0)")
                        else:
                            if st.button("🔄 Compute Transition Path", type="primary", key="compute_trans"):
                                with st.spinner("Computing optimal transition..."):
                                    try:
                                        # Get covariance
                                        _, cov, _ = _compute_statistics(data, transition_assets)
                                        
                                        if cov is None:
                                            st.error("Failed to compute covariance matrix")
                                        else:
                                            result = pdrift.transition_drift_uncertainty(
                                                initial_weights=current_weights,
                                                target_weights=target_weights,
                                                cov=cov,
                                                time_horizon=time_horizon_days,
                                                risk_aversion=risk_aversion,
                                                drift_uncertainty=drift_uncertainty,
                                                transaction_cost=transaction_cost_bps / 10000.0,
                                                num_steps=num_steps
                                            )
                                            
                                            col1, col2 = st.columns(2)
                                            with col1:
                                                st.metric("Expected Transition Cost", 
                                                         f"${result['expected_cost']:,.2f}")
                                            with col2:
                                                st.metric("Worst-Case Cost", 
                                                         f"${result['worst_case_cost']:,.2f}")
                                            
                                            # Trajectory visualization
                                            trajectory = np.array(result['trajectory'])
                                            times = result['times']
                                            
                                            fig_traj = go.Figure()
                                            colors = ['steelblue', 'orangered', 'green', 'purple', 'brown']
                                            
                                            for i, asset in enumerate(transition_assets):
                                                fig_traj.add_trace(go.Scatter(
                                                    x=times,
                                                    y=trajectory[:, i] * 100,
                                                    mode='lines',
                                                    name=asset,
                                                    line=dict(width=3, color=colors[i % len(colors)])
                                                ))
                                            
                                            fig_traj.update_layout(
                                                title="Portfolio Weight Trajectory",
                                                xaxis_title="Time (days)",
                                                yaxis_title="Weight (%)",
                                                height=500,
                                                hovermode='x unified'
                                            )
                                            st.plotly_chart(fig_traj, use_container_width=True)
                                            
                                            # Trading rates
                                            rates = np.array(result['trading_rates'])
                                            fig_rates = go.Figure()
                                            
                                            for i, asset in enumerate(transition_assets):
                                                fig_rates.add_trace(go.Scatter(
                                                    x=times,
                                                    y=rates[:, i],
                                                    mode='lines',
                                                    name=asset,
                                                    fill='tozeroy',
                                                    line=dict(color=colors[i % len(colors)])
                                                ))
                                            
                                            fig_rates.update_layout(
                                                title="Trading Velocity by Asset",
                                                xaxis_title="Time (days)",
                                                yaxis_title="Weight Change Rate",
                                                height=400,
                                                hovermode='x unified'
                                            )
                                            st.plotly_chart(fig_rates, use_container_width=True)
                                            
                                            # Summary table
                                            st.markdown("#### 📊 Transition Summary")
                                            transition_summary = pd.DataFrame({
                                                'Asset': transition_assets,
                                                'Current (%)': [c*100 for c in current_weights],
                                                'Target (%)': [t*100 for t in target_weights],
                                                'Change (%)': [(t-c)*100 for t, c in zip(target_weights, current_weights)],
                                                'Final (%)': [trajectory[-1, i]*100 for i in range(len(transition_assets))]
                                            })
                                            st.dataframe(transition_summary.style.format({
                                                'Current (%)': '{:.2f}',
                                                'Target (%)': '{:.2f}',
                                                'Change (%)': '{:+.2f}',
                                                'Final (%)': '{:.2f}'
                                            }), use_container_width=True)
                                            
                                            st.success(f"""
                                            **Transition Complete!**
                                            - Rebalancing over {time_horizon_days} days
                                            - Transaction cost: {transaction_cost_bps} bps
                                            - Total cost: ${result['expected_cost']:,.2f}
                                            """)
                                            
                                    except Exception as e:
                                        st.error(f"Transition computation failed: {e}")
                                        import traceback
                                        with st.expander("Error Details"):
                                            st.code(traceback.format_exc())
        
        with drift_tabs[3]:
            st.markdown("### ⚠️ Risk Measures under Drift Uncertainty")
            st.markdown("""
            Compute **VaR** (Value at Risk) and **CVaR** (Expected Shortfall) with drift uncertainty.
            
            **VaR**: Maximum loss at confidence level α  
            **CVaR**: Average loss in worst (1-α) scenarios
            """)
            
            if 'historical_data' not in st.session_state or st.session_state.historical_data is None:
                st.warning("⚠️ Please load data first from the Data Loader page")
            else:
                data = st.session_state.historical_data
                symbols = _get_symbols(data)
                
                if len(symbols) >= 2:
                    risk_assets = st.multiselect(
                        "Select Assets for Risk Analysis",
                        symbols,
                        default=symbols[:min(4, len(symbols))],
                        key="risk_assets"
                    )
                    
                    if len(risk_assets) >= 2:
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            risk_weights_input = st.text_input(
                                "Weights (comma-separated)",
                                value=",".join([f"{1.0/len(risk_assets):.3f}"] * len(risk_assets)),
                                help="Enter weights summing to 1.0"
                            )
                        with col2:
                            var_alpha = st.slider(
                                "Confidence Level",
                                0.90, 0.99, 0.95, 0.01,
                                help="VaR/CVaR confidence level"
                            )
                        with col3:
                            risk_horizon = st.number_input(
                                "Time Horizon (days)",
                                1, 30, 1,
                                help="Risk measurement period"
                            )
                        
                        if st.button("📊 Compute Risk Measures", type="primary", key="compute_risk"):
                            with st.spinner("Calculating risk metrics..."):
                                try:
                                    # Parse weights
                                    risk_weights = [float(x.strip()) for x in risk_weights_input.split(",")]
                                    
                                    if len(risk_weights) != len(risk_assets):
                                        st.error(f"Need {len(risk_assets)} weights, got {len(risk_weights)}")
                                    elif abs(sum(risk_weights) - 1.0) > 0.01:
                                        st.error(f"Weights must sum to 1.0, got {sum(risk_weights):.3f}")
                                    else:
                                        # Compute statistics
                                        mu, cov, _ = _compute_statistics(data, risk_assets)
                                        
                                        if mu is None or cov is None:
                                            st.error("Failed to compute statistics")
                                        else:
                                            # Compute VaR and CVaR
                                            var_result = pdrift.var_drift_uncertainty(
                                                mu=mu,
                                                cov=cov,
                                                weights=risk_weights,
                                                time_horizon=risk_horizon,
                                                confidence_level=var_alpha,
                                                drift_uncertainty=drift_uncertainty
                                            )
                                            
                                            cvar_result = pdrift.cvar_drift_uncertainty(
                                                mu=mu,
                                                cov=cov,
                                                weights=risk_weights,
                                                time_horizon=risk_horizon,
                                                confidence_level=var_alpha,
                                                drift_uncertainty=drift_uncertainty
                                            )
                                            
                                            # Display main metrics
                                            col1, col2 = st.columns(2)
                                            with col1:
                                                st.metric(
                                                    f"Value at Risk ({var_alpha*100:.0f}%)",
                                                    f"{var_result*100:.2f}%",
                                                    help=f"Maximum loss at {var_alpha*100:.0f}% confidence"
                                                )
                                            with col2:
                                                st.metric(
                                                    f"CVaR / Expected Shortfall ({var_alpha*100:.0f}%)",
                                                    f"{cvar_result*100:.2f}%",
                                                    delta=f"{(cvar_result-var_result)*100:.2f}%",
                                                    help="Average loss beyond VaR threshold"
                                                )
                                            
                                            # Comparison across confidence levels
                                            st.markdown("#### 📈 Risk Metrics Across Confidence Levels")
                                            conf_levels = [0.90, 0.95, 0.975, 0.99, 0.995]
                                            risk_comparison = []
                                            
                                            start_time = time.time()
                                            progress = st.progress(0)
                                            time_text = st.empty()
                                            for i, alpha in enumerate(conf_levels):
                                                var_val = pdrift.var_drift_uncertainty(
                                                    mu=mu, cov=cov, weights=risk_weights,
                                                    time_horizon=risk_horizon,
                                                    confidence_level=alpha,
                                                    drift_uncertainty=drift_uncertainty
                                                )
                                                cvar_val = pdrift.cvar_drift_uncertainty(
                                                    mu=mu, cov=cov, weights=risk_weights,
                                                    time_horizon=risk_horizon,
                                                    confidence_level=alpha,
                                                    drift_uncertainty=drift_uncertainty
                                                )
                                                risk_comparison.append({
                                                    'Confidence': f"{alpha*100:.1f}%",
                                                    'VaR (%)': var_val * 100,
                                                    'CVaR (%)': cvar_val * 100,
                                                    'CVaR - VaR (%)': (cvar_val - var_val) * 100
                                                })
                                                progress_pct = (i + 1) / len(conf_levels)
                                                progress.progress(progress_pct)
                                                time_text.text(f"⏱️ Estimated time remaining: {estimate_remaining_time(start_time, (i + 1), len(conf_levels))}")
                                            progress.empty()
                                            time_text.empty()
                                            
                                            risk_df = pd.DataFrame(risk_comparison)
                                            
                                            # Visualization
                                            fig_risk = go.Figure()
                                            fig_risk.add_trace(go.Bar(
                                                x=risk_df['Confidence'],
                                                y=risk_df['VaR (%)'],
                                                name='VaR',
                                                marker_color='steelblue'
                                            ))
                                            fig_risk.add_trace(go.Bar(
                                                x=risk_df['Confidence'],
                                                y=risk_df['CVaR (%)'],
                                                name='CVaR',
                                                marker_color='orangered'
                                            ))
                                            fig_risk.update_layout(
                                                title="Risk Measures by Confidence Level",
                                                xaxis_title="Confidence Level",
                                                yaxis_title="Loss (%)",
                                                barmode='group',
                                                height=400,
                                                hovermode='x unified'
                                            )
                                            st.plotly_chart(fig_risk, use_container_width=True)
                                            
                                            st.dataframe(risk_df.style.format({
                                                'VaR (%)': '{:.3f}',
                                                'CVaR (%)': '{:.3f}',
                                                'CVaR - VaR (%)': '{:.3f}'
                                            }), use_container_width=True)
                                            
                                            # Portfolio details
                                            st.markdown("#### 📋 Portfolio Composition")
                                            portfolio_detail = pd.DataFrame({
                                                'Asset': risk_assets,
                                                'Weight (%)': [w*100 for w in risk_weights],
                                                'Expected Return (%)': [m*100 for m in mu]
                                            })
                                            st.dataframe(portfolio_detail.style.format({
                                                'Weight (%)': '{:.2f}',
                                                'Expected Return (%)': '{:.3f}'
                                            }), use_container_width=True)
                                            
                                            st.info(f"""
                                            **Risk Analysis Summary:**
                                            - Portfolio: {len(risk_assets)} assets
                                            - Time horizon: {risk_horizon} day(s)
                                            - Drift uncertainty: {drift_uncertainty*100:.1f}%
                                            - CVaR represents expected loss in worst {(1-var_alpha)*100:.1f}% of scenarios
                                            - Current settings show {var_alpha*100:.0f}% confidence that losses won't exceed {var_result*100:.2f}%
                                            """)
                                            
                                except ValueError as e:
                                    st.error(f"Invalid input: {e}")
                                except Exception as e:
                                    st.error(f"Risk computation failed: {e}")
                                    import traceback
                                    with st.expander("Error Details"):
                                        st.code(traceback.format_exc())

elif optimization_mode == "Regime Switching Jump Diffusion":
    # MRSJD Mode
    if not REGIME_PORTFOLIO_AVAILABLE:
        st.error("⚠️ **Regime Portfolio Module Not Available**")
        st.markdown("""
        The Markov Regime Switching Jump Diffusion module requires Rust bindings to be built.
        
        **To enable this feature:**
        1. Ensure `optimiz-r` library is built: `cd optimiz-r && cargo build --release --no-default-features`
        2. Build bindings: `cd rust-arblab/rust_python_bindings && maturin develop --release`
        3. Restart Streamlit
        
        **For now**, you can explore the implementation in the Jupyter notebook:
        `examples/notebooks/regime_switching_jump_diffusion.ipynb`
        """)
    else:
        st.markdown("""
        ### 📊 Markov Regime Switching Jump Diffusion Portfolio Optimization
        
        Advanced portfolio optimization combining:
        - **Regime Switching**: Dynamic adaptation to market states (Bull/Normal/Bear)
        - **Jump Processes**: Capture tail risk and rare events (crashes, rallies)
        - **Optimal Control**: Hamilton-Jacobi-Bellman equation solver
        
        **Use Case**: Institutional investors needing robust strategies that adapt to changing market conditions.
        """)
        
        mrsjd_tabs = st.tabs([
            "🎯 Optimal Policy",
            "📈 Simulation & Backtest",
            "⚙️ Calibration"
        ])
        
        with mrsjd_tabs[0]:
            st.markdown("#### Compute Optimal Portfolio Allocation")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("**Model Parameters:**")
                risk_free_rate = st.number_input("Risk-Free Rate (%)", 0.0, 10.0, 2.0, 0.5) / 100
                risk_aversion = st.slider("Risk Aversion (γ)", 1.0, 10.0, 2.0, 0.5,
                                         help="Higher values = more conservative")
                time_horizon = st.number_input("Time Horizon (years)", 0.25, 5.0, 1.0, 0.25)
            
            with col2:
                st.markdown("**Numerical Settings:**")
                n_wealth_points = st.selectbox("Grid Points", [50, 100, 200], index=1,
                                              help="More points = higher accuracy but slower")
                max_iterations = st.number_input("Max Iterations", 100, 2000, 500, 100)
                tolerance = st.selectbox("Convergence Tolerance", [1e-4, 1e-5, 1e-6], index=1)
            
            st.markdown("---")
            st.markdown("**Regime Parameters:**")
            
            regime_cols = st.columns(3)
            
            with regime_cols[0]:
                st.markdown("**🟢 Bull Regime**")
                bull_drift = st.number_input("Drift (μ)", 0.0, 0.5, 0.15, 0.01, key="bull_drift")
                bull_vol = st.number_input("Volatility (σ)", 0.05, 0.5, 0.15, 0.01, key="bull_vol")
                bull_lambda = st.number_input("Jump Intensity (λ)", 0.0, 5.0, 0.5, 0.1, key="bull_lambda")
            
            with regime_cols[1]:
                st.markdown("**🔵 Normal Regime**")
                normal_drift = st.number_input("Drift (μ)", 0.0, 0.5, 0.08, 0.01, key="normal_drift")
                normal_vol = st.number_input("Volatility (σ)", 0.05, 0.5, 0.20, 0.01, key="normal_vol")
                normal_lambda = st.number_input("Jump Intensity (λ)", 0.0, 5.0, 1.0, 0.1, key="normal_lambda")
            
            with regime_cols[2]:
                st.markdown("**🔴 Bear Regime**")
                bear_drift = st.number_input("Drift (μ)", -0.3, 0.2, -0.05, 0.01, key="bear_drift")
                bear_vol = st.number_input("Volatility (σ)", 0.05, 0.8, 0.30, 0.01, key="bear_vol")
                bear_lambda = st.number_input("Jump Intensity (λ)", 0.0, 10.0, 2.0, 0.1, key="bear_lambda")
            
            st.markdown("---")
            st.markdown("**Transition Rates (Q Matrix):**")
            st.caption("Probability of switching regimes per unit time")
            
            q_cols = st.columns(3)
            with q_cols[0]:
                q01 = st.number_input("Bull → Normal", 0.0, 5.0, 0.5, 0.1)
                q02 = st.number_input("Bull → Bear", 0.0, 5.0, 0.1, 0.05)
            with q_cols[1]:
                q10 = st.number_input("Normal → Bull", 0.0, 5.0, 0.3, 0.1)
                q12 = st.number_input("Normal → Bear", 0.0, 5.0, 0.3, 0.1)
            with q_cols[2]:
                q20 = st.number_input("Bear → Bull", 0.0, 5.0, 0.1, 0.05)
                q21 = st.number_input("Bear → Normal", 0.0, 5.0, 0.5, 0.1)
            
            if st.button("🚀 Solve HJB Equation", type="primary", use_container_width=True):
                with st.spinner("Solving Hamilton-Jacobi-Bellman equations..."):
                    try:
                        # Create configuration
                        # Build transition rate matrix
                        Q = np.array([
                            [-(q01+q02), q01, q02],
                            [q10, -(q10+q12), q12],
                            [q20, q21, -(q20+q21)]
                        ])
                        
                        # Note: This is a placeholder - actual API may differ
                        # Need to check the exact Python API from regime_portfolio_bindings.rs
                        st.info("HJB solver integration in progress. See notebook for working example.")
                        
                        st.code(f"""
# Example usage (from notebook):
import hft_py.regime_portfolio as rp

config = rp.calibrate_model_from_data(returns, regimes)
optimizer = rp.PyRegimeSwitchingPortfolio(config)
result = optimizer.optimize()

# Access results:
wealth_grid = result.wealth_grid
value_functions = result.value_functions  # Per regime
optimal_policies = result.optimal_policies  # Per regime
""", language="python")
                        
                    except Exception as e:
                        st.error(f"Optimization failed: {e}")
                        import traceback
                        with st.expander("Error Details"):
                            st.code(traceback.format_exc())
        
        with mrsjd_tabs[1]:
            st.markdown("#### Monte Carlo Simulation & Backtest")
            st.info("🚧 Simulation interface coming soon. See notebook for examples.")
            
            st.markdown("""
            The simulation module allows you to:
            - Generate wealth trajectories under optimal policy
            - Backtest on historical data with regime detection
            - Compare against buy-and-hold and constant-mix strategies
            - Analyze risk metrics (Sharpe, drawdown, VaR/CVaR)
            
            **Current Status**: Available in Jupyter notebook `regime_switching_jump_diffusion.ipynb`
            """)
        
        with mrsjd_tabs[2]:
            st.markdown("#### Calibrate from Market Data")
            
            if 'historical_data' not in st.session_state or st.session_state.historical_data is None:
                st.warning("⚠️ Please load data first from the Data Loader page")
            else:
                data = st.session_state.historical_data
                
                # Get available symbols
                if isinstance(data, dict):
                    symbols = list(data.keys())
                    if len(symbols) > 0:
                        selected_symbol = st.selectbox("Select Symbol for Calibration", symbols)
                        
                        if selected_symbol:
                            symbol_data = data[selected_symbol]
                            if 'close' in symbol_data.columns:
                                # Calculate returns
                                returns = symbol_data['close'].pct_change().dropna()
                                
                                st.markdown(f"**Data Summary for {selected_symbol}:**")
                                metric_cols = st.columns(4)
                                with metric_cols[0]:
                                    st.metric("Observations", len(returns))
                                with metric_cols[1]:
                                    st.metric("Mean Return", f"{returns.mean():.4f}")
                                with metric_cols[2]:
                                    st.metric("Volatility", f"{returns.std():.4f}")
                                with metric_cols[3]:
                                    st.metric("Skewness", f"{returns.skew():.3f}")
                                
                                if st.button("📊 Auto-Calibrate Model", type="primary"):
                                    with st.spinner("Calibrating model parameters..."):
                                        try:
                                            returns_list = returns.tolist()
                                            
                                            # Calibrate without regimes first
                                            config = regime_portfolio.calibrate_model_from_data(returns_list, None)
                                            
                                            st.success("✅ Calibration complete!")
                                            
                                            st.markdown("**Calibrated Parameters:**")
                                            st.json({
                                                "risk_free_rate": config.risk_free_rate,
                                                "risk_aversion": config.risk_aversion,
                                                "time_horizon": config.time_horizon,
                                                "regime_drifts": config.regime_drifts,
                                                "regime_vols": config.regime_vols,
                                                "jump_intensities": config.jump_intensities
                                            })
                                            
                                            st.info("💡 Use these calibrated values in the Optimal Policy tab")
                                            
                                        except Exception as e:
                                            st.error(f"Calibration failed: {e}")
                                            import traceback
                                            with st.expander("Error Details"):
                                                st.code(traceback.format_exc())
                else:
                    st.info("Data format not supported for calibration. Please load symbol data from Data Loader.")

else:
    # Original Multi-Factor Analysis Mode
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Stock Analysis", "🎯 Portfolio Optimization", "📈 Results & Backtest", "🎨 Sparse Factor Analysis"])

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
                                }).map(color_score, subset=['Combined Score'])
                                
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
    
    with tab4:
        st.markdown("### 🎨 Sparse Factor Analysis with PCA")
        st.markdown("Use Sparse PCA to identify the most important latent factors driving your portfolio returns.")
        
        if 'historical_data' not in st.session_state or st.session_state.historical_data is None:
            st.warning("⚠️ Please load data first from the Data Loader page")
        else:
            try:
                # Import sparse_meanrev module
                sys.path.insert(0, str(project_root / 'python'))
                from sparse_meanrev import sparse_pca, compute_risk_metrics
                
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
                
                if len(symbols) < 3:
                    st.warning("⚠️ Need at least 3 assets for sparse PCA analysis")
                else:
                    st.markdown(f"**Available Symbols:** {len(symbols)}")
                    
                    # Configuration
                    col1, col2 = st.columns(2)
                    with col1:
                        n_components = st.slider("Number of Components", 1, min(5, len(symbols)//2), 3)
                        lambda_param = st.slider("Sparsity Parameter (λ)", 0.0, 1.0, 0.2, 0.05,
                                                help="Higher values = more sparse (fewer assets per component)")
                    with col2:
                        max_iter = st.slider("Max Iterations", 100, 2000, 1000, 100)
                        lookback_days = st.slider("Lookback Days", 50, 500, 200, 50)
                    
                    # Select symbols for analysis
                    selected_symbols = st.multiselect(
                        "Select Assets for Sparse PCA",
                        symbols,
                        default=symbols[:min(10, len(symbols))],
                        help="Select 3-20 assets"
                    )
                    
                    if len(selected_symbols) < 3:
                        st.info("Please select at least 3 assets")
                    elif len(selected_symbols) > 20:
                        st.warning("Maximum 20 assets recommended for clear visualization")
                    else:
                        if st.button("🎨 Run Sparse PCA Analysis", type="primary"):
                            with st.spinner("Computing sparse principal components..."):
                                try:
                                    # Build returns matrix
                                    returns_dict = {}
                                    
                                    for symbol in selected_symbols:
                                        if isinstance(data, dict):
                                            df = data[symbol]
                                        elif isinstance(data, pd.DataFrame):
                                            if 'symbol' in data.columns:
                                                df = data[data['symbol'] == symbol].copy()
                                            else:
                                                df = data.copy()
                                        else:
                                            continue
                                        
                                        close_col = None
                                        for col in df.columns:
                                            if col.lower() == 'close':
                                                close_col = col
                                                break
                                        
                                        if close_col is None:
                                            continue
                                        
                                        prices = df[close_col].values[-lookback_days:]
                                        if len(prices) > 10:
                                            returns = np.diff(prices) / prices[:-1]
                                            returns_dict[symbol] = returns
                                    
                                    if len(returns_dict) < 3:
                                        st.error("Not enough valid data for selected symbols")
                                    else:
                                        # Align returns to same length
                                        min_len = min(len(r) for r in returns_dict.values())
                                        returns_df = pd.DataFrame({
                                            sym: ret[-min_len:] for sym, ret in returns_dict.items()
                                        })
                                        
                                        # Run Sparse PCA
                                        st.info(f"Running Sparse PCA on {len(returns_df.columns)} assets over {len(returns_df)} periods...")
                                        result = sparse_pca(returns_df, n_components=n_components, lambda_=lambda_param, max_iter=max_iter)
                                        
                                        # Display results
                                        st.success("✅ Sparse PCA Complete!")
                                        st.write(result.summary())
                                        
                                        # Component weights heatmap
                                        st.markdown("### 📊 Component Loadings (Sparse Weights)")
                                        
                                        weights_df = pd.DataFrame(
                                            result.weights,
                                            columns=returns_df.columns,
                                            index=[f'PC{i+1}' for i in range(n_components)]
                                        )
                                        
                                        fig_heatmap = go.Figure(data=go.Heatmap(
                                            z=weights_df.values,
                                            x=weights_df.columns,
                                            y=weights_df.index,
                                            colorscale='RdBu',
                                            zmid=0,
                                            text=np.round(weights_df.values, 3),
                                            texttemplate='%{text}',
                                            textfont={"size": 10},
                                            colorbar=dict(title="Weight")
                                        ))
                                        
                                        fig_heatmap.update_layout(
                                            title='Sparse Component Loadings',
                                            xaxis_title='Assets',
                                            yaxis_title='Components',
                                            height=300 + n_components * 50
                                        )
                                        
                                        st.plotly_chart(fig_heatmap, use_container_width=True)
                                        
                                        # Variance explained
                                        st.markdown("### 📈 Variance Explained")
                                        
                                        col1, col2 = st.columns(2)
                                        
                                        with col1:
                                            fig_var = go.Figure()
                                            fig_var.add_trace(go.Bar(
                                                x=[f'PC{i+1}' for i in range(n_components)],
                                                y=result.variance_explained * 100,
                                                marker_color='lightblue',
                                                text=[f'{v:.1f}%' for v in result.variance_explained * 100],
                                                textposition='auto'
                                            ))
                                            fig_var.update_layout(
                                                title='Variance Explained by Component',
                                                yaxis_title='Variance (%)',
                                                height=400
                                            )
                                            st.plotly_chart(fig_var, use_container_width=True)
                                        
                                        with col2:
                                            fig_sparsity = go.Figure()
                                            fig_sparsity.add_trace(go.Bar(
                                                x=[f'PC{i+1}' for i in range(n_components)],
                                                y=result.sparsity * 100,
                                                marker_color='lightcoral',
                                                text=[f'{s:.1f}%' for s in result.sparsity * 100],
                                                textposition='auto'
                                            ))
                                            fig_sparsity.update_layout(
                                                title='Sparsity Level by Component',
                                                yaxis_title='Sparsity (%)',
                                                height=400
                                            )
                                            st.plotly_chart(fig_sparsity, use_container_width=True)
                                        
                                        # Component portfolios
                                        st.markdown("### 💼 Component-Based Portfolios")
                                        
                                        for i in range(n_components):
                                            with st.expander(f"📊 PC{i+1} Portfolio (Variance: {result.variance_explained[i]*100:.1f}%)"):
                                                weights = result.get_portfolio(i)
                                                non_zero = weights[weights.abs() > 1e-6].sort_values(key=abs, ascending=False)
                                                
                                                col1, col2 = st.columns([1, 2])
                                                
                                                with col1:
                                                    st.markdown("**Top Holdings:**")
                                                    for asset, weight in non_zero.head(10).items():
                                                        st.write(f"• {asset}: {weight:.4f}")
                                                
                                                with col2:
                                                    # Compute portfolio returns
                                                    portfolio_returns = (returns_df * weights).sum(axis=1)
                                                    
                                                    # Compute risk metrics
                                                    metrics = compute_risk_metrics(portfolio_returns)
                                                    
                                                    st.markdown("**Risk Metrics:**")
                                                    metric_col1, metric_col2 = st.columns(2)
                                                    with metric_col1:
                                                        st.metric("Sharpe Ratio", f"{metrics.sharpe_ratio:.3f}")
                                                        st.metric("Sortino Ratio", f"{metrics.sortino_ratio:.3f}")
                                                        st.metric("Max Drawdown", f"{metrics.max_drawdown*100:.2f}%")
                                                    with metric_col2:
                                                        st.metric("Volatility", f"{metrics.volatility*100:.2f}%")
                                                        st.metric("VaR (95%)", f"{metrics.var_95*100:.3f}%")
                                                        st.metric("Skewness", f"{metrics.skewness:.3f}")
                                        
                                        # Usage recommendations
                                        st.markdown("### 💡 How to Use These Results")
                                        st.info("""
                                        **Sparse PCA helps you:**
                                        1. **Factor Identification**: Each component represents a latent market factor
                                        2. **Portfolio Construction**: Use sparse weights to build factor-based portfolios
                                        3. **Risk Management**: Components with high variance may indicate systematic risks
                                        4. **Asset Selection**: Non-zero weights show which assets are most important for each factor
                                        
                                        **Interpretation Tips:**
                                        - Higher sparsity = cleaner, more interpretable factors
                                        - First component usually captures market-wide movements
                                        - Later components capture sector/style-specific patterns
                                        - Use these factors for strategic asset allocation or hedging
                                        """)
                                        
                                except Exception as e:
                                    st.error(f"Error during sparse PCA analysis: {str(e)}")
                                    import traceback
                                    with st.expander("Show error details"):
                                        st.code(traceback.format_exc())
                                        
            except ImportError as e:
                st.error("Could not import sparse_meanrev module. Please ensure optimizr is installed.")
                st.code(f"pip install -e ./optimiz-r")
                st.info("Or build it locally: `cd optimiz-r && maturin develop --release`")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🎯 Portfolio Optimizer Lab | Advanced Multi-Factor Selection</p>
</div>
""", unsafe_allow_html=True)
