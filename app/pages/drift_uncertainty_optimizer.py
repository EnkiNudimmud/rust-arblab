"""
Drift Uncertainty Portfolio Optimizer
Robust portfolio optimization accounting for uncertain expected returns
Based on Bismuth-Guéant-Pu paper
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import shared UI components
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.ui_components import render_sidebar_navigation, apply_custom_css

# Try to import drift uncertainty module
try:
    import hft_py.portfolio_drift as pdrift
    DRIFT_AVAILABLE = True
except ImportError:
    DRIFT_AVAILABLE = False

st.set_page_config(page_title="Drift Uncertainty Optimizer", page_icon="🛡️", layout="wide")

# Render sidebar navigation and apply CSS
render_sidebar_navigation(current_page="Drift Uncertainty Optimizer")
apply_custom_css()

st.markdown('<h1 class="lab-header">🛡️ Drift Uncertainty Portfolio Optimizer</h1>', unsafe_allow_html=True)
st.markdown("### Robust portfolio optimization under uncertain expected returns")

# Module availability check
if not DRIFT_AVAILABLE:
    st.error("⚠️ **Drift Uncertainty Module Not Available**")
    st.markdown("""
    The drift uncertainty optimization module requires Rust bindings to be built.
    
    **To enable this feature:**
    1. Fix compilation errors in `rust_core/src/orderbook.rs` (see `docs/DRIFT_UNCERTAINTY_IMPLEMENTATION.md`)
    2. Build bindings: `cd rust_python_bindings && maturin develop`
    3. Restart Streamlit
    
    **For now**, you can explore the implementation in the Jupyter notebook:
    📓 `examples/notebooks/portfolio_drift_uncertainty.ipynb`
    
    **Documentation**: See `docs/DRIFT_UNCERTAINTY_IMPLEMENTATION.md` for full details
    """)
    
    with st.expander("📚 About Drift Uncertainty Optimization"):
        st.markdown("""
        ### What is Drift Uncertainty?
        
        Traditional portfolio optimization assumes we know expected returns precisely. 
        In reality, expected returns are **uncertain** - they could lie anywhere in a range.
        
        **Drift uncertainty** models this by assuming expected returns μ lie in [μ̂ - δ, μ̂ + δ], 
        where δ is the uncertainty parameter.
        
        ### Why Use Robust Optimization?
        
        - **Worst-case protection**: Optimizes for the worst-case scenario
        - **Risk management**: Accounts for estimation error in expected returns
        - **Stability**: Less sensitive to parameter misspecification
        
        ### Features Implemented
        
        1. **Robust Portfolio Choice**: Maximize utility in worst-case scenario
        2. **Optimal Liquidation**: Minimize costs when selling a position over time
        3. **Portfolio Transition**: Optimal rebalancing from current to target weights
        4. **Risk Measures**: VaR and CVaR under drift uncertainty
        
        ### Mathematical Framework
        
        Uses **CARA utility**: U(W) = -exp(-γW) where γ is risk aversion.
        
        **Optimization**: max_w min_{μ ∈ [μ̂-δ, μ̂+δ]} E[U(w^T μ - (γ/2)w^T Σ w)]
        
        ### References
        
        **Paper**: Alexis Bismuth, Olivier Guéant, and Jiang Pu. 
        "Portfolio choice, portfolio liquidation, and portfolio transition under drift uncertainty." 
        *SIAM Journal on Financial Mathematics*, 2017.
        """)
    
    st.stop()

# Module is available - show full interface
st.success("✅ **Drift Uncertainty Module Loaded Successfully**")
st.markdown("---")

# Sidebar parameters
with st.sidebar:
    st.markdown("### 🎛️ Optimization Parameters")
    
    risk_aversion = st.slider(
        "Risk Aversion (γ)", 
        0.5, 10.0, 2.0, 0.5,
        help="Higher values = more risk-averse. Typical range: 1-5"
    )
    
    drift_uncertainty = st.slider(
        "Drift Uncertainty (δ)", 
        0.0, 0.15, 0.02, 0.01,
        help="Uncertainty range for expected returns. 0 = no uncertainty, 0.05 = ±5%"
    )
    
    time_horizon = st.slider(
        "Time Horizon (days)", 
        1, 30, 10, 1,
        help="Investment or rebalancing period"
    )
    
    transaction_cost_bps = st.slider(
        "Transaction Cost (bps)", 
        0, 100, 10, 5,
        help="Cost per trade in basis points (1 bps = 0.01%)"
    )
    
    num_steps = st.slider(
        "Time Steps", 
        20, 200, 100, 10,
        help="Number of discretization steps for time-dependent optimization"
    )
    
    st.markdown("---")
    st.markdown("### 📚 Documentation")
    st.markdown("[Implementation Guide](../docs/DRIFT_UNCERTAINTY_IMPLEMENTATION.md)")
    st.markdown("[Jupyter Notebook](../examples/notebooks/portfolio_drift_uncertainty.ipynb)")

# Helper functions
def get_symbols(data):
    """Extract symbols from various data formats"""
    if isinstance(data, dict):
        return list(data.keys())
    elif isinstance(data, pd.DataFrame):
        if 'symbol' in data.columns:
            return data['symbol'].unique().tolist()
    return []

def compute_statistics(data, symbols):
    """Compute mean returns and covariance matrix"""
    try:
        returns_list = []
        symbols_used = []
        
        for symbol in symbols:
            if isinstance(data, dict):
                df = data[symbol]
            elif isinstance(data, pd.DataFrame) and 'symbol' in data.columns:
                df = data[data['symbol'] == symbol].copy()
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

# Main tabs
tabs = st.tabs([
    "🛡️ Robust Portfolio", 
    "📉 Liquidation", 
    "🔄 Portfolio Transition",
    "⚠️ Risk Analysis"
])

# TAB 1: Robust Portfolio Choice
with tabs[0]:
    st.markdown("### 🛡️ Robust Portfolio Choice under Drift Uncertainty")
    
    with st.expander("ℹ️ About This Method"):
        st.markdown("""
        **Objective**: Find portfolio weights that maximize utility in the **worst-case scenario** 
        when expected returns are uncertain.
        
        **Formula**: max_w min_{μ ∈ [μ̂-δ, μ̂+δ]} E[U(w^T μ - (γ/2)w^T Σ w)]
        
        **Utility**: CARA (Constant Absolute Risk Aversion): U(W) = -exp(-γW)
        
        **Output**: Portfolio weights that perform well even if expected returns are misestimated.
        """)
    
    if 'historical_data' not in st.session_state or st.session_state.historical_data is None:
        st.warning("⚠️ Please load data first from the Data Loader page")
        if st.button("💾 Go to Data Loader", key="robust_loader"):
            st.switch_page("pages/data_loader.py")
    else:
        data = st.session_state.historical_data
        symbols = get_symbols(data)
        
        if len(symbols) < 2:
            st.warning("⚠️ Please load at least 2 symbols for portfolio analysis")
        else:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                selected_assets = st.multiselect(
                    "Select Assets for Portfolio",
                    symbols,
                    default=symbols[:min(5, len(symbols))],
                    key="robust_assets",
                    help="Select 2-10 assets for optimization"
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
                            mu, cov, symbols_used = compute_statistics(data, selected_assets)
                            
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
                                    st.metric("Expected Return", f"{result['expected_return']*100:.2f}%")
                                
                                with col2:
                                    st.metric(
                                        "Worst-Case Return", 
                                        f"{result['worst_case_return']*100:.2f}%",
                                        delta=f"{(result['worst_case_return']-result['expected_return'])*100:.2f}%"
                                    )
                                
                                with col3:
                                    st.metric("Portfolio Variance", f"{result['variance']:.4f}")
                                
                                with col4:
                                    st.metric("Utility Value", f"{result['utility']:.4f}")
                                
                                # Weights visualization
                                weights_array = np.array(result['weights'])
                                
                                fig_weights = go.Figure(data=[
                                    go.Bar(
                                        x=symbols_used,
                                        y=weights_array * 100,
                                        marker_color='steelblue',
                                        text=[f"{w*100:.1f}%" for w in weights_array],
                                        textposition='auto',
                                        hovertemplate='<b>%{x}</b><br>Weight: %{y:.2f}%<extra></extra>'
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
                                
                                progress_bar = st.progress(0)
                                
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
                                    progress_bar.progress((i + 1) / len(delta_levels))
                                
                                progress_bar.empty()
                                
                                sens_df = pd.DataFrame(sensitivity_data)
                                
                                # Plot sensitivity
                                fig_sens = go.Figure()
                                
                                fig_sens.add_trace(go.Scatter(
                                    x=sens_df['Uncertainty (%)'],
                                    y=sens_df['Expected Return (%)'],
                                    mode='lines+markers',
                                    name='Expected Return',
                                    line=dict(color='green', width=3),
                                    marker=dict(size=8)
                                ))
                                
                                fig_sens.add_trace(go.Scatter(
                                    x=sens_df['Uncertainty (%)'],
                                    y=sens_df['Worst-Case Return (%)'],
                                    mode='lines+markers',
                                    name='Worst-Case Return',
                                    line=dict(color='red', width=3),
                                    marker=dict(size=8)
                                ))
                                
                                fig_sens.update_layout(
                                    title="Return Sensitivity to Drift Uncertainty",
                                    xaxis_title="Drift Uncertainty (%)",
                                    yaxis_title="Return (%)",
                                    height=400,
                                    hovermode='x unified'
                                )
                                
                                st.plotly_chart(fig_sens, use_container_width=True)
                                
                                st.dataframe(
                                    sens_df.style.format({
                                        'Expected Return (%)': '{:.3f}',
                                        'Worst-Case Return (%)': '{:.3f}',
                                        'Utility': '{:.4f}'
                                    }),
                                    use_container_width=True
                                )
                                
                                # Position sizes
                                st.markdown("#### 💰 Position Sizes")
                                
                                position_df = pd.DataFrame({
                                    'Asset': symbols_used,
                                    'Weight (%)': weights_array * 100,
                                    'Dollar Amount ($)': weights_array * initial_wealth,
                                    'Shares (if $100/share)': (weights_array * initial_wealth) / 100
                                })
                                
                                st.dataframe(
                                    position_df.style.format({
                                        'Weight (%)': '{:.2f}',
                                        'Dollar Amount ($)': '${:,.2f}',
                                        'Shares (if $100/share)': '{:.0f}'
                                    }),
                                    use_container_width=True
                                )
                                
                                st.success(f"""
                                **✅ Optimization Complete!**
                                - Optimized {len(symbols_used)} assets
                                - Risk aversion: γ = {risk_aversion}
                                - Drift uncertainty: δ = {drift_uncertainty*100:.1f}%
                                - Worst-case protection: {(result['expected_return']-result['worst_case_return'])*100:.2f}% buffer
                                """)
                                
                        except Exception as e:
                            st.error(f"❌ Optimization failed: {e}")
                            import traceback
                            with st.expander("Error Details"):
                                st.code(traceback.format_exc())

# TAB 2: Optimal Liquidation
with tabs[1]:
    st.markdown("### 📉 Optimal Liquidation Strategy")
    
    with st.expander("ℹ️ About This Method"):
        st.markdown("""
        **Objective**: Minimize trading costs when liquidating a position over time.
        
        **Method**: Exponential decay liquidation with drift uncertainty
        
        **Balances**:
        - **Market impact**: Selling too fast increases price impact
        - **Drift risk**: Selling too slow exposes to uncertain price drift
        
        **Output**: Optimal trading schedule showing how many shares to sell at each time step.
        """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        liq_position = st.number_input(
            "Position to Liquidate (shares)",
            min_value=1.0,
            value=10000.0,
            step=1000.0,
            format="%.0f",
            help="Total number of shares to sell"
        )
    
    with col2:
        liq_time = st.number_input(
            "Time Horizon (days)",
            min_value=1,
            value=time_horizon,
            step=1,
            help="Period over which to liquidate"
        )
    
    with col3:
        temp_impact = st.number_input(
            "Temporary Impact (λ)",
            min_value=0.0001,
            value=0.01,
            step=0.001,
            format="%.4f",
            help="Market impact parameter (higher = more impact)"
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
                    st.metric(
                        "Worst-Case Cost", 
                        f"${result['worst_case_cost']:,.2f}",
                        delta=f"+${result['worst_case_cost']-result['expected_cost']:,.2f}"
                    )
                
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
                        fillcolor='rgba(70, 130, 180, 0.2)',
                        hovertemplate='Day %{x:.1f}<br>Remaining: %{y:,.0f} shares<extra></extra>'
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
                        fillcolor='rgba(255, 69, 0, 0.2)',
                        hovertemplate='Day %{x:.1f}<br>Rate: %{y:,.0f} shares/day<extra></extra>'
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
                **✅ Liquidation Strategy Computed!**
                - Uses exponential decay schedule
                - Minimizes market impact while managing drift risk
                - Total expected cost: ${result['expected_cost']:,.2f}
                - Worst-case buffer: ${result['worst_case_cost']-result['expected_cost']:,.2f}
                """)
                
            except Exception as e:
                st.error(f"❌ Liquidation computation failed: {e}")
                import traceback
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())

# TAB 3: Portfolio Transition
with tabs[2]:
    st.markdown("### 🔄 Portfolio Transition Strategy")
    
    with st.expander("ℹ️ About This Method"):
        st.markdown("""
        **Objective**: Optimally rebalance from current portfolio to target portfolio.
        
        **Method**: Minimize transition costs with transaction cost consideration
        
        **Accounts for**:
        - Transaction costs (bid-ask spread, fees)
        - Drift uncertainty during transition
        - Market impact
        
        **Output**: Optimal weight trajectory showing how portfolio weights should evolve over time.
        """)
    
    st.info("💡 **Tip**: Enter current and target weights for each asset. Weights should sum to 100%.")
    
    if 'historical_data' not in st.session_state or st.session_state.historical_data is None:
        st.warning("⚠️ Please load data first from the Data Loader page")
    else:
        data = st.session_state.historical_data
        symbols = get_symbols(data)
        
        if len(symbols) >= 2:
            transition_assets = st.multiselect(
                "Select Assets for Transition",
                symbols,
                default=symbols[:min(3, len(symbols))],
                key="transition_assets",
                help="Select 2-5 assets for rebalancing"
            )
            
            if len(transition_assets) >= 2:
                col1, col2 = st.columns(2)
                
                equal_weight = 100.0 / len(transition_assets)
                current_weights = []
                target_weights = []
                
                # Current weights
                with col1:
                    st.markdown("**Current Weights (%)**")
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
                                _, cov, _ = compute_statistics(data, transition_assets)
                                
                                if cov is None:
                                    st.error("Failed to compute covariance matrix")
                                else:
                                    result = pdrift.transition_drift_uncertainty(
                                        initial_weights=current_weights,
                                        target_weights=target_weights,
                                        cov=cov,
                                        time_horizon=time_horizon,
                                        risk_aversion=risk_aversion,
                                        drift_uncertainty=drift_uncertainty,
                                        transaction_cost=transaction_cost_bps / 10000.0,
                                        num_steps=num_steps
                                    )
                                    
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.metric("Expected Transition Cost", f"${result['expected_cost']:,.2f}")
                                    
                                    with col2:
                                        st.metric("Worst-Case Cost", f"${result['worst_case_cost']:,.2f}")
                                    
                                    # Trajectory visualization
                                    trajectory = np.array(result['trajectory'])
                                    times = result['times']
                                    
                                    fig_traj = go.Figure()
                                    colors = ['steelblue', 'orangered', 'green', 'purple', 'brown', 'pink', 'gray']
                                    
                                    for i, asset in enumerate(transition_assets):
                                        fig_traj.add_trace(go.Scatter(
                                            x=times,
                                            y=trajectory[:, i] * 100,
                                            mode='lines',
                                            name=asset,
                                            line=dict(width=3, color=colors[i % len(colors)]),
                                            hovertemplate=f'<b>{asset}</b><br>Day %{{x:.1f}}<br>Weight: %{{y:.2f}}%<extra></extra>'
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
                                    
                                    st.dataframe(
                                        transition_summary.style.format({
                                            'Current (%)': '{:.2f}',
                                            'Target (%)': '{:.2f}',
                                            'Change (%)': '{:+.2f}',
                                            'Final (%)': '{:.2f}'
                                        }),
                                        use_container_width=True
                                    )
                                    
                                    st.success(f"""
                                    **✅ Transition Complete!**
                                    - Rebalancing over {time_horizon} days
                                    - Transaction cost: {transaction_cost_bps} bps
                                    - Total cost: ${result['expected_cost']:,.2f}
                                    """)
                                    
                            except Exception as e:
                                st.error(f"❌ Transition computation failed: {e}")
                                import traceback
                                with st.expander("Error Details"):
                                    st.code(traceback.format_exc())

# TAB 4: Risk Analysis
with tabs[3]:
    st.markdown("### ⚠️ Risk Measures under Drift Uncertainty")
    
    with st.expander("ℹ️ About These Metrics"):
        st.markdown("""
        **VaR (Value at Risk)**: Maximum loss at a given confidence level
        - Example: 95% VaR = 5% means there's a 5% chance of losing more than 5%
        
        **CVaR (Conditional Value at Risk / Expected Shortfall)**: Average loss beyond VaR
        - Example: If 95% VaR = 5%, CVaR might be 7%, meaning average loss in worst 5% of cases is 7%
        
        **Why with drift uncertainty?**
        - Traditional VaR/CVaR assume we know expected returns precisely
        - With drift uncertainty, we compute **worst-case** risk measures
        - More conservative and robust to model uncertainty
        """)
    
    if 'historical_data' not in st.session_state or st.session_state.historical_data is None:
        st.warning("⚠️ Please load data first from the Data Loader page")
    else:
        data = st.session_state.historical_data
        symbols = get_symbols(data)
        
        if len(symbols) >= 2:
            risk_assets = st.multiselect(
                "Select Assets for Risk Analysis",
                symbols,
                default=symbols[:min(4, len(symbols))],
                key="risk_assets",
                help="Select 2-10 assets"
            )
            
            if len(risk_assets) >= 2:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    risk_weights_input = st.text_input(
                        "Weights (comma-separated)",
                        value=",".join([f"{1.0/len(risk_assets):.3f}"] * len(risk_assets)),
                        help="Enter weights summing to 1.0, e.g., 0.4,0.3,0.3"
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
                                st.error(f"❌ Need {len(risk_assets)} weights, got {len(risk_weights)}")
                            elif abs(sum(risk_weights) - 1.0) > 0.01:
                                st.error(f"❌ Weights must sum to 1.0, got {sum(risk_weights):.3f}")
                            else:
                                mu, cov, _ = compute_statistics(data, risk_assets)
                                
                                if mu is None or cov is None:
                                    st.error("Failed to compute statistics")
                                else:
                                    # Compute VaR and CVaR
                                    var_result = pdrift.var_drift_uncertainty(
                                        mu=mu, cov=cov, weights=risk_weights,
                                        time_horizon=risk_horizon,
                                        confidence_level=var_alpha,
                                        drift_uncertainty=drift_uncertainty
                                    )
                                    
                                    cvar_result = pdrift.cvar_drift_uncertainty(
                                        mu=mu, cov=cov, weights=risk_weights,
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
                                    
                                    progress = st.progress(0)
                                    
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
                                        progress.progress((i + 1) / len(conf_levels))
                                    
                                    progress.empty()
                                    
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
                                    
                                    st.dataframe(
                                        risk_df.style.format({
                                            'VaR (%)': '{:.3f}',
                                            'CVaR (%)': '{:.3f}',
                                            'CVaR - VaR (%)': '{:.3f}'
                                        }),
                                        use_container_width=True
                                    )
                                    
                                    # Portfolio details
                                    st.markdown("#### 📋 Portfolio Composition")
                                    
                                    portfolio_detail = pd.DataFrame({
                                        'Asset': risk_assets,
                                        'Weight (%)': [w*100 for w in risk_weights],
                                        'Expected Return (%)': [m*100 for m in mu]
                                    })
                                    
                                    st.dataframe(
                                        portfolio_detail.style.format({
                                            'Weight (%)': '{:.2f}',
                                            'Expected Return (%)': '{:.3f}'
                                        }),
                                        use_container_width=True
                                    )
                                    
                                    st.info(f"""
                                    **Risk Analysis Summary:**
                                    - Portfolio: {len(risk_assets)} assets
                                    - Time horizon: {risk_horizon} day(s)
                                    - Drift uncertainty: {drift_uncertainty*100:.1f}%
                                    - CVaR represents expected loss in worst {(1-var_alpha)*100:.1f}% of scenarios
                                    - Current settings show {var_alpha*100:.0f}% confidence that losses won't exceed {var_result*100:.2f}%
                                    """)
                                    
                        except ValueError as e:
                            st.error(f"❌ Invalid input: {e}")
                        except Exception as e:
                            st.error(f"❌ Risk computation failed: {e}")
                            import traceback
                            with st.expander("Error Details"):
                                st.code(traceback.format_exc())

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🛡️ Drift Uncertainty Portfolio Optimizer | Based on Bismuth-Guéant-Pu (2017)</p>
    <p style='font-size: 0.9em;'>Implementation: <code>rust_core/src/portfolio_drift_uncertainty.rs</code></p>
</div>
""", unsafe_allow_html=True)
