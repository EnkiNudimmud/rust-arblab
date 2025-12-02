"""
Chiarella Model Lab
Agent-based market dynamics with chartist-fundamentalist interaction
"""

import streamlit as st
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

st.set_page_config(page_title="Chiarella Model Lab", page_icon="🌀", layout="wide")

# Render sidebar navigation and apply CSS
render_sidebar_navigation(current_page="Chiarella Model Lab")
apply_custom_css()

st.markdown('<h1 class="lab-header">🌀 Chiarella Model Lab</h1>', unsafe_allow_html=True)
st.markdown("### Agent-based price dynamics with chartist-fundamentalist interaction")
st.markdown("---")

# Sidebar parameters
with st.sidebar:
    st.markdown("### 🎛️ Model Parameters")
    
    st.markdown("#### Agent Parameters")
    beta_f = st.slider("Fundamentalist Strength (β_f)", 0.0, 2.0, 0.5, 0.1)
    beta_c = st.slider("Chartist Strength (β_c)", 0.0, 2.0, 1.0, 0.1)
    gamma = st.slider("Switching Rate (γ)", 0.0, 5.0, 1.0, 0.1)
    
    st.markdown("---")
    st.markdown("#### Market Parameters")
    P_fundamental = st.number_input("Fundamental Price", value=100.0, step=5.0)
    volatility = st.slider("Noise Volatility", 0.0, 0.5, 0.1, 0.01)
    
    st.markdown("---")
    st.markdown("#### Simulation Settings")
    T = st.slider("Time Horizon", 100, 2000, 500, 100)
    P0 = st.number_input("Initial Price", value=95.0, step=1.0)

# Main content
tab1, tab2, tab3 = st.tabs(["📊 Price Dynamics", "🔄 Bifurcation Analysis", "📈 Trading Signals"])

with tab1:
    st.markdown("### Agent-Based Price Simulation")
    
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🚀 Simulate", type="primary", use_container_width=True):
            with st.spinner("Simulating agent dynamics..."):
                # Simulate Chiarella model
                dt = 0.1
                n_steps = int(T / dt)
                
                prices = np.zeros(n_steps)
                fundamentalists = np.zeros(n_steps)
                chartists = np.zeros(n_steps)
                
                prices[0] = P0
                fundamentalists[0] = 0.5
                chartists[0] = 0.5
                
                for t in range(1, n_steps):
                    # Current fractions
                    n_f = fundamentalists[t-1]
                    n_c = chartists[t-1]
                    
                    # Demands
                    D_f = beta_f * (P_fundamental - prices[t-1])
                    D_c = beta_c * (prices[t-1] - prices[max(0, t-10)])
                    
                    # Price update
                    noise = np.random.normal(0, volatility)
                    prices[t] = prices[t-1] + 0.1 * (n_f * D_f + n_c * D_c) + noise
                    
                    # Agent switching (discrete choice)
                    profit_f = D_f * (prices[t] - prices[t-1])
                    profit_c = D_c * (prices[t] - prices[t-1])
                    
                    exp_f = np.exp(gamma * profit_f)
                    exp_c = np.exp(gamma * profit_c)
                    
                    fundamentalists[t] = exp_f / (exp_f + exp_c)
                    chartists[t] = exp_c / (exp_f + exp_c)
                
                time_grid = np.arange(n_steps) * dt
                
                # Plot results
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('Market Price', 'Agent Fractions'),
                    row_heights=[0.6, 0.4],
                    vertical_spacing=0.1
                )
                
                # Price chart
                fig.add_trace(
                    go.Scatter(x=time_grid, y=prices, name='Market Price', 
                             line=dict(color='blue', width=2)),
                    row=1, col=1
                )
                fig.add_hline(y=P_fundamental, line_dash="dash", line_color="red", 
                            row=1, col=1, annotation_text="Fundamental")
                
                # Agent fractions
                fig.add_trace(
                    go.Scatter(x=time_grid, y=fundamentalists, name='Fundamentalists',
                             line=dict(color='green', width=2)),
                    row=2, col=1
                )
                fig.add_trace(
                    go.Scatter(x=time_grid, y=chartists, name='Chartists',
                             line=dict(color='orange', width=2)),
                    row=2, col=1
                )
                
                fig.update_xaxes(title_text="Time", row=2, col=1)
                fig.update_yaxes(title_text="Price", row=1, col=1)
                fig.update_yaxes(title_text="Fraction", row=2, col=1)
                fig.update_layout(height=800, showlegend=True, hovermode='x unified')
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Final Price", f"${prices[-1]:.2f}")
                with col2:
                    st.metric("Price Volatility", f"{np.std(prices):.2f}")
                with col3:
                    st.metric("Mean Deviation", f"${np.mean(np.abs(prices - P_fundamental)):.2f}")
                with col4:
                    st.metric("Final Fundamentalists", f"{fundamentalists[-1]:.2%}")
                
                # Regime detection
                st.markdown("#### 📊 Market Regime")
                if fundamentalists[-1] > 0.6:
                    st.success("🟢 Fundamentalist-dominated: Price converging to fundamental value")
                elif chartists[-1] > 0.6:
                    st.warning("🟡 Chartist-dominated: Trend-following behavior, potential bubbles")
                else:
                    st.info("🔵 Mixed regime: Balanced agent behavior")
    
    with col1:
        st.markdown("""
        #### About the Chiarella Model
        
        The model simulates two types of traders:
        
        **Fundamentalists** (Green):
        - Trade based on deviation from fundamental value
        - Provide stabilizing force
        - Profit from mean reversion
        
        **Chartists** (Orange):
        - Trade based on recent trends
        - Create momentum and overshooting
        - Can amplify price moves
        
        Agents switch between strategies based on recent profitability, 
        creating endogenous regime changes and realistic price dynamics.
        """)

with tab2:
    st.markdown("### Bifurcation Analysis")
    
    st.markdown("""
    Analyze how model dynamics change with parameters. The **bifurcation parameter** 
    $\\Lambda = \\frac{\\alpha \\cdot \\gamma}{\\beta \\cdot \\delta}$ determines market behavior:
    
    - $\\Lambda < 0.67$: **Stable** (mean-reverting)
    - $0.67 \\leq \\Lambda \\leq 1.5$: **Mixed** (complex dynamics)
    - $\\Lambda > 1.5$: **Unstable** (trending, bubbles possible)
    """)
    
    analysis_type = st.radio(
        "Analysis Type",
        ["Bifurcation Diagram", "Phase Portrait", "Stability Map"],
        horizontal=True
    )
    
    if analysis_type == "Bifurcation Diagram":
        st.markdown("#### Bifurcation Diagram: Price vs Parameter")
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            param_to_vary = st.selectbox("Vary Parameter", ["alpha", "beta", "gamma", "delta"])
            param_min = st.number_input("Min Value", value=0.1, step=0.1, format="%.2f")
            param_max = st.number_input("Max Value", value=2.0, step=0.1, format="%.2f")
            n_steps_param = st.slider("Number of Steps", 20, 100, 50)
            simulation_length = st.slider("Simulation Length", 500, 5000, 2000, 100)
        
        with col1:
            if st.button("🔬 Generate Bifurcation Diagram", type="primary"):
                with st.spinner("Computing bifurcation diagram..."):
                    param_values = np.linspace(param_min, param_max, n_steps_param)
                    
                    # Store steady-state prices for each parameter value
                    steady_states = {p: [] for p in param_values}
                    lambda_values = []
                    
                    for param_val in param_values:
                        # Set up parameters
                        params = {
                            'alpha': beta_c,
                            'beta': beta_f,
                            'gamma': gamma,
                            'delta': 0.2
                        }
                        params[param_to_vary] = param_val
                        
                        # Calculate Lambda
                        Lambda = (params['alpha'] * params['gamma']) / (params['beta'] * params['delta']) if params['delta'] > 0 else 1.0
                        lambda_values.append(Lambda)
                        
                        # Simulate
                        dt = 0.1
                        n_sim = int(simulation_length / dt)
                        prices = np.zeros(n_sim)
                        trend = 0.0
                        prices[0] = P0
                        
                        for t in range(1, n_sim):
                            # Chiarella dynamics
                            mispricing = prices[t-1] - P_fundamental
                            d_price = params['alpha'] * trend - params['beta'] * mispricing
                            d_trend = params['gamma'] * (prices[t-1] - prices[max(0, t-10)]) - params['delta'] * trend
                            
                            prices[t] = prices[t-1] + dt * d_price + np.random.normal(0, volatility)
                            trend = trend + dt * d_trend + np.random.normal(0, volatility * 0.5)
                        
                        # Take steady-state prices (last 20% of simulation)
                        steady_start = int(0.8 * n_sim)
                        steady_states[param_val] = prices[steady_start:]
                    
                    # Plot bifurcation diagram
                    fig = make_subplots(
                        rows=2, cols=1,
                        subplot_titles=(f'Bifurcation Diagram: Price vs {param_to_vary}', 'Bifurcation Parameter Λ'),
                        vertical_spacing=0.12,
                        row_heights=[0.6, 0.4]
                    )
                    
                    # Scatter plot of steady-state prices
                    for i, param_val in enumerate(param_values):
                        prices_sample = steady_states[param_val][::10]  # Sample every 10th point
                        fig.add_trace(
                            go.Scatter(
                                x=[param_val] * len(prices_sample),
                                y=prices_sample,
                                mode='markers',
                                marker={'size': 2, 'color': 'blue', 'opacity': 0.3},
                                showlegend=False,
                                hovertemplate=f'{param_to_vary}={param_val:.2f}<br>Price=%{{y:.2f}}<extra></extra>'
                            ),
                            row=1, col=1
                        )
                    
                    # Add fundamental price line
                    fig.add_hline(y=P_fundamental, line_dash="dash", line_color="red", 
                                row=1, col=1, annotation_text="Fundamental")
                    
                    # Plot Lambda values
                    fig.add_trace(
                        go.Scatter(
                            x=param_values,
                            y=lambda_values,
                            mode='lines+markers',
                            name='Λ',
                            line={'color': 'purple', 'width': 2}
                        ),
                        row=2, col=1
                    )
                    
                    # Add regime boundaries
                    fig.add_hline(y=0.67, line_dash="dash", line_color="green", 
                                row=2, col=1, annotation_text="Λ=0.67 (stable)")
                    fig.add_hline(y=1.5, line_dash="dash", line_color="orange", 
                                row=2, col=1, annotation_text="Λ=1.5 (unstable)")
                    
                    fig.update_xaxes(title_text=param_to_vary, row=1, col=1)
                    fig.update_xaxes(title_text=param_to_vary, row=2, col=1)
                    fig.update_yaxes(title_text="Price", row=1, col=1)
                    fig.update_yaxes(title_text="Λ", row=2, col=1)
                    fig.update_layout(height=800, hovermode='closest')
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("#### 💡 Interpretation")
                    st.info("""
                    - **Single cluster**: System converges to stable equilibrium
                    - **Multiple clusters**: System has multiple attractors (bimodal)
                    - **Spread increases**: Higher volatility and potential for bubbles/crashes
                    - **Λ crossing 1.5**: Transition from stable to chaotic dynamics
                    """)
    
    elif analysis_type == "Phase Portrait":
        st.markdown("#### Phase Portrait: Price vs Trend")
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            n_trajectories = st.slider("Number of Trajectories", 1, 10, 5)
            traj_length = st.slider("Trajectory Length", 100, 1000, 500, 50)
        
        with col1:
            if st.button("📊 Generate Phase Portrait", type="primary"):
                with st.spinner("Computing phase portrait..."):
                    fig = go.Figure()
                    
                    # Plot nullclines
                    price_range = np.linspace(P_fundamental * 0.8, P_fundamental * 1.2, 100)
                    
                    # dp/dt = 0 => trend = β·(p - p_f) / α
                    trend_nullcline = beta_f * (price_range - P_fundamental) / beta_c if beta_c > 0 else np.zeros_like(price_range)
                    
                    fig.add_trace(go.Scatter(
                        x=price_range,
                        y=trend_nullcline,
                        mode='lines',
                        name='dp/dt = 0',
                        line={'color': 'red', 'dash': 'dash', 'width': 2}
                    ))
                    
                    # dtrend/dt = 0 => trend ≈ 0 (assuming steady state)
                    fig.add_hline(y=0, line_dash="dash", line_color="blue", 
                                annotation_text="dtrend/dt = 0")
                    
                    # Simulate multiple trajectories
                    for traj_id in range(n_trajectories):
                        dt = 0.1
                        n_sim = int(traj_length / dt)
                        
                        # Random initial conditions
                        prices = np.zeros(n_sim)
                        trends = np.zeros(n_sim)
                        prices[0] = P_fundamental * (0.9 + 0.2 * np.random.random())
                        trends[0] = (-5 + 10 * np.random.random())
                        
                        for t in range(1, n_sim):
                            mispricing = prices[t-1] - P_fundamental
                            d_price = beta_c * trends[t-1] - beta_f * mispricing
                            d_trend = gamma * (prices[t-1] - prices[max(0, t-10)]) - 0.2 * trends[t-1]
                            
                            prices[t] = prices[t-1] + dt * d_price + np.random.normal(0, volatility * 0.5)
                            trends[t] = trends[t-1] + dt * d_trend + np.random.normal(0, volatility * 0.3)
                        
                        # Plot trajectory
                        fig.add_trace(go.Scatter(
                            x=prices,
                            y=trends,
                            mode='lines',
                            name=f'Trajectory {traj_id+1}',
                            line={'width': 1.5},
                            opacity=0.7
                        ))
                        
                        # Mark start and end
                        fig.add_trace(go.Scatter(
                            x=[prices[0]],
                            y=[trends[0]],
                            mode='markers',
                            marker={'symbol': 'circle', 'size': 8, 'color': 'green'},
                            showlegend=False,
                            hovertext='Start'
                        ))
                        fig.add_trace(go.Scatter(
                            x=[prices[-1]],
                            y=[trends[-1]],
                            mode='markers',
                            marker={'symbol': 'square', 'size': 8, 'color': 'red'},
                            showlegend=False,
                            hovertext='End'
                        ))
                    
                    # Mark equilibrium
                    fig.add_trace(go.Scatter(
                        x=[P_fundamental],
                        y=[0],
                        mode='markers',
                        marker={'symbol': 'star', 'size': 15, 'color': 'gold'},
                        name='Equilibrium'
                    ))
                    
                    fig.update_layout(
                        title="Phase Portrait: Price vs Trend Dynamics",
                        xaxis_title="Price",
                        yaxis_title="Trend",
                        height=600,
                        hovermode='closest'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("#### 💡 Interpretation")
                    st.info("""
                    - **Spiral**: Oscillatory convergence to equilibrium
                    - **Direct convergence**: Fast mean-reversion
                    - **Divergence**: Unstable system (chartists dominate)
                    - **Limit cycle**: Sustained oscillations
                    """)
    
    else:  # Stability Map
        st.markdown("#### Stability Map: Parameter Space")
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            param_x = st.selectbox("X-axis", ["alpha", "beta", "gamma", "delta"], index=0)
            param_y = st.selectbox("Y-axis", ["alpha", "beta", "gamma", "delta"], index=1)
            resolution = st.slider("Resolution", 10, 50, 25)
        
        with col1:
            if st.button("🗺️ Generate Stability Map", type="primary"):
                with st.spinner("Computing stability map..."):
                    # Create parameter grid
                    x_range = np.linspace(0.1, 2.0, resolution)
                    y_range = np.linspace(0.1, 2.0, resolution)
                    
                    lambda_grid = np.zeros((resolution, resolution))
                    
                    for i, x_val in enumerate(x_range):
                        for j, y_val in enumerate(y_range):
                            params = {
                                'alpha': beta_c,
                                'beta': beta_f,
                                'gamma': gamma,
                                'delta': 0.2
                            }
                            params[param_x] = x_val
                            params[param_y] = y_val
                            
                            # Calculate Lambda
                            Lambda = (params['alpha'] * params['gamma']) / (params['beta'] * params['delta'])
                            lambda_grid[j, i] = Lambda  # Note: j for y, i for x
                    
                    # Create heatmap
                    fig = go.Figure(data=go.Heatmap(
                        x=x_range,
                        y=y_range,
                        z=lambda_grid,
                        colorscale=[
                            [0, 'green'],      # Λ < 0.67: stable
                            [0.67/3, 'green'],
                            [0.67/3, 'yellow'], # 0.67 ≤ Λ ≤ 1.5: mixed
                            [1.5/3, 'yellow'],
                            [1.5/3, 'red'],     # Λ > 1.5: unstable
                            [1, 'red']
                        ],
                        colorbar={'title': 'Λ'},
                        hovertemplate=f'{param_x}=%{{x:.2f}}<br>{param_y}=%{{y:.2f}}<br>Λ=%{{z:.2f}}<extra></extra>'
                    ))
                    
                    # Add contour lines
                    fig.add_trace(go.Contour(
                        x=x_range,
                        y=y_range,
                        z=lambda_grid,
                        contours={'start': 0.67, 'end': 1.5, 'size': 0.83},
                        line={'color': 'black', 'width': 2},
                        showscale=False,
                        hoverinfo='skip'
                    ))
                    
                    fig.update_layout(
                        title=f"Stability Map: Λ in ({param_x}, {param_y}) space",
                        xaxis_title=param_x,
                        yaxis_title=param_y,
                        height=600
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("#### 💡 Interpretation")
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.success("**🟢 Green: Stable**\nΛ < 0.67\nMean-reversion dominates")
                    with col_b:
                        st.warning("**🟡 Yellow: Mixed**\n0.67 ≤ Λ ≤ 1.5\nComplex dynamics")
                    with col_c:
                        st.error("**🔴 Red: Unstable**\nΛ > 1.5\nTrending, bubbles")

with tab3:
    st.markdown("### Trading Signals from Agent Dynamics")
    
    if 'historical_data' not in st.session_state or st.session_state.historical_data is None:
        st.warning("⚠️ Please load data first from the Data Loader page")
        if st.button("💾 Go to Data Loader"):
            st.switch_page("pages/data_loader.py")
    else:
        import pandas as pd
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
            st.warning("No data available")
        else:
            selected_symbol = st.selectbox("Select Symbol", symbols, key="chiarella_symbol")
            
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
            
            prices = df[close_col].values
            
            # Mode selection
            st.markdown("#### 🎯 Analysis Mode")
            analysis_mode = st.radio(
                "Select Mode",
                ["📊 Trading Signals", "🫧 Bubble Detection", "⚙️ Calibrate Parameters"],
                horizontal=True
            )
            
            st.markdown("---")
            
            # Estimate fundamental price
            st.markdown("#### ⚙️ Configuration")
            col1, col2 = st.columns(2)
            
            with col1:
                fund_method = st.selectbox("Fundamental Price", ["EMA", "SMA", "Median"], 
                                           help="Method to estimate fundamental value")
                fund_window = st.slider("Fundamental Window", 20, 200, 100,
                                       help="Lookback period for fundamental estimate")
            
            with col2:
                signal_sensitivity = st.slider("Signal Sensitivity", 0.5, 2.0, 1.0, 0.1,
                                              help="Multiplier for signal strength")
                min_regime_duration = st.slider("Min Regime Duration", 5, 50, 20,
                                               help="Minimum bars to confirm regime")
            
            # Calibration option
            if analysis_mode == "⚙️ Calibrate Parameters":
                st.markdown("#### 🔧 Auto-Calibrate from Data")
                calibrate_col1, calibrate_col2 = st.columns([2, 1])
                
                with calibrate_col2:
                    calibration_window = st.slider("Calibration Window", 50, 500, 200,
                                                   help="Number of recent bars to use")
                
                with calibrate_col1:
                    if st.button("🔬 Calibrate Model", type="primary"):
                        with st.spinner("Calibrating Chiarella model from data..."):
                            # Use recent data for calibration
                            cal_prices = prices[-calibration_window:]
                            cal_returns = np.diff(cal_prices) / cal_prices[:-1]
                            
                            # Estimate fundamental
                            if fund_method == "EMA":
                                cal_fundamental = pd.Series(cal_prices).ewm(span=fund_window).mean().values
                            elif fund_method == "SMA":
                                cal_fundamental = pd.Series(cal_prices).rolling(fund_window).mean().values
                            else:
                                cal_fundamental = pd.Series(cal_prices).rolling(fund_window).median().values
                            
                            cal_fundamental = pd.Series(cal_fundamental).bfill().values
                            cal_mispricing = cal_prices - cal_fundamental
                            
                            # Estimate volatility
                            returns_vol = np.std(cal_returns)
                            
                            # Estimate mean reversion speed (beta_f)
                            # Higher autocorrelation → lower beta_f (slower reversion)
                            autocorr = np.corrcoef(cal_mispricing[:-1], cal_mispricing[1:])[0, 1]
                            estimated_beta_f = max(0.1, min(2.0, 1.0 - abs(autocorr)))
                            
                            # Estimate trend following strength (beta_c)
                            # Higher momentum → higher beta_c
                            momentum = pd.Series(cal_returns).rolling(10).mean().std()
                            estimated_beta_c = max(0.1, min(2.0, momentum * 100))
                            
                            # Estimate switching rate (gamma)
                            # Higher volatility → higher switching
                            estimated_gamma = max(0.1, min(5.0, returns_vol * 50))
                            
                            # Display calibrated parameters
                            st.success("✅ Calibration Complete!")
                            
                            calib_col1, calib_col2, calib_col3, calib_col4 = st.columns(4)
                            with calib_col1:
                                st.metric("β_f (Fundamentalist)", f"{estimated_beta_f:.2f}",
                                         help="Mean reversion strength")
                            with calib_col2:
                                st.metric("β_c (Chartist)", f"{estimated_beta_c:.2f}",
                                         help="Trend following strength")
                            with calib_col3:
                                st.metric("γ (Switching)", f"{estimated_gamma:.2f}",
                                         help="Agent switching rate")
                            with calib_col4:
                                Lambda_calibrated = (estimated_beta_c * estimated_gamma) / (estimated_beta_f * 0.2)
                                st.metric("Λ (Bifurcation)", f"{Lambda_calibrated:.2f}",
                                         help="System stability parameter")
                            
                            st.info(f"""
                            **📊 Calibration Insights:**
                            - Autocorrelation: {autocorr:.3f} → {"Strong" if abs(autocorr) > 0.3 else "Weak"} mean reversion
                            - Momentum volatility: {momentum:.4f} → {"High" if momentum > 0.01 else "Low"} trend strength
                            - Return volatility: {returns_vol:.4f} → {"High" if returns_vol > 0.02 else "Low"} switching rate
                            - Regime: {"🟢 Stable" if Lambda_calibrated < 0.67 else "🟡 Mixed" if Lambda_calibrated <= 1.5 else "🔴 Unstable"}
                            """)
                            
                            # Store calibrated params in session state
                            st.session_state['calibrated_params'] = {
                                'beta_f': estimated_beta_f,
                                'beta_c': estimated_beta_c,
                                'gamma': estimated_gamma,
                                'Lambda': Lambda_calibrated
                            }
            
            # Bubble detection mode
            elif analysis_mode == "🫧 Bubble Detection":
                st.markdown("#### 🫧 Sector Bubble Analysis")
                st.info("""
                **Bubble Indicators:**
                - 🔴 **Critical Overvaluation**: Price >> Fundamental + High volatility + Trending regime
                - 🟡 **Warning**: Moderate overvaluation with increasing volatility
                - 🟢 **Normal**: Price near fundamental or undervalued
                """)
                
                bubble_sensitivity = st.slider("Bubble Detection Sensitivity", 0.5, 2.0, 1.0, 0.1,
                                              help="Lower = more sensitive to bubbles")
                vol_threshold = st.slider("Volatility Threshold %", 1.0, 10.0, 3.0, 0.5,
                                         help="Elevated volatility indicator")
            
            # Main button text changes based on mode
            button_text = {
                "📊 Trading Signals": "⚡ Generate Trading Signals",
                "🫧 Bubble Detection": "🫧 Detect Bubble Risk",
                "⚙️ Calibrate Parameters": "⚡ Generate Signals (Use Calibrated)"
            }
            
            if st.button(button_text.get(analysis_mode, "⚡ Generate Signals"), type="primary"):
                with st.spinner("Computing agent-based signals..."):
                    # Use calibrated parameters if available
                    if 'calibrated_params' in st.session_state and analysis_mode == "⚙️ Calibrate Parameters":
                        cal_params = st.session_state['calibrated_params']
                        active_beta_f = cal_params['beta_f']
                        active_beta_c = cal_params['beta_c']
                        active_gamma = cal_params['gamma']
                    else:
                        active_beta_f = beta_f
                        active_beta_c = beta_c
                        active_gamma = gamma
                    
                    # Estimate fundamental price
                    if fund_method == "EMA":
                        fundamental = pd.Series(prices).ewm(span=fund_window).mean().values
                    elif fund_method == "SMA":
                        fundamental = pd.Series(prices).rolling(fund_window).mean().values
                    else:  # Median
                        fundamental = pd.Series(prices).rolling(fund_window).median().values
                    
                    # Fill NaN with first valid value
                    fundamental = pd.Series(fundamental).bfill().values
                    
                    # Compute mispricing and trend
                    mispricing = prices - fundamental  # type: ignore[operator]
                    mispricing_pct = mispricing / fundamental
                    
                    # Estimate trend (moving average of returns)
                    returns = np.diff(prices, prepend=prices[0])
                    trend = pd.Series(returns).rolling(20).mean().fillna(0).values
                    
                    # Compute Lambda (bifurcation parameter) over time
                    Lambda = (active_beta_c * active_gamma) / (active_beta_f * 0.2) if active_beta_f > 0 else 1.0
                    
                    # Compute rolling volatility
                    rolling_vol = pd.Series(returns).rolling(20).std().fillna(0).values * 100  # type: ignore[operator] # As percentage
                    
                    # Regime detection
                    regimes = []
                    for i in range(len(prices)):
                        if Lambda < 0.67:
                            regimes.append('mean_reverting')
                        elif Lambda > 1.5:
                            regimes.append('trending')
                        else:
                            regimes.append('mixed')
                    
                    # BUBBLE DETECTION
                    if analysis_mode == "🫧 Bubble Detection":
                        bubble_scores = []
                        bubble_levels = []
                        
                        for i in range(max(fund_window, min_regime_duration), len(prices)):
                            # Bubble indicators
                            overvaluation = mispricing_pct[i]  # How far above fundamental
                            vol_elevated = rolling_vol[i] > (vol_threshold if 'vol_threshold' in locals() else 3.0)
                            is_trending = regimes[i] == 'trending' or Lambda > 1.5
                            momentum_strong = abs(trend[i]) > 0.002
                            
                            # Bubble score (0-1)
                            bubble_score = 0.0
                            
                            # Strong overvaluation (40% weight)
                            if overvaluation > 0.05 / (bubble_sensitivity if 'bubble_sensitivity' in locals() else 1.0):
                                bubble_score += 0.4 * min(overvaluation / 0.2, 1.0)
                            
                            # Elevated volatility (25% weight)
                            if vol_elevated:
                                bubble_score += 0.25
                            
                            # Trending regime (20% weight)
                            if is_trending:
                                bubble_score += 0.20
                            
                            # Strong momentum (15% weight)
                            if momentum_strong and overvaluation > 0:
                                bubble_score += 0.15
                            
                            bubble_scores.append(bubble_score)
                            
                            # Classify bubble level
                            if bubble_score > 0.7:
                                bubble_levels.append('🔴 CRITICAL')
                            elif bubble_score > 0.4:
                                bubble_levels.append('🟡 WARNING')
                            else:
                                bubble_levels.append('🟢 NORMAL')
                        
                        # Display bubble analysis
                        st.markdown("---")
                        st.markdown("### 🫧 Bubble Risk Assessment")
                        
                        current_bubble = bubble_levels[-1] if bubble_levels else '🟢 NORMAL'
                        current_score = bubble_scores[-1] if bubble_scores else 0.0
                        
                        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                        with metric_col1:
                            st.metric("Bubble Level", current_bubble)
                        with metric_col2:
                            st.metric("Bubble Score", f"{current_score:.2f}")
                        with metric_col3:
                            st.metric("Overvaluation", f"{mispricing_pct[-1]:.2%}")
                        with metric_col4:
                            st.metric("Current Vol", f"{rolling_vol[-1]:.2f}%")
                        
                        # Bubble timeline
                        critical_periods = sum(1 for level in bubble_levels if '🔴' in level)
                        warning_periods = sum(1 for level in bubble_levels if '🟡' in level)
                        
                        st.markdown(f"""
                        **📊 Historical Analysis ({len(bubble_levels)} periods):**
                        - 🔴 Critical bubble risk: {critical_periods} periods ({critical_periods/len(bubble_levels)*100:.1f}%)
                        - 🟡 Warning level: {warning_periods} periods ({warning_periods/len(bubble_levels)*100:.1f}%)
                        - 🟢 Normal conditions: {len(bubble_levels) - critical_periods - warning_periods} periods
                        """)
                        
                        if current_bubble == '🔴 CRITICAL':
                            st.error("""
                            ⚠️ **CRITICAL BUBBLE WARNING**
                            - Price significantly above fundamental value
                            - High volatility detected
                            - Trending/unstable regime
                            - Consider reducing exposure or hedging
                            """)
                        elif current_bubble == '🟡 WARNING':
                            st.warning("""
                            ⚠️ **Bubble Warning Signs Detected**
                            - Moderate overvaluation
                            - Monitor volatility and momentum
                            - Consider taking profits or tightening stops
                            """)
                        else:
                            st.success("""
                            ✅ **Normal Market Conditions**
                            - Price near fundamental value
                            - Normal volatility levels
                            - No significant bubble indicators
                            """)
                        
                        # Plot bubble score over time
                        bubble_fig = go.Figure()
                        
                        bubble_times = df.index[max(fund_window, min_regime_duration):]
                        
                        bubble_fig.add_trace(go.Scatter(
                            x=bubble_times,
                            y=bubble_scores,
                            name='Bubble Score',
                            line={'color': 'red', 'width': 2},
                            fill='tozeroy',
                            fillcolor='rgba(255,0,0,0.1)'
                        ))
                        
                        bubble_fig.add_hline(y=0.7, line_dash="dash", line_color="red",
                                           annotation_text="Critical (0.7)")
                        bubble_fig.add_hline(y=0.4, line_dash="dash", line_color="orange",
                                           annotation_text="Warning (0.4)")
                        
                        bubble_fig.update_layout(
                            title="Bubble Risk Score Over Time",
                            xaxis_title="Time",
                            yaxis_title="Bubble Score (0-1)",
                            height=400,
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(bubble_fig, use_container_width=True)
                        
                        # Skip normal signal generation in bubble mode
                        st.stop()
                    
                    # Generate signals based on regime (for normal trading mode)
                    signals = []
                    for i in range(max(fund_window, min_regime_duration), len(prices)):
                        regime = regimes[i]
                        
                        if regime == 'mean_reverting':
                            # Fundamentalist strategy: fade mispricings
                            if mispricing_pct[i] < -0.02 * signal_sensitivity:
                                signals.append({'time': df.index[i], 'signal': 'LONG_FUND', 
                                              'strength': abs(mispricing_pct[i]), 'regime': regime})
                            elif mispricing_pct[i] > 0.02 * signal_sensitivity:
                                signals.append({'time': df.index[i], 'signal': 'SHORT_FUND',
                                              'strength': abs(mispricing_pct[i]), 'regime': regime})
                        
                        elif regime == 'trending':
                            # Chartist strategy: follow momentum
                            if trend[i] > 0.001 * signal_sensitivity:
                                signals.append({'time': df.index[i], 'signal': 'LONG_CHART',
                                              'strength': abs(trend[i]), 'regime': regime})
                            elif trend[i] < -0.001 * signal_sensitivity:
                                signals.append({'time': df.index[i], 'signal': 'SHORT_CHART',
                                              'strength': abs(trend[i]), 'regime': regime})
                        
                        else:  # mixed
                            # Balanced strategy: combine both
                            fund_signal = -mispricing_pct[i]  # Buy when undervalued
                            chart_signal = trend[i]  # Follow trend
                            combined = 0.5 * fund_signal + 0.5 * chart_signal
                            
                            if combined > 0.01 * signal_sensitivity:
                                signals.append({'time': df.index[i], 'signal': 'LONG_MIXED',
                                              'strength': abs(combined), 'regime': regime})
                            elif combined < -0.01 * signal_sensitivity:
                                signals.append({'time': df.index[i], 'signal': 'SHORT_MIXED',
                                              'strength': abs(combined), 'regime': regime})
                    
                    # Display metrics
                    col_a, col_b, col_c, col_d = st.columns(4)
                    
                    with col_a:
                        st.metric("Λ (Bifurcation)", f"{Lambda:.2f}")
                    with col_b:
                        current_regime = regimes[-1]
                        st.metric("Current Regime", current_regime.title())
                    with col_c:
                        st.metric("Total Signals", len(signals))
                    with col_d:
                        current_mispricing = mispricing_pct[-1]
                        st.metric("Mispricing", f"{current_mispricing:.2%}")
                    
                    # Plot signals
                    fig = make_subplots(
                        rows=3, cols=1,
                        subplot_titles=('Price & Fundamental Value', 'Mispricing %', 'Trend'),
                        vertical_spacing=0.1,
                        row_heights=[0.4, 0.3, 0.3]
                    )
                    
                    # Price and fundamental
                    fig.add_trace(
                        go.Scatter(x=df.index, y=prices, name='Price',
                                 line={'color': 'blue', 'width': 2}),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=df.index, y=fundamental, name='Fundamental',
                                 line={'color': 'red', 'dash': 'dash', 'width': 2}),
                        row=1, col=1
                    )
                    
                    # Add signals on price chart
                    long_signals = [s for s in signals if 'LONG' in s['signal']]
                    short_signals = [s for s in signals if 'SHORT' in s['signal']]
                    
                    if long_signals:
                        long_prices = [prices[df.index.get_loc(s['time'])] for s in long_signals]
                        fig.add_trace(
                            go.Scatter(
                                x=[s['time'] for s in long_signals],
                                y=long_prices,
                                mode='markers',
                                name='Long',
                                marker={'symbol': 'triangle-up', 'size': 8, 'color': 'green'}
                            ),
                            row=1, col=1
                        )
                    
                    if short_signals:
                        short_prices = [prices[df.index.get_loc(s['time'])] for s in short_signals]
                        fig.add_trace(
                            go.Scatter(
                                x=[s['time'] for s in short_signals],
                                y=short_prices,
                                mode='markers',
                                name='Short',
                                marker={'symbol': 'triangle-down', 'size': 8, 'color': 'red'}
                            ),
                            row=1, col=1
                        )
                    
                    # Mispricing
                    fig.add_trace(
                        go.Scatter(x=df.index, y=mispricing_pct * 100, name='Mispricing %',
                                 line={'color': 'purple', 'width': 2}),
                        row=2, col=1
                    )
                    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
                    
                    # Trend
                    fig.add_trace(
                        go.Scatter(x=df.index, y=trend * 100, name='Trend',  # type: ignore[operator]
                                 line={'color': 'orange', 'width': 2}),
                        row=3, col=1
                    )
                    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)
                    
                    fig.update_xaxes(title_text="Date", row=3, col=1)
                    fig.update_yaxes(title_text="Price", row=1, col=1)
                    fig.update_yaxes(title_text="%", row=2, col=1)
                    fig.update_yaxes(title_text="%", row=3, col=1)
                    fig.update_layout(height=900, showlegend=True, hovermode='x unified')
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Signal breakdown
                    st.markdown("#### 📊 Signal Breakdown by Type")
                    if signals:
                        signal_types = {}
                        for s in signals:
                            sig_type = s['signal']
                            signal_types[sig_type] = signal_types.get(sig_type, 0) + 1
                        
                        breakdown_df = pd.DataFrame([
                            {'Signal Type': k, 'Count': v, 'Percentage': f"{v/len(signals)*100:.1f}%"}
                            for k, v in signal_types.items()
                        ])
                        st.dataframe(breakdown_df, use_container_width=True)
                    else:
                        st.info("No signals generated with current parameters")
        
        # Strategy explanation
        with st.expander("📚 Regime-Based Trading Logic"):
            st.markdown("""
            ### Agent-Based Signal Generation
            
            The Chiarella model adapts trading strategy based on market regime (determined by Λ):
            
            #### 🟢 Mean-Reverting Regime (Λ < 0.67)
            **Fundamentalist Strategy Dominant**
            - **Long**: Price < Fundamental (undervalued)
            - **Short**: Price > Fundamental (overvalued)
            - **Logic**: Fundamentalists dominate → prices revert to fair value
            
            #### 🔴 Trending Regime (Λ > 1.5)
            **Chartist Strategy Dominant**
            - **Long**: Positive trend/momentum
            - **Short**: Negative trend/momentum
            - **Logic**: Chartists dominate → follow the trend
            
            #### 🟡 Mixed Regime (0.67 ≤ Λ ≤ 1.5)
            **Balanced Strategy**
            - **Combined Signal**: 50% fundamental + 50% chartist
            - **Logic**: Neither force dominates → use both signals
            
            ### Advantages
            ✅ **Adaptive**: Automatically switches strategy based on market conditions  
            ✅ **Theory-driven**: Based on mathematical model of agent interactions  
            ✅ **Regime-aware**: Recognizes when mean-reversion vs momentum works  
            ✅ **Risk-conscious**: Avoids fighting dominant market forces
            """)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🌀 Chiarella Model Lab | Part of HFT Arbitrage Lab</p>
</div>
""", unsafe_allow_html=True)
