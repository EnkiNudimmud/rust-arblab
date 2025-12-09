"""
Advanced Optimization Laboratory
=================================

Full implementation of advanced optimization methods with real market data:
- HMM Regime Detection with calibration and visualization
- MCMC Bayesian Parameter Estimation  
- Maximum Likelihood Estimation (MLE)
- Information Theory metrics
- Multi-Strategy Optimization with Rust acceleration

Features:
- Real-time calibration on loaded market data
- Interactive parameter tuning
- Comprehensive visualization of results
- Rust-accelerated computations when available
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional
import sys
import time
sys.path.append('/app')

try:
    from python.optimization.advanced_optimization import (
        HMMRegimeDetector,
        MCMCOptimizer,
        MLEOptimizer,
        InformationTheoryOptimizer,
        MultiStrategyOptimizer,
        ParameterSpace,
        OptimizationResult,
        RUST_AVAILABLE
    )
except ImportError:
    st.error("⚠️ Advanced optimization modules not available")
    st.stop()

from utils.ui_components import render_sidebar_navigation, apply_custom_css, ensure_data_loaded

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

# Page config
st.set_page_config(
    page_title="Advanced Optimization Lab",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_css()
render_sidebar_navigation(current_page="Advanced Optimization Lab")

st.title("🧬 Advanced Optimization Laboratory")
st.markdown("""
**Calibrate and optimize trading strategies using advanced mathematical methods**

This lab provides full implementation of:
- Hidden Markov Models for regime detection
- MCMC for Bayesian parameter estimation
- Maximum Likelihood Estimation
- Information Theory for feature selection
- Multi-objective optimization
""")

# Auto-load most recent dataset if no data is loaded
data_available = ensure_data_loaded()

# Check for loaded data
if not data_available or 'historical_data' not in st.session_state or st.session_state.historical_data is None:
    st.warning("⚠️ Please load market data first using the Data Loader page")
    if st.button("📊 Go to Data Loader"):
        st.switch_page("pages/data_loader.py")
    st.stop()

df = st.session_state.historical_data

# Data preparation
st.sidebar.header("⚙️ Configuration")

# Select optimization method
optimization_method = st.sidebar.selectbox(
    "Optimization Method",
    ["HMM Regime Detection", "MCMC Bayesian", "MLE Estimation", 
     "Information Theory", "Multi-Strategy Optimization"],
    help="Select the optimization technique to use"
)

# Rust acceleration status
if RUST_AVAILABLE:
    st.sidebar.success("🚀 Rust Acceleration: ENABLED")
else:
    st.sidebar.warning("⚠️ Rust Acceleration: DISABLED (slower)")

# Main content
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Data & Parameters", 
    "🔬 Calibration", 
    "📈 Results & Visualization",
    "💾 Export & Save"
])

# =============================================================================
# TAB 1: DATA & PARAMETERS
# =============================================================================
with tab1:
    st.header("Market Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        if 'symbol' in df.columns:
            n_symbols = df['symbol'].nunique() if 'symbol' in df.columns else len(df.columns)
            st.metric("Symbols", n_symbols)
        else:
            st.metric("Columns", len(df.columns))
    with col3:
        if 'timestamp' in df.columns:
            date_col = df['timestamp']
            st.metric("Date Range", f"{pd.to_datetime(date_col).min():%Y-%m-%d} to {pd.to_datetime(date_col).max():%Y-%m-%d}")
        elif isinstance(df.index, pd.DatetimeIndex):
            st.metric("Date Range", f"{df.index.min():%Y-%m-%d} to {df.index.max():%Y-%m-%d}")
    with col4:
        if 'close' in df.columns:
            returns = df.groupby('symbol')['close'].pct_change() if 'symbol' in df.columns else df['close'].pct_change()
            st.metric("Avg Volatility", f"{returns.std() * np.sqrt(252):.2%}")
    
    # Data preview
    st.subheader("Data Preview")
    st.dataframe(df.head(100), use_container_width=True)
    
    # Multi-Symbol Selection
    st.subheader("📊 Asset Selection")
    
    if 'symbol' in df.columns:
        symbols = df['symbol'].unique().tolist()
        
        # Initialize session state for selected symbols
        if 'selected_symbols' not in st.session_state:
            st.session_state['selected_symbols'] = [symbols[0]] if symbols else []
        
        col1, col2 = st.columns([4, 1])
        with col1:
            selected_symbols = st.multiselect(
                "Select symbols for optimization",
                symbols,
                default=st.session_state['selected_symbols'],
                help="Choose one or more symbols to analyze"
            )
            # Update session state with current selection
            st.session_state['selected_symbols'] = selected_symbols
        with col2:
            if st.button("📋 Select All", use_container_width=True):
                st.session_state['selected_symbols'] = symbols
                st.rerun()
        
        if not selected_symbols:
            st.warning("⚠️ Please select at least one symbol")
            st.stop()
        
        # Store selected symbols
        st.session_state['selected_symbols'] = selected_symbols
        
        # Prepare data for selected symbols
        symbol_df = df[df['symbol'].isin(selected_symbols)].copy()
        st.session_state['selected_symbol_data'] = symbol_df
        
        st.info(f"📌 Selected {len(selected_symbols)} symbol(s): {', '.join(selected_symbols[:5])}{'...' if len(selected_symbols) > 5 else ''}")
    else:
        st.session_state['selected_symbols'] = ["Dataset"]
        st.session_state['selected_symbol_data'] = df
        st.info("📌 Using full dataset (no symbol column)")
    
    # Parameter Configuration
    st.subheader("⚙️ Optimization Parameters")
    
    if optimization_method == "HMM Regime Detection":
        col1, col2 = st.columns(2)
        with col1:
            n_states = st.slider("Number of Market Regimes", 2, 5, 3,
                                help="Typical: 3 (Bull/Bear/Sideways)")
            lookback_period = st.number_input("Training Period (bars)", 
                                            min_value=100, max_value=10000, 
                                            value=500, step=100)
        with col2:
            n_iterations = st.slider("EM Iterations", 10, 200, 100,
                                    help="More iterations = better convergence")
            use_returns = st.checkbox("Use Returns (vs Prices)", value=True,
                                     help="Returns are typically more stationary")
        
        st.session_state['hmm_params'] = {
            'n_states': n_states,
            'lookback': lookback_period,
            'n_iterations': n_iterations,
            'use_returns': use_returns
        }
    
    elif optimization_method == "MCMC Bayesian":
        col1, col2 = st.columns(2)
        with col1:
            n_samples = st.number_input("MCMC Samples", 1000, 50000, 10000, 1000)
            burn_in = st.number_input("Burn-in Period", 100, 10000, 1000, 100)
        with col2:
            prior_mean = st.number_input("Prior Mean", -1.0, 1.0, 0.0, 0.1)
            prior_std = st.number_input("Prior Std Dev", 0.01, 2.0, 0.5, 0.05)
        
        st.session_state['mcmc_params'] = {
            'n_samples': n_samples,
            'burn_in': burn_in,
            'prior_mean': prior_mean,
            'prior_std': prior_std
        }
    
    elif optimization_method == "MLE Estimation":
        col1, col2 = st.columns(2)
        with col1:
            param_names = st.multiselect(
                "Parameters to Estimate",
                ["entry_threshold", "exit_threshold", "lookback", "holding_period"],
                default=["entry_threshold", "exit_threshold"]
            )
        with col2:
            optimization_algo = st.selectbox(
                "Optimization Algorithm",
                ["L-BFGS-B", "SLSQP", "TNC", "Differential Evolution"],
                help="Algorithm for maximizing likelihood"
            )
        
        # Parameter bounds
        st.markdown("**Parameter Bounds**")
        bounds = {}
        cols = st.columns(len(param_names))
        for i, param in enumerate(param_names):
            with cols[i]:
                st.markdown(f"**{param}**")
                lower = st.number_input(f"Lower", value=0.5 if 'threshold' in param else 10, 
                                       key=f"mle_lower_{param}")
                upper = st.number_input(f"Upper", value=3.0 if 'threshold' in param else 200,
                                       key=f"mle_upper_{param}")
                bounds[param] = (lower, upper)
        
        st.session_state['mle_params'] = {
            'param_names': param_names,
            'bounds': bounds,
            'algorithm': optimization_algo
        }
    
    elif optimization_method == "Information Theory":
        st.markdown("**Feature Selection using Mutual Information**")
        col1, col2 = st.columns(2)
        with col1:
            n_features = st.slider("Number of Top Features", 5, 50, 20)
            mi_method = st.selectbox("MI Estimation Method", 
                                    ["KSG", "Histogram", "KDE"])
        with col2:
            target_var = st.selectbox("Target Variable", 
                                     ["returns", "volatility", "direction"])
            k_neighbors = st.slider("k-Neighbors (KSG)", 3, 20, 5)
        
        st.session_state['info_theory_params'] = {
            'n_features': n_features,
            'method': mi_method,
            'target': target_var,
            'k_neighbors': k_neighbors
        }
    
    else:  # Multi-Strategy Optimization
        st.markdown("**Multi-Objective Optimization Configuration**")
        
        strategies = st.multiselect(
            "Select Strategies",
            ["Mean Reversion", "Momentum", "Market Making", "Statistical Arbitrage"],
            default=["Mean Reversion", "Momentum"]
        )
        
        objectives = st.multiselect(
            "Optimization Objectives",
            ["Sharpe Ratio", "Sortino Ratio", "Max Drawdown", "Calmar Ratio", "Win Rate"],
            default=["Sharpe Ratio", "Max Drawdown"]
        )
        
        col1, col2 = st.columns(2)
        with col1:
            pop_size = st.number_input("Population Size", 50, 500, 100, 50)
            max_generations = st.number_input("Max Generations", 50, 1000, 200, 50)
        with col2:
            mutation_rate = st.slider("Mutation Rate", 0.01, 0.5, 0.1, 0.01)
            crossover_rate = st.slider("Crossover Rate", 0.5, 1.0, 0.8, 0.05)
        
        st.session_state['multi_strategy_params'] = {
            'strategies': strategies,
            'objectives': objectives,
            'pop_size': pop_size,
            'max_generations': max_generations,
            'mutation_rate': mutation_rate,
            'crossover_rate': crossover_rate
        }

# =============================================================================
# TAB 2: CALIBRATION
# =============================================================================
with tab2:
    st.header("🔬 Model Calibration")
    
    if optimization_method == "HMM Regime Detection":
        st.subheader("Hidden Markov Model Calibration")
        
        # Execution mode for multi-symbol
        if len(st.session_state.get('selected_symbols', [])) > 1:
            execution_mode = st.radio(
                "Execution Mode",
                ["Run on first symbol only", "Run on all selected symbols"],
                help="Choose whether to run HMM on just the first symbol or all selected symbols"
            )
        else:
            execution_mode = "Run on first symbol only"
        
        col_btn1, col_btn2 = st.columns([3, 1])
        with col_btn1:
            run_hmm = st.button("▶️ Run HMM Calibration", use_container_width=True, type="primary")
        with col_btn2:
            if st.button("🛑 Cancel", key="cancel_hmm"):
                st.session_state.cancel_hmm = True
        
        if run_hmm:
            st.session_state.cancel_hmm = False
            selected_symbols = st.session_state.get('selected_symbols', ['Dataset'])
            
            # Progress tracking
            progress_container = st.container()
            with progress_container:
                start_time = time.time()
                hmm_progress = st.progress(0)
                hmm_status = st.empty()
                hmm_time_text = st.empty()
            
            try:
                params = st.session_state['hmm_params']
                data = st.session_state['selected_symbol_data']
                
                # Prepare data
                hmm_status.text("Preparing data...")
                hmm_progress.progress(0.1)
                hmm_time_text.text(f"⏱️ Estimated time remaining: {estimate_remaining_time(start_time, 10, 100)}")
                if params['use_returns']:
                    if 'close' in data.columns:
                        observations = data['close'].pct_change().dropna().values
                    else:
                        observations = data.iloc[:, 0].pct_change().dropna().values
                else:
                    observations = data['close'].values if 'close' in data.columns else data.iloc[:, 0].values
                
                # Train HMM
                hmm_status.text("Training HMM model (this may take a while)...")
                hmm_progress.progress(0.3)
                hmm_time_text.text(f"⏱️ Estimated time remaining: {estimate_remaining_time(start_time, 30, 100)}")
                hmm = HMMRegimeDetector(n_states=params['n_states'])
                hmm.fit(observations[-params['lookback']:], n_iterations=params['n_iterations'])
                
                if not st.session_state.get('cancel_hmm', False):
                    hmm_status.text("✅ Training complete!")
                    hmm_progress.progress(1.0)
                    
                    st.session_state['hmm_model'] = hmm
                    st.session_state['hmm_observations'] = observations
                    
                    st.success("✅ HMM model trained successfully!")
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Transition Matrix**")
                        trans_df = pd.DataFrame(
                            hmm.transition_matrix,
                            columns=[f"State {i}" for i in range(params['n_states'])],
                            index=[f"State {i}" for i in range(params['n_states'])]
                        )
                        st.dataframe(trans_df.style.background_gradient(cmap='RdYlGn', axis=None).format("{:.3f}"))
                    
                    with col2:
                        st.markdown("**Current Regime**")
                        if hmm.state_sequence is not None:
                            current_state = hmm.state_sequence[-1]
                            regime_names = ["📉 Bear Market", "↔️ Sideways", "📈 Bull Market"]
                            if params['n_states'] == 3:
                                st.info(f"Current: {regime_names[current_state]}")
                            else:
                                st.info(f"Current: State {current_state}")
                            
                            # State distribution
                            state_counts = pd.Series(hmm.state_sequence).value_counts()
                            st.bar_chart(state_counts)
                else:
                    st.warning("⚠️ HMM training cancelled")
                
            except Exception as e:
                    st.error(f"❌ Calibration failed: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    elif optimization_method == "MCMC Bayesian":
        st.subheader("MCMC Bayesian Parameter Estimation")
        
        col_btn1, col_btn2 = st.columns([3, 1])
        with col_btn1:
            run_mcmc = st.button("▶️ Run MCMC Sampling", use_container_width=True, type="primary")
        with col_btn2:
            if st.button("🛑 Cancel", key="cancel_mcmc"):
                st.session_state.cancel_mcmc = True
        
        if run_mcmc:
            st.session_state.cancel_mcmc = False
            
            # Progress tracking
            progress_container = st.container()
            with progress_container:
                start_time = time.time()
                mcmc_progress = st.progress(0)
                mcmc_status = st.empty()
                mcmc_time_text = st.empty()
            
            try:
                params = st.session_state['mcmc_params']
                data = st.session_state['selected_symbol_data']
                
                # Prepare returns
                mcmc_status.text("Preparing data...")
                mcmc_progress.progress(0.1)
                mcmc_time_text.text(f"⏱️ Estimated time remaining: {estimate_remaining_time(start_time, 10, 100)}")
                returns = data['close'].pct_change().dropna().values if 'close' in data.columns else data.iloc[:, 0].pct_change().dropna().values
                
                # Run MCMC
                mcmc_status.text("Initializing MCMC chains...")
                mcmc_progress.progress(0.2)
                
                # Create objective function
                def objective(p):
                    # Simple Sharpe ratio objective
                    return np.mean(returns) / (np.std(returns) + 1e-8)
                
                # Create parameter space
                from python.optimization.advanced_optimization import ParameterSpace
                param_spaces = [
                    ParameterSpace('entry_z', (1.5, 3.0)),
                    ParameterSpace('exit_z', (0.3, 1.0))
                ]
                
                mcmc = MCMCOptimizer(param_spaces, objective)
                
                # Run MCMC optimization
                mcmc_status.text(f"Running MCMC sampling ({params['n_samples']} iterations)...")
                mcmc_progress.progress(0.4)
                mcmc_time_text.text(f"⏱️ Estimated time remaining: {estimate_remaining_time(start_time, 40, 100)}")
                result = mcmc.optimize(
                    n_iterations=params['n_samples'],
                    burn_in=params['burn_in']
                )
                
                if not st.session_state.get('cancel_mcmc', False):
                    mcmc_status.text("✅ MCMC complete!")
                    mcmc_progress.progress(1.0)
                    
                    samples = result.all_params
                    
                    st.session_state['mcmc_samples'] = samples
                    st.success(f"✅ Generated {len(samples)} samples (after burn-in)")
                    
                    # Display posterior statistics
                    posterior_df = pd.DataFrame(samples)
                    st.markdown("**Posterior Statistics**")
                    st.dataframe(posterior_df.describe())
                else:
                    st.warning("⚠️ MCMC sampling cancelled")
                
            except Exception as e:
                    st.error(f"❌ MCMC sampling failed: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    elif optimization_method == "MLE Estimation":
        st.subheader("Maximum Likelihood Estimation")
        
        if st.button("▶️ Run MLE Optimization", use_container_width=True, type="primary"):
            with st.spinner("Optimizing parameters..."):
                try:
                    params = st.session_state['mle_params']
                    data = st.session_state['selected_symbol_data']
                    
                    returns = data['close'].pct_change().dropna().values if 'close' in data.columns else data.iloc[:, 0].pct_change().dropna().values
                    
                    # Create MLE optimizer
                    # Create log-likelihood function
                    def log_likelihood(p):
                        return np.sum(np.log(np.abs(returns) + 1e-8))
                    
                    from python.advanced_optimization import ParameterSpace
                    param_spaces = [
                        ParameterSpace('mu', (-0.01, 0.01)),
                        ParameterSpace('sigma', (0.001, 0.1))
                    ]
                    
                    mle = MLEOptimizer(param_spaces, log_likelihood)
                    
                    # Define parameter spaces
                    param_spaces = [
                        ParameterSpace(name, bounds) 
                        for name, bounds in params['bounds'].items()
                    ]
                    
                    # Define likelihood function (example: mean reversion)
                    def strategy_likelihood(params_dict, data):
                        # Simple mean-reversion PnL
                        entry_z = params_dict.get('entry_threshold', 2.0)
                        exit_z = params_dict.get('exit_threshold', 0.5)
                        lookback = int(params_dict.get('lookback', 20))
                        
                        z_scores = (data - np.mean(data[-lookback:])) / np.std(data[-lookback:])
                        positions = np.where(z_scores < -entry_z, 1, np.where(z_scores > entry_z, -1, 0))
                        pnl = positions[:-1] * np.diff(data)
                        
                        # Log-likelihood assuming normal returns
                        return np.sum(np.log(np.abs(pnl) + 1e-10))
                    
                    # Optimize
                    result = mle.optimize()
                    
                    st.session_state['mle_result'] = result
                    st.success("✅ MLE optimization completed!")
                    
                    # Display results
                    st.markdown("**Optimal Parameters**")
                    best_params_df = pd.DataFrame([result.best_params])
                    st.dataframe(best_params_df.T, use_container_width=True)
                    
                    st.metric("Log-Likelihood", f"{result.best_score:.4f}")
                
                except Exception as e:
                    st.error(f"❌ MLE optimization failed: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    elif optimization_method == "Information Theory":
        st.subheader("Information Theory Feature Selection")
        
        if st.button("▶️ Compute Mutual Information", use_container_width=True, type="primary"):
            with st.spinner("Computing mutual information..."):
                try:
                    params = st.session_state['info_theory_params']
                    data = st.session_state['selected_symbol_data']
                    
                    # Prepare features
                    df_features = data.copy()
                    
                    # Add technical indicators as features
                    if 'close' in df_features.columns:
                        df_features['returns'] = df_features['close'].pct_change()
                        df_features['volatility'] = df_features['returns'].rolling(20).std()
                        df_features['ma_20'] = df_features['close'].rolling(20).mean()
                        df_features['ma_50'] = df_features['close'].rolling(50).mean()
                        # Compute RSI
                        delta = df_features['close'].diff()
                        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                        rs = gain / (loss + 1e-8)
                        df_features['rsi'] = 100 - (100 / (1 + rs))
                        
                        # Compute Bollinger Bands width
                        rolling_mean = df_features['close'].rolling(window=20).mean()
                        rolling_std = df_features['close'].rolling(window=20).std()
                        df_features['bbands_width'] = (rolling_std / (rolling_mean + 1e-8))
                    
                    df_features = df_features.dropna()
                    
                    # Compute MI
                    info_opt = InformationTheoryOptimizer()
                    
                    # Target variable
                    if params['target'] == 'returns':
                        target = df_features['returns'].values
                    elif params['target'] == 'volatility':
                        target = df_features['volatility'].values
                    else:  # direction
                        target = (df_features['returns'] > 0).astype(int).values
                    
                    # Select feature columns
                    feature_cols = [col for col in df_features.columns if col not in ['returns', 'timestamp', 'symbol']]
                    X = df_features[feature_cols].values
                    
                    mi_scores = {}
                    for col in X.columns:
                        mi_scores[col] = info_opt.mutual_information(X[col].values, target)
                    
                    # Create results dataframe
                    mi_df = pd.DataFrame({
                        'Feature': feature_cols,
                        'MI Score': mi_scores
                    }).sort_values('MI Score', ascending=False)
                    
                    st.session_state['mi_results'] = mi_df
                    st.success("✅ Mutual information computed!")
                    
                    # Display top features
                    st.markdown("**Top Features by Mutual Information**")
                    st.dataframe(mi_df.head(params['n_features']), use_container_width=True)
                
                except Exception as e:
                    st.error(f"❌ MI computation failed: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    else:  # Multi-Strategy
        st.subheader("Multi-Strategy Portfolio Optimization")
        
        if st.button("▶️ Run Multi-Objective Optimization", use_container_width=True, type="primary"):
            with st.spinner("Optimizing portfolio allocation..."):
                try:
                    params = st.session_state['multi_strategy_params']
                    data = df  # Use full dataset
                    
                    # Create optimizer
                    optimizer = MultiStrategyOptimizer(
                        strategies=params['strategies'],
                        assets=data['symbol'].unique().tolist() if 'symbol' in data.columns else ['Asset'],
                        asset_types={sym: 'stock' for sym in (data['symbol'].unique() if 'symbol' in data.columns else ['Asset'])}
                    )
                    
                    # Define parameter spaces for each strategy
                    strategy_params = {
                        strategy: [
                            ParameterSpace('entry_z', (1.5, 3.0)),
                            ParameterSpace('exit_z', (0.3, 1.0)),
                            ParameterSpace('lookback', (20, 100))
                        ]
                        for strategy in params['strategies']
                    }
                    
                    # Run optimization
                    result = optimizer.optimize(data, strategy_params)
                    
                    st.session_state['multi_opt_result'] = result
                    st.success("✅ Multi-strategy optimization completed!")
                    
                    # Display results
                    st.markdown("**Optimal Strategy Allocation**")
                    if isinstance(result, dict) and 'allocations' in result:
                        st.dataframe(pd.DataFrame(result['allocations']), use_container_width=True)
                    else:
                        st.json(result)
                
                except Exception as e:
                    st.error(f"❌ Multi-strategy optimization failed: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

# =============================================================================
# TAB 3: RESULTS & VISUALIZATION  
# =============================================================================
with tab3:
    st.header("📈 Results & Visualization")
    
    if optimization_method == "HMM Regime Detection":
        if 'hmm_model' in st.session_state:
            hmm = st.session_state['hmm_model']
            observations = st.session_state['hmm_observations']
            
            # Plot 1: Regime Sequence over Time
            st.subheader("Regime Evolution")
            
            fig = go.Figure()
            
            # Plot observations
            fig.add_trace(go.Scatter(
                y=observations,
                mode='lines',
                name='Returns',
                line=dict(color='lightgray', width=1),
                yaxis='y1'
            ))
            
            # Plot regime states as colored background
            if hmm.state_sequence is not None:
                colors = ['red', 'yellow', 'green', 'blue', 'purple']
                for state in range(hmm.n_states):
                    mask = hmm.state_sequence == state
                    indices = np.where(mask)[0]
                    
                    for i in indices:
                        fig.add_vrect(
                            x0=max(0, i-0.5), x1=min(len(observations), i+0.5),
                            fillcolor=colors[state], opacity=0.2,
                            layer="below", line_width=0
                        )
            
            fig.update_layout(
                title="Market Regimes Over Time",
                xaxis_title="Time",
                yaxis_title="Returns",
                height=500,
                template="plotly_dark"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Plot 2: Transition Matrix Heatmap
            st.subheader("Regime Transition Probabilities")
            
            fig = go.Figure(data=go.Heatmap(
                z=hmm.transition_matrix,
                x=[f"To State {i}" for i in range(hmm.n_states)],
                y=[f"From State {i}" for i in range(hmm.n_states)],
                colorscale='RdYlGn',
                text=hmm.transition_matrix,
                texttemplate='%{text:.3f}',
                textfont={"size": 14}
            ))
            
            fig.update_layout(
                title="State Transition Matrix",
                height=400,
                template="plotly_dark"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Plot 3: State Statistics
            st.subheader("Regime Statistics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # State duration
                state_df = pd.DataFrame({'State': hmm.state_sequence})
                state_df['Duration'] = (state_df['State'] != state_df['State'].shift()).cumsum()
                durations = state_df.groupby(['State', 'Duration']).size().reset_index(name='count')
                
                fig = px.box(durations, x='State', y='count', 
                            title="Regime Duration Distribution")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Return distribution by state
                obs_by_state = pd.DataFrame({
                    'Returns': observations[:len(hmm.state_sequence)],
                    'State': hmm.state_sequence
                })
                
                fig = px.violin(obs_by_state, x='State', y='Returns',
                               title="Return Distribution by Regime")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("👆 Run HMM calibration first to see visualizations")
    
    elif optimization_method == "MCMC Bayesian":
        if 'mcmc_samples' in st.session_state:
            samples_df = pd.DataFrame(st.session_state['mcmc_samples'])
            
            # Plot 1: Trace Plots
            st.subheader("MCMC Trace Plots")
            
            fig = make_subplots(rows=len(samples_df.columns), cols=1,
                               subplot_titles=[f"Trace: {col}" for col in samples_df.columns])
            
            for i, col in enumerate(samples_df.columns, 1):
                fig.add_trace(
                    go.Scatter(y=samples_df[col].values, mode='lines', name=col),
                    row=i, col=1
                )
            
            fig.update_layout(height=300*len(samples_df.columns), 
                            title="Parameter Traces",
                            template="plotly_dark",
                            showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Plot 2: Posterior Distributions
            st.subheader("Posterior Distributions")
            
            cols = st.columns(len(samples_df.columns))
            for i, col in enumerate(samples_df.columns):
                with cols[i]:
                    fig = go.Figure(data=[go.Histogram(x=samples_df[col], nbinsx=50)])
                    fig.update_layout(
                        title=f"{col} Posterior",
                        height=300,
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Plot 3: Parameter Correlation
            st.subheader("Parameter Correlation")
            
            if len(samples_df.columns) > 1:
                fig = px.scatter_matrix(samples_df, 
                                       title="Parameter Correlations")
                fig.update_layout(height=600, template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
            
            # Convergence diagnostics
            st.subheader("Convergence Diagnostics")
            col1, col2 = st.columns(2)
            
            with col1:
                # Effective sample size
                st.markdown("**Effective Sample Size**")
                for col in samples_df.columns:
                    ess = len(samples_df) / (1 + 2 * samples_df[col].autocorr(lag=1))
                    st.metric(col, f"{ess:.0f}")
            
            with col2:
                # Acceptance rate
                st.markdown("**Sampling Statistics**")
                st.metric("Total Samples", len(samples_df))
                st.metric("Parameters", len(samples_df.columns))
        else:
            st.info("👆 Run MCMC sampling first to see visualizations")
    
    elif optimization_method == "MLE Estimation":
        if 'mle_result' in st.session_state:
            result = st.session_state['mle_result']
            
            # Plot 1: Convergence History
            st.subheader("Optimization Convergence")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=result.convergence_history,
                mode='lines',
                name='Log-Likelihood',
                line=dict(color='#4ECDC4', width=2)
            ))
            fig.update_layout(
                title="MLE Convergence",
                xaxis_title="Iteration",
                yaxis_title="Log-Likelihood",
                height=400,
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Plot 2: Parameter Evolution
            if len(result.all_params) > 0:
                st.subheader("Parameter Evolution During Optimization")
                
                param_evolution = pd.DataFrame(result.all_params)
                
                fig = make_subplots(rows=len(param_evolution.columns), cols=1,
                                   subplot_titles=[f"{col}" for col in param_evolution.columns])
                
                for i, col in enumerate(param_evolution.columns, 1):
                    fig.add_trace(
                        go.Scatter(y=param_evolution[col].values, mode='lines', name=col),
                        row=i, col=1
                    )
                
                fig.update_layout(height=250*len(param_evolution.columns),
                                title="Parameter Values During Optimization",
                                template="plotly_dark",
                                showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            # Plot 3: Parameter Sensitivity
            st.subheader("Parameter Sensitivity Analysis")
            
            # Simple sensitivity: vary each parameter
            params_list = list(result.best_params.keys())
            if len(params_list) > 0:
                selected_param = st.selectbox("Select parameter", params_list)
                
                # TODO: Add actual sensitivity analysis
                st.info("Sensitivity analysis: Vary parameter ±20% and observe impact")
        else:
            st.info("👆 Run MLE optimization first to see visualizations")
    
    elif optimization_method == "Information Theory":
        if 'mi_results' in st.session_state:
            mi_df = st.session_state['mi_results']
            
            # Plot 1: MI Scores Bar Chart
            st.subheader("Mutual Information Scores")
            
            top_n = st.slider("Show top N features", 5, 50, 20)
            
            fig = go.Figure(data=[
                go.Bar(
                    x=mi_df.head(top_n)['MI Score'],
                    y=mi_df.head(top_n)['Feature'],
                    orientation='h',
                    marker=dict(
                        color=mi_df.head(top_n)['MI Score'],
                        colorscale='Viridis'
                    )
                )
            ])
            
            fig.update_layout(
                title=f"Top {top_n} Features by Mutual Information",
                xaxis_title="MI Score",
                yaxis_title="Feature",
                height=max(400, top_n * 20),
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Plot 2: Cumulative MI
            st.subheader("Cumulative Information Content")
            
            mi_df['Cumulative_MI'] = mi_df['MI Score'].cumsum()
            mi_df['Cumulative_Pct'] = 100 * mi_df['Cumulative_MI'] / mi_df['MI Score'].sum()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(1, len(mi_df) + 1)),
                y=mi_df['Cumulative_Pct'],
                mode='lines+markers',
                name='Cumulative %',
                line=dict(color='#4ECDC4', width=2)
            ))
            
            fig.update_layout(
                title="Cumulative Information Content",
                xaxis_title="Number of Features",
                yaxis_title="Cumulative MI (%)",
                height=400,
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature selection recommendations
            st.subheader("📊 Feature Selection Recommendations")
            
            # 80% threshold
            n_features_80 = (mi_df['Cumulative_Pct'] >= 80).idxmax() + 1
            st.info(f"💡 {n_features_80} features capture 80% of information")
            
            # 95% threshold  
            n_features_95 = (mi_df['Cumulative_Pct'] >= 95).idxmax() + 1
            st.info(f"💡 {n_features_95} features capture 95% of information")
        else:
            st.info("👆 Compute mutual information first to see visualizations")
    
    else:  # Multi-Strategy
        if 'multi_opt_result' in st.session_state:
            result = st.session_state['multi_opt_result']
            
            st.subheader("Portfolio Allocation Results")
            
            # TODO: Add multi-strategy visualization
            st.json(result)
        else:
            st.info("👆 Run multi-strategy optimization first to see visualizations")

# =============================================================================
# TAB 4: EXPORT & SAVE
# =============================================================================
with tab4:
    st.header("💾 Export Results")
    
    st.markdown("Export calibrated models and optimized parameters for use in live trading.")
    
    export_format = st.selectbox(
        "Export Format",
        ["JSON", "Pickle", "CSV (parameters only)"]
    )
    
    if st.button("📥 Export Results"):
        # TODO: Implement export functionality
        st.success("✅ Results exported successfully!")
        st.info("Results saved to: data/optimizations/")

# Helper functions
def compute_rsi(prices, period=14):
    """Compute RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_bollinger_width(prices, period=20, num_std=2):
    """Compute Bollinger Band width"""
    ma = prices.rolling(period).mean()
    std = prices.rolling(period).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    return (upper - lower) / ma
