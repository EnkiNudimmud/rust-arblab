"""
Enhanced Live Trading Configuration Components
===============================================

This file contains UI components for:
1. Virtual portfolio management
2. Advanced parameter optimization (HMM, MCMC, MLE, Information Theory)
3. Multi-strategy/multi-asset configuration
4. Portfolio merging with labs
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Optional
from datetime import datetime
from itertools import product

try:
    from python.virtual_portfolio import VirtualPortfolio, load_portfolio, list_portfolios
except ImportError:
    list_portfolios = lambda: []
    VirtualPortfolio = None
    load_portfolio = None

try:
    from python.advanced_optimization import (
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
    HMMRegimeDetector = None
    MCMCOptimizer = None
    MLEOptimizer = None
    InformationTheoryOptimizer = None
    MultiStrategyOptimizer = None
    ParameterSpace = None
    OptimizationResult = None
    RUST_AVAILABLE = False

try:
    from python.adaptive_strategies import (
        AdaptiveMeanReversion,
        AdaptiveMomentum,
        AdaptiveStatArb
    )
except ImportError:
    AdaptiveMeanReversion = None
    AdaptiveMomentum = None
    AdaptiveStatArb = None

# Available strategies dictionary
AVAILABLE_STRATEGIES = {
    'Mean Reversion': 'meanrev',
    'Statistical Arbitrage': 'stat_arb',
    'Pairs Trading': 'pairs',
    'Market Making': 'market_making'
}

def configure_virtual_portfolio():
    """Configure virtual portfolio settings"""
    
    with st.expander("💼 Virtual Portfolio", expanded=False):
        st.markdown("**Portfolio Management**")
        
        # Portfolio selection/creation
        existing_portfolios = list_portfolios()
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if existing_portfolios:
                portfolio_name = st.selectbox(
                    "Select Portfolio",
                    ["[Create New]"] + existing_portfolios,
                    key="portfolio_selector"
                )
            else:
                portfolio_name = "[Create New]"
                st.info("No existing portfolios found")
        
        with col2:
            if st.button("🔄 Refresh"):
                st.rerun()
        
        # Create new or load existing
        if portfolio_name == "[Create New]":
            new_name = st.text_input("Portfolio Name", value="live_trading_" + datetime.now().strftime("%Y%m%d"))
            initial_cash = st.number_input("Initial Cash", value=100000.0, min_value=1000.0, step=1000.0)
            
            if st.button("Create Portfolio"):
                from python.virtual_portfolio import VirtualPortfolio
                portfolio = VirtualPortfolio(name=new_name, initial_cash=initial_cash)
                st.session_state.virtual_portfolio = portfolio
                st.success(f"✓ Created portfolio: {new_name}")
                st.rerun()
        else:
            # Load existing portfolio
            if 'virtual_portfolio' not in st.session_state or \
               st.session_state.virtual_portfolio.name != portfolio_name:
                from python.virtual_portfolio import load_portfolio
                portfolio = load_portfolio(portfolio_name)
                st.session_state.virtual_portfolio = portfolio
                st.success(f"✓ Loaded portfolio: {portfolio_name}")
        
        # Show portfolio summary
        if 'virtual_portfolio' in st.session_state:
            portfolio = st.session_state.virtual_portfolio
            if portfolio is not None and hasattr(portfolio, 'get_metrics'):
                metrics = portfolio.get_metrics()
            else:
                metrics = {'total_value': 0, 'cash': 0, 'total_pnl': 0, 'total_pnl_pct': 0}
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Value", f"${metrics['total_value']:,.2f}")
            with col2:
                st.metric("Cash", f"${metrics['cash']:,.2f}")
            with col3:
                st.metric("P&L", f"${metrics['total_pnl']:,.2f}", 
                         delta=f"{metrics['total_pnl_pct']:.2f}%")
            
            # Portfolio actions
            st.markdown("**Actions**")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📊 View Portfolio Details"):
                    st.session_state['show_portfolio_details'] = True
            
            with col2:
                if st.button("🔗 Merge with Lab Portfolio"):
                    st.session_state['show_merge_dialog'] = True


def configure_regime_detection():
    """Configure HMM regime detection with adaptive strategies"""
    
    with st.expander("🔮 Regime-Adaptive Trading", expanded=False):
        st.markdown("**Auto-Adapt Strategy Parameters to Market Conditions**")
        
        if RUST_AVAILABLE:
            st.success("🚀 Rust-Accelerated HMM: ON")
        else:
            st.warning("⚠️ Rust OFF (slower HMM)")
        
        enable_adaptive = st.checkbox(
            "Enable Adaptive Strategies",
            value=False,
            help="Automatically detect market regimes and adapt strategy parameters"
        )
        
        if enable_adaptive:
            # Strategy selection for adaptation
            st.markdown("**Adaptive Strategy Type**")
            adaptive_strategy_type = st.selectbox(
                "Base Strategy",
                ["Mean Reversion", "Momentum", "Statistical Arbitrage"],
                help="Strategy to make regime-adaptive"
            )
            
            # HMM configuration
            st.markdown("**HMM Configuration**")
            col1, col2 = st.columns(2)
            
            with col1:
                n_states = st.slider(
                    "Number of Regimes",
                    2, 5, 3,
                    help="Typical: 3 (bull/bear/sideways)"
                )
                
                lookback_period = st.number_input(
                    "Training Period (bars)",
                    min_value=100,
                    max_value=5000,
                    value=500,
                    step=100,
                    help="Historical data for HMM training"
                )
            
            with col2:
                update_frequency = st.number_input(
                    "Update Frequency (bars)",
                    min_value=10,
                    max_value=500,
                    value=100,
                    step=10,
                    help="Bars between HMM retraining"
                )
                
                auto_retrain = st.checkbox(
                    "Auto-Retrain",
                    value=True,
                    help="Automatically retrain HMM periodically"
                )
            
            # Base parameters
            st.markdown("**Base Strategy Parameters**")
            with st.form("adaptive_params_form"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    entry_thresh = st.number_input("Entry Threshold", 0.5, 5.0, 2.0, 0.1)
                    exit_thresh = st.number_input("Exit Threshold", 0.1, 2.0, 0.5, 0.1)
                
                with col2:
                    pos_size = st.number_input("Position Size", 0.1, 2.0, 1.0, 0.1)
                    stop_loss_pct = st.number_input("Stop Loss (%)", 0.5, 10.0, 2.0, 0.5)
                
                with col3:
                    take_profit_pct = st.number_input("Take Profit (%)", 1.0, 20.0, 5.0, 0.5)
                    max_holding = st.number_input("Max Holding (bars)", 5, 100, 20, 5)
                
                submitted = st.form_submit_button("🚀 Initialize Adaptive Strategy")
                
                if submitted:
                    # Check if adaptive strategies are available
                    if AdaptiveMeanReversion is None:
                        st.error("❌ Adaptive strategies module not available. Check imports.")
                        return
                    
                    # Create adaptive strategy
                    base_config = {
                        'entry_threshold': entry_thresh,
                        'exit_threshold': exit_thresh,
                        'position_size': pos_size,
                        'stop_loss': stop_loss_pct / 100,
                        'take_profit': take_profit_pct / 100,
                        'max_holding_period': max_holding
                    }
                    
                    try:
                        if adaptive_strategy_type == "Mean Reversion":
                            if AdaptiveMeanReversion is None:
                                raise ImportError("AdaptiveMeanReversion not available")
                            strategy = AdaptiveMeanReversion(
                                n_regimes=n_states,
                                lookback_period=lookback_period,
                                update_frequency=update_frequency,
                                base_config=base_config
                            )
                        elif adaptive_strategy_type == "Momentum":
                            if AdaptiveMomentum is None:
                                raise ImportError("AdaptiveMomentum not available")
                            strategy = AdaptiveMomentum(
                                n_regimes=n_states,
                                lookback_period=lookback_period,
                                update_frequency=update_frequency,
                                base_config=base_config
                            )
                        else:  # Statistical Arbitrage
                            if AdaptiveStatArb is None:
                                raise ImportError("AdaptiveStatArb not available")
                            strategy = AdaptiveStatArb(
                                n_regimes=n_states,
                                lookback_period=lookback_period,
                                update_frequency=update_frequency,
                                base_config=base_config
                            )
                        
                        st.session_state['adaptive_strategy'] = strategy
                        st.session_state['adaptive_enabled'] = True
                        st.success(f"✓ Initialized {adaptive_strategy_type} with {n_states} regimes")
                        st.rerun()
                    
                    except Exception as e:
                        st.error(f"❌ Failed to initialize: {str(e)}")
            
            # Show current regime if strategy initialized
            if st.session_state.get('adaptive_strategy'):
                strategy = st.session_state['adaptive_strategy']
                
                st.markdown("---")
                st.markdown("**Current Status**")
                
                if strategy.hmm_trained and strategy.current_regime is not None:
                    regime_names = ["📉 Bear Market", "↔️ Sideways", "📈 Bull Market"]
                    current_regime = strategy.current_regime
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Current Regime", regime_names[current_regime])
                    
                    with col2:
                        current_config = strategy.get_current_config()
                        st.metric("Entry Threshold", f"{current_config.entry_threshold:.2f}")
                    
                    with col3:
                        st.metric("Position Size", f"{current_config.position_size:.2f}x")
                    
                    # Show regime transition probabilities
                    trans_matrix = strategy.get_transition_probabilities()
                    if trans_matrix is not None:
                        st.markdown("**Regime Transition Probabilities**")
                        trans_df = pd.DataFrame(
                            trans_matrix,
                            index=[f"State {i}" for i in range(n_states)],
                            columns=[f"State {i}" for i in range(n_states)]
                        )
                        st.dataframe(trans_df.style.format("{:.3f}"), use_container_width=True)
                    
                    # Performance by regime
                    regime_perf = strategy.get_regime_performance()
                    if not regime_perf.empty:
                        st.markdown("**Performance by Regime**")
                        st.dataframe(regime_perf, use_container_width=True)
                
                else:
                    st.info("⏳ Waiting for sufficient data to train HMM...")
                
                # Manual retrain button
                if st.button("🔄 Retrain HMM Now"):
                    st.session_state['force_hmm_retrain'] = True
                    st.info("Will retrain on next update")


def configure_parameter_optimization():
    """Configure advanced parameter optimization"""
    
    with st.expander("⚙️ Parameter Optimization", expanded=False):
        st.markdown("**Advanced Optimization Methods**")
        
        optimization_method = st.selectbox(
            "Optimization Method",
            [
                "None - Use Manual Parameters",
                "MCMC - Bayesian Sampling",
                "MLE - Maximum Likelihood",
                "Information Theory - Feature Selection",
                "Grid Search - Exhaustive",
                "Differential Evolution"
            ]
        )
        
        # Show method description
        method_descriptions = {
            "MCMC - Bayesian Sampling": "🔬 Explores parameter space using Markov Chain Monte Carlo. Best for complex posteriors and uncertainty quantification.",
            "MLE - Maximum Likelihood": "📊 Finds parameters that maximize likelihood of observed data. Fast and efficient for well-behaved distributions.",
            "Information Theory - Feature Selection": "🧠 Uses mutual information to select features, then optimizes parameters. Good for high-dimensional spaces.",
            "Grid Search - Exhaustive": "🔍 Tests all combinations in a grid. Thorough but computationally expensive.",
            "Differential Evolution": "🧬 Evolutionary algorithm that's robust to local minima. Good for complex, non-convex objectives."
        }
        
        if optimization_method in method_descriptions:
            st.info(method_descriptions[optimization_method])
        
        if optimization_method != "None - Use Manual Parameters":
            st.markdown("**Parameter Space**")
            
            # Define parameter ranges for selected strategy
            strategy = st.session_state.get('live_strategy')
            if strategy:
                st.text(f"Optimizing: {strategy}")
                
                # Example parameter spaces (would be strategy-specific)
                with st.form("param_space_form"):
                    st.markdown("Define search ranges:")
                    
                    entry_z_range = st.slider(
                        "Entry Z-Score Range",
                        0.5, 5.0, (1.5, 3.0), 0.1
                    )
                    
                    exit_z_range = st.slider(
                        "Exit Z-Score Range",
                        0.1, 2.0, (0.3, 1.0), 0.1
                    )
                    
                    lookback_range = st.slider(
                        "Lookback Period Range",
                        20, 200, (40, 100), 10
                    )
                    
                    # Method-specific parameters
                    method_params = {}
                    if "MCMC" in optimization_method:
                        st.markdown("**MCMC Settings**")
                        method_params['n_iterations'] = st.number_input(
                            "MCMC Iterations",
                            1000, 50000, 10000, 1000
                        )
                        method_params['burn_in'] = st.number_input(
                            "Burn-in Period",
                            100, 10000, 1000, 100
                        )
                    elif "Grid Search" in optimization_method:
                        st.markdown("**Grid Search Settings**")
                        method_params['grid_points'] = st.slider(
                            "Points per dimension",
                            3, 10, 5, 1
                        )
                        total_evals = method_params['grid_points'] ** 3
                        st.caption(f"Total evaluations: {total_evals}")
                    elif "Differential Evolution" in optimization_method:
                        st.markdown("**DE Settings**")
                        method_params['maxiter'] = st.number_input(
                            "Max Iterations",
                            100, 5000, 1000, 100
                        )
                        method_params['popsize'] = st.slider(
                            "Population Size",
                            5, 30, 15, 1
                        )
                    
                    submitted = st.form_submit_button("🚀 Run Optimization")
                    
                    if submitted:
                        with st.spinner(f"Running {optimization_method}..."):
                            run_parameter_optimization(
                                method=optimization_method,
                                param_ranges={
                                    'entry_z': entry_z_range,
                                    'exit_z': exit_z_range,
                                    'lookback': lookback_range
                                },
                                **method_params
                            )
            else:
                st.warning("⚠️ Enable a strategy first to optimize parameters")
        
        # Show optimization results if available
        if 'optimization_result' in st.session_state:
            result = st.session_state.optimization_result
            
            st.markdown("---")
            st.markdown("### 🎯 Optimization Results")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Method", result.method)
            with col2:
                st.metric("Best Score", f"{result.best_score:.4f}")
            with col3:
                st.metric("Iterations", result.n_iterations)
            
            st.markdown("**Optimal Parameters**")
            params_df = pd.DataFrame([result.best_params])
            st.dataframe(params_df, use_container_width=True)
            
            # Show parameter distributions for MCMC
            if result.method == "MCMC" and len(result.all_params) > 0:
                st.markdown("**Parameter Distributions**")
                for param_name in result.best_params.keys():
                    values = [p[param_name] for p in result.all_params]
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=values,
                        name=param_name,
                        nbinsx=30,
                        marker_color='#4ECDC4'
                    ))
                    fig.update_layout(
                        title=f"Distribution of {param_name}",
                        xaxis_title=param_name,
                        yaxis_title="Frequency",
                        template="plotly_dark",
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Convergence plot
            col1, col2 = st.columns([3, 1])
            with col1:
                if result.convergence_history and len(result.convergence_history) > 0:
                    plot_optimization_convergence(result)
            with col2:
                st.markdown("**Actions**")
                if st.button("✅ Apply Parameters"):
                    st.session_state.live_strategy_params = result.best_params
                    st.success("Parameters applied!")
                    st.rerun()
                if st.button("🗑️ Clear Results"):
                    del st.session_state.optimization_result
                    st.rerun()


def configure_multi_strategy_mode():
    """Configure multi-strategy, multi-asset trading"""
    
    with st.expander("🎯 Multi-Strategy Mode", expanded=False):
        st.markdown("**Multiple Strategies × Multiple Assets**")
        
        enable_multi = st.checkbox(
            "Enable Multi-Strategy Mode",
            value=False,
            help="Run multiple strategies across different asset types"
        )
        
        if enable_multi:
            st.markdown("**Asset Types**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                enable_stocks = st.checkbox("📈 Stocks", value=True)
                enable_crypto = st.checkbox("₿ Crypto", value=True)
            
            with col2:
                enable_etfs = st.checkbox("📊 ETFs", value=False)
                enable_options = st.checkbox("📜 Options", value=False)
            
            asset_types = []
            if enable_stocks:
                asset_types.append('stock')
            if enable_crypto:
                asset_types.append('crypto')
            if enable_etfs:
                asset_types.append('etf')
            if enable_options:
                asset_types.append('option')
            
            st.session_state['enabled_asset_types'] = asset_types
            
            # Strategy allocation
            st.markdown("**Strategy Selection**")
            
            available_strategies = list(AVAILABLE_STRATEGIES.keys())
            selected_strategies = st.multiselect(
                "Active Strategies",
                available_strategies,
                default=available_strategies[:2] if len(available_strategies) >= 2 else available_strategies
            )
            
            if selected_strategies:
                st.markdown("**Capital Allocation**")
                
                allocations = {}
                remaining = 100.0
                
                for strategy in selected_strategies[:-1]:
                    alloc = st.slider(
                        f"{strategy} (%)",
                        0.0, remaining, min(20.0, remaining),
                        1.0,
                        key=f"alloc_{strategy}"
                    )
                    allocations[strategy] = alloc
                    remaining -= alloc
                
                # Last strategy gets remainder
                allocations[selected_strategies[-1]] = remaining
                st.info(f"{selected_strategies[-1]}: {remaining:.1f}%")
                
                # Save configuration
                st.session_state['multi_strategy_config'] = {
                    'strategies': selected_strategies,
                    'allocations': allocations,
                    'asset_types': asset_types
                }
                
                # Optimize allocation
                if st.button("🎯 Optimize Allocation"):
                    with st.spinner("Running multi-objective optimization..."):
                        optimize_multi_strategy_allocation()


def train_hmm_model(n_states: int, lookback: int):
    """Train HMM model on historical data"""
    try:
        from python.advanced_optimization import HMMRegimeDetector
        
        # Get historical returns from buffer
        buffer = st.session_state.get('live_data_buffer', [])
        if len(buffer) < lookback:
            st.warning(f"Need at least {lookback} data points. Current: {len(buffer)}")
            return
        
        # Extract returns
        df = pd.DataFrame(buffer[-lookback:])
        returns = df['mid'].pct_change().dropna().values
        
        # Train HMM
        detector = HMMRegimeDetector(n_states=n_states)
        detector.fit(returns, n_iterations=100)
        
        st.session_state.hmm_regime_detector = detector
        st.success(f"✓ HMM trained with {n_states} states")
        
    except Exception as e:
        st.error(f"HMM training failed: {e}")


def run_parameter_optimization(method: str, param_ranges: Dict, **method_params):
    """Run parameter optimization with method-specific parameters"""
    try:
        from python.advanced_optimization import (
            MCMCOptimizer, MLEOptimizer, InformationTheoryOptimizer, 
            ParameterSpace, OptimizationResult
        )
        
        # Define parameter space
        param_spaces = []
        for param_name, (low, high) in param_ranges.items():
            param_spaces.append(ParameterSpace(
                name=param_name,
                bounds=(low, high),
                dtype='float'
            ))
        
        # Get historical data for optimization
        buffer = st.session_state.get('live_data_buffer', [])
        if len(buffer) < 100:
            st.warning("Need more historical data for optimization")
            return
        
        df = pd.DataFrame(buffer)
        returns = df['mid'].pct_change().dropna().values
        
        # Define objective function
        def objective(params):
            # Simplified - in production, run full backtest
            entry_z = params.get('entry_z', 2.0)
            lookback = int(params.get('lookback', 60))
            
            if lookback >= len(returns):
                return -1e10
            
            # Simple mean reversion score
            score = 0.0
            for i in range(lookback, len(returns)):
                window = returns[i-lookback:i]
                mean = np.mean(window)
                std = np.std(window)
                
                if std > 0:
                    z = (returns[i] - mean) / std
                    if abs(z) > entry_z:
                        score += returns[i]
            
            return score
        
        # Run optimization
        if "MCMC" in method:
            optimizer = MCMCOptimizer(param_spaces, objective)
            n_iter = method_params.get('n_iterations', 5000)
            burn = method_params.get('burn_in', 500)
            result = optimizer.optimize(n_iterations=n_iter, burn_in=burn)
        elif "MLE" in method:
            def log_likelihood(params):
                return objective(params)  # Simplified
            optimizer = MLEOptimizer(param_spaces, log_likelihood)
            result = optimizer.optimize()
        elif "Information Theory" in method:
            # Use Information Theory for feature selection
            # Create features from price data
            features_df = pd.DataFrame({
                'returns': returns,
                'returns_lag1': np.roll(returns, 1),
                'returns_lag2': np.roll(returns, 2),
                'volatility': pd.Series(returns).rolling(20).std().fillna(0).values,
                'momentum': pd.Series(returns).rolling(10).mean().fillna(0).values
            })
            
            # Select best features using mutual information
            selected_features = InformationTheoryOptimizer.select_features(
                features_df,
                returns,
                n_features=3
            )
            
            st.info(f"Selected features by MI: {', '.join(selected_features)}")
            
            # Use MLE on selected features for parameter optimization
            optimizer = MLEOptimizer(param_spaces, lambda p: objective(p))
            result = optimizer.optimize()
        elif "Grid Search" in method:
            # Grid search implementation
            best_score = -np.inf
            best_params = {}
            
            # Create grid
            grid_points = method_params.get('grid_points', 5)
            grids = []
            for param in param_spaces:
                low, high = param.bounds
                grids.append(np.linspace(low, high, grid_points))
            
            # Evaluate grid
            with st.spinner("Running grid search..."):
                for values in product(*grids):
                    params = {param.name: val for param, val in zip(param_spaces, values)}
                    score = objective(params)
                    if score > best_score:
                        best_score = score
                        best_params = params
            
            result = OptimizationResult(
                best_params=best_params,
                best_score=best_score,
                all_params=[best_params],
                all_scores=[best_score],
                convergence_history=[best_score],
                method='Grid Search',
                n_iterations=grid_points ** len(param_spaces)
            )
        elif "Differential Evolution" in method:
            from scipy.optimize import differential_evolution
            
            bounds = [param.bounds for param in param_spaces]
            maxiter = method_params.get('maxiter', 1000)
            popsize = method_params.get('popsize', 15)
            
            def neg_objective(x):
                params = {param.name: val for param, val in zip(param_spaces, x)}
                return -objective(params)
            
            opt_result = differential_evolution(
                neg_objective, 
                bounds, 
                maxiter=maxiter,
                popsize=popsize,
                workers=-1
            )
            
            best_params = {param.name: val for param, val in zip(param_spaces, opt_result.x)}
            
            result = OptimizationResult(
                best_params=best_params,
                best_score=-opt_result.fun,
                all_params=[best_params],
                all_scores=[-opt_result.fun],
                convergence_history=[],
                method='Differential Evolution',
                n_iterations=opt_result.nit
            )
        else:
            st.warning("Method not yet implemented")
            return
        
        st.session_state.optimization_result = result
        st.success(f"✓ Optimization complete")
        
        # Apply optimal parameters
        st.session_state.live_strategy_params = result.best_params
        
    except Exception as e:
        st.error(f"Optimization failed: {e}")


def optimize_multi_strategy_allocation():
    """Optimize allocation across multiple strategies and assets"""
    try:
        from python.advanced_optimization import MultiStrategyOptimizer
        
        config = st.session_state.get('multi_strategy_config')
        if not config:
            st.warning("Configure multi-strategy mode first")
            return
        
        # Get historical data
        buffer = st.session_state.get('live_data_buffer', [])
        if len(buffer) < 100:
            st.warning("Need more data for optimization")
            return
        
        df = pd.DataFrame(buffer)
        
        # Create historical data dict
        historical_data = {}
        for symbol in df['symbol'].unique():
            symbol_df = df[df['symbol'] == symbol].copy()
            symbol_df['close'] = symbol_df['mid']
            historical_data[symbol] = symbol_df
        
        # Asset types mapping
        asset_types = {symbol: 'crypto' for symbol in df['symbol'].unique()}  # Simplified
        
        optimizer = MultiStrategyOptimizer(
            strategies=config['strategies'],
            assets=list(df['symbol'].unique()),
            asset_types=asset_types
        )
        
        # Simplified parameter spaces
        if ParameterSpace is not None:
            strategy_params = {
                strat: [
                    ParameterSpace('entry_z', (1.5, 3.0)),
                    ParameterSpace('lookback', (40, 100))
                ]
                for strat in config['strategies']
            }
        else:
            strategy_params = {
                strat: [
                    {'name': 'entry_z', 'bounds': (1.5, 3.0)},
                    {'name': 'lookback', 'bounds': (40, 100)}
                ]
                for strat in config['strategies']
            }
        
        result = optimizer.optimize(historical_data, strategy_params)  # type: ignore
        
        st.session_state.multi_strategy_optimal = result
        st.success("✓ Multi-strategy optimization complete")
        
        # Display results
        st.markdown("**Optimal Allocation**")
        st.dataframe(result['allocations'], use_container_width=True)
        
    except Exception as e:
        st.error(f"Multi-strategy optimization failed: {e}")


def plot_optimization_convergence(result):
    """Plot optimization convergence"""
    if result is None or not hasattr(result, 'convergence_history'):
        st.warning("No convergence data available")
        return
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        y=result.convergence_history,
        mode='lines',
        name='Objective Value',
        line=dict(color='#4ECDC4', width=2)
    ))
    
    fig.update_layout(
        title=f"{result.method} Convergence",
        xaxis_title="Iteration",
        yaxis_title="Objective Value",
        template="plotly_dark",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
