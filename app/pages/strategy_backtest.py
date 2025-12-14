"""
Strategy Backtesting Module
============================

Backtest multiple trading strategies on historical data:
- Mean Reversion
- Pairs Trading
- Triangular Arbitrage
- Market Making
- Statistical Arbitrage



"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from typing import Dict, List, Any
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from python.strategies import meanrev
from python.strategies.definitions import AVAILABLE_STRATEGIES
from python.strategies.executor import StrategyExecutor, StrategyConfig
from utils.ui_components import render_sidebar_navigation, apply_custom_css

def render():
    """Render the strategy backtesting page"""
    # Render sidebar navigation and apply CSS
    render_sidebar_navigation(current_page="Strategy Backtest")
    apply_custom_css()
    # Initialize session state
    if 'historical_data' not in st.session_state:
        st.session_state.historical_data = None
    if 'backtest_results' not in st.session_state:
        st.session_state.backtest_results = None
    if 'selected_strategy' not in st.session_state:
        st.session_state.selected_strategy = None
    
    st.title("⚡ Strategy Backtesting")
    st.markdown("Select and backtest trading strategies on historical data")
    
    # Check if data is loaded
    if st.session_state.historical_data is None:
        st.warning("⚠️ No historical data loaded. Please load data first from the Data Loading page.")
        if st.button("Go to Data Loading", key="goto_data_loading_btn"):
            st.session_state.page = "📊 Data Loading"
            st.rerun()
        return
    
    # Main layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Strategy Configuration")
        
        # Strategy selection
        strategy_names = list(AVAILABLE_STRATEGIES.keys())
        strategy_names.extend(["Mean Reversion (PCA)", "Mean Reversion (CARA)", "Mean Reversion (Sharpe)"])
        
        selected_strategy = st.selectbox(
            "Select Strategy",
            strategy_names,
            help="Choose a trading strategy to backtest"
        )
        
        st.session_state.selected_strategy = selected_strategy
        
        st.markdown("---")
        
        # Display strategy info
        if selected_strategy in AVAILABLE_STRATEGIES:
            strategy_def = AVAILABLE_STRATEGIES[selected_strategy]
            st.markdown(f"**{strategy_def.display_name}**")
            st.caption(strategy_def.description)
            
            st.markdown("#### Parameters")
            params = configure_strategy_params(strategy_def)
        else:
            # Mean reversion strategies
            st.markdown("**Mean Reversion Portfolio**")
            st.caption("Construct mean-reverting portfolios using PCA, CARA utility, or Sharpe optimization")
            
            st.markdown("#### Parameters")
            params = configure_meanrev_params()
        
        st.markdown("---")
        
        # Backtest configuration
        st.markdown("#### Backtest Settings")
        initial_capital = st.number_input(
            "Initial Capital ($)",
            min_value=1000,
            max_value=10000000,
            value=100000,
            step=10000
        )
        
        transaction_cost = st.number_input(
            "Transaction Cost (bps)",
            min_value=0.0,
            max_value=100.0,
            value=10.0,
            step=1.0,
            help="Transaction cost in basis points (1 bps = 0.01%)"
        )
        
        # Run backtest button
        if st.button("🚀 Run Backtest", type="primary", use_container_width=True):
            run_backtest(selected_strategy, params, initial_capital, transaction_cost)
    
    with col2:
        st.markdown("### Backtest Results")
        
        if st.session_state.backtest_results is not None:
            display_backtest_results()
        else:
            st.info("Configure strategy parameters and click 'Run Backtest' to see results")
            
            # Show strategy comparison option
            st.markdown("---")
            st.markdown("### Multi-Strategy Comparison")
            
            if st.button("🔥 Compare All Strategies", use_container_width=True):
                compare_strategies(initial_capital, transaction_cost)

def configure_strategy_params(strategy_def) -> Dict[str, Any]:
    """Configure parameters for a strategy"""
    params = {}
    
    for param_name, default_value in strategy_def.parameters.items():
        description = strategy_def.param_descriptions.get(param_name, param_name)
        
        if isinstance(default_value, str):
            params[param_name] = st.text_input(
                description,
                value=default_value
            )
        elif isinstance(default_value, float):
            params[param_name] = st.number_input(
                description,
                value=default_value,
                format="%.4f"
            )
        elif isinstance(default_value, int):
            params[param_name] = st.number_input(
                description,
                value=default_value,
                step=1
            )
    
    return params

def configure_meanrev_params() -> Dict[str, Any]:
    """Configure mean reversion strategy parameters"""
    params = {}
    
    params['entry_z'] = st.number_input(
        "Entry Z-Score",
        min_value=0.5,
        max_value=5.0,
        value=2.0,
        step=0.1,
        help="Z-score threshold to enter positions"
    )
    
    params['exit_z'] = st.number_input(
        "Exit Z-Score",
        min_value=0.0,
        max_value=2.0,
        value=0.5,
        step=0.1,
        help="Z-score threshold to exit positions"
    )
    
    params['lookback'] = st.number_input(
        "Lookback Window",
        min_value=10,
        max_value=500,
        value=60,
        step=10,
        help="Rolling window for statistics calculation"
    )
    
    params['gamma'] = st.number_input(
        "Risk Aversion (CARA)",
        min_value=0.1,
        max_value=10.0,
        value=2.0,
        step=0.1,
        help="Risk aversion parameter for CARA utility"
    )
    
    params['risk_free'] = st.number_input(
        "Risk-Free Rate",
        min_value=0.0,
        max_value=0.2,
        value=0.02,
        step=0.01,
        help="Annual risk-free rate for Sharpe calculation"
    )
    
    return params

def run_backtest(strategy_name: str, params: Dict[str, Any], initial_capital: float, transaction_cost: float):
    """Run backtest for selected strategy"""
    
    with st.spinner(f"Running backtest for {strategy_name}..."):
        try:
            df = st.session_state.historical_data
            
            # Prepare price data
            if 'close' not in df.columns:
                st.error("Data must contain 'close' column")
                return
            
            # Pivot to wide format (timestamp x symbol)
            if 'symbol' in df.columns:
                price_df = df.pivot_table(
                    index='timestamp',
                    columns='symbol',
                    values='close'
                ).sort_index()
            else:
                # Single symbol
                price_df = df.set_index('timestamp')[['close']]
            
            # Remove any NaN
            price_df = price_df.dropna()
            
            if len(price_df) < 30:
                st.error("Insufficient data points for backtesting (need at least 30)")
                return
            
            # Run strategy-specific backtest
            if "Mean Reversion" in strategy_name:
                results = backtest_meanrev(strategy_name, price_df, params, transaction_cost)
            elif strategy_name in AVAILABLE_STRATEGIES:
                results = backtest_general_strategy(strategy_name, price_df, params, initial_capital, transaction_cost)
            else:
                st.error(f"Unknown strategy: {strategy_name}")
                return
            
            st.session_state.backtest_results = results
            st.session_state.strategy_params = params
            st.success("✅ Backtest completed successfully!")
            st.rerun()
            
        except Exception as e:
            st.error(f"Backtest failed: {e}")
            import traceback
            st.code(traceback.format_exc())

def backtest_meanrev(strategy_name: str, price_df: pd.DataFrame, params: Dict, transaction_cost: float) -> Dict:
    """Backtest mean reversion strategies"""
    
    # Compute returns
    returns = np.log(price_df).diff().dropna()
    
    results = {}
    
    if "PCA" in strategy_name:
        # PCA-based portfolio
        pcs, pca_info = meanrev.pca_portfolios(returns, n_components=min(5, returns.shape[1]))
        weights = pcs[0, :]  # First principal component
        weights = weights / (np.sum(np.abs(weights)) + 1e-12)
        
        aligned_prices = price_df.loc[returns.index]
        portfolio_series = pd.Series(
            aligned_prices.values @ weights,
            index=returns.index
        )
        
        backtest = meanrev.backtest_with_costs(
            portfolio_series,
            entry_z=params['entry_z'],
            exit_z=params['exit_z'],
            transaction_cost=transaction_cost / 10000  # Convert bps to fraction
        )
        
        results['portfolio_series'] = portfolio_series
        results['weights'] = weights
        results['method'] = 'PCA'
        
    elif "CARA" in strategy_name:
        # CARA optimal portfolio
        expected_returns = returns.mean().values
        covariance = returns.cov().values
        
        cara_result = meanrev.cara_optimal_weights(expected_returns, covariance, gamma=params['gamma'])
        weights = np.array(cara_result['weights'], dtype=float)
        weights = weights / (np.sum(np.abs(weights)) + 1e-12)
        
        aligned_prices = price_df.loc[returns.index]
        portfolio_series = pd.Series(
            aligned_prices.values @ weights,
            index=returns.index
        )
        
        backtest = meanrev.backtest_with_costs(
            portfolio_series,
            entry_z=params['entry_z'],
            exit_z=params['exit_z'],
            transaction_cost=transaction_cost / 10000
        )
        
        results['portfolio_series'] = portfolio_series
        results['weights'] = weights
        results['method'] = 'CARA'
        
    elif "Sharpe" in strategy_name:
        # Sharpe optimal portfolio
        expected_returns = returns.mean().values
        covariance = returns.cov().values
        
        sharpe_result = meanrev.sharpe_optimal_weights(
            expected_returns,
            covariance,
            risk_free_rate=params['risk_free']
        )
        weights = np.array(sharpe_result['weights'], dtype=float)
        weights = weights / (np.sum(np.abs(weights)) + 1e-12)
        
        aligned_prices = price_df.loc[returns.index]
        portfolio_series = pd.Series(
            aligned_prices.values @ weights,
            index=returns.index
        )
        
        backtest = meanrev.backtest_with_costs(
            portfolio_series,
            entry_z=params['entry_z'],
            exit_z=params['exit_z'],
            transaction_cost=transaction_cost / 10000
        )
        
        results['portfolio_series'] = portfolio_series
        results['weights'] = weights
        results['method'] = 'Sharpe'
    
    # Extract backtest results and convert to proper format
    pnl_data = backtest['pnl']
    
    # Convert pnl list to Series if needed
    if isinstance(pnl_data, list):
        pnl_series = pd.Series(pnl_data, index=portfolio_series.index[-len(pnl_data):])
    else:
        pnl_series = pnl_data
    
    # Calculate equity curve (starting capital + pnl)
    initial_capital = 100000.0
    results['equity'] = initial_capital + pnl_series
    results['pnl'] = pnl_series
    results['sharpe'] = backtest['sharpe']
    results['max_drawdown'] = backtest.get('max_drawdown', 0.0)
    results['total_return'] = pnl_series.iloc[-1] if len(pnl_series) > 0 else 0.0
    results['total_costs'] = backtest.get('total_costs', 0.0)
    
    # Convert positions list to Series if needed
    positions_data = backtest.get('positions', [])
    if isinstance(positions_data, list):
        results['positions'] = pd.Series(positions_data, index=portfolio_series.index[-len(positions_data):])
    else:
        results['positions'] = positions_data
    
    # Calculate z-score for the portfolio series (for display purposes)
    window = 20
    rolling_mean = portfolio_series.rolling(window).mean()
    rolling_std = portfolio_series.rolling(window).std()
    results['z_score'] = (portfolio_series - rolling_mean) / (rolling_std + 1e-10)
    
    results['returns'] = backtest.get('returns', [])
    results['symbols'] = price_df.columns.tolist()
    
    return results

def backtest_general_strategy(strategy_name: str, price_df: pd.DataFrame, params: Dict, initial_capital: float, transaction_cost: float) -> Dict:
    """Backtest general strategies using StrategyExecutor"""
    
    config = StrategyConfig(
        strategy_name=strategy_name,
        parameters=params,
        mode='backtest',
        initial_capital=initial_capital
    )
    
    executor = StrategyExecutor(config)
    
    # Simple implementation - this would need to be expanded based on strategy
    # For now, return a placeholder
    results = {
        'equity': pd.Series([initial_capital] * len(price_df), index=price_df.index),
        'sharpe': 0.0,
        'max_drawdown': 0.0,
        'total_return': 0.0,
        'trades': [],
        'symbols': price_df.columns.tolist()
    }
    
    st.warning(f"Strategy {strategy_name} backtest implementation is a placeholder. Full implementation pending.")
    
    return results

def display_backtest_results():
    """Display backtest results with visualizations"""
    
    results = st.session_state.backtest_results
    
    # Performance metrics
    st.markdown("#### 📊 Performance Metrics")
    
    cols = st.columns(4)
    with cols[0]:
        total_return = results.get('total_return', 0.0) * 100
        st.metric("Total Return", f"{total_return:.2f}%")
    with cols[1]:
        sharpe = results.get('sharpe', 0.0)
        st.metric("Sharpe Ratio", f"{sharpe:.3f}")
    with cols[2]:
        max_dd = results.get('max_drawdown', 0.0) * 100
        st.metric("Max Drawdown", f"{max_dd:.2f}%")
    with cols[3]:
        n_trades = len(results.get('trades', []))
        st.metric("Total Trades", n_trades)
    
    st.markdown("---")
    
    # Tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Equity Curve", "🎯 Portfolio", "📊 Trades", "📉 Analysis"])
    
    with tab1:
        display_equity_curve(results)
    
    with tab2:
        display_portfolio_weights(results)
    
    with tab3:
        display_trades(results)
    
    with tab4:
        display_analysis(results)

def display_equity_curve(results: Dict):
    """Display equity curve with drawdowns"""
    
    equity = results.get('equity', pd.Series())
    
    if len(equity) == 0:
        st.info("No equity curve data available")
        return
    
    # Calculate returns and cumulative returns
    returns = equity.pct_change().fillna(0)
    cum_returns = (1 + returns).cumprod()
    
    # Calculate drawdown
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    
    # Create figure with subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=("Equity Curve", "Drawdown")
    )
    
    # Equity curve
    fig.add_trace(
        go.Scatter(
            x=equity.index,
            y=equity.values,
            name='Equity',
            line=dict(color='#00ff00', width=2)
        ),
        row=1, col=1
    )
    
    # Drawdown
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values * 100,
            name='Drawdown',
            fill='tozeroy',
            line=dict(color='#ff0000', width=1)
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=600,
        template="plotly_dark",
        showlegend=True,
        hovermode='x unified'
    )
    
    fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)

def display_portfolio_weights(results: Dict):
    """Display portfolio weights if available"""
    
    weights = results.get('weights', None)
    symbols = results.get('symbols', [])
    
    if weights is not None and len(symbols) > 0:
        # Create bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=symbols,
            y=weights,
            marker_color=['green' if w > 0 else 'red' for w in weights],
            text=[f"{w:.3f}" for w in weights],
            textposition='outside'
        ))
        
        fig.update_layout(
            title=f"Portfolio Weights ({results.get('method', 'Strategy')})",
            xaxis_title="Symbol",
            yaxis_title="Weight",
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display as table
        weight_df = pd.DataFrame({
            'Symbol': symbols,
            'Weight': weights,
            'Weight %': [f"{w*100:.2f}%" for w in weights]
        })
        st.dataframe(weight_df, use_container_width=True)
    else:
        st.info("No portfolio weight information available for this strategy")

def display_trades(results: Dict):
    """Display trade log"""
    
    trades = results.get('trades', [])
    
    if len(trades) > 0:
        trades_df = pd.DataFrame(trades)
        st.dataframe(trades_df, use_container_width=True, height=400)
        
        # Trade statistics
        if 'pnl' in trades_df.columns:
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_pnl = trades_df['pnl'].mean()
                st.metric("Avg P&L per Trade", f"${avg_pnl:.2f}")
            with col2:
                win_rate = (trades_df['pnl'] > 0).mean() * 100
                st.metric("Win Rate", f"{win_rate:.1f}%")
            with col3:
                total_pnl = trades_df['pnl'].sum()
                st.metric("Total P&L", f"${total_pnl:.2f}")
    else:
        st.info("No trades executed in this backtest")

def display_analysis(results: Dict):
    """Display additional analysis"""
    
    # Z-score chart for mean reversion
    z_score = results.get('z_score', pd.Series())
    if len(z_score) > 0:
        st.markdown("#### Z-Score Evolution")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=z_score.index,
            y=z_score.values,
            name='Z-Score',
            line=dict(color='cyan')
        ))
        
        # Add entry/exit thresholds
        params = st.session_state.strategy_params
        entry_z = params.get('entry_z', 2.0)
        exit_z = params.get('exit_z', 0.5)
        
        fig.add_hline(y=entry_z, line_dash="dash", line_color="green", annotation_text="Entry+")
        fig.add_hline(y=-entry_z, line_dash="dash", line_color="green", annotation_text="Entry-")
        fig.add_hline(y=exit_z, line_dash="dot", line_color="yellow", annotation_text="Exit+")
        fig.add_hline(y=-exit_z, line_dash="dot", line_color="yellow", annotation_text="Exit-")
        
        fig.update_layout(
            template="plotly_dark",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Returns distribution
    equity = results.get('equity', pd.Series())
    if len(equity) > 1:
        st.markdown("#### Returns Distribution")
        
        returns = equity.pct_change().dropna()
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=returns * 100,
            nbinsx=50,
            name='Returns',
            marker_color='steelblue'
        ))
        
        fig.update_layout(
            xaxis_title="Return (%)",
            yaxis_title="Frequency",
            template="plotly_dark",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean Return", f"{returns.mean()*100:.3f}%")
        with col2:
            st.metric("Std Dev", f"{returns.std()*100:.3f}%")
        with col3:
            st.metric("Skewness", f"{returns.skew():.3f}")
        with col4:
            st.metric("Kurtosis", f"{returns.kurtosis():.3f}")
        
        # Enhanced Risk Metrics using optimizr
        st.markdown("---")
        st.markdown("#### 🎯 Comprehensive Risk Metrics")
        
        try:
            # Try to import compute_risk_metrics from sparse_meanrev
            sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'python'))
            from sparse_meanrev import compute_risk_metrics
            
            # Compute comprehensive metrics
            risk_metrics = compute_risk_metrics(returns.values, risk_free_rate=0.0, periods_per_year=252)
            
            st.success("✅ Using Rust-accelerated risk metrics")
            
            # Display in organized columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Return Metrics**")
                st.metric("Sharpe Ratio", f"{risk_metrics.sharpe_ratio:.4f}")
                st.metric("Sortino Ratio", f"{risk_metrics.sortino_ratio:.4f}")
                st.metric("Calmar Ratio", f"{risk_metrics.calmar_ratio:.4f}")
            
            with col2:
                st.markdown("**Risk Metrics**")
                st.metric("Volatility (Ann.)", f"{risk_metrics.volatility*100:.2f}%")
                st.metric("Downside Dev (Ann.)", f"{risk_metrics.downside_deviation*100:.2f}%")
                st.metric("Max Drawdown", f"{risk_metrics.max_drawdown*100:.2f}%")
            
            with col3:
                st.markdown("**Tail Risk**")
                st.metric("VaR (95%)", f"{risk_metrics.var_95*100:.3f}%")
                st.metric("CVaR (95%)", f"{risk_metrics.cvar_95*100:.3f}%")
                st.metric("Max DD Duration", f"{risk_metrics.max_drawdown_duration} periods")
            
            # Detailed explanation
            with st.expander("ℹ️ Understanding Risk Metrics"):
                st.markdown("""
                **Sharpe Ratio**: Risk-adjusted return (higher is better). Measures excess return per unit of volatility.
                
                **Sortino Ratio**: Like Sharpe but only penalizes downside volatility (higher is better).
                
                **Calmar Ratio**: Return divided by max drawdown (higher is better).
                
                **Volatility**: Annualized standard deviation of returns.
                
                **Downside Deviation**: Volatility of negative returns only.
                
                **VaR (Value at Risk)**: Expected loss at 95% confidence level (1 in 20 days worse than this).
                
                **CVaR (Conditional VaR)**: Average loss beyond the VaR threshold (tail risk).
                
                **Max DD Duration**: Longest period underwater (below previous high).
                """)
            
        except ImportError:
            st.info("Install optimizr for enhanced risk metrics: `cd optimiz-r && maturin develop --release`")

def compare_strategies(initial_capital: float, transaction_cost: float):
    """Compare multiple strategies"""
    
    st.markdown("---")
    st.markdown("### 🔥 Multi-Strategy Comparison")
    
    # Strategy selection for comparison
    all_strategies = ["Mean Reversion (PCA)", "Mean Reversion (CARA)", "Mean Reversion (Sharpe)"]
    
    selected_strategies = st.multiselect(
        "Select Strategies to Compare",
        all_strategies,
        default=all_strategies,
        help="Choose 2 or more strategies to compare"
    )
    
    if len(selected_strategies) < 2:
        st.warning("⚠️ Please select at least 2 strategies to compare")
        return
    
    # Global parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        entry_z = st.number_input("Entry Z-Score", 0.5, 5.0, 2.0, 0.1, key="comp_entry_z")
    with col2:
        exit_z = st.number_input("Exit Z-Score", 0.0, 2.0, 0.5, 0.1, key="comp_exit_z")
    with col3:
        gamma = st.number_input("CARA γ", 0.1, 10.0, 2.0, 0.1, key="comp_gamma")
    
    if st.button("🚀 Run Comparison", type="primary", use_container_width=True):
        with st.spinner("Running all strategies..."):
            # Prepare data
            data = st.session_state.historical_data
            
            # Convert DataFrame to dict format if needed
            if isinstance(data, pd.DataFrame):
                if 'symbol' in data.columns:
                    symbols = data['symbol'].unique()[:10]  # Limit to 10 symbols
                    price_dict = {}
                    for symbol in symbols:
                        symbol_data = data[data['symbol'] == symbol].copy()
                        if 'timestamp' in symbol_data.columns:
                            symbol_data = symbol_data.set_index('timestamp')
                        if 'close' in symbol_data.columns:
                            price_dict[symbol] = symbol_data['close']
                    
                    price_df = pd.DataFrame(price_dict)
                else:
                    st.error("Data format not supported for multi-strategy comparison")
                    return
            else:
                st.error("Please load data in the correct format")
                return
            
            if price_df.empty or len(price_df.columns) < 3:
                st.error("Need at least 3 symbols for strategy comparison")
                return
            
            # Run all selected strategies
            results_dict = {}
            params = {
                'entry_z': entry_z,
                'exit_z': exit_z,
                'gamma': gamma,
                'risk_free': 0.02,
                'lookback': 60
            }
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, strategy_name in enumerate(selected_strategies):
                status_text.text(f"Running {strategy_name}...")
                
                try:
                    if "Mean Reversion" in strategy_name:
                        result = backtest_meanrev(strategy_name, price_df, params, transaction_cost)
                        results_dict[strategy_name] = result
                except Exception as e:
                    st.error(f"Failed to run {strategy_name}: {str(e)}")
                
                progress_bar.progress((idx + 1) / len(selected_strategies))
            
            progress_bar.empty()
            status_text.empty()
            
            if results_dict:
                display_multi_strategy_comparison(results_dict, initial_capital)
            else:
                st.error("No strategies completed successfully")


def display_multi_strategy_comparison(results_dict: Dict, initial_capital: float):
    """Display comprehensive multi-strategy comparison"""
    
    st.markdown("---")
    st.markdown("### 📊 Strategy Performance Comparison")
    
    # Extract equity curves
    equity_curves = {}
    for name, result in results_dict.items():
        equity_curves[name] = result.get('equity', pd.Series())
    
    # Summary metrics at top
    st.markdown("#### 📈 Summary Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate metrics for each strategy
    metrics = {}
    for name, equity in equity_curves.items():
        if len(equity) > 0:
            final_value = equity.iloc[-1]
            total_return = ((final_value - initial_capital) / initial_capital) * 100
            
            returns = equity.pct_change().dropna()
            sharpe = np.sqrt(252) * returns.mean() / (returns.std() + 1e-10)
            
            running_max = equity.expanding().max()
            drawdown = ((equity - running_max) / running_max).min()
            
            metrics[name] = {
                'final_value': final_value,
                'total_return': total_return,
                'sharpe': sharpe,
                'max_drawdown': drawdown
            }
    
    with col1:
        best_return = max(metrics.items(), key=lambda x: x[1]['total_return'])
        st.metric("🏆 Best Return", best_return[0].split('(')[1].strip(')'), 
                 f"{best_return[1]['total_return']:.2f}%")
    
    with col2:
        best_sharpe = max(metrics.items(), key=lambda x: x[1]['sharpe'])
        st.metric("⭐ Best Sharpe", best_sharpe[0].split('(')[1].strip(')'),
                 f"{best_sharpe[1]['sharpe']:.2f}")
    
    with col3:
        best_dd = max(metrics.items(), key=lambda x: x[1]['max_drawdown'])  # Least negative
        st.metric("🛡️ Lowest Drawdown", best_dd[0].split('(')[1].strip(')'),
                 f"{best_dd[1]['max_drawdown']:.2%}")
    
    with col4:
        avg_return = np.mean([m['total_return'] for m in metrics.values()])
        st.metric("📊 Avg Return", "All Strategies", f"{avg_return:.2f}%")
    
    # Detailed metrics table
    st.markdown("#### 📋 Performance Metrics")
    
    metrics_data = []
    for name, m in metrics.items():
        method = name.split('(')[1].strip(')')
        metrics_data.append({
            'Strategy': method,
            'Final Value': f"${m['final_value']:,.2f}",
            'Total Return': f"{m['total_return']:.2f}%",
            'Sharpe Ratio': f"{m['sharpe']:.2f}",
            'Max Drawdown': f"{m['max_drawdown']:.2%}",
            'Risk-Adj Return': f"{(m['total_return'] / abs(m['max_drawdown'] * 100) if m['max_drawdown'] != 0 else 0):.2f}"
        })
    
    # Sort by total return
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df['_sort'] = [float(r.strip('%')) for r in metrics_df['Total Return']]
    metrics_df = metrics_df.sort_values('_sort', ascending=False).drop('_sort', axis=1)
    
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    # Equity curves comparison
    st.markdown("#### 📈 Equity Curves")
    
    tab1, tab2, tab3 = st.tabs(["Absolute Value", "Normalized Returns", "Drawdowns"])
    
    with tab1:
        fig = go.Figure()
        
        colors = ['#00ff00', '#00ffff', '#ff00ff', '#ffff00', '#ff8800']
        for idx, (name, equity) in enumerate(equity_curves.items()):
            method = name.split('(')[1].strip(')')
            fig.add_trace(go.Scatter(
                x=equity.index,
                y=equity.values,
                name=method,
                mode='lines',
                line=dict(width=2, color=colors[idx % len(colors)]),
                hovertemplate='%{y:$,.2f}<extra></extra>'
            ))
        
        fig.update_layout(
            title="Equity Curves Comparison",
            xaxis_title="Time",
            yaxis_title="Portfolio Value ($)",
            hovermode='x unified',
            height=500,
            template="plotly_dark",
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        fig = go.Figure()
        
        for idx, (name, equity) in enumerate(equity_curves.items()):
            method = name.split('(')[1].strip(')')
            # Normalize to percentage returns
            normalized = ((equity - initial_capital) / initial_capital) * 100
            
            fig.add_trace(go.Scatter(
                x=normalized.index,
                y=normalized.values,
                name=method,
                mode='lines',
                line=dict(width=2, color=colors[idx % len(colors)]),
                hovertemplate='%{y:.2f}%<extra></extra>'
            ))
        
        fig.update_layout(
            title="Normalized Returns (%)",
            xaxis_title="Time",
            yaxis_title="Return (%)",
            hovermode='x unified',
            height=500,
            template="plotly_dark"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        fig = go.Figure()
        
        for idx, (name, equity) in enumerate(equity_curves.items()):
            method = name.split('(')[1].strip(')')
            # Calculate drawdown
            running_max = equity.expanding().max()
            drawdown = ((equity - running_max) / running_max) * 100
            
            fig.add_trace(go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                name=method,
                mode='lines',
                fill='tozeroy',
                line=dict(width=1, color=colors[idx % len(colors)]),
                hovertemplate='%{y:.2f}%<extra></extra>'
            ))
        
        fig.update_layout(
            title="Drawdown Comparison",
            xaxis_title="Time",
            yaxis_title="Drawdown (%)",
            hovermode='x unified',
            height=500,
            template="plotly_dark"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Risk-Return scatter
    st.markdown("#### 🎯 Risk-Return Profile")
    
    col_scatter, col_rankings = st.columns([2, 1])
    
    with col_scatter:
        returns_list = [m['total_return'] for m in metrics.values()]
        drawdowns_list = [abs(m['max_drawdown']) * 100 for m in metrics.values()]
        sharpes_list = [m['sharpe'] for m in metrics.values()]
        names_list = [n.split('(')[1].strip(')') for n in metrics.keys()]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=drawdowns_list,
            y=returns_list,
            mode='markers+text',
            text=names_list,
            textposition='top center',
            marker=dict(
                size=[max(10, min(30, abs(s) * 8)) for s in sharpes_list],
                color=returns_list,
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Return %"),
                line=dict(width=2, color='white')
            ),
            hovertemplate='<b>%{text}</b><br>Return: %{y:.2f}%<br>Max DD: %{x:.2f}%<extra></extra>'
        ))
        
        fig.update_layout(
            title="Risk-Return Profile (Size = Sharpe Ratio)",
            xaxis_title="Max Drawdown (%)",
            yaxis_title="Total Return (%)",
            height=400,
            template="plotly_dark"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col_rankings:
        st.markdown("##### 🏆 Rankings")
        
        st.markdown("**By Total Return:**")
        sorted_by_return = sorted(metrics.items(), key=lambda x: x[1]['total_return'], reverse=True)
        for idx, (name, m) in enumerate(sorted_by_return[:3]):
            medal = ["🥇", "🥈", "🥉"][idx]
            method = name.split('(')[1].strip(')')
            st.caption(f"{medal} {method}: {m['total_return']:.2f}%")
        
        st.markdown("**By Sharpe Ratio:**")
        sorted_by_sharpe = sorted(metrics.items(), key=lambda x: x[1]['sharpe'], reverse=True)
        for idx, (name, m) in enumerate(sorted_by_sharpe[:3]):
            medal = ["🥇", "🥈", "🥉"][idx]
            method = name.split('(')[1].strip(')')
            st.caption(f"{medal} {method}: {m['sharpe']:.2f}")
        
        st.markdown("**By Risk-Adjusted Return:**")
        sorted_by_risk_adj = sorted(
            metrics.items(), 
            key=lambda x: x[1]['total_return'] / abs(x[1]['max_drawdown'] * 100) if x[1]['max_drawdown'] != 0 else 0,
            reverse=True
        )
        for idx, (name, m) in enumerate(sorted_by_risk_adj[:3]):
            medal = ["🥇", "🥈", "🥉"][idx]
            method = name.split('(')[1].strip(')')
            risk_adj = m['total_return'] / abs(m['max_drawdown'] * 100) if m['max_drawdown'] != 0 else 0
            st.caption(f"{medal} {method}: {risk_adj:.2f}")
    
    # Portfolio weights comparison
    st.markdown("#### ⚖️ Portfolio Weights Comparison")
    
    weights_data = {}
    symbols = None
    
    for name, result in results_dict.items():
        if 'weights' in result and 'symbols' in result:
            method = name.split('(')[1].strip(')')
            weights_data[method] = result['weights']
            if symbols is None:
                symbols = result['symbols']
    
    if weights_data and symbols:
        weights_df = pd.DataFrame(weights_data, index=symbols)
        
        fig = go.Figure()
        
        for method in weights_df.columns:
            fig.add_trace(go.Bar(
                name=method,
                x=symbols,
                y=weights_df[method],
                text=[f"{w:.3f}" for w in weights_df[method]],
                textposition='outside'
            ))
        
        fig.update_layout(
            title="Portfolio Weights by Strategy",
            xaxis_title="Symbol",
            yaxis_title="Weight",
            barmode='group',
            height=400,
            template="plotly_dark"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Recommendations
    with st.expander("💡 Strategy Recommendations"):
        best_overall = max(metrics.items(), key=lambda x: x[1]['sharpe'])
        best_name = best_overall[0].split('(')[1].strip(')')
        
        st.markdown(f"""
        **Best Overall Strategy:** {best_name}
        - Highest Sharpe Ratio: {best_overall[1]['sharpe']:.2f}
        - Total Return: {best_overall[1]['total_return']:.2f}%
        - Max Drawdown: {best_overall[1]['max_drawdown']:.2%}
        
        **Key Insights:**
        - Use **{sorted_by_return[0][0].split('(')[1].strip(')')}** for maximum absolute returns
        - Use **{sorted_by_sharpe[0][0].split('(')[1].strip(')')}** for best risk-adjusted performance
        - Consider combining multiple strategies for better diversification
        """)

# Execute the render function when page is loaded
if __name__ == "__main__":
    render()
