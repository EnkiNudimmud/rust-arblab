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

from python import meanrev
from python.strategies.definitions import AVAILABLE_STRATEGIES
from python.strategies.executor import StrategyExecutor, StrategyConfig

def render():
    """Render the strategy backtesting page"""
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
    
    # Extract backtest results
    results['equity'] = backtest['equity']
    results['sharpe'] = backtest['sharpe']
    results['max_drawdown'] = backtest.get('max_drawdown', 0.0)
    results['total_return'] = backtest.get('total_return', 0.0)
    results['trades'] = backtest.get('trades', [])
    results['z_score'] = backtest.get('z', pd.Series())
    results['positions'] = backtest.get('position', pd.Series())
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

def compare_strategies(initial_capital: float, transaction_cost: float):
    """Compare multiple strategies"""
    
    st.info("🚧 Multi-strategy comparison coming soon!")
    st.markdown("""
    This feature will allow you to:
    - Run multiple strategies simultaneously
    - Compare performance metrics side-by-side
    - Visualize equity curves together
    - Identify best-performing strategies
    """)

# Execute the render function when page is loaded
if __name__ == "__main__":
    render()
