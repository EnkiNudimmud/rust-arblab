"""
Arbitrage Analysis Module
=========================

Apply arbitrage strategies directly on loaded historical data:
- Triangular Arbitrage (cross-rate opportunities)
- Statistical Arbitrage (pairs trading)
- Mean Reversion Arbitrage (PCA, CARA, Sharpe)
- Multi-pair comparison and portfolio construction
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from typing import Dict, List, Any, Tuple
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from python import meanrev

def render():
    """Render the arbitrage analysis page"""
    st.title("🔺 Arbitrage Analysis")
    st.markdown("Apply arbitrage strategies directly on your loaded historical data")
    
    # Check if data is loaded
    if 'historical_data' not in st.session_state or st.session_state.historical_data is None:
        st.warning("⚠️ No historical data loaded. Please load data first from the Data Loading page.")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("📊 Go to Data Loading", type="primary", use_container_width=True):
                st.switch_page("pages/data_loader.py")
        return
    
    # Get loaded data
    data = st.session_state.historical_data
    symbols = list(data.keys())
    
    st.success(f"✅ Data loaded: {len(symbols)} symbols | Date range: {get_date_range_str(data)}")
    
    # Strategy selection
    st.markdown("---")
    st.markdown("### 🎯 Select Arbitrage Strategy")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        strategy = st.radio(
            "Strategy Type",
            [
                "📉 Mean Reversion Arbitrage",
                "🔄 Statistical Arbitrage (Pairs)",
                "🔺 Triangular Arbitrage",
                "📊 Multi-Strategy Comparison"
            ],
            help="Choose an arbitrage strategy to analyze"
        )
    
    with col2:
        if strategy == "📉 Mean Reversion Arbitrage":
            render_mean_reversion_arb(data, symbols)
        elif strategy == "🔄 Statistical Arbitrage (Pairs)":
            render_pairs_trading_arb(data, symbols)
        elif strategy == "🔺 Triangular Arbitrage":
            render_triangular_arb(data, symbols)
        elif strategy == "📊 Multi-Strategy Comparison":
            render_multi_strategy(data, symbols)


def render_mean_reversion_arb(data: Dict, symbols: List[str]):
    """Mean reversion arbitrage using PCA/CARA/Sharpe"""
    st.markdown("#### Mean Reversion Arbitrage")
    st.caption("Construct mean-reverting portfolios to exploit temporary price deviations")
    
    with st.expander("📖 Theory", expanded=False):
        st.markdown(r"""
        **Ornstein-Uhlenbeck Process:**
        $$dS_t = \theta(\mu - S_t)dt + \sigma dW_t$$
        
        **Three Approaches:**
        - **PCA**: Find principal component (most mean-reverting direction)
        - **CARA**: Utility maximization $w^* = \frac{1}{\gamma} \Sigma^{-1} \mu$
        - **Sharpe**: Maximum risk-adjusted return
        """)
    
    # Configuration
    col1, col2 = st.columns(2)
    
    with col1:
        entry_z = st.slider("Entry Z-Score", 0.5, 4.0, 2.0, 0.1, 
                           help="Enter position when |z| > entry_z")
        exit_z = st.slider("Exit Z-Score", 0.0, 2.0, 0.5, 0.1,
                          help="Exit position when |z| < exit_z")
    
    with col2:
        gamma = st.number_input("CARA Risk Aversion (γ)", 0.1, 10.0, 2.0, 0.1,
                               help="Higher = more risk-averse")
        risk_free = st.number_input("Risk-Free Rate", 0.0, 0.1, 0.02, 0.001,
                                    format="%.4f")
    
    transaction_cost = st.slider("Transaction Cost (bps)", 0.0, 50.0, 10.0, 1.0) / 10000
    
    # Symbol selection
    selected_symbols = st.multiselect(
        "Select Symbols for Portfolio",
        symbols,
        default=symbols[:min(10, len(symbols))],
        help="Select at least 2 symbols"
    )
    
    if len(selected_symbols) < 2:
        st.warning("⚠️ Please select at least 2 symbols")
        return
    
    if st.button("🚀 Run Mean Reversion Analysis", type="primary", use_container_width=True):
        with st.spinner("Computing mean-reverting portfolios..."):
            results = compute_mean_reversion(data, selected_symbols, {
                'entry_z': entry_z,
                'exit_z': exit_z,
                'gamma': gamma,
                'risk_free': risk_free,
                'transaction_cost': transaction_cost
            })
            
            if results:
                display_mean_reversion_results(results, selected_symbols)


def render_pairs_trading_arb(data: Dict, symbols: List[str]):
    """Statistical arbitrage through pairs trading"""
    st.markdown("#### Statistical Arbitrage (Pairs Trading)")
    st.caption("Exploit mean-reverting spreads between cointegrated pairs")
    
    with st.expander("📖 Theory", expanded=False):
        st.markdown(r"""
        **Hedge Ratio:**
        $$y_t = \beta x_t + c + \epsilon_t$$
        
        **Spread:**
        $$s_t = y_t - \beta x_t$$
        
        **Z-Score Signal:**
        $$z_t = \frac{s_t - \mu_s}{\sigma_s}$$
        
        - **Long spread** when $z < -2$ (spread too low)
        - **Short spread** when $z > 2$ (spread too high)
        """)
    
    # Configuration
    col1, col2 = st.columns(2)
    
    with col1:
        window = st.slider("Rolling Window", 20, 200, 50, 10,
                          help="Window for computing rolling mean/std")
        entry_z = st.slider("Entry Z-Score", 1.0, 4.0, 2.0, 0.1)
    
    with col2:
        exit_z = st.slider("Exit Z-Score", 0.0, 2.0, 0.5, 0.1)
        transaction_cost = st.slider("Transaction Cost (bps)", 0.0, 50.0, 10.0, 1.0) / 10000
    
    # Pair selection
    st.markdown("**Select Pair**")
    col1, col2 = st.columns(2)
    
    with col1:
        symbol_1 = st.selectbox("Asset 1", symbols, index=0)
    with col2:
        symbol_2 = st.selectbox("Asset 2", [s for s in symbols if s != symbol_1], 
                                index=0 if len(symbols) > 1 else 0)
    
    if st.button("🚀 Run Pairs Analysis", type="primary", use_container_width=True):
        with st.spinner("Computing pairs trading strategy..."):
            results = compute_pairs_trading(data, symbol_1, symbol_2, {
                'window': window,
                'entry_z': entry_z,
                'exit_z': exit_z,
                'transaction_cost': transaction_cost
            })
            
            if results:
                display_pairs_results(results, symbol_1, symbol_2)


def render_triangular_arb(data: Dict, symbols: List[str]):
    """Triangular arbitrage detection"""
    st.markdown("#### Triangular Arbitrage")
    st.caption("Detect and exploit cross-rate pricing inefficiencies")
    
    with st.expander("📖 Theory", expanded=False):
        st.markdown(r"""
        **Three assets form a triangle:** A, B, C
        
        **Forward path product:**
        $$P_{\text{forward}} = \frac{A}{B} \times \frac{B}{C} \times \frac{C}{A}$$
        
        **Perfect arbitrage-free condition:**
        $$P_{\text{forward}} = 1$$
        
        **Profit opportunity when:**
        $$|P_{\text{forward}} - 1| > \text{threshold} + \text{costs}$$
        """)
    
    # Configuration
    col1, col2 = st.columns(2)
    
    with col1:
        threshold = st.number_input("Arbitrage Threshold", 0.0001, 0.01, 0.001, 
                                   format="%.4f",
                                   help="Minimum deviation to signal arbitrage")
    
    with col2:
        transaction_cost = st.slider("Total Transaction Cost (bps)", 0.0, 100.0, 30.0, 1.0) / 10000
    
    # Triangle selection
    st.markdown("**Select Triangle**")
    
    if len(symbols) < 3:
        st.warning("⚠️ Need at least 3 symbols for triangular arbitrage")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        symbol_a = st.selectbox("Asset A", symbols, index=0)
    with col2:
        symbol_b = st.selectbox("Asset B", [s for s in symbols if s != symbol_a], 
                                index=0)
    with col3:
        available_c = [s for s in symbols if s not in [symbol_a, symbol_b]]
        symbol_c = st.selectbox("Asset C", available_c, 
                                index=0 if available_c else 0)
    
    if st.button("🚀 Detect Arbitrage Opportunities", type="primary", use_container_width=True):
        with st.spinner("Scanning for triangular arbitrage..."):
            results = compute_triangular_arb(data, symbol_a, symbol_b, symbol_c, {
                'threshold': threshold,
                'transaction_cost': transaction_cost
            })
            
            if results:
                display_triangular_results(results, symbol_a, symbol_b, symbol_c)


def render_multi_strategy(data: Dict, symbols: List[str]):
    """Compare multiple arbitrage strategies"""
    st.markdown("#### Multi-Strategy Comparison")
    st.caption("Run and compare all arbitrage strategies simultaneously")
    
    # Global parameters
    col1, col2 = st.columns(2)
    
    with col1:
        transaction_cost = st.slider("Transaction Cost (bps)", 0.0, 50.0, 10.0, 1.0) / 10000
        entry_z = st.slider("Entry Z-Score", 1.0, 4.0, 2.0, 0.1)
    
    with col2:
        exit_z = st.slider("Exit Z-Score", 0.0, 2.0, 0.5, 0.1)
        gamma = st.number_input("CARA γ", 0.1, 10.0, 2.0, 0.1)
    
    # Symbol selection
    selected_symbols = st.multiselect(
        "Select Symbols",
        symbols,
        default=symbols[:min(10, len(symbols))],
        help="Select at least 3 symbols for best results"
    )
    
    if len(selected_symbols) < 3:
        st.warning("⚠️ Please select at least 3 symbols for multi-strategy analysis")
        return
    
    if st.button("🚀 Run All Strategies", type="primary", use_container_width=True):
        with st.spinner("Running all arbitrage strategies..."):
            all_results = {}
            
            # Mean reversion
            with st.status("Computing mean reversion..."):
                mr_results = compute_mean_reversion(data, selected_symbols, {
                    'entry_z': entry_z,
                    'exit_z': exit_z,
                    'gamma': gamma,
                    'risk_free': 0.02,
                    'transaction_cost': transaction_cost
                })
                if mr_results:
                    all_results['Mean Reversion'] = mr_results
            
            # Pairs trading (top 3 pairs)
            with st.status("Computing pairs trading..."):
                pairs_results = []
                for i in range(min(3, len(selected_symbols) - 1)):
                    result = compute_pairs_trading(data, selected_symbols[i], 
                                                   selected_symbols[i+1], {
                        'window': 50,
                        'entry_z': entry_z,
                        'exit_z': exit_z,
                        'transaction_cost': transaction_cost
                    })
                    if result:
                        pairs_results.append(result)
                if pairs_results:
                    all_results['Pairs Trading'] = pairs_results
            
            # Triangular arbitrage
            if len(selected_symbols) >= 3:
                with st.status("Computing triangular arbitrage..."):
                    tri_result = compute_triangular_arb(data, 
                                                        selected_symbols[0],
                                                        selected_symbols[1],
                                                        selected_symbols[2], {
                        'threshold': 0.001,
                        'transaction_cost': transaction_cost * 3  # 3 legs
                    })
                    if tri_result:
                        all_results['Triangular'] = tri_result
            
            if all_results:
                display_multi_strategy_results(all_results)
            else:
                st.error("❌ No strategies produced results")


# ============================================================================
# COMPUTATION FUNCTIONS
# ============================================================================

def compute_mean_reversion(data: Dict, symbols: List[str], params: Dict) -> Dict:
    """Compute mean reversion strategies"""
    try:
        # Extract prices
        prices = extract_prices(data, symbols)
        
        # Compute log returns
        rets = np.log(prices).diff().dropna()
        
        results = {}
        
        # 1. PCA Method
        try:
            pcs, pca_info = meanrev.pca_portfolios(rets, n_components=min(3, len(symbols)))
            pc1_weights = pcs[0, :]
            pc1_weights = pc1_weights / (np.sum(np.abs(pc1_weights)) + 1e-12)
            
            aligned_prices = prices.loc[rets.index]
            pc1_series = (aligned_prices.values @ pc1_weights)
            pc1_series = pd.Series(pc1_series, index=rets.index)
            
            backtest = meanrev.backtest_with_costs(
                pc1_series,
                entry_z=params['entry_z'],
                exit_z=params['exit_z'],
                transaction_cost=params['transaction_cost']
            )
            
            results['PCA'] = {
                'series': pc1_series,
                'weights': pc1_weights,
                'backtest': backtest,
                'variance_explained': pca_info['variance_explained'][0]
            }
        except Exception as e:
            st.warning(f"PCA method failed: {e}")
        
        # 2. CARA Method
        try:
            expected_returns = rets.mean().values
            covariance = rets.cov().values
            
            cara_result = meanrev.cara_optimal_weights(expected_returns, covariance, 
                                                      gamma=params['gamma'])
            cara_w = np.array(cara_result['weights'], dtype=float)
            cara_w = cara_w / (np.sum(np.abs(cara_w)) + 1e-12)
            
            aligned_prices = prices.loc[rets.index]
            cara_series = aligned_prices.values @ cara_w
            cara_series = pd.Series(cara_series, index=rets.index)
            
            backtest = meanrev.backtest_with_costs(
                cara_series,
                entry_z=params['entry_z'],
                exit_z=params['exit_z'],
                transaction_cost=params['transaction_cost']
            )
            
            results['CARA'] = {
                'series': cara_series,
                'weights': cara_w,
                'backtest': backtest,
                'utility': cara_result.get('expected_utility', 0)
            }
        except Exception as e:
            st.warning(f"CARA method failed: {e}")
        
        # 3. Sharpe Method
        try:
            expected_returns = rets.mean().values
            covariance = rets.cov().values
            
            sharpe_result = meanrev.sharpe_optimal_weights(expected_returns, covariance,
                                                          risk_free_rate=params['risk_free'])
            sharpe_w = np.array(sharpe_result['weights'], dtype=float)
            sharpe_w = sharpe_w / (np.sum(np.abs(sharpe_w)) + 1e-12)
            
            aligned_prices = prices.loc[rets.index]
            sharpe_series = aligned_prices.values @ sharpe_w
            sharpe_series = pd.Series(sharpe_series, index=rets.index)
            
            backtest = meanrev.backtest_with_costs(
                sharpe_series,
                entry_z=params['entry_z'],
                exit_z=params['exit_z'],
                transaction_cost=params['transaction_cost']
            )
            
            results['Sharpe'] = {
                'series': sharpe_series,
                'weights': sharpe_w,
                'backtest': backtest,
                'sharpe_ratio': sharpe_result.get('expected_sharpe', 0)
            }
        except Exception as e:
            st.warning(f"Sharpe method failed: {e}")
        
        return results
    
    except Exception as e:
        st.error(f"Mean reversion computation failed: {e}")
        return {}


def compute_pairs_trading(data: Dict, symbol_1: str, symbol_2: str, params: Dict) -> Dict:
    """Compute pairs trading strategy"""
    try:
        # Extract prices
        prices = extract_prices(data, [symbol_1, symbol_2])
        
        x = prices.iloc[:, 0].values
        y = prices.iloc[:, 1].values
        
        # OLS regression for hedge ratio
        X = np.vstack([x, np.ones_like(x)]).T
        beta, c = np.linalg.lstsq(X, y, rcond=None)[0]
        
        # Compute spread
        spread = y - (beta * x + c)
        spread_series = pd.Series(spread, index=prices.index)
        
        # Rolling z-score
        window = params['window']
        mu = spread_series.rolling(window).mean()
        sig = spread_series.rolling(window).std().replace(0, 1e-9)
        z = (spread_series - mu) / sig
        
        # Trading signals
        positions = np.zeros_like(z.values)
        position = 0
        
        for i in range(len(z)):
            if np.isnan(z.iloc[i]):
                positions[i] = position
                continue
            
            # Entry signals
            if z.iloc[i] < -params['entry_z'] and position == 0:
                position = 1  # Long spread
            elif z.iloc[i] > params['entry_z'] and position == 0:
                position = -1  # Short spread
            
            # Exit signals
            elif abs(z.iloc[i]) < params['exit_z'] and position != 0:
                position = 0
            
            positions[i] = position
        
        positions = pd.Series(positions, index=prices.index)
        
        # Compute PnL
        spread_returns = spread_series.diff().fillna(0)
        gross_pnl = (positions.shift(1).fillna(0) * spread_returns).cumsum()
        
        # Account for transaction costs
        trades = positions.diff().abs()
        costs = (trades * params['transaction_cost']).cumsum()
        net_pnl = gross_pnl - costs
        
        # Cointegration test
        try:
            coint_result = meanrev.cointegration_test(prices.iloc[:, 0], prices.iloc[:, 1])
            p_value = coint_result.get('p_value', 1.0)
        except:
            p_value = 1.0
        
        return {
            'spread': spread_series,
            'zscore': z,
            'positions': positions,
            'gross_pnl': gross_pnl,
            'net_pnl': net_pnl,
            'beta': beta,
            'intercept': c,
            'p_value': p_value,
            'trades': trades.sum() / 2  # Each trade involves entry + exit
        }
    
    except Exception as e:
        st.error(f"Pairs trading computation failed: {e}")
        return {}


def compute_triangular_arb(data: Dict, symbol_a: str, symbol_b: str, 
                           symbol_c: str, params: Dict) -> Dict:
    """Compute triangular arbitrage opportunities"""
    try:
        # Extract prices
        prices = extract_prices(data, [symbol_a, symbol_b, symbol_c])
        
        pa = prices.iloc[:, 0]
        pb = prices.iloc[:, 1]
        pc = prices.iloc[:, 2]
        
        # Forward path: A/B * B/C * C/A = (A/B) * (B/C) * (C/A)
        forward = (pa / pb) * (pb / pc) * (pc / pa)
        
        # Arbitrage signal
        deviation = np.abs(forward - 1.0)
        
        # Opportunities (above threshold and cost)
        net_threshold = params['threshold'] + params['transaction_cost']
        opportunities = deviation > net_threshold
        
        # Theoretical profit (before costs)
        theoretical_profit = deviation.copy()
        theoretical_profit[~opportunities] = 0
        
        # Net profit (after costs)
        net_profit = theoretical_profit - params['transaction_cost']
        net_profit[net_profit < 0] = 0
        
        # Cumulative PnL
        cumulative_pnl = net_profit.cumsum() * 10000  # Scale for visibility
        
        return {
            'forward': forward,
            'deviation': deviation,
            'opportunities': opportunities,
            'theoretical_profit': theoretical_profit,
            'net_profit': net_profit,
            'cumulative_pnl': cumulative_pnl,
            'num_opportunities': opportunities.sum(),
            'avg_profit': net_profit[net_profit > 0].mean() if (net_profit > 0).any() else 0
        }
    
    except Exception as e:
        st.error(f"Triangular arbitrage computation failed: {e}")
        return {}


# ============================================================================
# DISPLAY FUNCTIONS
# ============================================================================

def display_mean_reversion_results(results: Dict, symbols: List[str]):
    """Display mean reversion arbitrage results"""
    st.markdown("---")
    st.markdown("### 📊 Mean Reversion Results")
    
    # Performance metrics
    methods = list(results.keys())
    cols = st.columns(len(methods))
    
    for i, method in enumerate(methods):
        with cols[i]:
            backtest = results[method]['backtest']
            pnl = backtest['pnl']
            
            st.metric(
                f"{method} Method",
                f"${pnl.iloc[-1]:,.2f}",
                f"{(pnl.iloc[-1] / 100000 * 100):.2f}%"
            )
            
            sharpe = compute_sharpe(pnl)
            max_dd = compute_max_drawdown(pnl)
            
            st.caption(f"Sharpe: {sharpe:.2f}")
            st.caption(f"Max DD: {max_dd:.2%}")
    
    # PnL comparison chart
    fig = go.Figure()
    
    for method in methods:
        pnl = results[method]['backtest']['pnl']
        fig.add_trace(go.Scatter(
            x=pnl.index,
            y=pnl.values,
            name=method,
            mode='lines'
        ))
    
    fig.update_layout(
        title="Cumulative PnL Comparison",
        xaxis_title="Date",
        yaxis_title="PnL ($)",
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Portfolio weights
    st.markdown("#### Portfolio Weights")
    
    weights_df = pd.DataFrame({
        method: results[method]['weights']
        for method in methods
    }, index=symbols)
    
    fig = go.Figure()
    
    for method in methods:
        fig.add_trace(go.Bar(
            name=method,
            x=symbols,
            y=weights_df[method].values
        ))
    
    fig.update_layout(
        title="Portfolio Weights by Method",
        xaxis_title="Symbol",
        yaxis_title="Weight",
        barmode='group',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed metrics table
    with st.expander("📋 Detailed Metrics"):
        metrics_data = []
        for method in methods:
            backtest = results[method]['backtest']
            pnl = backtest['pnl']
            
            metrics_data.append({
                'Method': method,
                'Total Return': f"${pnl.iloc[-1]:,.2f}",
                'Return %': f"{(pnl.iloc[-1] / 100000 * 100):.2f}%",
                'Sharpe Ratio': f"{compute_sharpe(pnl):.2f}",
                'Max Drawdown': f"{compute_max_drawdown(pnl):.2%}",
                'Num Trades': backtest.get('num_trades', 0),
                'Win Rate': f"{backtest.get('win_rate', 0):.2%}"
            })
        
        st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)


def display_pairs_results(results: Dict, symbol_1: str, symbol_2: str):
    """Display pairs trading results"""
    st.markdown("---")
    st.markdown(f"### 📊 Pairs Trading: {symbol_1} vs {symbol_2}")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Net PnL", f"${results['net_pnl'].iloc[-1]:,.2f}")
    
    with col2:
        st.metric("Hedge Ratio (β)", f"{results['beta']:.4f}")
    
    with col3:
        st.metric("Cointegration p-value", f"{results['p_value']:.4f}",
                 delta="Cointegrated" if results['p_value'] < 0.05 else "Not cointegrated")
    
    with col4:
        st.metric("Number of Trades", f"{int(results['trades'])}")
    
    # Charts
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=("Spread", "Z-Score & Signals", "Cumulative PnL"),
        vertical_spacing=0.1,
        row_heights=[0.3, 0.3, 0.4]
    )
    
    # Spread
    fig.add_trace(
        go.Scatter(x=results['spread'].index, y=results['spread'].values,
                  name="Spread", line=dict(color='blue')),
        row=1, col=1
    )
    
    # Z-score with entry/exit levels
    fig.add_trace(
        go.Scatter(x=results['zscore'].index, y=results['zscore'].values,
                  name="Z-Score", line=dict(color='purple')),
        row=2, col=1
    )
    fig.add_hline(y=2, line_dash="dash", line_color="red", row=2, col=1,
                 annotation_text="Entry (short)")
    fig.add_hline(y=-2, line_dash="dash", line_color="green", row=2, col=1,
                 annotation_text="Entry (long)")
    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)
    
    # PnL
    fig.add_trace(
        go.Scatter(x=results['net_pnl'].index, y=results['net_pnl'].values,
                  name="Net PnL", line=dict(color='green'), fill='tozeroy'),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=results['gross_pnl'].index, y=results['gross_pnl'].values,
                  name="Gross PnL", line=dict(color='lightgreen', dash='dash')),
        row=3, col=1
    )
    
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="Spread", row=1, col=1)
    fig.update_yaxes(title_text="Z-Score", row=2, col=1)
    fig.update_yaxes(title_text="PnL ($)", row=3, col=1)
    
    fig.update_layout(height=900, showlegend=True, hovermode='x unified')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Trade analysis
    with st.expander("📋 Trade Analysis"):
        returns = results['net_pnl'].diff().dropna()
        winning_trades = returns[returns > 0]
        losing_trades = returns[returns < 0]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Winning Trades**")
            st.write(f"Count: {len(winning_trades)}")
            st.write(f"Avg Profit: ${winning_trades.mean():.2f}" if len(winning_trades) > 0 else "Avg Profit: $0.00")
            st.write(f"Total Profit: ${winning_trades.sum():.2f}" if len(winning_trades) > 0 else "Total Profit: $0.00")
        
        with col2:
            st.markdown("**Losing Trades**")
            st.write(f"Count: {len(losing_trades)}")
            st.write(f"Avg Loss: ${losing_trades.mean():.2f}" if len(losing_trades) > 0 else "Avg Loss: $0.00")
            st.write(f"Total Loss: ${losing_trades.sum():.2f}" if len(losing_trades) > 0 else "Total Loss: $0.00")


def display_triangular_results(results: Dict, symbol_a: str, symbol_b: str, symbol_c: str):
    """Display triangular arbitrage results"""
    st.markdown("---")
    st.markdown(f"### 📊 Triangular Arbitrage: {symbol_a} → {symbol_b} → {symbol_c}")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Opportunities Found", f"{int(results['num_opportunities'])}")
    
    with col2:
        st.metric("Avg Profit per Opportunity", f"{results['avg_profit']:.4%}")
    
    with col3:
        st.metric("Cumulative PnL", f"${results['cumulative_pnl'].iloc[-1]:,.2f}")
    
    with col4:
        max_dev = results['deviation'].max()
        st.metric("Max Deviation", f"{max_dev:.4%}")
    
    # Charts
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=("Forward Path Product", "Deviation from Parity", "Cumulative PnL"),
        vertical_spacing=0.1,
        row_heights=[0.3, 0.3, 0.4]
    )
    
    # Forward path
    fig.add_trace(
        go.Scatter(x=results['forward'].index, y=results['forward'].values,
                  name="Forward Path", line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", row=1, col=1,
                 annotation_text="Parity (1.0)")
    
    # Deviation with opportunities highlighted
    fig.add_trace(
        go.Scatter(x=results['deviation'].index, y=results['deviation'].values,
                  name="Deviation", line=dict(color='orange')),
        row=2, col=1
    )
    
    # Highlight opportunities
    opps = results['opportunities']
    if opps.sum() > 0:
        fig.add_trace(
            go.Scatter(x=results['deviation'].index[opps],
                      y=results['deviation'].values[opps],
                      mode='markers',
                      name="Opportunities",
                      marker=dict(color='red', size=8)),
            row=2, col=1
        )
    
    # Cumulative PnL
    fig.add_trace(
        go.Scatter(x=results['cumulative_pnl'].index, y=results['cumulative_pnl'].values,
                  name="Cumulative PnL", line=dict(color='green'), fill='tozeroy'),
        row=3, col=1
    )
    
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="Forward Path", row=1, col=1)
    fig.update_yaxes(title_text="Deviation", row=2, col=1)
    fig.update_yaxes(title_text="PnL ($)", row=3, col=1)
    
    fig.update_layout(height=900, showlegend=True, hovermode='x unified')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Opportunity timeline
    with st.expander("📋 Opportunity Timeline"):
        opp_df = pd.DataFrame({
            'Timestamp': results['deviation'].index[results['opportunities']],
            'Deviation': results['deviation'].values[results['opportunities']],
            'Net Profit': results['net_profit'].values[results['opportunities']]
        })
        
        if len(opp_df) > 0:
            st.dataframe(opp_df, use_container_width=True)
        else:
            st.info("No arbitrage opportunities found with current parameters")


def display_multi_strategy_results(all_results: Dict):
    """Display multi-strategy comparison"""
    st.markdown("---")
    st.markdown("### 📊 Multi-Strategy Comparison")
    
    # Collect all PnL series
    pnl_series = {}
    
    # Mean reversion
    if 'Mean Reversion' in all_results:
        for method, data in all_results['Mean Reversion'].items():
            pnl_series[f"MR_{method}"] = data['backtest']['pnl']
    
    # Pairs trading
    if 'Pairs Trading' in all_results:
        for i, result in enumerate(all_results['Pairs Trading']):
            pnl_series[f"Pairs_{i+1}"] = result['net_pnl']
    
    # Triangular arbitrage
    if 'Triangular' in all_results:
        pnl_series['Triangular'] = all_results['Triangular']['cumulative_pnl']
    
    # Performance metrics
    st.markdown("#### Performance Metrics")
    
    metrics_data = []
    for strategy, pnl in pnl_series.items():
        metrics_data.append({
            'Strategy': strategy,
            'Total Return': f"${pnl.iloc[-1]:,.2f}",
            'Return %': f"{(pnl.iloc[-1] / 100000 * 100):.2f}%",
            'Sharpe Ratio': f"{compute_sharpe(pnl):.2f}",
            'Max Drawdown': f"{compute_max_drawdown(pnl):.2%}",
            'Volatility': f"{compute_volatility(pnl):.2%}"
        })
    
    st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)
    
    # Comparison chart
    fig = go.Figure()
    
    for strategy, pnl in pnl_series.items():
        fig.add_trace(go.Scatter(
            x=pnl.index,
            y=pnl.values,
            name=strategy,
            mode='lines'
        ))
    
    fig.update_layout(
        title="Strategy Performance Comparison",
        xaxis_title="Date",
        yaxis_title="Cumulative PnL ($)",
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Risk-return scatter
    st.markdown("#### Risk-Return Profile")
    
    returns = []
    volatilities = []
    strategy_names = []
    
    for strategy, pnl in pnl_series.items():
        returns.append((pnl.iloc[-1] / 100000) * 100)  # Return %
        volatilities.append(compute_volatility(pnl) * 100)  # Volatility %
        strategy_names.append(strategy)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=volatilities,
        y=returns,
        mode='markers+text',
        text=strategy_names,
        textposition='top center',
        marker=dict(size=12, color=returns, colorscale='RdYlGn', showscale=True,
                   colorbar=dict(title="Return %"))
    ))
    
    fig.update_layout(
        title="Risk-Return Profile",
        xaxis_title="Volatility (%)",
        yaxis_title="Return (%)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def extract_prices(data: Dict, symbols: List[str]) -> pd.DataFrame:
    """Extract close prices from loaded data"""
    prices = {}
    
    for symbol in symbols:
        if symbol in data:
            df = data[symbol]
            if 'close' in df.columns:
                prices[symbol] = df['close']
            elif 'Close' in df.columns:
                prices[symbol] = df['Close']
    
    return pd.DataFrame(prices)


def get_date_range_str(data: Dict) -> str:
    """Get date range string from loaded data"""
    if not data:
        return "N/A"
    
    first_symbol = list(data.keys())[0]
    df = data[first_symbol]
    
    if len(df) == 0:
        return "N/A"
    
    start = df.index[0].strftime("%Y-%m-%d")
    end = df.index[-1].strftime("%Y-%m-%d")
    
    return f"{start} to {end}"


def compute_sharpe(pnl: pd.Series, periods_per_year: int = 252) -> float:
    """Compute annualized Sharpe ratio"""
    returns = pnl.diff().dropna()
    
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    
    return (returns.mean() / returns.std()) * np.sqrt(periods_per_year)


def compute_max_drawdown(pnl: pd.Series) -> float:
    """Compute maximum drawdown"""
    cummax = pnl.cummax()
    drawdown = (pnl - cummax) / (cummax + 1e-9)
    
    return drawdown.min()


def compute_volatility(pnl: pd.Series, periods_per_year: int = 252) -> float:
    """Compute annualized volatility"""
    returns = pnl.diff().dropna()
    
    if len(returns) == 0:
        return 0.0
    
    return returns.std() * np.sqrt(periods_per_year)


# Entry point
if __name__ == "__main__":
    render()
