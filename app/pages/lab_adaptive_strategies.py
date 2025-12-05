"""
Adaptive Strategies Laboratory
===============================

Test and visualize regime-adaptive trading strategies with:
- Real-time HMM regime detection
- Parameter adaptation by regime
- 3D performance visualizations
- Heatmaps of regime transitions
- Strategy comparison across regimes
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
sys.path.append('/app')

try:
    from python.adaptive_strategies import (
        AdaptiveMeanReversion,
        AdaptiveMomentum,
        AdaptiveStatArb,
        RegimeConfig
    )
    from python.advanced_optimization import RUST_AVAILABLE
except ImportError:
    st.error("⚠️ Adaptive strategies not available")
    st.stop()

from utils.ui_components import render_sidebar_navigation, apply_custom_css

# Page config
st.set_page_config(
    page_title="Adaptive Strategies Lab",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_css()
render_sidebar_navigation()

st.title("🎯 Adaptive Strategies Laboratory")
st.markdown("""
**Test regime-adaptive trading strategies with real market data**

Features:
- HMM automatic regime detection
- Parameter adaptation per market condition  
- Performance tracking by regime
- Interactive 3D visualizations
""")

# Check for data
if 'historical_data' not in st.session_state or st.session_state.historical_data is None:
    st.warning("⚠️ Please load market data first using the Data Loader page")
    if st.button("📊 Go to Data Loader"):
        st.switch_page("pages/data_loader.py")
    st.stop()

df = st.session_state.historical_data

# Sidebar configuration
st.sidebar.header("⚙️ Strategy Configuration")

# Strategy selection
strategy_type = st.sidebar.selectbox(
    "Strategy Type",
    ["Mean Reversion", "Momentum", "Statistical Arbitrage"],
    help="Select adaptive strategy type"
)

# HMM configuration
st.sidebar.subheader("🔮 Regime Detection")
n_regimes = st.sidebar.slider("Number of Regimes", 2, 5, 3)
lookback_period = st.sidebar.number_input("Training Period (bars)", 100, 2000, 500, 50)
update_frequency = st.sidebar.number_input("Update Frequency (bars)", 10, 500, 100, 10)

# Rust status
if RUST_AVAILABLE:
    st.sidebar.success("🚀 Rust Acceleration: ON")
else:
    st.sidebar.warning("⚠️ Rust: OFF (slower)")

# Base parameters
st.sidebar.subheader("📊 Base Parameters")
with st.sidebar.expander("Strategy Settings", expanded=False):
    entry_threshold = st.slider("Entry Threshold", 0.5, 5.0, 2.0, 0.1)
    exit_threshold = st.slider("Exit Threshold", 0.1, 2.0, 0.5, 0.1)
    position_size = st.slider("Position Size", 0.1, 2.0, 1.0, 0.1)
    stop_loss = st.slider("Stop Loss (%)", 0.5, 10.0, 2.0, 0.5) / 100
    take_profit = st.slider("Take Profit (%)", 1.0, 20.0, 5.0, 0.5) / 100
    max_holding = st.number_input("Max Holding Period", 5, 100, 20, 5)

base_config = {
    'entry_threshold': entry_threshold,
    'exit_threshold': exit_threshold,
    'position_size': position_size,
    'stop_loss': stop_loss,
    'take_profit': take_profit,
    'max_holding_period': max_holding
}

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Strategy Testing",
    "🔬 Regime Analysis",
    "📈 Performance Visualization",
    "🎨 Advanced Viz (3D & Heatmaps)"
])

# =============================================================================
# TAB 1: STRATEGY TESTING
# =============================================================================
with tab1:
    st.header("Strategy Backtesting")
    
    # Symbol selection - now supports multiple symbols
    if 'symbol' in df.columns:
        symbols = df['symbol'].unique().tolist()
        
        # Initialize session state for selected symbols
        if 'adaptive_selected_symbols' not in st.session_state:
            st.session_state['adaptive_selected_symbols'] = [symbols[0]] if symbols else []
        
        col1, col2 = st.columns([3, 1])
        with col1:
            selected_symbols = st.multiselect(
                "Select Symbol(s)",
                symbols,
                default=st.session_state['adaptive_selected_symbols'],
                help="Select one or more symbols for backtesting"
            )
            # Update session state with current selection
            st.session_state['adaptive_selected_symbols'] = selected_symbols
        with col2:
            if st.button("📊 Select All"):
                st.session_state['adaptive_selected_symbols'] = symbols
                st.rerun()
        
        if not selected_symbols:
            st.warning("⚠️ Please select at least one symbol")
            st.stop()
        
        test_data = df[df['symbol'].isin(selected_symbols)].copy()
        symbol_display = ", ".join(selected_symbols) if len(selected_symbols) <= 5 else f"{len(selected_symbols)} symbols"
    else:
        selected_symbols = ["Dataset"]
        symbol_display = "Dataset"
        test_data = df.copy()
    
    if 'timestamp' in test_data.columns and 'symbol' not in test_data.columns:
        test_data = test_data.set_index('timestamp').sort_index()
    
    st.markdown(f"**Testing on:** {symbol_display} ({len(test_data)} bars total)")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Adjust max test period based on number of symbols for performance
        max_test = min(len(test_data), 5000 if len(selected_symbols) == 1 else 2000)
        default_test = min(1000 if len(selected_symbols) == 1 else 500, len(test_data))
        
        test_period = st.slider(
            "Test Period (bars)",
            min_value=200,
            max_value=max_test,
            value=default_test,
            step=100,
            help=f"Reduced default for multi-symbol testing. Processing {len(selected_symbols)} symbols..."
        )
    
    with col2:
        if st.button("🚀 Run Backtest", type="primary"):
            progress_bar = st.progress(0, text="Initializing strategy...")
            with st.spinner(f"Running adaptive strategy backtest on {len(selected_symbols)} symbol(s)..."):
                # Optimize HMM update frequency for multi-symbol
                opt_update_freq = update_frequency * len(selected_symbols) if len(selected_symbols) > 1 else update_frequency
                
                # Initialize strategy
                progress_bar.progress(10, text="Creating strategy instance...")
                if strategy_type == "Mean Reversion":
                    strategy = AdaptiveMeanReversion(
                        n_regimes=n_regimes,
                        lookback_period=lookback_period,
                        update_frequency=opt_update_freq,
                        base_config=base_config
                    )
                elif strategy_type == "Momentum":
                    strategy = AdaptiveMomentum(
                        n_regimes=n_regimes,
                        lookback_period=lookback_period,
                        update_frequency=opt_update_freq,
                        base_config=base_config
                    )
                else:  # Statistical Arbitrage
                    strategy = AdaptiveStatArb(
                        n_regimes=n_regimes,
                        lookback_period=lookback_period,
                        update_frequency=opt_update_freq,
                        base_config=base_config
                    )
                
                # Initialize portfolio tracking across all symbols
                portfolio_value = 100000.0
                cash = portfolio_value
                positions: dict[str, dict[str, float] | None] = {sym: None for sym in selected_symbols}
                all_trades = []
                equity_curve = [portfolio_value]
                regime_history = []
                
                # For multi-symbol: process each symbol's data
                if 'symbol' in test_data.columns and len(selected_symbols) > 1:
                    # Multi-symbol backtest - OPTIMIZED
                    # Prepare data for all symbols first
                    symbol_dfs = {}
                    max_bars = 0
                    for symbol in selected_symbols:
                        symbol_data = test_data[test_data['symbol'] == symbol].copy()
                        if 'timestamp' in symbol_data.columns:
                            symbol_data = symbol_data.set_index('timestamp').sort_index()
                        
                        symbol_df = symbol_data.iloc[-test_period:] if len(symbol_data) > test_period else symbol_data
                        symbol_dfs[symbol] = symbol_df
                        max_bars = max(max_bars, len(symbol_df))
                    
                    # Use first symbol for regime detection to avoid redundant HMM calls
                    ref_symbol = selected_symbols[0]
                    ref_df = symbol_dfs[ref_symbol]
                    
                    progress_bar.progress(20, text="Preparing data...")
                    
                    # Process bars for all symbols simultaneously
                    total_bars = min(max_bars, len(ref_df)) - lookback_period
                    for idx, i in enumerate(range(lookback_period, min(max_bars, len(ref_df)))):
                        # Update progress every 50 bars
                        if idx % 50 == 0:
                            progress = min(95, 20 + int((idx / total_bars) * 75))
                            progress_bar.progress(progress, text=f"Processing bar {idx}/{total_bars}...")
                        
                        # Update regime once per bar using reference symbol
                        ref_window = ref_df.iloc[:i+1]
                        _ = strategy.generate_signal(ref_window, ref_symbol, {ref_symbol: positions[ref_symbol]})
                        
                        current_regime = strategy.current_regime if strategy.current_regime is not None else 1
                        regime_history.append(current_regime)
                        
                        # Process each symbol with the current regime
                        for symbol in selected_symbols:
                            if i >= len(symbol_dfs[symbol]):
                                continue
                                
                            symbol_df = symbol_dfs[symbol]
                            if i < lookback_period or i >= len(symbol_df):
                                continue
                            
                            window = symbol_df.iloc[:i+1]
                            current_price = window['close'].iloc[-1]
                            
                            # Generate signal (regime already updated above)
                            current_positions = {symbol: positions[symbol]}
                            signal = strategy.generate_signal(window, symbol, current_positions)
                            
                            # Execute signal
                            if signal:
                                if signal['action'] == 'open' and positions[symbol] is None:
                                    # Open position
                                    allocation = cash / len(selected_symbols)  # Equal allocation
                                    shares = (allocation * signal['size']) / current_price
                                    positions[symbol] = {
                                        'side': signal['side'],
                                        'shares': shares,
                                        'entry_price': current_price,
                                        'entry_bar': i,
                                        'regime': signal['regime']
                                    }
                                    cash -= shares * current_price
                                    
                                    all_trades.append({
                                        'symbol': symbol,
                                        'bar': i,
                                        'action': 'open',
                                        'side': signal['side'],
                                        'price': current_price,
                                        'regime': signal['regime'],
                                        'reason': signal.get('reason', 'signal')
                                    })
                                
                                elif signal['action'] == 'close' and positions[symbol] is not None:
                                    # Close position
                                    position = positions[symbol]
                                    assert position is not None  # Type narrowing for type checker
                                    shares = position['shares']
                                    side = position['side']
                                    entry_price = position['entry_price']
                                    entry_bar = position['entry_bar']
                                    
                                    if side == 'long':
                                        pnl = shares * (current_price - entry_price)
                                    else:
                                        pnl = shares * (entry_price - current_price)
                                    
                                    cash += shares * current_price
                                    
                                    all_trades.append({
                                        'symbol': symbol,
                                        'bar': i,
                                        'action': 'close',
                                        'side': side,
                                        'price': current_price,
                                        'pnl': pnl,
                                        'pnl_pct': (pnl / (shares * entry_price)) * 100,
                                        'holding_period': i - entry_bar,
                                        'regime': signal['regime'],
                                        'reason': signal.get('reason', 'signal')
                                    })
                                    
                                    positions[symbol] = None
                        
                        # Update portfolio value once per bar
                        portfolio_value = cash
                        for sym, pos in positions.items():
                            if pos is not None and i < len(symbol_dfs[sym]):
                                sym_price = symbol_dfs[sym].iloc[i]['close']
                                if pos['side'] == 'long':
                                    portfolio_value += pos['shares'] * sym_price
                                else:
                                    portfolio_value += pos['shares'] * (2 * pos['entry_price'] - sym_price)
                        
                        equity_curve.append(portfolio_value)
                    
                    trades = all_trades
                    selected_symbol = selected_symbols[0]  # For compatibility with single-symbol code
                    test_df = ref_df
                    
                else:
                    # Single symbol backtest (original logic)
                    progress_bar.progress(20, text="Preparing data...")
                    test_df = test_data.iloc[-test_period:].copy()
                    selected_symbol = selected_symbols[0]
                    position = None
                    trades = []
                    
                    total_bars = len(test_df) - lookback_period
                    for idx, i in enumerate(range(lookback_period, len(test_df))):
                        # Update progress every 50 bars
                        if idx % 50 == 0:
                            progress = min(95, 20 + int((idx / total_bars) * 75))
                            progress_bar.progress(progress, text=f"Processing bar {idx}/{total_bars}...")
                        
                        window = test_df.iloc[:i+1]
                        current_price = window['close'].iloc[-1]
                        
                        # Generate signal
                        current_positions = {selected_symbol: position} if position else {}
                        signal = strategy.generate_signal(window, selected_symbol, current_positions)
                        
                        # Track regime
                        current_regime = strategy.current_regime if strategy.current_regime is not None else 1
                        regime_history.append(current_regime)
                        
                        # Execute signal
                        if signal:
                            if signal['action'] == 'open' and position is None:
                                # Open position
                                shares = (cash * signal['size']) / current_price
                                position = {
                                    'side': signal['side'],
                                    'shares': shares,
                                    'entry_price': current_price,
                                    'entry_bar': i,
                                    'regime': signal['regime']
                                }
                                cash -= shares * current_price
                                
                                trades.append({
                                    'bar': i,
                                    'action': 'open',
                                    'side': signal['side'],
                                    'price': current_price,
                                    'regime': signal['regime'],
                                    'reason': signal.get('reason', 'signal')
                                })
                            
                            elif signal['action'] == 'close' and position is not None:
                                # Close position
                                shares = position['shares']
                                entry_price = position['entry_price']
                                
                                if position['side'] == 'long':
                                    pnl = shares * (current_price - entry_price)
                                else:  # short
                                    pnl = shares * (entry_price - current_price)
                                
                                cash += shares * current_price
                                
                                strategy.record_trade(
                                    symbol=selected_symbol,
                                    action='close',
                                    regime=position['regime'],
                                    entry_price=entry_price,
                                    exit_price=current_price,
                                    pnl=pnl
                                )
                                
                                trades.append({
                                    'bar': i,
                                    'action': 'close',
                                    'price': current_price,
                                    'pnl': pnl,
                                    'pnl_pct': (pnl / (shares * entry_price)) * 100,
                                    'holding_period': i - position['entry_bar'],
                                    'regime': position['regime'],
                                    'reason': signal.get('reason', 'signal')
                                })
                                
                                position = None
                        
                        # Update portfolio value
                        if position:
                            if position['side'] == 'long':
                                portfolio_value = cash + position['shares'] * current_price
                            else:  # short
                                portfolio_value = cash + position['shares'] * (2 * position['entry_price'] - current_price)
                        else:
                            portfolio_value = cash
                        
                        equity_curve.append(portfolio_value)
                
                # Store results
                progress_bar.progress(100, text="Finalizing results...")
                st.session_state['backtest_results'] = {
                    'strategy': strategy,
                    'trades': trades,
                    'equity_curve': equity_curve,
                    'regime_history': regime_history,
                    'test_data': test_df,
                    'final_value': portfolio_value,
                    'initial_value': 100000.0
                }
                
                progress_bar.empty()
                st.success(f"✓ Backtest completed! Processed {len(selected_symbols)} symbol(s) with {len(trades)} trades")
    
    # Show results
    if 'backtest_results' in st.session_state:
        results = st.session_state['backtest_results']
        
        # Performance metrics
        st.subheader("📊 Performance Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        total_return = ((results['final_value'] / results['initial_value']) - 1) * 100
        n_trades = len([t for t in results['trades'] if t['action'] == 'close'])
        
        if n_trades > 0:
            winning_trades = [t for t in results['trades'] if t['action'] == 'close' and t.get('pnl', 0) > 0]
            win_rate = (len(winning_trades) / n_trades) * 100
            avg_pnl = np.mean([t.get('pnl', 0) for t in results['trades'] if t['action'] == 'close'])
        else:
            win_rate = 0
            avg_pnl = 0
        
        with col1:
            st.metric("Total Return", f"{total_return:.2f}%")
        with col2:
            st.metric("Total Trades", n_trades)
        with col3:
            st.metric("Win Rate", f"{win_rate:.1f}%")
        with col4:
            st.metric("Avg P&L", f"${avg_pnl:.2f}")
        
        # Equity curve
        st.subheader("💰 Equity Curve")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            y=results['equity_curve'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#4ECDC4', width=2)
        ))
        
        # Add trade markers
        close_trades = [t for t in results['trades'] if t['action'] == 'close']
        if close_trades:
            profitable = [t for t in close_trades if t.get('pnl', 0) > 0]
            losses = [t for t in close_trades if t.get('pnl', 0) <= 0]
            
            if profitable:
                fig.add_trace(go.Scatter(
                    x=[t['bar'] - lookback_period for t in profitable],
                    y=[results['equity_curve'][t['bar'] - lookback_period] for t in profitable],
                    mode='markers',
                    name='Profitable',
                    marker=dict(color='green', size=10, symbol='triangle-up')
                ))
            
            if losses:
                fig.add_trace(go.Scatter(
                    x=[t['bar'] - lookback_period for t in losses],
                    y=[results['equity_curve'][t['bar'] - lookback_period] for t in losses],
                    mode='markers',
                    name='Loss',
                    marker=dict(color='red', size=10, symbol='triangle-down')
                ))
        
        fig.update_layout(
            title="Portfolio Value Over Time",
            xaxis_title="Bar",
            yaxis_title="Portfolio Value ($)",
            height=500,
            template="plotly_dark",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Trade list
        st.subheader("📋 Trade History")
        if close_trades:
            trade_df = pd.DataFrame(close_trades)
            trade_df['regime_name'] = trade_df['regime'].map({
                0: "Bear", 1: "Sideways", 2: "Bull"
            })
            
            # Build column list based on what's available
            display_cols = ['bar', 'price', 'pnl', 'pnl_pct', 'holding_period', 'regime_name', 'reason']
            if 'symbol' in trade_df.columns:
                display_cols = ['symbol'] + display_cols
            
            # Only show columns that exist in the dataframe
            display_cols = [col for col in display_cols if col in trade_df.columns]
            
            st.dataframe(
                trade_df[display_cols],
                use_container_width=True
            )
        else:
            st.info("No completed trades yet")

# =============================================================================
# TAB 2: REGIME ANALYSIS
# =============================================================================
with tab2:
    st.header("🔬 Market Regime Analysis")
    
    if 'backtest_results' in st.session_state:
        results = st.session_state['backtest_results']
        strategy = results['strategy']
        
        # Regime statistics
        st.subheader("📊 Regime Statistics")
        
        regime_perf = strategy.get_regime_performance()
        if not regime_perf.empty:
            st.dataframe(regime_perf, use_container_width=True)
        
        # Transition matrix
        st.subheader("🔄 Regime Transition Matrix")
        
        trans_matrix = strategy.get_transition_probabilities()
        if trans_matrix is not None:
            regime_names = ["Bear", "Sideways", "Bull"][:n_regimes]
            
            fig = go.Figure(data=go.Heatmap(
                z=trans_matrix,
                x=[f"To {name}" for name in regime_names],
                y=[f"From {name}" for name in regime_names],
                colorscale='RdYlGn',
                text=trans_matrix,
                texttemplate='%{text:.3f}',
                textfont={"size": 14},
                colorbar=dict(title="Probability")
            ))
            
            fig.update_layout(
                title="State Transition Probabilities",
                height=400,
                template="plotly_dark"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Emission parameters
        st.subheader("📈 Regime Characteristics")
        
        emission_params = strategy.get_emission_params()
        if emission_params:
            col1, col2, col3 = st.columns(3)
            
            for i, (mean, var) in enumerate(emission_params):
                regime_name = ["Bear", "Sideways", "Bull"][i] if i < 3 else f"Regime {i}"
                
                # Handle NaN/inf values gracefully
                if not np.isfinite(mean) or not np.isfinite(var) or var < 0:
                    with [col1, col2, col3][i]:
                        st.metric(
                            regime_name,
                            "Not yet fitted",
                            delta="Waiting for data"
                        )
                else:
                    std = np.sqrt(var)
                    with [col1, col2, col3][i]:
                        st.metric(
                            regime_name,
                            f"μ={mean:.4f}",
                            delta=f"σ={std:.4f}"
                        )
        
        # Regime timeline
        st.subheader("⏱️ Regime Evolution")
        
        regime_history = results['regime_history']
        test_df = results['test_data']
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=("Price", "Market Regime"),
            row_heights=[0.7, 0.3]
        )
        
        # Price chart
        fig.add_trace(
            go.Scatter(
                y=test_df['close'].values[lookback_period:],
                mode='lines',
                name='Price',
                line=dict(color='white', width=1)
            ),
            row=1, col=1
        )
        
        # Regime colored background
        colors = ['red', 'yellow', 'green', 'blue', 'purple']
        for regime in range(n_regimes):
            mask = np.array(regime_history) == regime
            indices = np.where(mask)[0]
            
            for idx in indices:
                fig.add_vrect(
                    x0=max(0, idx-0.5),
                    x1=min(len(regime_history), idx+0.5),
                    fillcolor=colors[regime],
                    opacity=0.2,
                    layer="below",
                    line_width=0,
                    row=1, col=1
                )
        
        # Regime indicator
        fig.add_trace(
            go.Scatter(
                y=regime_history,
                mode='lines',
                name='Regime',
                line=dict(color='cyan', width=2)
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=700,
            template="plotly_dark",
            showlegend=True,
            hovermode='x unified'
        )
        
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Regime ID", row=2, col=1)
        fig.update_xaxes(title_text="Bar", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("👆 Run a backtest first to see regime analysis")

# =============================================================================
# TAB 3: PERFORMANCE VISUALIZATION
# =============================================================================
with tab3:
    st.header("📈 Performance Metrics")
    
    if 'backtest_results' in st.session_state:
        results = st.session_state['backtest_results']
        close_trades = [t for t in results['trades'] if t['action'] == 'close']
        
        if close_trades:
            trade_df = pd.DataFrame(close_trades)
            
            # P&L distribution by regime
            st.subheader("💰 P&L Distribution by Regime")
            
            regime_names = {0: "Bear", 1: "Sideways", 2: "Bull"}
            trade_df['regime_name'] = trade_df['regime'].map(regime_names)
            
            fig = px.box(
                trade_df,
                x='regime_name',
                y='pnl',
                color='regime_name',
                title="P&L Distribution by Market Regime",
                labels={'pnl': 'P&L ($)', 'regime_name': 'Market Regime'}
            )
            
            fig.update_layout(
                height=500,
                template="plotly_dark",
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Holding period analysis
            st.subheader("⏱️ Holding Period Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(
                    trade_df,
                    x='holding_period',
                    color='regime_name',
                    title="Holding Period Distribution",
                    labels={'holding_period': 'Bars Held'}
                )
                fig.update_layout(template="plotly_dark", height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.scatter(
                    trade_df,
                    x='holding_period',
                    y='pnl_pct',
                    color='regime_name',
                    size=abs(trade_df['pnl']),
                    title="P&L vs Holding Period",
                    labels={'holding_period': 'Bars Held', 'pnl_pct': 'P&L (%)'}
                )
                fig.update_layout(template="plotly_dark", height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Win rate by regime
            st.subheader("🎯 Win Rate by Regime")
            
            win_rate_data = []
            for regime in trade_df['regime'].unique():
                regime_trades = trade_df[trade_df['regime'] == regime]
                wins = (regime_trades['pnl'] > 0).sum()
                total = len(regime_trades)
                win_rate = (wins / total * 100) if total > 0 else 0
                
                win_rate_data.append({
                    'regime': regime_names[regime],
                    'win_rate': win_rate,
                    'trades': total
                })
            
            wr_df = pd.DataFrame(win_rate_data)
            
            fig = go.Figure(data=[
                go.Bar(
                    x=wr_df['regime'],
                    y=wr_df['win_rate'],
                    text=wr_df['trades'],
                    texttemplate='%{y:.1f}%<br>(%{text} trades)',
                    textposition='outside',
                    marker=dict(
                        color=wr_df['win_rate'],
                        colorscale='RdYlGn',
                        cmin=0,
                        cmax=100
                    )
                )
            ])
            
            fig.update_layout(
                title="Win Rate by Market Regime",
                xaxis_title="Market Regime",
                yaxis_title="Win Rate (%)",
                height=400,
                template="plotly_dark"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("No trades completed yet")
    
    else:
        st.info("👆 Run a backtest first to see performance metrics")

# =============================================================================
# TAB 4: ADVANCED VISUALIZATIONS
# =============================================================================
with tab4:
    st.header("🎨 Advanced Visualizations")
    
    if 'backtest_results' in st.session_state:
        results = st.session_state['backtest_results']
        close_trades = [t for t in results['trades'] if t['action'] == 'close']
        
        if close_trades:
            trade_df = pd.DataFrame(close_trades)
            
            # 3D scatter plot: Price x Holding Period x P&L
            st.subheader("🌐 3D Performance Space")
            
            fig = go.Figure(data=[go.Scatter3d(
                x=trade_df['price'],
                y=trade_df['holding_period'],
                z=trade_df['pnl'],
                mode='markers',
                marker=dict(
                    size=abs(trade_df['pnl']) / 10,
                    color=trade_df['regime'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Regime"),
                    line=dict(width=0.5, color='white')
                ),
                text=[f"Regime: {r}<br>P&L: ${p:.2f}<br>Period: {h}" 
                      for r, p, h in zip(trade_df['regime'], trade_df['pnl'], trade_df['holding_period'])],
                hoverinfo='text'
            )])
            
            fig.update_layout(
                title="3D Trade Performance",
                scene=dict(
                    xaxis_title="Entry Price",
                    yaxis_title="Holding Period (bars)",
                    zaxis_title="P&L ($)",
                    bgcolor='rgb(10,10,10)'
                ),
                height=700,
                template="plotly_dark"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Parameter sensitivity heatmap
            st.subheader("🔥 Parameter Sensitivity Heatmap")
            
            st.info("💡 This would show how different parameter combinations perform in each regime")
            
            # Simulated sensitivity data
            regimes = ["Bear", "Sideways", "Bull"]
            params = ["Entry Threshold", "Exit Threshold", "Position Size", "Stop Loss"]
            
            sensitivity_data = np.random.randn(len(regimes), len(params)) * 0.5 + 0.5
            
            fig = go.Figure(data=go.Heatmap(
                z=sensitivity_data,
                x=params,
                y=regimes,
                colorscale='RdYlGn',
                text=sensitivity_data,
                texttemplate='%{text:.2f}',
                textfont={"size": 12},
                colorbar=dict(title="Sharpe Ratio")
            ))
            
            fig.update_layout(
                title="Parameter Sensitivity by Regime",
                height=400,
                template="plotly_dark"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Regime transition flow (Sankey diagram)
            st.subheader("🌊 Regime Transition Flow")
            
            regime_history = results['regime_history']
            
            # Count transitions
            transitions = {}
            for i in range(len(regime_history) - 1):
                from_regime = regime_history[i]
                to_regime = regime_history[i + 1]
                key = (from_regime, to_regime)
                transitions[key] = transitions.get(key, 0) + 1
            
            # Create Sankey
            regime_names = ["Bear", "Sideways", "Bull"]
            
            source = []
            target = []
            value = []
            
            for (from_r, to_r), count in transitions.items():
                if from_r != to_r:  # Exclude self-loops
                    source.append(from_r)
                    target.append(to_r + n_regimes)  # Offset target
                    value.append(count)
            
            fig = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=[f"{regime_names[i]} (t)" for i in range(n_regimes)] + 
                          [f"{regime_names[i]} (t+1)" for i in range(n_regimes)],
                    color=["red", "yellow", "green"] * 2
                ),
                link=dict(
                    source=source,
                    target=target,
                    value=value
                )
            )])
            
            fig.update_layout(
                title="Regime Transition Flow",
                height=400,
                template="plotly_dark"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Correlation matrix heatmap
            st.subheader("📊 Trade Metrics Correlation")
            
            numeric_cols = trade_df[['pnl', 'pnl_pct', 'holding_period', 'price']].select_dtypes(include=[np.number])
            corr_df = numeric_cols.corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_df.values,
                x=corr_df.columns,
                y=corr_df.columns,
                colorscale='RdBu',
                zmid=0,
                text=corr_df.values,
                texttemplate='%{text:.2f}',
                textfont={"size": 12},
                colorbar=dict(title="Correlation")
            ))
            
            fig.update_layout(
                title="Trade Metrics Correlation Matrix",
                height=500,
                template="plotly_dark"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("No trades completed yet")
    
    else:
        st.info("👆 Run a backtest first to see advanced visualizations")

# Footer
st.markdown("---")
st.markdown("**💡 Tip:** Adjust HMM parameters in the sidebar to see how regime detection affects strategy performance")
