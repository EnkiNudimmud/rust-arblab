"""
Portfolio View Module
=====================

Track and manage virtual portfolio:
- Portfolio constituents and allocation
- Value evolution over time
- P&L tracking
- Performance analytics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from python.rust_bridge import get_connector

def render():
    """Render the portfolio view page"""
    # Initialize session state
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = {
            'positions': {},
            'cash': 100000.0,
            'initial_capital': 100000.0,
            'history': []
        }
    if 'historical_data' not in st.session_state:
        st.session_state.historical_data = None
    
    st.title("💼 Portfolio View")
    st.markdown("Monitor your virtual portfolio and track performance")
    
    # Portfolio overview
    portfolio = st.session_state.portfolio
    
    # Top metrics
    display_portfolio_metrics()
    
    st.markdown("---")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Holdings",
        "📈 Performance",
        "💰 P&L Analysis",
        "🎯 Allocation"
    ])
    
    with tab1:
        display_holdings()
    
    with tab2:
        display_performance()
    
    with tab3:
        display_pnl_analysis()
    
    with tab4:
        display_allocation()
    
    # Sidebar - portfolio management
    with st.sidebar:
        st.markdown("### Portfolio Management")
        manage_portfolio()

def display_portfolio_metrics():
    """Display top-level portfolio metrics"""
    
    portfolio = st.session_state.portfolio
    
    # Calculate current portfolio value
    total_value = portfolio['cash']
    positions_value = 0.0
    
    # Get current prices for positions
    if len(portfolio['positions']) > 0:
        positions_value = calculate_positions_value(portfolio['positions'])
        total_value += positions_value
    
    # Calculate P&L
    initial_capital = 100000.0  # Default
    if len(portfolio['history']) > 0:
        initial_capital = portfolio['history'][0].get('total_value', 100000.0)
    
    total_pnl = total_value - initial_capital
    total_pnl_pct = (total_pnl / initial_capital) * 100 if initial_capital > 0 else 0.0
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Value",
            f"${total_value:,.2f}",
            delta=f"{total_pnl_pct:+.2f}%"
        )
    
    with col2:
        st.metric(
            "Cash",
            f"${portfolio['cash']:,.2f}",
            delta=f"{(portfolio['cash']/total_value)*100:.1f}% of total"
        )
    
    with col3:
        st.metric(
            "Positions Value",
            f"${positions_value:,.2f}",
            delta=f"{(positions_value/total_value)*100:.1f}% of total"
        )
    
    with col4:
        n_positions = len(portfolio['positions'])
        st.metric(
            "Open Positions",
            n_positions
        )

def calculate_positions_value(positions: Dict) -> float:
    """Calculate total value of all positions"""
    
    total = 0.0
    
    for symbol, position in positions.items():
        try:
            # Try to get current price
            current_price = get_current_price(symbol)
            if current_price > 0:
                qty = position.get('quantity', 0)
                total += qty * current_price
            else:
                # Fallback to average price
                qty = position.get('quantity', 0)
                avg_price = position.get('avg_price', 0)
                total += qty * avg_price
        except Exception:
            # Use average price as fallback
            qty = position.get('quantity', 0)
            avg_price = position.get('avg_price', 0)
            total += qty * avg_price
    
    return total

def get_current_price(symbol: str) -> float:
    """Get current price for a symbol"""
    
    # Try to get from live data buffer
    if st.session_state.get('live_trading_active', False):
        buffer = st.session_state.get('live_data_buffer', [])
        symbol_data = [d for d in buffer if d['symbol'] == symbol]
        if len(symbol_data) > 0:
            return symbol_data[-1]['mid']
    
    # Try to fetch from connector
    try:
        connector_name = st.session_state.get('data_source', 'finnhub')
        connector = get_connector(connector_name)
        
        if hasattr(connector, 'fetch_orderbook_sync'):
            ob = connector.fetch_orderbook_sync(symbol)
            if isinstance(ob, dict) and ob.get('bids') and ob.get('asks'):
                bid = float(ob['bids'][0][0])
                ask = float(ob['asks'][0][0])
                return (bid + ask) / 2
    except Exception:
        pass
    
    return 0.0

def display_holdings():
    """Display current holdings"""
    
    portfolio = st.session_state.portfolio
    positions = portfolio['positions']
    
    if len(positions) == 0:
        st.info("No open positions")
        
        st.markdown("""
        ### How to add positions:
        
        1. **From Backtesting**: Run a backtest on the Strategy Backtest page
        2. **From Live Trading**: Execute trades on the Live Trading page
        3. **Manual Entry**: Use the sidebar to manually add positions
        """)
        return
    
    # Build positions DataFrame
    positions_data = []
    
    for symbol, position in positions.items():
        qty = position.get('quantity', 0)
        avg_price = position.get('avg_price', 0)
        entry_date = position.get('entry_date', datetime.now())
        
        current_price = get_current_price(symbol)
        if current_price == 0:
            current_price = avg_price
        
        market_value = qty * current_price
        cost_basis = qty * avg_price
        pnl = market_value - cost_basis
        pnl_pct = (pnl / cost_basis * 100) if cost_basis > 0 else 0
        
        positions_data.append({
            'Symbol': symbol,
            'Quantity': qty,
            'Avg Price': avg_price,
            'Current Price': current_price,
            'Market Value': market_value,
            'Cost Basis': cost_basis,
            'P&L': pnl,
            'P&L %': pnl_pct,
            'Entry Date': entry_date
        })
    
    df = pd.DataFrame(positions_data)
    
    # Format and display
    st.dataframe(
        df.style.format({
            'Quantity': '{:.4f}',
            'Avg Price': '${:.2f}',
            'Current Price': '${:.2f}',
            'Market Value': '${:,.2f}',
            'Cost Basis': '${:,.2f}',
            'P&L': '${:,.2f}',
            'P&L %': '{:+.2f}%'
        }).background_gradient(subset=['P&L %'], cmap='RdYlGn', vmin=-10, vmax=10),
        use_container_width=True,
        height=400
    )
    
    # Summary statistics
    st.markdown("### Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_market_value = df['Market Value'].sum()
        st.metric("Total Market Value", f"${total_market_value:,.2f}")
    
    with col2:
        total_cost = df['Cost Basis'].sum()
        st.metric("Total Cost Basis", f"${total_cost:,.2f}")
    
    with col3:
        total_pnl = df['P&L'].sum()
        st.metric("Total P&L", f"${total_pnl:,.2f}")
    
    with col4:
        avg_pnl_pct = df['P&L %'].mean()
        st.metric("Avg P&L %", f"{avg_pnl_pct:+.2f}%")

def display_performance():
    """Display portfolio performance over time"""
    
    portfolio = st.session_state.portfolio
    history = portfolio.get('history', [])
    
    if len(history) < 2:
        st.info("Not enough history to display performance. Portfolio value is tracked over time as you trade.")
        
        # Show current state
        st.markdown("### Current Portfolio Snapshot")
        
        total_value = portfolio['cash'] + calculate_positions_value(portfolio['positions'])
        
        fig = go.Figure()
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=total_value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Portfolio Value"},
            delta={'reference': 100000},
            gauge={
                'axis': {'range': [None, 150000]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 100000], 'color': "lightgray"},
                    {'range': [100000, 150000], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 100000
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        return
    
    # Convert history to DataFrame
    df = pd.DataFrame(history)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # Calculate returns
    df['returns'] = df['total_value'].pct_change()
    df['cum_returns'] = (1 + df['returns']).cumprod() - 1
    
    # Calculate drawdown
    running_max = df['total_value'].expanding().max()
    df['drawdown'] = (df['total_value'] - running_max) / running_max
    
    # Create figure with subplots
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=("Portfolio Value", "Cumulative Returns", "Drawdown")
    )
    
    # Portfolio value
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['total_value'],
            name='Value',
            line={'color': 'cyan', 'width': 2},
            fill='tozeroy'
        ),
        row=1, col=1
    )
    
    # Cumulative returns
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['cum_returns'] * 100,
            name='Returns',
            line={'color': 'green', 'width': 2}
        ),
        row=2, col=1
    )
    
    # Drawdown
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['drawdown'] * 100,
            name='Drawdown',
            fill='tozeroy',
            line={'color': 'red', 'width': 1}
        ),
        row=3, col=1
    )
    
    fig.update_layout(
        height=700,
        template="plotly_dark",
        showlegend=False,
        hovermode='x unified'
    )
    
    fig.update_yaxes(title_text="Value ($)", row=1, col=1)
    fig.update_yaxes(title_text="Return (%)", row=2, col=1)
    fig.update_yaxes(title_text="DD (%)", row=3, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance metrics
    st.markdown("### Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_return = df['cum_returns'].iloc[-1] * 100
        st.metric("Total Return", f"{total_return:+.2f}%")
    
    with col2:
        sharpe = calculate_sharpe_ratio(df['returns'])
        st.metric("Sharpe Ratio", f"{sharpe:.3f}")
    
    with col3:
        max_dd = df['drawdown'].min() * 100
        st.metric("Max Drawdown", f"{max_dd:.2f}%")
    
    with col4:
        volatility = df['returns'].std() * np.sqrt(252) * 100  # Annualized
        st.metric("Volatility (Ann.)", f"{volatility:.2f}%")

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe ratio"""
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    
    # Annualize
    mean_return = returns.mean() * 252
    std_return = returns.std() * np.sqrt(252)
    
    sharpe = (mean_return - risk_free_rate) / std_return
    return sharpe

def display_pnl_analysis():
    """Display P&L analysis"""
    
    portfolio = st.session_state.portfolio
    trades = portfolio.get('trades', [])
    
    if len(trades) == 0:
        st.info("No trades executed yet. P&L analysis will appear after trading activity.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(trades)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # Calculate P&L per trade
    if 'pnl' in df.columns:
        # P&L over time
        df['cum_pnl'] = df['pnl'].cumsum()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['cum_pnl'],
            mode='lines+markers',
            name='Cumulative P&L',
            line={'color': 'gold', 'width': 2}
        ))
        
        fig.update_layout(
            title="Cumulative P&L Over Time",
            xaxis_title="Date",
            yaxis_title="P&L ($)",
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # P&L distribution
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=df['pnl'],
                nbinsx=30,
                marker_color='steelblue'
            ))
            fig.update_layout(
                title="P&L Distribution",
                xaxis_title="P&L ($)",
                yaxis_title="Frequency",
                template="plotly_dark",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Win rate analysis
            wins = (df['pnl'] > 0).sum()
            losses = (df['pnl'] < 0).sum()
            
            fig = go.Figure()
            fig.add_trace(go.Pie(
                labels=['Wins', 'Losses', 'Breakeven'],
                values=[wins, losses, len(df) - wins - losses],
                marker_colors=['green', 'red', 'gray']
            ))
            fig.update_layout(
                title="Win/Loss Ratio",
                template="plotly_dark",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        st.markdown("### P&L Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_win = df[df['pnl'] > 0]['pnl'].mean() if wins > 0 else 0
            st.metric("Avg Win", f"${avg_win:.2f}")
        
        with col2:
            avg_loss = df[df['pnl'] < 0]['pnl'].mean() if losses > 0 else 0
            st.metric("Avg Loss", f"${avg_loss:.2f}")
        
        with col3:
            win_rate = (wins / len(df) * 100) if len(df) > 0 else 0
            st.metric("Win Rate", f"{win_rate:.1f}%")
        
        with col4:
            total_pnl = df['pnl'].sum()
            st.metric("Total P&L", f"${total_pnl:,.2f}")

def display_allocation():
    """Display portfolio allocation"""
    
    portfolio = st.session_state.portfolio
    positions = portfolio['positions']
    
    if len(positions) == 0:
        st.info("No positions to display allocation")
        return
    
    # Calculate allocation
    total_value = portfolio['cash'] + calculate_positions_value(positions)
    
    allocation_data = []
    
    # Cash allocation
    cash_pct = (portfolio['cash'] / total_value) * 100
    allocation_data.append({
        'Asset': 'Cash',
        'Value': portfolio['cash'],
        'Percentage': cash_pct
    })
    
    # Position allocations
    for symbol, position in positions.items():
        qty = position.get('quantity', 0)
        current_price = get_current_price(symbol)
        if current_price == 0:
            current_price = position.get('avg_price', 0)
        
        value = qty * current_price
        pct = (value / total_value) * 100
        
        allocation_data.append({
            'Asset': symbol,
            'Value': value,
            'Percentage': pct
        })
    
    df = pd.DataFrame(allocation_data)
    df = df.sort_values('Value', ascending=False)
    
    # Pie chart
    fig = go.Figure()
    
    fig.add_trace(go.Pie(
        labels=df['Asset'],
        values=df['Value'],
        textinfo='label+percent',
        marker=dict(line=dict(color='#000000', width=2))
    ))
    
    fig.update_layout(
        title="Portfolio Allocation",
        template="plotly_dark",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Table
    st.dataframe(
        df.style.format({
            'Value': '${:,.2f}',
            'Percentage': '{:.2f}%'
        }),
        use_container_width=True
    )

def manage_portfolio():
    """Portfolio management controls in sidebar"""
    
    st.markdown("#### Quick Actions")
    
    # Reset portfolio
    if st.button("🔄 Reset Portfolio", use_container_width=True):
        if st.session_state.get('confirm_reset', False):
            st.session_state.portfolio = {
                'cash': 100000.0,
                'positions': {},
                'history': [],
                'trades': []
            }
            st.session_state.confirm_reset = False
            st.success("Portfolio reset!")
            st.rerun()
        else:
            st.session_state.confirm_reset = True
            st.warning("Click again to confirm reset")
    
    # Add manual position
    with st.expander("➕ Add Manual Position"):
        symbol = st.text_input("Symbol", key="manual_symbol")
        quantity = st.number_input("Quantity", value=1.0, key="manual_qty")
        price = st.number_input("Price", value=100.0, key="manual_price")
        
        if st.button("Add Position", key="add_position_btn"):
            if symbol:
                st.session_state.portfolio['positions'][symbol] = {
                    'quantity': quantity,
                    'avg_price': price,
                    'entry_date': datetime.now()
                }
                st.success(f"Added {symbol} position")
                st.rerun()
    
    # Export portfolio
    if st.button("📥 Export Portfolio", use_container_width=True):
        portfolio_json = pd.Series(st.session_state.portfolio).to_json()
        st.download_button(
            "Download JSON",
            portfolio_json,
            file_name=f"portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

# Execute the render function when page is loaded
if __name__ == "__main__":
    render()
