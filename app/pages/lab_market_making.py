"""
Market Making Lab - Liquidity Provision and Spread Capture Strategies
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.ui_components import render_sidebar_navigation, apply_custom_css

# Page configuration
st.set_page_config(
    page_title="Market Making Lab",
    page_icon="🌊",
    layout="wide"
)

# Apply custom styling and navigation
apply_custom_css()
render_sidebar_navigation(current_page="Market Making Lab")

# Initialize session state
if 'historical_data' not in st.session_state:
    st.session_state.historical_data = None
if 'theme_mode' not in st.session_state:
    st.session_state.theme_mode = 'dark'
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {'positions': {}, 'cash': 100000.0}
if 'mm_inventory' not in st.session_state:
    st.session_state.mm_inventory = 0

# Header
st.markdown('<h1 class="lab-header">🌊 Market Making Lab</h1>', unsafe_allow_html=True)
st.markdown("**Liquidity provision with bid-ask spread capture and inventory management**")

# Check if data is loaded
if st.session_state.historical_data is None:
    st.warning("⚠️ No historical data loaded. Please load data first.")
    if st.button("💾 Go to Data Loader"):
        st.switch_page("pages/data_loader.py")
    st.stop()

data = st.session_state.historical_data

# Ensure data is a DataFrame
if not isinstance(data, pd.DataFrame):
    st.error("❌ Invalid data format. Please reload data from Data Loader.")
    if st.button("💾 Go to Data Loader"):
        st.switch_page("pages/data_loader.py")
    st.stop()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Spread Analysis",
    "⚖️ Inventory Management",
    "🎯 Quote Optimization",
    "📈 P&L Simulation"
])

with tab1:
    st.markdown("### Bid-Ask Spread Analysis")
    
    # Prepare data
    if 'symbol' in data.columns:
        available_symbols = data['symbol'].unique().tolist()
        selected_symbol = st.selectbox("Select Asset", available_symbols)
        symbol_data = data[data['symbol'] == selected_symbol].copy()
        symbol_data = symbol_data.set_index('timestamp')
    else:
        available_symbols = [col for col in data.columns if col not in ['timestamp', 'date', 'Date']]
        selected_symbol = st.selectbox("Select Asset", available_symbols)
        symbol_data = pd.DataFrame(data[selected_symbol])
        symbol_data.columns = ['close']
    
    # Extract or estimate bid/ask
    if 'bid' in symbol_data.columns and 'ask' in symbol_data.columns:
        mid_price = (symbol_data['bid'] + symbol_data['ask']) / 2
        spread = symbol_data['ask'] - symbol_data['bid']
    else:
        # Estimate from close and volatility
        mid_price = symbol_data['close'].apply(pd.to_numeric, errors='coerce')
        volatility = mid_price.pct_change().rolling(20).std()
        spread = mid_price * volatility * 0.1  # Estimated spread
    
    spread_bps = (spread / mid_price * 10000).fillna(0)
    
    st.info(f"📊 Analyzing {selected_symbol} - {len(mid_price)} data points")
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_spread = spread_bps.mean()
        st.metric("Avg Spread (bps)", f"{avg_spread:.2f}")
    
    with col2:
        current_spread = spread_bps.iloc[-1]
        st.metric("Current Spread (bps)", f"{current_spread:.2f}")
    
    with col3:
        spread_vol = spread_bps.std()
        st.metric("Spread Volatility", f"{spread_vol:.2f}")
    
    with col4:
        current_mid = mid_price.iloc[-1]
        st.metric("Mid Price", f"${current_mid:.2f}")
    
    # Spread time series
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.6, 0.4],
        subplot_titles=('Mid Price', 'Spread (bps)'),
        vertical_spacing=0.1
    )
    
    fig.add_trace(go.Scatter(
        x=mid_price.index, y=mid_price,
        name='Mid Price', line=dict(color='#667eea', width=2)
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=spread_bps.index, y=spread_bps,
        name='Spread', line=dict(color='#764ba2', width=1),
        fill='tozeroy'
    ), row=2, col=1)
    
    fig.update_layout(
        height=600,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Spread distribution
    st.markdown("#### Spread Distribution")
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=spread_bps.dropna(),
        nbinsx=50,
        name='Spread Distribution',
        marker_color='#667eea'
    ))
    
    fig.update_layout(
        title='Histogram of Spreads (bps)',
        xaxis_title='Spread (bps)',
        yaxis_title='Frequency',
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("### Inventory Management")
    
    st.markdown("""
    Market makers must manage inventory risk from accumulating positions.
    Key strategies:
    - **Symmetric quotes**: Equal distance from mid when flat
    - **Inventory skew**: Widen far side, tighten near side to mean-revert
    - **Position limits**: Max inventory constraints
    """)
    
    # Inventory settings
    col1, col2 = st.columns(2)
    
    with col1:
        max_inventory = st.slider("Max Inventory (units)", 10, 1000, 100)
    
    with col2:
        inventory_penalty = st.slider("Inventory Penalty Factor", 0.0, 1.0, 0.3, 0.05)
    
    # Simulate inventory evolution
    st.markdown("#### Inventory Simulation")
    
    if st.button("🎲 Run Simulation", type="primary"):
        with st.spinner("Simulating market making..."):
            # Initialize
            inventory = np.zeros(len(mid_price))
            cash = 0
            
            # Estimate arrival rate (trades per period)
            volatility = mid_price.pct_change().std()
            base_arrival_rate = 0.5  # Base probability of trade
            
            for i in range(1, len(mid_price)):
                current_inv = inventory[i-1]
                price = mid_price.iloc[i]
                
                # Adjust quotes based on inventory
                inv_skew = (current_inv / max_inventory) * inventory_penalty
                
                # Bid/ask spread (bps)
                base_spread = spread_bps.iloc[i] / 2  # Half spread
                
                bid_spread = base_spread * (1 + inv_skew)
                ask_spread = base_spread * (1 - inv_skew)
                
                bid_price = price * (1 - bid_spread / 10000)
                ask_price = price * (1 + ask_spread / 10000)
                
                # Simulate order arrivals
                bid_hit = np.random.random() < base_arrival_rate * (1 + inv_skew/2)
                ask_hit = np.random.random() < base_arrival_rate * (1 - inv_skew/2)
                
                # Update inventory
                if bid_hit and current_inv < max_inventory:
                    inventory[i] = current_inv + 1
                    cash -= bid_price
                elif ask_hit and current_inv > -max_inventory:
                    inventory[i] = current_inv - 1
                    cash += ask_price
                else:
                    inventory[i] = current_inv
            
            # Store results
            st.session_state.mm_inventory_series = inventory
            st.session_state.mm_cash = cash
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                final_inv = inventory[-1]
                st.metric("Final Inventory", f"{final_inv:.0f} units")
            
            with col2:
                final_value = cash + final_inv * mid_price.iloc[-1]
                st.metric("Total P&L", f"${final_value:.2f}")
            
            with col3:
                max_inv = np.abs(inventory).max()
                st.metric("Peak Inventory", f"{max_inv:.0f} units")
            
            # Plot inventory
            fig = make_subplots(
                rows=2, cols=1,
                row_heights=[0.5, 0.5],
                subplot_titles=('Mid Price', 'Inventory'),
                vertical_spacing=0.1
            )
            
            fig.add_trace(go.Scatter(
                x=mid_price.index, y=mid_price,
                name='Mid Price', line=dict(color='#667eea', width=2)
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=mid_price.index, y=inventory,
                name='Inventory', line=dict(color='#764ba2', width=2),
                fill='tozeroy'
            ), row=2, col=1)
            
            # Add inventory limits
            fig.add_hline(y=max_inventory, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=-max_inventory, line_dash="dash", line_color="red", row=2, col=1)
            
            fig.update_layout(height=600, hovermode='x unified')
            
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("### Quote Optimization")
    
    st.markdown("""
    Optimal quote placement balances:
    - **Adverse selection risk**: Price moving against you after fill
    - **Fill probability**: Tighter quotes = more fills
    - **Spread capture**: Wider quotes = more profit per fill
    """)
    
    # Avellaneda-Stoikov parameters
    st.markdown("#### Avellaneda-Stoikov Model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        risk_aversion = st.slider("Risk Aversion (γ)", 0.01, 1.0, 0.1, 0.01)
    
    with col2:
        order_arrival_rate = st.slider("Order Arrival Rate (λ)", 0.1, 10.0, 1.0, 0.1)
    
    # Calculate optimal spread
    sigma = mid_price.pct_change().std() * np.sqrt(252)  # Annual volatility
    
    # Avellaneda-Stoikov optimal spread
    # δ_bid + δ_ask = γ * σ² * (T - t) + (2/γ) * ln(1 + γ/λ)
    time_to_end = 1.0  # Assume 1 year horizon
    
    optimal_spread = risk_aversion * sigma**2 * time_to_end + (2/risk_aversion) * np.log(1 + risk_aversion/order_arrival_rate)
    optimal_spread_bps = optimal_spread * 10000
    
    st.info(f"📊 Optimal Half-Spread: **{optimal_spread_bps:.2f} bps**")
    
    # Visualize quote placement
    current_price = mid_price.iloc[-1]
    optimal_bid = current_price * (1 - optimal_spread)
    optimal_ask = current_price * (1 + optimal_spread)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Optimal Bid", f"${optimal_bid:.2f}")
    
    with col2:
        st.metric("Mid Price", f"${current_price:.2f}")
    
    with col3:
        st.metric("Optimal Ask", f"${optimal_ask:.2f}")
    
    # Risk-return tradeoff
    st.markdown("#### Risk-Return Tradeoff")
    
    gamma_range = np.linspace(0.01, 1.0, 50)
    spreads = []
    
    for g in gamma_range:
        spread = g * sigma**2 * time_to_end + (2/g) * np.log(1 + g/order_arrival_rate)
        spreads.append(spread * 10000)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=gamma_range,
        y=spreads,
        mode='lines',
        name='Optimal Spread',
        line=dict(color='#667eea', width=2)
    ))
    
    fig.add_vline(x=risk_aversion, line_dash="dash", line_color="red", 
                  annotation_text="Current γ")
    
    fig.update_layout(
        title='Optimal Spread vs Risk Aversion',
        xaxis_title='Risk Aversion (γ)',
        yaxis_title='Optimal Half-Spread (bps)',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.markdown("### P&L Simulation")
    
    st.markdown("#### Strategy Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        spread_multiplier = st.slider("Spread Multiplier", 0.5, 3.0, 1.0, 0.1)
    
    with col2:
        trade_size = st.slider("Trade Size (units)", 1, 100, 10)
    
    with col3:
        trading_fee_bps = st.slider("Trading Fee (bps)", 0, 20, 5)
    
    if st.button("📊 Simulate P&L", type="primary"):
        with st.spinner("Simulating market making strategy..."):
            # Calculate metrics
            returns = mid_price.pct_change().dropna()
            
            # Estimate spread capture
            avg_spread = spread_bps.mean() * spread_multiplier
            fee_cost = trading_fee_bps * 2  # Both sides
            net_spread = avg_spread - fee_cost
            
            # Simulate trades (assume both sides hit proportionally to time)
            n_periods = len(returns)
            trade_frequency = 0.1  # 10% of periods have trades
            n_trades = int(n_periods * trade_frequency)
            
            # P&L components
            spread_pnl = n_trades * trade_size * mid_price.mean() * (net_spread / 10000)
            
            # Inventory risk (assume some directional exposure)
            avg_inventory = trade_size * 0.3  # Average 30% of trade size
            inventory_pnl = avg_inventory * (mid_price.iloc[-1] - mid_price.iloc[0])
            
            total_pnl = spread_pnl + inventory_pnl
            
            # Display results
            st.markdown("#### Simulated Performance")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Spread P&L", f"${spread_pnl:.2f}")
            
            with col2:
                st.metric("Inventory P&L", f"${inventory_pnl:.2f}")
            
            with col3:
                st.metric("Total P&L", f"${total_pnl:.2f}")
            
            with col4:
                roi = (total_pnl / (trade_size * mid_price.mean())) * 100
                st.metric("ROI", f"{roi:.2f}%")
            
            # Simulate daily P&L
            daily_spread_pnl = np.random.normal(
                spread_pnl / n_periods,
                spread_pnl / n_periods * 0.3,
                n_periods
            )
            
            daily_inventory_pnl = returns * avg_inventory * mid_price
            
            total_daily_pnl = daily_spread_pnl + daily_inventory_pnl
            cumulative_pnl = total_daily_pnl.cumsum()
            
            # Plot P&L
            fig = make_subplots(
                rows=2, cols=1,
                row_heights=[0.6, 0.4],
                subplot_titles=('Cumulative P&L', 'Daily P&L'),
                vertical_spacing=0.1
            )
            
            fig.add_trace(go.Scatter(
                x=cumulative_pnl.index,
                y=cumulative_pnl.values,
                name='Cumulative P&L',
                line=dict(color='#667eea', width=2),
                fill='tozeroy'
            ), row=1, col=1)
            
            colors = ['green' if x > 0 else 'red' for x in total_daily_pnl]
            fig.add_trace(go.Bar(
                x=total_daily_pnl.index,
                y=total_daily_pnl.values,
                name='Daily P&L',
                marker_color=colors
            ), row=2, col=1)
            
            fig.update_layout(
                height=600,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            st.markdown("#### Performance Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                sharpe = total_daily_pnl.mean() / total_daily_pnl.std() * np.sqrt(252) if total_daily_pnl.std() > 0 else 0
                st.metric("Sharpe Ratio", f"{sharpe:.2f}")
            
            with col2:
                win_rate = (total_daily_pnl > 0).sum() / len(total_daily_pnl) * 100
                st.metric("Win Rate", f"{win_rate:.1f}%")
            
            with col3:
                max_dd = (cumulative_pnl - cumulative_pnl.expanding().max()).min()
                st.metric("Max Drawdown", f"${max_dd:.2f}")
            
            with col4:
                profit_factor = total_daily_pnl[total_daily_pnl > 0].sum() / abs(total_daily_pnl[total_daily_pnl < 0].sum())
                st.metric("Profit Factor", f"{profit_factor:.2f}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888;'>
    <p>🌊 Market making provides liquidity while capturing bid-ask spreads and managing inventory risk</p>
</div>
""", unsafe_allow_html=True)
