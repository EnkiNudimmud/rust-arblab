"""
Portfolio Analytics Lab
Advanced portfolio optimization, risk metrics, and performance analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
import sys
from pathlib import Path
import json
import os
from datetime import datetime

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import shared UI components
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.ui_components import render_sidebar_navigation, apply_custom_css

st.set_page_config(page_title="Portfolio Analytics Lab", page_icon="📊", layout="wide")

# Portfolio persistence directory
PORTFOLIO_DIR = project_root / "data" / "portfolios"
PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)
LAST_PORTFOLIO_FILE = PORTFOLIO_DIR / "last_portfolio.json"

def save_portfolio_to_disk(portfolio_data, filename="last_portfolio.json"):
    """Save portfolio to disk for persistence"""
    try:
        filepath = PORTFOLIO_DIR / filename
        # Convert numpy arrays to lists for JSON serialization
        portfolio_copy = portfolio_data.copy()
        for symbol, pos in portfolio_copy.get('positions', {}).items():
            for key, value in pos.items():
                if isinstance(value, (np.integer, np.floating)):
                    pos[key] = float(value)
        
        with open(filepath, 'w') as f:
            json.dump(portfolio_copy, f, indent=2, default=str)
        return True
    except Exception as e:
        st.error(f"Error saving portfolio: {str(e)}")
        return False

def load_portfolio_from_disk(filename="last_portfolio.json"):
    """Load portfolio from disk"""
    try:
        filepath = PORTFOLIO_DIR / filename
        if filepath.exists():
            with open(filepath, 'r') as f:
                portfolio_data = json.load(f)
            return portfolio_data
        return None
    except Exception as e:
        st.error(f"Error loading portfolio: {str(e)}")
        return None

def list_saved_portfolios():
    """List all saved portfolios"""
    try:
        portfolios = []
        for file in PORTFOLIO_DIR.glob("*.json"):
            if file.name != "last_portfolio.json":
                try:
                    with open(file, 'r') as f:
                        data = json.load(f)
                    portfolios.append({
                        'filename': file.name,
                        'name': data.get('name', 'Unnamed'),
                        'created_at': data.get('created_at', 'Unknown'),
                        'initial_capital': data.get('initial_capital', 0)
                    })
                except:
                    pass
        return sorted(portfolios, key=lambda x: x.get('created_at', ''), reverse=True)
    except Exception as e:
        return []

# Initialize session state and load last portfolio
if 'portfolio' not in st.session_state:
    # Try to load last saved portfolio
    last_portfolio = load_portfolio_from_disk()
    if last_portfolio:
        st.session_state.portfolio = last_portfolio
    else:
        st.session_state.portfolio = {
            'positions': {},
            'cash': 100000.0,
            'initial_capital': 100000.0,
            'history': []
        }
if 'historical_data' not in st.session_state:
    st.session_state.historical_data = None

# Render sidebar navigation and apply CSS
render_sidebar_navigation(current_page="Portfolio Analytics Lab")
apply_custom_css()

st.markdown('<h1 class="lab-header">📊 Portfolio Analytics Lab</h1>', unsafe_allow_html=True)
st.markdown("### Advanced portfolio optimization and risk analysis")

# Show notification if portfolio was loaded from disk
if 'portfolio_loaded_notification' not in st.session_state:
    if 'name' in st.session_state.portfolio and st.session_state.portfolio.get('positions'):
        st.info(f"📂 Loaded saved portfolio: **{st.session_state.portfolio['name']}** (from data/portfolios/)")
    st.session_state['portfolio_loaded_notification'] = True

st.markdown("---")

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📊 Current Portfolio", "🎯 Optimization", "📉 Risk Analysis", "📈 Performance"])

with tab1:
    st.markdown("### Current Portfolio Status")
    
    portfolio = st.session_state.portfolio
    
    # Portfolio header with name and metadata
    if 'name' in portfolio:
        col_name, col_date = st.columns([3, 1])
        with col_name:
            st.markdown(f"#### 💼 {portfolio['name']}")
        with col_date:
            if 'created_at' in portfolio:
                st.caption(f"Created: {portfolio['created_at']}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_value = portfolio['cash']
    for symbol, pos in portfolio['positions'].items():
        total_value += pos['quantity'] * pos['avg_price']
    
    pnl = total_value - portfolio['initial_capital']
    pnl_pct = (pnl / portfolio['initial_capital']) * 100 if portfolio['initial_capital'] > 0 else 0
    
    with col1:
        st.markdown(f'<div class="metric-card"><strong>Total Value</strong><br/><span style="font-size: 1.5rem; color: #667eea;">${total_value:,.2f}</span></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><strong>Cash</strong><br/><span style="font-size: 1.5rem; color: #667eea;">${portfolio["cash"]:,.2f}</span></div>', unsafe_allow_html=True)
    with col3:
        color = "#10b981" if pnl >= 0 else "#ef4444"
        st.markdown(f'<div class="metric-card"><strong>P&L</strong><br/><span style="font-size: 1.5rem; color: {color};">${pnl:,.2f}</span></div>', unsafe_allow_html=True)
    with col4:
        color = "#10b981" if pnl_pct >= 0 else "#ef4444"
        st.markdown(f'<div class="metric-card"><strong>Return</strong><br/><span style="font-size: 1.5rem; color: {color};">{pnl_pct:+.2f}%</span></div>', unsafe_allow_html=True)
    
    # Display portfolio metrics if available
    if 'expected_return' in portfolio:
        st.markdown("#### Expected Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Expected Return", f"{portfolio['expected_return']:.2%}")
        with col2:
            st.metric("Volatility", f"{portfolio['volatility']:.2%}")
        with col3:
            st.metric("Sharpe Ratio", f"{portfolio['sharpe_ratio']:.2f}")
    
    # Calculate advanced risk metrics if historical data is available
    if st.session_state.historical_data is not None and portfolio['positions']:
        try:
            data = st.session_state.historical_data
            
            # Get portfolio returns
            portfolio_returns = None
            if 'optimization_results' in st.session_state:
                opt = st.session_state['optimization_results']
                prices = opt['prices']
                weights = opt['weights']
                returns = prices.pct_change().dropna()
                portfolio_returns = (returns * weights).sum(axis=1)
            elif len(portfolio['positions']) > 0:
                # Calculate from current positions
                assets = list(portfolio['positions'].keys())
                if isinstance(data.index, pd.MultiIndex):
                    prices_df = data.reset_index()
                    value_col = 'close' if 'close' in prices_df.columns else 'Close'
                    prices = prices_df.pivot(index='timestamp', columns='symbol', values=value_col)
                    prices = prices[assets]
                elif 'symbol' in data.columns:
                    value_col = 'close' if 'close' in data.columns else 'Close'
                    prices = data.pivot(index='timestamp', columns='symbol', values=value_col)
                    prices = prices[assets]
                else:
                    prices = data[assets]
                
                prices = prices.apply(pd.to_numeric, errors='coerce').dropna()
                returns = prices.pct_change().dropna()
                
                # Calculate weights from current positions
                weights = []
                for asset in assets:
                    weights.append(portfolio['positions'][asset].get('weight', 1.0 / len(assets)))
                weights = np.array(weights)
                
                portfolio_returns = (returns * weights).sum(axis=1)
            
            if portfolio_returns is not None and len(portfolio_returns) > 0:
                st.markdown("#### Advanced Risk Metrics")
                
                # Calculate benchmark (equal-weighted portfolio or market proxy)
                benchmark_returns = returns.mean(axis=1) if 'returns' in locals() else portfolio_returns
                
                # Information Ratio
                excess_returns = portfolio_returns - benchmark_returns
                tracking_error = excess_returns.std() * np.sqrt(252)
                information_ratio = (excess_returns.mean() * 252 / tracking_error) if tracking_error > 0 else 0.0
                
                # Sortino Ratio
                downside_returns = portfolio_returns[portfolio_returns < 0]
                downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0.0
                risk_free_rate = 0.04  # 4% annual
                sortino_ratio = ((portfolio_returns.mean() * 252 - risk_free_rate) / downside_std) if downside_std > 0 else 0.0
                
                # Calmar Ratio
                cumulative_returns = (1 + portfolio_returns).cumprod() - 1
                running_max = cumulative_returns.cummax()
                drawdown = cumulative_returns - running_max
                max_drawdown = drawdown.min()
                annualized_return = portfolio_returns.mean() * 252
                calmar_ratio = (annualized_return / abs(max_drawdown)) if max_drawdown != 0 else 0.0
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Information Ratio", f"{information_ratio:.2f}",
                             help="Alpha generation vs benchmark. Higher is better (>0.5 is good)")
                with col2:
                    st.metric("Sortino Ratio", f"{sortino_ratio:.2f}",
                             help="Downside risk-adjusted return. Higher is better (>2 is excellent)")
                with col3:
                    st.metric("Calmar Ratio", f"{calmar_ratio:.2f}",
                             help="Return vs maximum drawdown. Higher is better (>0.5 is good)")
        except Exception as e:
            st.warning(f"Could not calculate advanced metrics: {str(e)}")
    
    st.markdown("#### Holdings")
    if portfolio['positions']:
        holdings_data = []
        for symbol, pos in portfolio['positions'].items():
            weight_pct = pos.get('weight', 0) * 100
            holdings_data.append({
                'Symbol': symbol,
                'Quantity': f"{pos['quantity']:.4f}",
                'Avg Price': f"${pos['avg_price']:.2f}",
                'Current Value': f"${pos['quantity'] * pos['avg_price']:,.2f}",
                'Weight': f"{weight_pct:.2f}%"
            })
        st.dataframe(pd.DataFrame(holdings_data), use_container_width=True, hide_index=True)
        
        # Portfolio Visualization
        st.markdown("#### Portfolio Visualization")
        
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            # Enhanced bar chart showing allocation
            symbols = [h['Symbol'] for h in holdings_data]
            values = [float(h['Current Value'].replace('$', '').replace(',', '')) for h in holdings_data]
            weights = [float(h['Weight'].replace('%', '')) for h in holdings_data]
            
            # Sort by value descending
            sorted_indices = np.argsort(values)[::-1]
            symbols_sorted = [symbols[i] for i in sorted_indices]
            values_sorted = [values[i] for i in sorted_indices]
            weights_sorted = [weights[i] for i in sorted_indices]
            
            # Create color scale based on weight
            colors = ['#667eea' if w >= 10 else '#764ba2' if w >= 5 else '#a8c0ff' for w in weights_sorted]
            
            fig_bar = go.Figure(data=[
                go.Bar(
                    x=values_sorted,
                    y=symbols_sorted,
                    orientation='h',
                    marker=dict(
                        color=colors,
                        line=dict(color='white', width=1)
                    ),
                    text=[f"${v:,.0f} ({w:.1f}%)" for v, w in zip(values_sorted, weights_sorted)],
                    textposition='auto',
                    hovertemplate='<b>%{y}</b><br>Value: $%{x:,.2f}<extra></extra>'
                )
            ])
            
            fig_bar.update_layout(
                title="Position Size by Asset",
                xaxis_title="Position Value ($)",
                yaxis_title="",
                height=max(350, len(symbols) * 25),
                showlegend=False,
                margin=dict(l=80, r=20, t=40, b=40)
            )
            
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with viz_col2:
            # Treemap for hierarchical view
            fig_tree = go.Figure(go.Treemap(
                labels=symbols,
                parents=["Portfolio"] * len(symbols),
                values=values,
                text=[f"{s}<br>${v:,.0f}<br>{w:.1f}%" for s, v, w in zip(symbols, values, weights)],
                textposition="middle center",
                marker=dict(
                    colorscale='Viridis',
                    colorbar=dict(title="Weight (%)", thickness=15),
                    line=dict(width=2, color='white')
                ),
                hovertemplate='<b>%{label}</b><br>Value: $%{value:,.2f}<extra></extra>'
            ))
            
            fig_tree.update_layout(
                title="Portfolio Composition (Treemap)",
                height=max(350, len(symbols) * 25),
                margin=dict(l=0, r=0, t=40, b=0)
            )
            
            st.plotly_chart(fig_tree, use_container_width=True)
        
        # Weight distribution visualization
        st.markdown("##### Weight Distribution")
        
        # Create weight bands
        weight_bands = {
            'Large (>15%)': sum(1 for w in weights if w > 15),
            'Medium (5-15%)': sum(1 for w in weights if 5 <= w <= 15),
            'Small (<5%)': sum(1 for w in weights if w < 5)
        }
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Large Positions", weight_bands['Large (>15%)'], help="Positions with >15% allocation")
        with col2:
            st.metric("Medium Positions", weight_bands['Medium (5-15%)'], help="Positions with 5-15% allocation")
        with col3:
            st.metric("Small Positions", weight_bands['Small (<5%)'], help="Positions with <5% allocation")
        
        # Concentration metrics
        top_5_concentration = sum(sorted(weights, reverse=True)[:min(5, len(weights))])
        herfindahl_index = sum(w**2 for w in weights) / 100  # Normalized
        
        conc_col1, conc_col2 = st.columns(2)
        with conc_col1:
            st.metric("Top 5 Concentration", f"{top_5_concentration:.1f}%", 
                     help="Percentage of portfolio in top 5 positions")
        with conc_col2:
            st.metric("Herfindahl Index", f"{herfindahl_index:.2f}",
                     help="Portfolio concentration (0=equal weight, 100=single asset)")
    else:
        st.info("No positions currently held. Optimize a portfolio in the Optimization tab to get started!")
    
    # Portfolio Management Section
    st.markdown("---")
    st.markdown("#### 📂 Portfolio Management")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("##### Saved Portfolios")
        saved_portfolios = list_saved_portfolios()
        
        if saved_portfolios:
            # Create a display for saved portfolios
            portfolio_options = [f"{p['name']} ({p['created_at']}) - ${p['initial_capital']:,.0f}" 
                               for p in saved_portfolios]
            
            selected_idx = st.selectbox(
                "Load a saved portfolio:",
                range(len(portfolio_options)),
                format_func=lambda i: portfolio_options[i],
                key="portfolio_selector"
            )
            
            col_load, col_delete = st.columns(2)
            with col_load:
                if st.button("📥 Load Portfolio", type="primary", use_container_width=True):
                    loaded = load_portfolio_from_disk(saved_portfolios[selected_idx]['filename'])
                    if loaded:
                        st.session_state.portfolio = loaded
                        st.success(f"✅ Loaded portfolio: {loaded['name']}")
                        st.rerun()
            
            with col_delete:
                if st.button("🗑️ Delete Portfolio", type="secondary", use_container_width=True):
                    try:
                        filepath = PORTFOLIO_DIR / saved_portfolios[selected_idx]['filename']
                        filepath.unlink()
                        st.success("✅ Portfolio deleted")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error deleting portfolio: {str(e)}")
        else:
            st.info("No saved portfolios found. Create one in the Optimization tab!")
    
    with col2:
        st.markdown("##### Quick Actions")
        
        if portfolio['positions']:
            # Save current portfolio
            if st.button("💾 Save Current Portfolio", use_container_width=True):
                save_portfolio_to_disk(portfolio, "last_portfolio.json")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                portfolio_name = portfolio.get('name', 'Portfolio')
                safe_name = portfolio_name.replace(' ', '_').replace('/', '_')
                save_portfolio_to_disk(portfolio, f"{safe_name}_{timestamp}.json")
                st.success("✅ Portfolio saved!")
                st.rerun()
            
            # Clear portfolio
            if st.button("🗑️ Clear Current Portfolio", use_container_width=True):
                st.session_state.portfolio = {
                    'positions': {},
                    'cash': 100000.0,
                    'initial_capital': 100000.0,
                    'history': []
                }
                st.success("✅ Portfolio cleared")
                st.rerun()
        
        # Show storage location
        st.caption(f"📁 Storage: {PORTFOLIO_DIR}")
        st.caption(f"📊 Portfolios saved: {len(saved_portfolios)}")

with tab2:
    st.markdown("### Portfolio Optimization")
    
    if st.session_state.historical_data is None:
        st.warning("⚠️ Please load historical data first")
        if st.button("💾 Load Data"):
            st.switch_page("pages/data_loader.py")
    else:
        data = st.session_state.historical_data
        
        # Debug: Show data structure
        with st.expander("🔍 Data Info", expanded=False):
            st.write(f"Data shape: {data.shape}")
            st.write(f"Index type: {type(data.index)}")
            st.write(f"Columns: {list(data.columns)[:10]}")
            st.write(f"First few rows:")
            st.dataframe(data.head())
        
        # Extract available symbols based on data structure
        if isinstance(data.index, pd.MultiIndex):
            # Multi-index data (timestamp, symbol)
            available_symbols = data.index.get_level_values('symbol').unique().tolist() if 'symbol' in data.index.names else []
        elif 'symbol' in data.columns:
            # Long format: each row has timestamp, symbol, and prices
            available_symbols = data['symbol'].unique().tolist()
        else:
            # Wide format: columns are symbols
            available_symbols = [col for col in data.columns if col not in ['Date', 'date', 'timestamp', 'Datetime', 'datetime', 'time', 'Time']]
        
        # Filter out index symbols (common indices that shouldn't be in portfolios)
        index_symbols = {'^GSPC', '^DJI', '^IXIC', '^RUT', '^VIX', '^TNX', '^FTSE', '^N225', 
                        '^HSI', 'SPY', 'QQQ', 'DIA', 'IWM', 'VXX'}  # Common market indices and ETFs
        index_prefixes = ('^', '$')  # Symbols starting with ^ or $ are usually indices
        
        available_symbols = [
            symbol for symbol in available_symbols 
            if symbol not in index_symbols and not symbol.startswith(index_prefixes)
        ]
        
        if not available_symbols:
            st.error("❌ No tradeable asset columns found in data (indices excluded). Please check your data format.")
            st.stop()
        
        # Show helpful info about optimization
        if not st.session_state.portfolio or len(st.session_state.portfolio) == 0:
            st.info("💡 You can run portfolio optimization even without existing positions. "
                   "The optimizer will suggest optimal weights for selected assets.")
        
        st.markdown("#### Mean-Variance Optimization")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_assets = st.multiselect(
                "Select Assets for Portfolio",
                available_symbols,
                default=available_symbols[:5] if len(available_symbols) >= 5 else available_symbols
            )
        
        with col2:
            target_return = st.slider("Target Return (%/year)", 0.0, 50.0, 10.0, 1.0)
            risk_free_rate = st.number_input("Risk-free Rate (%)", value=4.0, step=0.5) / 100
        
        if len(selected_assets) >= 2:
            if st.button("🎯 Optimize Portfolio", type="primary"):
                with st.spinner("Computing optimal weights..."):
                    try:
                        # Calculate returns - ensure numeric data
                        if isinstance(data.index, pd.MultiIndex):
                            # Multi-index: pivot to wide format
                            prices_df = data.reset_index()
                            value_col = 'close' if 'close' in prices_df.columns else 'Close'
                            prices = prices_df.pivot(index='timestamp', columns='symbol', values=value_col)
                            prices = prices[selected_assets]
                        elif 'symbol' in data.columns:
                            # Long format: pivot to wide format
                            value_col = 'close' if 'close' in data.columns else 'Close'
                            prices = data.pivot(index='timestamp', columns='symbol', values=value_col)
                            prices = prices[selected_assets]
                        else:
                            # Already in wide format
                            prices = data[selected_assets]
                        
                        # Convert to numeric and clean
                        prices = prices.apply(pd.to_numeric, errors='coerce')
                        
                        # Drop rows with any NaN
                        prices = prices.dropna()
                        
                        # Check if we have enough data
                        if len(prices) < 2:
                            st.error("❌ Insufficient valid price data after cleaning. Need at least 2 data points.")
                            st.info(f"Data points available: {len(prices)}")
                            st.stop()
                        
                        if len(prices) < 2:
                            st.error("❌ Insufficient data after cleaning. Need at least 2 data points.")
                            st.stop()
                        
                        returns = prices.pct_change().dropna()
                        
                        # Calculate statistics
                        mean_returns = returns.mean() * 252  # Annualized
                        cov_matrix = returns.cov() * 252
                        
                        # Validate results
                        if mean_returns.isna().any() or np.isinf(mean_returns).any():
                            st.error("❌ Invalid returns calculated. Check your price data.")
                            st.stop()
                        
                        # Try Rust optimization if available
                        try:
                            import rust_connector
                            
                            # CARA utility optimization
                            risk_aversion = 2.0
                            result = rust_connector.cara_optimal_weights_rust(
                                mean_returns.values.tolist(),
                                cov_matrix.values.tolist(),
                                risk_aversion
                            )
                            
                            # Handle dict or list return type
                            if isinstance(result, dict):
                                weights = result.get('weights', list(result.values()))
                            else:
                                weights = result
                            
                            st.success("✅ Optimization completed using Rust backend")
                        except Exception as e:
                            # Fallback to equal weights
                            st.warning(f"Using equal weights (Rust optimization unavailable: {str(e)})")
                            weights = np.ones(len(selected_assets)) / len(selected_assets)
                        
                        # Display results
                        st.markdown("#### Optimal Weights")
                        
                        # Ensure weights are numpy array of floats
                        if isinstance(weights, dict):
                            weights = list(weights.values())
                        weights = np.array(weights, dtype=float)
                        
                        # Portfolio metrics - ensure numeric values
                        mean_returns_array = mean_returns.values if hasattr(mean_returns, 'values') else np.array(mean_returns, dtype=float)
                        cov_matrix_array = cov_matrix.values if hasattr(cov_matrix, 'values') else np.array(cov_matrix, dtype=float)
                        
                        port_return = float(np.dot(weights, mean_returns_array))
                        port_vol = float(np.sqrt(np.dot(weights, np.dot(cov_matrix_array, weights))))
                        sharpe = float((port_return - risk_free_rate) / port_vol) if port_vol > 0 else 0.0
                        
                        # Save to session state for use in other tabs and management
                        st.session_state['optimization_results'] = {
                            'weights': weights,
                            'assets': selected_assets,
                            'prices': prices,
                            'returns': returns,
                            'mean_returns': mean_returns,
                            'cov_matrix': cov_matrix,
                            'port_return': port_return,
                            'port_vol': port_vol,
                            'sharpe': sharpe,
                            'risk_free_rate': risk_free_rate
                        }
                        st.session_state['portfolio_weights'] = {
                            'weights': weights,
                            'assets': selected_assets
                        }
                        st.session_state['selected_assets'] = selected_assets
                        
                        weights_df = pd.DataFrame({
                            'Asset': selected_assets,
                            'Weight': [f"{float(w):.2%}" for w in weights]
                        })
                        st.dataframe(weights_df, use_container_width=True, hide_index=True)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Expected Return", f"{port_return:.2%}")
                        with col2:
                            st.metric("Volatility", f"{port_vol:.2%}")
                        with col3:
                            st.metric("Sharpe Ratio", f"{sharpe:.2f}")
                        
                        # Efficient frontier plot
                        st.markdown("#### Efficient Frontier")
                        
                        n_portfolios = 1000
                        results = np.zeros((3, n_portfolios))
                        
                        for i in range(n_portfolios):
                            w = np.random.random(len(selected_assets))
                            w /= w.sum()
                            
                            results[0, i] = np.dot(w, mean_returns)
                            results[1, i] = np.sqrt(np.dot(w, np.dot(cov_matrix, w)))
                            results[2, i] = (results[0, i] - risk_free_rate) / results[1, i]
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=results[1], y=results[0],
                            mode='markers',
                            marker=dict(
                                color=results[2],
                                colorscale='Viridis',
                                showscale=True,
                                colorbar=dict(title="Sharpe Ratio"),
                                size=5
                            ),
                            name='Random Portfolios'
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=[port_vol], y=[port_return],
                            mode='markers',
                            marker=dict(color='red', size=15, symbol='star'),
                            name='Optimal Portfolio'
                        ))
                        
                        fig.update_layout(
                            title='Efficient Frontier',
                            xaxis_title='Volatility (σ)',
                            yaxis_title='Expected Return',
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.success("✅ Optimization complete! Use the sections below to save, backtest, or export your portfolio.")
                        
                        with col1:
                            portfolio_name = st.text_input("Portfolio Name", value="Optimized Portfolio", key="portfolio_name_input")
                        
                        with col2:
                            initial_capital = st.number_input("Initial Capital ($)", value=100000.0, min_value=1000.0, step=1000.0, key="initial_capital_input")
                        
                        with col3:
                            st.markdown("<br/>", unsafe_allow_html=True)
                            if st.button("💾 Save as Current Portfolio", type="primary", use_container_width=True):
                                # Calculate positions based on weights and capital
                                positions = {}
                                for i, asset in enumerate(selected_assets):
                                    weight = float(weights[i])
                                    allocation = initial_capital * weight
                                    
                                    # Get last price for the asset
                                    if isinstance(prices, pd.DataFrame) and asset in prices.columns:
                                        last_price = float(prices[asset].iloc[-1])
                                    else:
                                        last_price = 100.0  # Default fallback
                                    
                                    quantity = allocation / last_price if last_price > 0 else 0
                                    
                                    positions[asset] = {
                                        'quantity': quantity,
                                        'avg_price': last_price,
                                        'weight': weight
                                    }
                                
                                # Update portfolio in session state
                                st.session_state.portfolio = {
                                    'name': portfolio_name,
                                    'positions': positions,
                                    'cash': 0.0,  # Fully invested
                                    'initial_capital': initial_capital,
                                    'history': [],
                                    'created_at': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    'expected_return': port_return,
                                    'volatility': port_vol,
                                    'sharpe_ratio': sharpe
                                }
                                
                                st.success(f"✅ Portfolio '{portfolio_name}' saved with ${initial_capital:,.2f} capital!")
                                st.rerun()
                        
                        # Backtest Section
                        st.markdown("---")
                        st.markdown("#### 📈 Backtest Portfolio")
                        
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            backtest_period = st.slider("Backtest Period (days)", 30, 365, 180, key="backtest_period_slider")
                        
                        with col2:
                            rebalance_freq = st.selectbox("Rebalancing Frequency", 
                                                         ["Daily", "Weekly", "Monthly", "Quarterly", "No Rebalancing"],
                                                         index=2,
                                                         key="rebalance_freq_select")
                        
                        if st.button("🚀 Run Backtest", type="secondary", use_container_width=True):
                            with st.spinner("Running backtest..."):
                                try:
                                    # Calculate portfolio returns over time
                                    backtest_prices = prices.tail(backtest_period)
                                    returns = backtest_prices.pct_change().dropna()
                                    
                                    # Calculate portfolio value over time
                                    portfolio_returns = (returns * weights).sum(axis=1)
                                    portfolio_value = initial_capital * (1 + portfolio_returns).cumprod()
                                    
                                    # Calculate metrics
                                    total_return = (portfolio_value.iloc[-1] / initial_capital - 1) * 100
                                    annualized_return = ((portfolio_value.iloc[-1] / initial_capital) ** (252 / len(portfolio_value)) - 1) * 100
                                    cumulative_returns = (1 + portfolio_returns).cumprod() - 1
                                    max_drawdown = (cumulative_returns - cumulative_returns.cummax()).min() * 100
                                    
                                    # Display metrics
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.metric("Total Return", f"{total_return:.2f}%")
                                    with col2:
                                        st.metric("Annualized Return", f"{annualized_return:.2f}%")
                                    with col3:
                                        st.metric("Max Drawdown", f"{max_drawdown:.2f}%")
                                    with col4:
                                        final_value = portfolio_value.iloc[-1]
                                        st.metric("Final Value", f"${final_value:,.2f}")
                                    
                                    # Plot portfolio value
                                    fig_backtest = go.Figure()
                                    fig_backtest.add_trace(go.Scatter(
                                        x=portfolio_value.index,
                                        y=portfolio_value.values,
                                        mode='lines',
                                        name='Portfolio Value',
                                        line=dict(color='#667eea', width=2)
                                    ))
                                    
                                    # Add initial capital line
                                    fig_backtest.add_hline(y=initial_capital, line_dash="dash", 
                                                          line_color="gray", 
                                                          annotation_text="Initial Capital")
                                    
                                    fig_backtest.update_layout(
                                        title=f'Portfolio Backtest ({backtest_period} days)',
                                        xaxis_title='Date',
                                        yaxis_title='Portfolio Value ($)',
                                        height=400,
                                        showlegend=True
                                    )
                                    
                                    st.plotly_chart(fig_backtest, use_container_width=True)
                                    
                                    # Asset contribution
                                    st.markdown("#### Asset Contribution to Returns")
                                    asset_contributions = (returns * weights).tail(30)
                                    
                                    fig_contrib = go.Figure()
                                    for i, asset in enumerate(selected_assets):
                                        fig_contrib.add_trace(go.Bar(
                                            name=asset,
                                            x=asset_contributions.index,
                                            y=asset_contributions.iloc[:, i] * 100
                                        ))
                                    
                                    fig_contrib.update_layout(
                                        title='Daily Asset Contributions (Last 30 Days)',
                                        xaxis_title='Date',
                                        yaxis_title='Contribution (%)',
                                        barmode='stack',
                                        height=350
                                    )
                                    
                                    st.plotly_chart(fig_contrib, use_container_width=True)
                                    
                                except Exception as e:
                                    st.error(f"Backtest error: {str(e)}")
                        
                        # Export Section
                        st.markdown("---")
                        st.markdown("#### 📥 Export Portfolio")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            # Export weights as CSV
                            weights_export_df = pd.DataFrame({
                                'Asset': selected_assets,
                                'Weight': weights,
                                'Expected_Return': port_return,
                                'Volatility': port_vol,
                                'Sharpe_Ratio': sharpe
                            })
                            csv_data = weights_export_df.to_csv(index=False)
                            st.download_button(
                                label="📊 Download Weights (CSV)",
                                data=csv_data,
                                file_name=f"{portfolio_name.replace(' ', '_')}_weights.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                        
                        with col2:
                            # Export as JSON
                            portfolio_json = {
                                'name': portfolio_name,
                                'initial_capital': initial_capital,
                                'assets': selected_assets,
                                'weights': weights.tolist(),
                                'metrics': {
                                    'expected_return': port_return,
                                    'volatility': port_vol,
                                    'sharpe_ratio': sharpe
                                },
                                'created_at': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                            import json
                            json_data = json.dumps(portfolio_json, indent=2)
                            st.download_button(
                                label="📄 Download Config (JSON)",
                                data=json_data,
                                file_name=f"{portfolio_name.replace(' ', '_')}_config.json",
                                mime="application/json",
                                use_container_width=True
                            )
                        
                        with col3:
                            # Export positions
                            if 'positions' in locals():
                                positions_df = pd.DataFrame([
                                    {
                                        'Asset': asset,
                                        'Quantity': pos['quantity'],
                                        'Price': pos['avg_price'],
                                        'Value': pos['quantity'] * pos['avg_price'],
                                        'Weight': pos['weight']
                                    }
                                    for asset, pos in positions.items()
                                ])
                                positions_csv = positions_df.to_csv(index=False)
                                st.download_button(
                                    label="💼 Download Positions (CSV)",
                                    data=positions_csv,
                                    file_name=f"{portfolio_name.replace(' ', '_')}_positions.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                        
                    except Exception as e:
                        st.error(f"Optimization error: {str(e)}")
        else:
            st.info("Please select at least 2 assets for optimization")
        
        # Portfolio Management UI (outside button to avoid rerun issues)
        if 'optimization_results' in st.session_state:
            opt = st.session_state['optimization_results']
            weights = opt['weights']
            selected_assets = opt['assets']
            prices = opt['prices']
            port_return = opt['port_return']
            port_vol = opt['port_vol']
            sharpe = opt['sharpe']
            
            st.markdown("---")
            st.markdown("#### 💼 Save & Manage Portfolio")
            
            # Initialize form values in session state if not exists
            if 'portfolio_name' not in st.session_state:
                st.session_state['portfolio_name'] = "Optimized Portfolio"
            if 'initial_capital' not in st.session_state:
                st.session_state['initial_capital'] = 100000.0
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                portfolio_name = st.text_input("Portfolio Name", value=st.session_state['portfolio_name'], key="portfolio_name_input")
                st.session_state['portfolio_name'] = portfolio_name
            
            with col2:
                initial_capital = st.number_input("Initial Capital ($)", value=st.session_state['initial_capital'], min_value=1000.0, step=1000.0, key="initial_capital_input")
                st.session_state['initial_capital'] = initial_capital
            
            with col3:
                st.markdown("<br/>", unsafe_allow_html=True)
                if st.button("💾 Save as Current Portfolio", type="primary", use_container_width=True):
                    # Calculate positions based on weights and capital
                    positions = {}
                    for i, asset in enumerate(selected_assets):
                        weight = float(weights[i])
                        allocation = initial_capital * weight
                        
                        # Get last price for the asset
                        if isinstance(prices, pd.DataFrame) and asset in prices.columns:
                            last_price = float(prices[asset].iloc[-1])
                        else:
                            last_price = 100.0  # Default fallback
                        
                        quantity = allocation / last_price if last_price > 0 else 0
                        
                        positions[asset] = {
                            'quantity': quantity,
                            'avg_price': last_price,
                            'weight': weight
                        }
                    
                    # Update portfolio in session state
                    portfolio_data = {
                        'name': portfolio_name,
                        'positions': positions,
                        'cash': 0.0,  # Fully invested
                        'initial_capital': initial_capital,
                        'history': [],
                        'created_at': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'expected_return': port_return,
                        'volatility': port_vol,
                        'sharpe_ratio': sharpe
                    }
                    st.session_state.portfolio = portfolio_data
                    
                    # Save to disk as last portfolio
                    save_portfolio_to_disk(portfolio_data, "last_portfolio.json")
                    
                    # Also save with timestamp for history
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    safe_name = portfolio_name.replace(' ', '_').replace('/', '_')
                    save_portfolio_to_disk(portfolio_data, f"{safe_name}_{timestamp}.json")
                    
                    st.success(f"✅ Portfolio '{portfolio_name}' saved with ${initial_capital:,.2f} capital!")
                    st.info(f"📁 Saved to: data/portfolios/")
                    st.balloons()
            
            # Backtest Section
            st.markdown("---")
            st.markdown("#### 📈 Backtest Portfolio")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                backtest_period = st.slider("Backtest Period (days)", 30, min(365, len(prices)), min(180, len(prices)), key="backtest_period_slider")
            
            with col2:
                rebalance_freq = st.selectbox("Rebalancing Frequency", 
                                             ["Daily", "Weekly", "Monthly", "Quarterly", "No Rebalancing"],
                                             index=2,
                                             key="rebalance_freq_select")
            
            if st.button("🚀 Run Backtest", type="secondary", use_container_width=True):
                with st.spinner("Running backtest..."):
                    try:
                        # Calculate portfolio returns over time
                        backtest_prices = prices.tail(backtest_period)
                        backtest_returns = backtest_prices.pct_change().dropna()
                        
                        # Calculate portfolio value over time
                        portfolio_returns = (backtest_returns * weights).sum(axis=1)
                        portfolio_value = initial_capital * (1 + portfolio_returns).cumprod()
                        
                        # Calculate metrics
                        total_return = (portfolio_value.iloc[-1] / initial_capital - 1) * 100
                        annualized_return = ((portfolio_value.iloc[-1] / initial_capital) ** (252 / len(portfolio_value)) - 1) * 100
                        cumulative_returns = (1 + portfolio_returns).cumprod() - 1
                        max_drawdown = (cumulative_returns - cumulative_returns.cummax()).min() * 100
                        
                        # Display metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Return", f"{total_return:.2f}%")
                        with col2:
                            st.metric("Annualized Return", f"{annualized_return:.2f}%")
                        with col3:
                            st.metric("Max Drawdown", f"{max_drawdown:.2f}%")
                        with col4:
                            final_value = portfolio_value.iloc[-1]
                            st.metric("Final Value", f"${final_value:,.2f}")
                        
                        # Plot portfolio value
                        fig_backtest = go.Figure()
                        fig_backtest.add_trace(go.Scatter(
                            x=portfolio_value.index,
                            y=portfolio_value.values,
                            mode='lines',
                            name='Portfolio Value',
                            line=dict(color='#667eea', width=2)
                        ))
                        
                        # Add initial capital line
                        fig_backtest.add_hline(y=initial_capital, line_dash="dash", 
                                              line_color="gray", 
                                              annotation_text="Initial Capital")
                        
                        fig_backtest.update_layout(
                            title=f'Portfolio Backtest ({backtest_period} days)',
                            xaxis_title='Date',
                            yaxis_title='Portfolio Value ($)',
                            height=400,
                            showlegend=True
                        )
                        
                        st.plotly_chart(fig_backtest, use_container_width=True)
                        
                        # Asset contribution
                        st.markdown("#### Asset Contribution to Returns")
                        asset_contributions = (backtest_returns * weights).tail(30)
                        
                        fig_contrib = go.Figure()
                        for i, asset in enumerate(selected_assets):
                            fig_contrib.add_trace(go.Bar(
                                name=asset,
                                x=asset_contributions.index,
                                y=asset_contributions.iloc[:, i] * 100
                            ))
                        
                        fig_contrib.update_layout(
                            title='Daily Asset Contributions (Last 30 Days)',
                            xaxis_title='Date',
                            yaxis_title='Contribution (%)',
                            barmode='stack',
                            height=350
                        )
                        
                        st.plotly_chart(fig_contrib, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Backtest error: {str(e)}")
            
            # Export Section
            st.markdown("---")
            st.markdown("#### 📥 Export Portfolio")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Export weights as CSV
                weights_export_df = pd.DataFrame({
                    'Asset': selected_assets,
                    'Weight': weights,
                    'Expected_Return': port_return,
                    'Volatility': port_vol,
                    'Sharpe_Ratio': sharpe
                })
                csv_data = weights_export_df.to_csv(index=False)
                st.download_button(
                    label="📊 Download Weights (CSV)",
                    data=csv_data,
                    file_name=f"{portfolio_name.replace(' ', '_')}_weights.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                # Export as JSON
                import json
                portfolio_json = {
                    'name': portfolio_name,
                    'initial_capital': initial_capital,
                    'assets': selected_assets,
                    'weights': weights.tolist(),
                    'metrics': {
                        'expected_return': port_return,
                        'volatility': port_vol,
                        'sharpe_ratio': sharpe
                    },
                    'created_at': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                json_data = json.dumps(portfolio_json, indent=2)
                st.download_button(
                    label="📄 Download Config (JSON)",
                    data=json_data,
                    file_name=f"{portfolio_name.replace(' ', '_')}_config.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            with col3:
                # Export positions based on current prices
                positions_export = []
                for i, asset in enumerate(selected_assets):
                    weight = float(weights[i])
                    allocation = initial_capital * weight
                    if isinstance(prices, pd.DataFrame) and asset in prices.columns:
                        last_price = float(prices[asset].iloc[-1])
                    else:
                        last_price = 100.0
                    quantity = allocation / last_price if last_price > 0 else 0
                    
                    positions_export.append({
                        'Asset': asset,
                        'Quantity': quantity,
                        'Price': last_price,
                        'Value': quantity * last_price,
                        'Weight': weight
                    })
                
                positions_df = pd.DataFrame(positions_export)
                positions_csv = positions_df.to_csv(index=False)
                st.download_button(
                    label="💼 Download Positions (CSV)",
                    data=positions_csv,
                    file_name=f"{portfolio_name.replace(' ', '_')}_positions.csv",
                    mime="text/csv",
                    use_container_width=True
                )

with tab3:
    st.markdown("### Risk Analysis")
    
    if 'portfolio_weights' not in st.session_state:
        st.info("💡 Optimize a portfolio in the Optimization tab first")
    else:
        weights_data = st.session_state['portfolio_weights']
        weights = weights_data['weights'] if isinstance(weights_data, dict) else weights_data
        selected_assets = weights_data.get('assets', []) if isinstance(weights_data, dict) else []
        data = st.session_state['historical_data']
        
        st.markdown("#### 🎯 Risk Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            confidence_level = st.slider("VaR Confidence Level", 90, 99, 95, 1,
                                        help="Confidence level for VaR calculation")
            time_horizon = st.slider("Time Horizon (days)", 1, 30, 10,
                                    help="Risk measurement period")
        
        with col2:
            num_simulations = st.slider("Monte Carlo Simulations", 1000, 50000, 10000, 1000,
                                       help="Number of scenarios for stress testing")
        
        if st.button("📊 Calculate Risk Metrics", type="primary"):
            with st.spinner("Computing risk analytics..."):
                # Extract returns for selected assets
                returns_dict = {}
                for asset in selected_assets:
                    if isinstance(data, dict):
                        df = data[asset]
                    else:
                        df = data[data['symbol'] == asset].copy() if 'symbol' in data.columns else data.copy()
                    
                    close_col = None
                    for col in df.columns:
                        if col.lower() == 'close':
                            close_col = col
                            break
                    
                    if close_col:
                        prices = df[close_col].values
                        returns_dict[asset] = np.diff(prices) / prices[:-1]
                
                # Align returns to same length
                min_length = min(len(r) for r in returns_dict.values())
                aligned_returns = {k: v[-min_length:] for k, v in returns_dict.items()}
                
                # Portfolio returns
                returns_matrix = np.array([aligned_returns[asset] for asset in selected_assets]).T
                portfolio_returns = returns_matrix @ weights
                
                # 1. Value at Risk (VaR)
                var_percentile = 100 - confidence_level
                var_historical = np.percentile(portfolio_returns, var_percentile) * np.sqrt(time_horizon)
                
                # Parametric VaR (assuming normal distribution)
                var_parametric = norm.ppf(var_percentile / 100) * np.std(portfolio_returns) * np.sqrt(time_horizon)
                
                # 2. Expected Shortfall (CVaR)
                threshold = np.percentile(portfolio_returns, var_percentile)
                cvar = portfolio_returns[portfolio_returns <= threshold].mean() * np.sqrt(time_horizon)
                
                # 3. Maximum Drawdown
                cum_returns = np.cumprod(1 + portfolio_returns)
                running_max = np.maximum.accumulate(cum_returns)
                drawdowns = (cum_returns - running_max) / running_max
                max_drawdown = np.min(drawdowns)
                
                # 4. Beta (if we have market proxy)
                # Using equal-weighted portfolio as market proxy
                market_returns = returns_matrix.mean(axis=1)
                covariance = np.cov(portfolio_returns, market_returns)[0, 1]
                market_variance = np.var(market_returns)
                beta = covariance / market_variance if market_variance > 0 else 1.0
                
                # Display metrics
                st.markdown("### 📊 Risk Metrics")
                
                col_a, col_b, col_c, col_d = st.columns(4)
                
                with col_a:
                    st.metric("VaR (Historical)", f"{var_historical*100:.2f}%",
                             help=f"Maximum loss at {confidence_level}% confidence")
                with col_b:
                    st.metric("VaR (Parametric)", f"{var_parametric*100:.2f}%",
                             help="VaR assuming normal distribution")
                with col_c:
                    st.metric("Expected Shortfall", f"{cvar*100:.2f}%",
                             help="Average loss beyond VaR")
                with col_d:
                    st.metric("Max Drawdown", f"{max_drawdown*100:.2f}%",
                             help="Largest peak-to-trough decline")
                
                col_e, col_f, col_g, col_h = st.columns(4)
                
                with col_e:
                    st.metric("Portfolio Beta", f"{beta:.2f}",
                             help="Sensitivity to market movements")
                with col_f:
                    volatility = np.std(portfolio_returns) * np.sqrt(252)
                    st.metric("Annualized Vol", f"{volatility*100:.2f}%")
                with col_g:
                    sharpe = np.mean(portfolio_returns) / np.std(portfolio_returns) * np.sqrt(252)
                    st.metric("Sharpe Ratio", f"{sharpe:.2f}")
                with col_h:
                    sortino_denominator = np.std(portfolio_returns[portfolio_returns < 0])
                    sortino = np.mean(portfolio_returns) / sortino_denominator * np.sqrt(252) if sortino_denominator > 0 else 0
                    st.metric("Sortino Ratio", f"{sortino:.2f}")
                
                # Monte Carlo Stress Testing
                st.markdown("### 🎲 Monte Carlo Stress Testing")
                
                # Simulate portfolio scenarios
                mean_return = np.mean(portfolio_returns)
                std_return = np.std(portfolio_returns)
                
                simulated_returns = np.random.normal(
                    mean_return * time_horizon,
                    std_return * np.sqrt(time_horizon),
                    num_simulations
                )
                
                # Stress scenarios
                scenarios = {
                    'Base Case': simulated_returns,
                    'Market Crash (-2σ)': np.random.normal(mean_return * time_horizon - 2 * std_return * np.sqrt(time_horizon),
                                                          std_return * np.sqrt(time_horizon), num_simulations),
                    'High Volatility (2x σ)': np.random.normal(mean_return * time_horizon,
                                                               2 * std_return * np.sqrt(time_horizon), num_simulations),
                    'Bull Market (+2σ)': np.random.normal(mean_return * time_horizon + 2 * std_return * np.sqrt(time_horizon),
                                                         std_return * np.sqrt(time_horizon), num_simulations)
                }
                
                # Plot scenarios
                fig = go.Figure()
                
                for scenario_name, scenario_returns in scenarios.items():
                    fig.add_trace(go.Histogram(
                        x=scenario_returns * 100,
                        name=scenario_name,
                        opacity=0.6,
                        nbinsx=50
                    ))
                
                fig.update_layout(
                    title=f'Portfolio Return Distribution ({time_horizon}-day horizon)',
                    xaxis_title='Return (%)',
                    yaxis_title='Frequency',
                    barmode='overlay',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Scenario statistics
                st.markdown("#### 📈 Scenario Analysis")
                
                scenario_stats = []
                for scenario_name, scenario_returns in scenarios.items():
                    scenario_stats.append({
                        'Scenario': scenario_name,
                        'Mean': f"{np.mean(scenario_returns)*100:.2f}%",
                        'Median': f"{np.median(scenario_returns)*100:.2f}%",
                        'Std Dev': f"{np.std(scenario_returns)*100:.2f}%",
                        f'VaR {confidence_level}%': f"{np.percentile(scenario_returns, var_percentile)*100:.2f}%",
                        'Min': f"{np.min(scenario_returns)*100:.2f}%",
                        'Max': f"{np.max(scenario_returns)*100:.2f}%"
                    })
                
                st.dataframe(pd.DataFrame(scenario_stats), use_container_width=True)
                
                # Risk decomposition
                st.markdown("### 🔍 Risk Decomposition")
                
                # Component VaR (marginal contribution to risk)
                component_vars = []
                for i, asset in enumerate(selected_assets):
                    # Perturb weight slightly
                    perturbed_weights = weights.copy()
                    epsilon = 0.01
                    perturbed_weights[i] += epsilon
                    perturbed_weights = perturbed_weights / perturbed_weights.sum()  # Renormalize
                    
                    perturbed_returns = returns_matrix @ perturbed_weights
                    perturbed_var = np.percentile(perturbed_returns, var_percentile) * np.sqrt(time_horizon)
                    
                    marginal_var = (perturbed_var - var_historical) / epsilon
                    component_var = marginal_var * weights[i]
                    
                    component_vars.append({
                        'Asset': asset,
                        'Weight': f"{weights[i]*100:.1f}%",
                        'Marginal VaR': f"{marginal_var*100:.3f}%",
                        'Component VaR': f"{component_var*100:.3f}%",
                        'Contribution': f"{(component_var/var_historical)*100:.1f}%" if var_historical != 0 else "N/A"
                    })
                
                st.dataframe(pd.DataFrame(component_vars), use_container_width=True)

with tab4:
    st.markdown("### Performance Attribution")
    
    if 'portfolio_weights' not in st.session_state:
        st.info("💡 Optimize a portfolio in the Optimization tab first")
    else:
        weights_data = st.session_state['portfolio_weights']
        weights = weights_data['weights'] if isinstance(weights_data, dict) else weights_data
        selected_assets = weights_data.get('assets', []) if isinstance(weights_data, dict) else []
        data = st.session_state['historical_data']
        
        st.markdown("#### 📊 Return Attribution Analysis")
        
        if st.button("🔍 Analyze Attribution", type="primary"):
            with st.spinner("Computing attribution..."):
                # Extract returns
                returns_dict = {}
                for asset in selected_assets:
                    if isinstance(data, dict):
                        df = data[asset]
                    else:
                        df = data[data['symbol'] == asset].copy() if 'symbol' in data.columns else data.copy()
                    
                    close_col = None
                    for col in df.columns:
                        if col.lower() == 'close':
                            close_col = col
                            break
                    
                    if close_col:
                        prices = df[close_col].values
                        returns_dict[asset] = np.diff(prices) / prices[:-1]
                
                # Align returns
                min_length = min(len(r) for r in returns_dict.values())
                aligned_returns = {k: v[-min_length:] for k, v in returns_dict.items()}
                
                # Portfolio returns
                returns_matrix = np.array([aligned_returns[asset] for asset in selected_assets]).T
                portfolio_returns = returns_matrix @ weights
                
                # Attribution metrics
                st.markdown("### 📈 Asset Contribution")
                
                attribution_data = []
                total_return = np.sum(portfolio_returns)
                
                for i, asset in enumerate(selected_assets):
                    asset_returns = aligned_returns[asset]
                    asset_contribution = np.sum(asset_returns) * weights[i]
                    contribution_pct = (asset_contribution / total_return * 100) if total_return != 0 else 0
                    
                    attribution_data.append({
                        'Asset': asset,
                        'Weight': f"{weights[i]*100:.1f}%",
                        'Asset Return': f"{np.sum(asset_returns)*100:.2f}%",
                        'Contribution': f"{asset_contribution*100:.2f}%",
                        '% of Total': f"{contribution_pct:.1f}%",
                        'Sharpe Ratio': f"{(np.mean(asset_returns) / np.std(asset_returns) * np.sqrt(252)):.2f}"
                    })
                
                st.dataframe(pd.DataFrame(attribution_data), use_container_width=True)
                
                # Contribution breakdown chart
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=[a['Asset'] for a in attribution_data],
                    y=[float(a['Contribution'].rstrip('%')) for a in attribution_data],
                    name='Return Contribution',
                    marker_color='lightblue'
                ))
                
                fig.update_layout(
                    title='Asset Return Contribution (%)',
                    xaxis_title='Asset',
                    yaxis_title='Contribution (%)',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Factor exposure (simplified: momentum and volatility)
                st.markdown("### 🔬 Factor Exposure")
                
                factor_data = []
                
                for i, asset in enumerate(selected_assets):
                    asset_returns = aligned_returns[asset]
                    
                    # Momentum factor (trailing 20-day return)
                    momentum = np.mean(asset_returns[-20:]) if len(asset_returns) >= 20 else np.mean(asset_returns)
                    
                    # Volatility factor
                    volatility = np.std(asset_returns)
                    
                    # Size factor (weight)
                    size_exposure = weights[i]
                    
                    factor_data.append({
                        'Asset': asset,
                        'Momentum': f"{momentum*100:.3f}%",
                        'Volatility': f"{volatility*100:.2f}%",
                        'Size Exposure': f"{size_exposure*100:.1f}%"
                    })
                
                st.dataframe(pd.DataFrame(factor_data), use_container_width=True)
                
                # Portfolio-level metrics
                st.markdown("### 📊 Portfolio Summary")
                
                col_a, col_b, col_c, col_d = st.columns(4)
                
                with col_a:
                    st.metric("Total Return", f"{total_return*100:.2f}%")
                with col_b:
                    ann_return = np.mean(portfolio_returns) * 252 * 100
                    st.metric("Annualized Return", f"{ann_return:.2f}%")
                with col_c:
                    ann_vol = np.std(portfolio_returns) * np.sqrt(252) * 100
                    st.metric("Annualized Vol", f"{ann_vol:.2f}%")
                with col_d:
                    sharpe = (np.mean(portfolio_returns) / np.std(portfolio_returns) * np.sqrt(252))
                    st.metric("Sharpe Ratio", f"{sharpe:.2f}")
    
    st.markdown("""
    ### Performance Metrics:
    
    - **Sharpe Ratio**: Risk-adjusted return
    - **Information Ratio**: Alpha generation vs benchmark
    - **Sortino Ratio**: Downside risk-adjusted return
    - **Calmar Ratio**: Return vs maximum drawdown
    - **Win Rate**: Percentage of profitable trades
    """)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>📊 Portfolio Analytics Lab | Part of HFT Arbitrage Lab</p>
</div>
""", unsafe_allow_html=True)
