"""
HFT Arbitrage Lab - Main Dashboard
A comprehensive platform for arbitrage strategy development and testing
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Page configuration
st.set_page_config(
    page_title="HFT Arbitrage Lab",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'historical_data' not in st.session_state:
    st.session_state.historical_data = None
if 'symbols' not in st.session_state:
    st.session_state.symbols = []  # Empty list for user to populate
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {
        'positions': {},
        'cash': 100000.0,
        'initial_capital': 100000.0
    }
if 'derivatives_data' not in st.session_state:
    st.session_state.derivatives_data = {}
if 'live_ws_status' not in st.session_state:
    st.session_state.live_ws_status = {}
if 'live_use_websocket' not in st.session_state:
    st.session_state.live_use_websocket = False

# Custom CSS for better aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 2rem 0;
    }
    
    .strategy-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s;
    }
    
    .strategy-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 12px rgba(0,0,0,0.2);
    }
    
    .metric-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    
    .status-online {
        color: #10b981;
        font-weight: bold;
    }
    
    .status-offline {
        color: #ef4444;
        font-weight: bold;
    }
    
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        font-weight: bold;
        padding: 0.75rem 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">⚡ HFT Arbitrage Lab</h1>', unsafe_allow_html=True)
st.markdown("### Advanced Trading Strategy Development & Backtesting Platform")

# Status indicators
col1, col2, col3, col4 = st.columns(4)

try:
    import hft_py
    rust_status = "🟢 Online"
    rust_class = "status-online"
except:
    rust_status = "🔴 Offline"
    rust_class = "status-offline"

try:
    from python.api_keys import get_finnhub_key
    api_key = get_finnhub_key()
    api_status = "🟢 Configured" if api_key else "🟡 Missing"
    api_class = "status-online" if api_key else "status-offline"
except:
    api_status = "🔴 Error"
    api_class = "status-offline"

with col1:
    st.markdown(f'<div class="metric-box"><strong>Rust Engine</strong><br/><span class="{rust_class}">{rust_status}</span></div>', unsafe_allow_html=True)

with col2:
    st.markdown(f'<div class="metric-box"><strong>API Keys</strong><br/><span class="{api_class}">{api_status}</span></div>', unsafe_allow_html=True)

with col3:
    st.markdown(f'<div class="metric-box"><strong>Market Data</strong><br/><span class="status-online">🟢 Finnhub</span></div>', unsafe_allow_html=True)

with col4:
    st.markdown(f'<div class="metric-box"><strong>Strategies</strong><br/><span class="status-online">🟢 12+ Active</span></div>', unsafe_allow_html=True)

st.markdown("---")

# Main content
st.markdown("## 🎯 Available Strategies")

# Strategy categories
tab1, tab2, tab3, tab4 = st.tabs(["📊 Statistical Arbitrage", "📈 Trend & Momentum", "🎲 Options Strategies", "⚡ Live Trading"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="strategy-card">
            <h3>📉 Mean Reversion</h3>
            <p>Trade price deviations from historical means using statistical signals</p>
            <ul>
                <li>Z-score based entry/exit</li>
                <li>Multiple symbol support (100+)</li>
                <li>Rust-optimized computations</li>
                <li>Regime-aware position sizing</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Launch Mean Reversion Lab", key="meanrev"):
            st.switch_page("pages/strategy_backtest.py")
    
    with col2:
        st.markdown("""
        <div class="strategy-card">
            <h3>🔄 Pairs Trading</h3>
            <p>Trade co-integrated pairs with statistical relationships</p>
            <ul>
                <li>Correlation analysis</li>
                <li>Cointegration tests</li>
                <li>Spread calculation</li>
                <li>Beta-hedged positions</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Launch Pairs Trading", key="pairs"):
            st.switch_page("pages/strategy_backtest.py")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="strategy-card">
            <h3>📊 Chiarella Model</h3>
            <p>Agent-based price dynamics with chartist-fundamentalist interaction</p>
            <ul>
                <li>Bifurcation analysis</li>
                <li>Regime detection</li>
                <li>Adaptive strategy selection</li>
                <li>Parameter estimation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Launch Chiarella Lab", key="chiarella"):
            st.info("Chiarella model page coming soon!")
    
    with col2:
        st.markdown("""
        <div class="strategy-card">
            <h3>🎯 PCA Arbitrage</h3>
            <p>Trade principal components for portfolio-level strategies</p>
            <ul>
                <li>Dimensionality reduction</li>
                <li>Factor extraction</li>
                <li>Orthogonal portfolios</li>
                <li>Reduced noise</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Launch PCA Lab", key="pca"):
            st.info("PCA arbitrage page coming soon!")

with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="strategy-card">
            <h3>📈 Momentum Trading</h3>
            <p>Ride trends with adaptive position sizing</p>
            <ul>
                <li>Multiple timeframe analysis</li>
                <li>Trend strength indicators</li>
                <li>Breakout detection</li>
                <li>Trailing stops</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Launch Momentum Lab", key="momentum"):
            st.info("Momentum trading page coming soon!")
    
    with col2:
        st.markdown("""
        <div class="strategy-card">
            <h3>🌊 Market Making</h3>
            <p>Provide liquidity and capture bid-ask spreads</p>
            <ul>
                <li>Order book analysis</li>
                <li>Inventory management</li>
                <li>Adverse selection protection</li>
                <li>Dynamic pricing</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Launch Market Making", key="mm"):
            st.info("Market making page coming soon!")

with tab3:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="strategy-card">
            <h3>🦋 Multi-Leg Spreads</h3>
            <p>Complex options strategies with defined risk/reward</p>
            <ul>
                <li>Butterfly spreads</li>
                <li>Iron condors</li>
                <li>Calendar spreads</li>
                <li>Ratio spreads</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Launch Options Lab", key="options"):
            st.switch_page("pages/derivatives.py")
    
    with col2:
        st.markdown("""
        <div class="strategy-card">
            <h3>📊 Volatility Trading</h3>
            <p>Trade implied vs realized volatility</p>
            <ul>
                <li>Straddles & strangles</li>
                <li>Volatility arbitrage</li>
                <li>Greeks hedging</li>
                <li>Risk reversal</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Launch Vol Trading", key="vol"):
            st.switch_page("pages/derivatives.py")

with tab4:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="strategy-card">
            <h3>⚡ Live WebSocket Trading</h3>
            <p>Real-time execution with multi-exchange connectivity</p>
            <ul>
                <li>Binance, Kraken, Coinbase</li>
                <li>Low-latency execution</li>
                <li>Real-time P&L</li>
                <li>Order management</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Launch Live Trading", key="live"):
            st.switch_page("pages/live_trading.py")
    
    with col2:
        st.markdown("""
        <div class="strategy-card">
            <h3>📊 Portfolio Monitor</h3>
            <p>Real-time portfolio tracking and risk management</p>
            <ul>
                <li>Multi-asset positions</li>
                <li>Real-time Greeks</li>
                <li>Risk metrics</li>
                <li>Alert system</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Launch Portfolio", key="launch_portfolio"):
            st.switch_page("pages/portfolio_view.py")

st.markdown("---")

# Quick stats
st.markdown("## 📈 Platform Capabilities")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Supported Assets", value="200+", delta="Stocks, Crypto, ETFs")

with col2:
    st.metric(label="Strategy Types", value="12+", delta="Statistical to ML-based")

with col3:
    st.metric(label="Data Sources", value="5+", delta="Finnhub, Binance, Kraken")

with col4:
    st.metric(label="Compute Speed", value="10-100x", delta="Rust optimization")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with ❤️ using Streamlit, Python & Rust | HFT Arbitrage Lab © 2025</p>
    <p>⚠️ For educational and research purposes only. Not financial advice.</p>
</div>
""", unsafe_allow_html=True)
