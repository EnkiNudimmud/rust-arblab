"""
HFT Arbitrage Lab - Homepage
A comprehensive platform for arbitrage strategy development and testing
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import shared UI components
from utils.ui_components import render_sidebar_navigation, apply_custom_css

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
    st.session_state.symbols = []
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {
        'positions': {},
        'cash': 100000.0,
        'initial_capital': 100000.0,
        'history': []
    }
if 'derivatives_data' not in st.session_state:
    st.session_state.derivatives_data = {}
if 'live_ws_status' not in st.session_state:
    st.session_state.live_ws_status = {}
if 'live_use_websocket' not in st.session_state:
    st.session_state.live_use_websocket = False

# Render sidebar navigation and apply custom CSS
render_sidebar_navigation(current_page="Home")
apply_custom_css()

# Header
st.markdown('<h1 class="main-header">⚡ HFT Arbitrage Lab</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced High-Frequency Trading Strategy Development & Backtesting Platform</p>', unsafe_allow_html=True)

# System Status
st.markdown("## 🖥️ System Status")

col1, col2, col3, col4 = st.columns(4)

try:
    import rust_connector
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
    st.markdown(f'<div class="status-card"><strong>Rust Engine</strong><br/><span class="{rust_class}">{rust_status}</span></div>', unsafe_allow_html=True)

with col2:
    st.markdown(f'<div class="status-card"><strong>API Keys</strong><br/><span class="{api_class}">{api_status}</span></div>', unsafe_allow_html=True)

with col3:
    st.markdown(f'<div class="status-card"><strong>Market Data</strong><br/><span class="status-online">🟢 Finnhub</span></div>', unsafe_allow_html=True)

with col4:
    st.markdown(f'<div class="status-card"><strong>Active Labs</strong><br/><span class="status-online">🟢 5 Labs</span></div>', unsafe_allow_html=True)

st.markdown("---")

# Overview
st.markdown("## 🎯 Platform Overview")

st.markdown("""
Welcome to **HFT Arbitrage Lab**, a comprehensive research and development platform for quantitative trading strategies. 
This platform combines cutting-edge mathematical models with high-performance computing to enable rapid strategy development, 
backtesting, and live trading execution.
""")

# Key Features
st.markdown("## ✨ Key Features")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
        <h3>📊 Data & Analytics</h3>
        <ul>
            <li>Multi-source market data</li>
            <li>200+ assets supported</li>
            <li>Real-time & historical</li>
            <li>Preset symbol lists</li>
            <li>Custom universes</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚀 Go to Data Loader", key="feature_data", use_container_width=True, type="primary"):
        st.switch_page("pages/data_loader.py")

with col2:
    st.markdown("""
    <div class="feature-card">
        <h3>🔬 Research Labs</h3>
        <ul>
            <li>Mean reversion analysis</li>
            <li>Rough Heston volatility</li>
            <li>Chiarella dynamics</li>
            <li>Signature methods</li>
            <li>Portfolio analytics</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚀 Explore Labs", key="feature_labs", use_container_width=True, type="primary"):
        st.switch_page("pages/lab_mean_reversion.py")

with col3:
    st.markdown("""
    <div class="feature-card">
        <h3>⚡ Live Trading</h3>
        <ul>
            <li>WebSocket connectivity</li>
            <li>Multi-exchange support</li>
            <li>Real-time execution</li>
            <li>Risk management</li>
            <li>Portfolio tracking</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚀 Start Trading", key="feature_live", use_container_width=True, type="primary"):
        st.switch_page("pages/live_trading.py")

st.markdown("---")

# Quick Start Guide
st.markdown("## 🚀 Quick Start Guide")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Load Data", "🔬 Run Lab", "⚡ Backtest Strategy", "🔴 Live Trade"])

with tab1:
    st.markdown("""
    ### Step 1: Load Market Data
    
    1. Navigate to **📊 Data & Market → Data Loader**
    2. Choose data source (Finnhub, Binance, etc.)
    3. Select symbols or use presets (Sectors, Indexes, ETFs)
    4. Set date range and fetch data
    5. Data will be stored in session state for analysis
    
    **💡 Tip:** Use sector presets to quickly load multiple related assets
    
    Use the **📊 Data & Market** section in the sidebar to access the Data Loader.
    """)

with tab2:
    st.markdown("""
    ### Step 2: Explore Research Labs
    
    Choose from 5 advanced research labs:
    
    - **📉 Mean Reversion Lab**: Statistical arbitrage with Z-scores
    - **📈 Rough Heston Lab**: Stochastic volatility modeling
    - **🌀 Chiarella Lab**: Agent-based market dynamics
    - **✍️ Signature Methods Lab**: Path signature analysis
    - **📊 Portfolio Analytics**: Risk metrics and optimization
    
    Each lab provides interactive analysis with real-time Rust computations.
    
    💡 **Use the sidebar** to navigate to any research lab.
    """)

with tab3:
    st.markdown("""
    ### Step 3: Backtest Strategies
    
    1. Go to **⚡ Trading Strategies** section
    2. Select strategy type (arbitrage, mean reversion, etc.)
    3. Configure parameters
    4. Run backtest on historical data
    5. Analyze results and optimize
    
    **Available Strategies:**
    - Statistical arbitrage
    - Pairs trading
    - Triangular arbitrage
    - Options strategies
    - Derivatives trading
    
    Navigate using the **⚡ Trading Strategies** section in the sidebar.
    """)

with tab4:
    st.markdown("""
    ### Step 4: Deploy Live Trading
    
    1. Navigate to **🔴 Live Trading** section
    2. Connect to exchange via WebSocket
    3. Select trading strategy
    4. Set risk parameters
    5. Enable auto-trading or manual execution
    
    **⚠️ Important:** 
    - Start with paper trading
    - Always set stop losses
    - Monitor positions actively
    - This is for educational purposes only
    
    Access live trading from the **🔴 Live Trading** section in the sidebar.
    """)

st.markdown("---")

# Statistics
st.markdown("## 📊 Platform Capabilities")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="status-card" style="text-align: center;"><div class="metric-big">200+</div><strong>Supported Assets</strong><br/>Stocks, Crypto, ETFs, Options</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="status-card" style="text-align: center;"><div class="metric-big">15+</div><strong>Strategies</strong><br/>Statistical to ML-based</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="status-card" style="text-align: center;"><div class="metric-big">5+</div><strong>Data Sources</strong><br/>Finnhub, Binance, Kraken</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="status-card" style="text-align: center;"><div class="metric-big">10-100x</div><strong>Speed Boost</strong><br/>Rust-powered analytics</div>', unsafe_allow_html=True)

st.markdown("---")

# Documentation
st.markdown("## 📚 Documentation")

with st.expander("📖 Learn More About HFT Arbitrage Lab"):
    st.markdown("""
    ### 🎓 Getting Started
    
    - **Installation**: Clone the repository and run `./setup_project.sh`
    - **Configuration**: Copy `api_keys.properties.example` to `api_keys.properties`
    - **Dependencies**: Python 3.11+, Rust 1.70+, required packages
    
    ### 📖 Core Concepts
    
    - **Statistical Arbitrage**: Exploit mean-reverting price relationships
    - **Rough Volatility**: Model volatility with fractional processes
    - **Agent-Based Models**: Simulate market dynamics with interacting agents
    - **Signature Methods**: Analyze price paths using path signatures
    
    ### 🛠️ Technical Stack
    
    - **Frontend**: Streamlit for interactive dashboards
    - **Backend**: Python + Rust for high-performance computing
    - **Data**: Finnhub, Binance, Kraken APIs
    - **Analytics**: NumPy, Pandas, SciPy, custom Rust modules
    
    ### 📝 Best Practices
    
    1. Always start with data exploration in the Data Loader
    2. Validate strategies thoroughly in Research Labs
    3. Backtest with realistic transaction costs
    4. Paper trade before live deployment
    5. Monitor risk metrics continuously
    
    ### 🔗 Resources
    
    - [GitHub Repository](https://github.com/ThotDjehuty/rust-hft-arbitrage-lab)
    - [Documentation](docs/README.md)
    - [Quick Start Guide](docs/QUICKSTART_APP.md)
    """)

st.markdown("---")

# Footer
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p style='font-size: 1.1rem;'>Built with ❤️ using <strong>Streamlit</strong>, <strong>Python</strong> & <strong>Rust</strong></p>
    <p style='font-weight: bold;'>HFT Arbitrage Lab © 2025</p>
    <p style='color: #ef4444; font-weight: bold;'>⚠️ For educational and research purposes only. Not financial advice.</p>
</div>
""", unsafe_allow_html=True)

# Hide streamlit default menu
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
