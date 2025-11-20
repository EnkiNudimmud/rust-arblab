"""
Multi-Strategy HFT Trading Platform
====================================

Unified multi-page Streamlit application for:
- Historical data loading & backtesting
- Real-time strategy execution
- Portfolio management & tracking
- Options & futures data

Author: Rust HFT Arbitrage Lab
"""

import streamlit as st
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Check Python version compatibility
try:
    from app.utils.compat import check_python_version, PYTHON_VERSION
    check_python_version()
except RuntimeError as e:
    st.error(f"⚠️ Python Version Error: {e}")
    st.stop()
except UserWarning as w:
    st.warning(f"⚠️ {w}")

# Page imports
from app.pages import data_loader, strategy_backtest, live_trading, portfolio_view, derivatives

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="HFT Trading Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        background-color: #f0f2f6;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def init_session_state():
    """Initialize session state variables"""
    defaults = {
        # Data management
        'historical_data': None,
        'data_source': 'finnhub',
        'symbols': ['AAPL', 'MSFT', 'GOOGL'],
        'date_range': None,
        
        # Strategy
        'selected_strategy': 'Mean Reversion',
        'strategy_params': {},
        'backtest_results': None,
        
        # Live trading
        'live_trading_active': False,
        'live_data_buffer': [],
        'websocket_connection': None,
        'trade_log': [],
        
        # Portfolio
        'portfolio': {
            'cash': 100000.0,
            'positions': {},
            'history': [],
            'trades': []
        },
        
        # Options/Futures
        'derivatives_data': {},
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ============================================================================
# NAVIGATION
# ============================================================================

# Sidebar navigation
with st.sidebar:
    st.markdown('<div class="main-header">🚀 HFT Lab</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Check Rust availability
    try:
        import rust_connector
        st.success("⚡ Rust acceleration enabled")
    except ImportError:
        st.warning("⚠️ Pure Python mode")
    
    st.markdown("---")
    
    # Navigation
    page = st.radio(
        "Navigation",
        [
            "📊 Data Loading",
            "⚡ Strategy Backtest", 
            "🔴 Live Trading",
            "💼 Portfolio View",
            "📈 Options & Futures"
        ],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Quick stats
    st.markdown("### Quick Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Portfolio Value", f"${st.session_state.portfolio['cash']:,.0f}")
    with col2:
        n_positions = len(st.session_state.portfolio['positions'])
        st.metric("Open Positions", n_positions)
    
    st.markdown("---")
    st.caption("Version 1.0.0 | Multi-Strategy Trading Platform")

# ============================================================================
# PAGE ROUTING
# ============================================================================

if page == "📊 Data Loading":
    data_loader.render()
elif page == "⚡ Strategy Backtest":
    strategy_backtest.render()
elif page == "🔴 Live Trading":
    live_trading.render()
elif page == "💼 Portfolio View":
    portfolio_view.render()
elif page == "📈 Options & Futures":
    derivatives.render()
