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
from utils.ui_components import render_sidebar_navigation, apply_custom_css, ensure_data_loaded
from utils.data_persistence import load_all_datasets, list_datasets, get_storage_stats

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

# Initialize theme mode
if 'theme_mode' not in st.session_state:
    st.session_state.theme_mode = 'dark'  # Default to dark mode for better visibility

# Initialize persisted datasets cache
if 'persisted_datasets' not in st.session_state:
    st.session_state.persisted_datasets = {}
    
# Load persisted datasets on first run (if not already loaded)
if 'datasets_loaded' not in st.session_state:
    try:
        datasets = list_datasets()
        if datasets:
            # Store metadata about available datasets
            st.session_state.persisted_datasets = {
                ds['name']: ds for ds in datasets
            }
            st.session_state.datasets_loaded = True
        else:
            st.session_state.datasets_loaded = True
    except Exception as e:
        # Silently fail if data directory doesn't exist yet
        st.session_state.datasets_loaded = True

# Auto-load most recent dataset if no data is loaded
ensure_data_loaded()

# Render sidebar navigation and apply custom CSS
render_sidebar_navigation(current_page="Home")
apply_custom_css()

# Header
st.markdown('<h1 class="main-header">⚡ HFT Arbitrage Lab</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced High-Frequency Trading Strategy Development & Backtesting Platform</p>', unsafe_allow_html=True)

# GitHub and Contact Info
col_gh, col_disclaimer = st.columns([1, 2])

with col_gh:
    st.markdown("""
    <div style='padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; text-align: center;'>
        <h3 style='color: white; margin: 0;'>📦 Open Source</h3>
        <p style='color: white; margin: 0.5rem 0;'>
            <a href='https://github.com/ThotDjehuty/rust-hft-arbitrage-lab' target='_blank' style='color: white; text-decoration: none; font-weight: bold;'>
                ⭐ GitHub Repository
            </a>
        </p>
        <p style='color: white; font-size: 0.9rem; margin: 0;'>Made by <strong>ThotDjehuty</strong></p>
    </div>
    """, unsafe_allow_html=True)

with col_disclaimer:
    st.warning("""
    **⚠️ RISK DISCLAIMER**
    
    This platform is for **educational and research purposes only**. Trading financial instruments involves substantial risk of loss. 
    
    - Not financial advice or recommendations
    - Past performance does not guarantee future results
    - Use at your own risk - the author assumes no liability
    - Always paper trade first and understand the strategies
    - Consult a licensed financial advisor before trading
    """)

st.markdown("---")

# System Status
st.markdown("## 🖥️ System Status")

col1, col2, col3, col4 = st.columns(4)

try:
    from python.rust_grpc_bridge import rust_connector as rust_connector  # type: ignore
    rust_status = "🟢 ENABLED (gRPC backend)"
    rust_class = "status-online"
    rust_available = True
except:
    rust_status = "🟡 FALLBACK (numpy/pandas)"
    rust_class = "status-offline"
    rust_available = False

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

# Show prominent message if Rust is not available
if not rust_available:
    st.error("""
    **⚠️ RUST ACCELERATION DISABLED - Performance will be significantly slower!**
    
    To enable 10-100x performance boost, build the Rust engine:
    ```bash
    cd rust_connector
    maturin develop --release
    ```
    Or use the full restart script: `./scripts/restart_all.sh`
    """)
else:
    st.success("""
    **🚀 RUST ACCELERATION ENABLED** - Enjoying 10-100x performance boost on computations!
    """)

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

st.info("""
**📖 How to Use This Platform:**

1. **Load Data** → Go to Data Loader and fetch historical price data for your assets
2. **Explore Labs** → Choose a strategy lab (Mean Reversion, Portfolio Analytics, etc.)
3. **Analyze** → Run analysis with interactive parameters and visualizations
4. **Optimize** → Fine-tune strategy parameters using backtesting
5. **Save & Export** → Save portfolios and export results for further analysis

💡 **Tip:** Start with the Portfolio Analytics Lab to create and optimize a diversified portfolio!
""")

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

# Strategy Explanations
st.markdown("## 📚 Strategy Library & Mathematical Foundations")

st.markdown("""
Explore the mathematical foundations and implementations of each trading strategy. 
Click on any strategy to learn how it works and navigate to its dedicated lab.
""")

with st.expander("📉 **Mean Reversion Strategy**", expanded=False):
    st.markdown("""
    ### Overview
    Mean reversion strategies exploit the tendency of asset prices to revert to their historical average.
    
    ### Mathematical Foundation
    
    **Z-Score Calculation:**
    """)
    st.latex(r"Z = \frac{P_t - \mu}{\sigma}")
    st.markdown(r"""
    Where:
    - $P_t$ = Current price
    - $\mu$ = Moving average (mean)
    - $\sigma$ = Standard deviation
    
    **Trading Signals:**
    - **Buy Signal**: $Z < -2$ (price below 2 standard deviations)
    - **Sell Signal**: $Z > 2$ (price above 2 standard deviations)
    - **Close Position**: $Z \\approx 0$ (price returns to mean)
    
    ### Implementation
    - Window: 20-50 periods (adjustable)
    - Entry threshold: Z-score ± 2
    - Exit: Mean crossing or opposite signal
    
    ### Best For
    - Range-bound markets
    - Pairs trading
    - Statistical arbitrage
    """)
    
    if st.button("🚀 Open Mean Reversion Lab", key="goto_meanrev"):
        st.switch_page("pages/lab_mean_reversion.py")

with st.expander("📊 **Portfolio Optimization (Markowitz)**", expanded=False):
    st.markdown("""
    ### Overview
    Modern Portfolio Theory (MPT) optimizes asset allocation to maximize expected return for a given risk level.
    
    ### Mathematical Foundation
    
    **Portfolio Return:**
    """)
    st.latex(r"R_p = \sum_{i=1}^{n} w_i \cdot R_i")
    
    st.markdown("**Portfolio Variance:**")
    st.latex(r"\sigma_p^2 = \sum_{i=1}^{n} \sum_{j=1}^{n} w_i w_j \sigma_i \sigma_j \rho_{ij}")
    
    st.markdown("**Sharpe Ratio:**")
    st.latex(r"SR = \frac{R_p - R_f}{\sigma_p}")
    
    st.markdown(r"""
    Where:
    - $w_i$ = Weight of asset $i$
    - $R_i$ = Expected return of asset $i$
    - $\sigma_i$ = Standard deviation of asset $i$
    - $\\rho_{ij}$ = Correlation between assets $i$ and $j$
    - $R_f$ = Risk-free rate
    
    ### Optimization Objective
    Maximize: $\\frac{R_p - R_f}{\sigma_p}$ subject to $\sum w_i = 1$
    
    ### Advanced Metrics
    - **Information Ratio**: Alpha generation vs benchmark
    - **Sortino Ratio**: Downside risk-adjusted returns
    - **Calmar Ratio**: Return vs maximum drawdown
    
    ### Best For
    - Multi-asset portfolios
    - Risk management
    - Long-term investing
    """)
    
    if st.button("🚀 Open Portfolio Analytics Lab", key="goto_portfolio"):
        st.switch_page("pages/lab_portfolio_analytics.py")

with st.expander("🌊 **Rough Heston Model**", expanded=False):
    st.markdown("""
    ### Overview
    The Rough Heston model extends the classic Heston stochastic volatility model with fractional Brownian motion.
    
    ### Mathematical Foundation
    
    **Asset Price Process:**
    """)
    st.latex(r"dS_t = \mu S_t dt + \sqrt{V_t} S_t dW_t^S")
    
    st.markdown("**Rough Volatility Process:**")
    st.latex(r"V_t = V_0 + \frac{1}{\Gamma(\alpha)} \int_0^t (t-s)^{\alpha-1} \kappa(\theta - V_s)ds + \xi \int_0^t (t-s)^{\alpha-1} \sqrt{V_s} dW_s^V")
    
    st.markdown(r"""
    Where:
    - $S_t$ = Asset price
    - $V_t$ = Instantaneous variance
    - $\\alpha$ = Hurst parameter (< 0.5 for rough paths)
    - $\kappa$ = Mean reversion speed
    - $\\theta$ = Long-term variance
    - $\\xi$ = Volatility of volatility
    - $\\rho$ = Correlation between price and volatility
    
    ### Key Features
    - Captures volatility clustering
    - Models extreme events better than standard models
    - Rough paths (H < 0.5) match empirical data
    
    ### Applications
    - Options pricing
    - Volatility forecasting
    - Risk management
    """)
    
    if st.button("🚀 Open Rough Heston Lab", key="goto_heston"):
        st.switch_page("pages/lab_rough_heston.py")

with st.expander("🌀 **Chiarella Model (Agent-Based)**", expanded=False):
    st.markdown("""
    ### Overview
    Agent-based model simulating market dynamics through interaction of fundamentalists, chartists, and noise traders.
    
    ### Mathematical Foundation
    
    **Price Dynamics:**
    """)
    st.latex(r"\frac{dP_t}{dt} = \lambda(D_t^f + D_t^c + D_t^n)")
    
    st.markdown("""
    **Agent Demands:**
    
    *Fundamentalists (mean reversion):*
    """)
    st.latex(r"D_t^f = \gamma_f (P^* - P_t)")
    
    st.markdown("*Chartists (trend following):*")
    st.latex(r"D_t^c = \gamma_c \cdot \text{Trend}(P_t)")
    
    st.markdown("*Noise Traders (random):*")
    st.latex(r"D_t^n = \epsilon_t \sim \mathcal{N}(0, \sigma_n^2)")
    
    st.markdown(r"""
    Where:
    - $P^*$ = Fundamental value
    - $\gamma_f$, $\gamma_c$ = Agent aggressiveness parameters
    - $\lambda$ = Market liquidity
    
    ### Market Regimes
    - **Stable**: Fundamentalists dominate
    - **Trending**: Chartists dominate
    - **Chaotic**: Balanced competition
    
    ### Applications
    - Market microstructure analysis
    - Regime detection
    - Behavioral finance insights
    """)
    
    if st.button("🚀 Open Chiarella Lab", key="goto_chiarella"):
        st.switch_page("pages/lab_chiarella.py")

with st.expander("✍️ **Signature Methods**", expanded=False):
    st.markdown("""
    ### Overview
    Path signature methods transform time series into a feature-rich representation for ML and pattern recognition.
    
    ### Mathematical Foundation
    
    **Path Signature:**
    """)
    st.latex(r"S(X)_{i_1,...,i_k} = \int_{0<t_1<...<t_k<T} dX^{i_1}_{t_1} \otimes ... \otimes dX^{i_k}_{t_k}")
    
    st.markdown("""
    **Truncated Signature (order N):**
    """)
    st.latex(r"S^{(N)}(X) = (1, S^{(1)}, S^{(2)}, ..., S^{(N)})")
    
    st.markdown("""
    Where:
    - $X_t$ = Path (price/return trajectory)
    - $S^{(k)}$ = k-th level signature terms
    - N = Truncation depth
    
    ### Key Properties
    - **Invariance**: Translation, time-reparameterization
    - **Information**: Captures all path properties
    - **Dimensionality**: Grows as $d^N$ (d = dimension, N = depth)
    
    ### Applications
    - Pattern recognition in price paths
    - Options pricing with path-dependent payoffs
    - Feature engineering for ML models
    - Order flow analysis
    
    ### Computational Notes
    - Uses logarithmic signatures for numerical stability
    - Signature depth typically N=2-4 for financial data
    """)
    
    if st.button("🚀 Open Signature Methods Lab", key="goto_signature"):
        st.switch_page("pages/lab_signature_optimal_stopping.py")

with st.expander("📈 **PCA Arbitrage**", expanded=False):
    st.markdown("""
    ### Overview
    Principal Component Analysis identifies common factors driving multi-asset price movements for statistical arbitrage.
    
    ### Mathematical Foundation
    
    **Covariance Decomposition:**
    """)
    st.latex(r"\Sigma = V \Lambda V^T")
    
    st.markdown(r"""
    Where:
    - $\Sigma$ = Covariance matrix of returns
    - $V$ = Eigenvectors (principal components)
    - $\Lambda$ = Eigenvalues (explained variance)
    
    **Factor Loadings:**
    """)
    st.latex(r"R_t = \mu + \sum_{k=1}^K \beta_k F_{k,t} + \epsilon_t")
    
    st.markdown("""
    ### Trading Strategy
    1. Extract principal components from correlated assets
    2. Identify factor exposures (loadings)
    3. Trade deviations from factor model
    4. Construct market-neutral portfolios
    
    ### Applications
    - Sector rotation
    - Statistical arbitrage
    - Risk factor hedging
    """)
    
    if st.button("🚀 Open PCA Arbitrage Lab", key="goto_pca"):
        st.switch_page("pages/lab_pca_arbitrage.py")

with st.expander("🚀 **Momentum Trading**", expanded=False):
    st.markdown("""
    ### Overview
    Momentum strategies exploit the tendency of assets with strong recent performance to continue outperforming.
    
    ### Mathematical Foundation
    
    **Momentum Score:**
    """)
    st.latex(r"MOM_t = \frac{P_t - P_{t-n}}{P_{t-n}} \times 100")
    
    st.markdown("**Crossover Signals (Moving Averages):**")
    st.latex(r"\text{Signal} = \text{sign}(MA_{short} - MA_{long})")
    
    st.markdown("""
    **Relative Strength Index (RSI):**
    """)
    st.latex(r"RSI = 100 - \frac{100}{1 + RS}, \quad RS = \frac{\text{Avg Gain}}{\text{Avg Loss}}")
    
    st.markdown("""
    ### Strategy Types
    - **Trend Following**: Buy strong trends, sell weak
    - **Breakout**: Enter on support/resistance breaks
    - **Crossover**: MA crossovers as entry signals
    
    ### Position Sizing
    - **Fixed**: Equal allocation
    - **Volatility-Adjusted**: Inverse volatility weighting
    - **Kelly Criterion**: Optimal leverage
    
    ### Risk Management
    - Stop-loss at support levels
    - Trailing stops for trend capture
    - Position scaling based on conviction
    """)
    
    if st.button("🚀 Open Momentum Trading Lab", key="goto_momentum"):
        st.switch_page("pages/lab_momentum.py")

with st.expander("💱 **Market Making**", expanded=False):
    st.markdown("""
    ### Overview
    Provide liquidity by continuously quoting bid and ask prices, profiting from the spread.
    
    ### Mathematical Foundation
    
    **Avellaneda-Stoikov Model:**
    
    Optimal bid/ask quotes:
    """)
    st.latex(r"p^{bid} = s - \delta^{bid}, \quad p^{ask} = s + \delta^{ask}")
    
    st.markdown("**Spread Optimization:**")
    st.latex(r"\delta^* = \gamma \sigma^2 (T-t) + \frac{2}{\gamma} \ln(1 + \frac{\gamma}{k})")
    
    st.markdown(r"""
    Where:
    - $s$ = Mid price
    - $\delta$ = Half-spread
    - $\gamma$ = Risk aversion
    - $\sigma$ = Volatility
    - $k$ = Order arrival rate
    - $T-t$ = Time to horizon
    
    ### Inventory Management
    """)
    st.latex(r"q_{max} = \frac{Capital \times MaxAllocation}{Price}")
    
    st.markdown("""
    ### Key Concepts
    - **Spread**: Profit per round-trip trade
    - **Inventory Risk**: Exposure to price moves
    - **Adverse Selection**: Risk of informed traders
    
    ### P&L Components
    - Spread capture (revenue)
    - Inventory risk (cost)
    - Transaction costs
    """)
    
    if st.button("🚀 Open Market Making Lab", key="goto_marketmaking"):
        st.switch_page("pages/lab_market_making.py")

st.markdown("---")

# Persisted Datasets Info
if st.session_state.persisted_datasets:
    st.markdown("## 💾 Persisted Datasets")
    
    try:
        stats = get_storage_stats()
        
        col_ds1, col_ds2, col_ds3, col_ds4 = st.columns(4)
        
        with col_ds1:
            st.markdown('<div class="status-card" style="text-align: center;"><div class="metric-big">📦</div><strong>Saved Datasets</strong><br/>' + str(stats['total_datasets']) + '</div>', unsafe_allow_html=True)
        
        with col_ds2:
            st.markdown('<div class="status-card" style="text-align: center;"><div class="metric-big">📊</div><strong>Total Rows</strong><br/>' + f"{stats['total_rows']:,}" + '</div>', unsafe_allow_html=True)
        
        with col_ds3:
            st.markdown('<div class="status-card" style="text-align: center;"><div class="metric-big">🏷️</div><strong>Unique Symbols</strong><br/>' + str(stats['total_symbols']) + '</div>', unsafe_allow_html=True)
        
        with col_ds4:
            st.markdown('<div class="status-card" style="text-align: center;"><div class="metric-big">💽</div><strong>Storage Used</strong><br/>' + f"{stats['total_size_mb']} MB" + '</div>', unsafe_allow_html=True)
        
        with st.expander("📋 View Saved Datasets", expanded=False):
            datasets = list(st.session_state.persisted_datasets.values())[:5]  # Show first 5
            for ds in datasets:
                st.markdown(f"**{ds['name']}** - {ds['source']} | {len(ds['symbols'])} symbols | {ds['row_count']:,} rows")
            
            if len(st.session_state.persisted_datasets) > 5:
                st.info(f"...and {len(st.session_state.persisted_datasets) - 5} more. Go to Data Loader to manage all datasets.")
            
            if st.button("🔄 Refresh Dataset List", use_container_width=True):
                st.session_state.datasets_loaded = False
                st.rerun()
    
    except Exception:
        pass  # Silently skip if data directory doesn't exist
    
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

# Backend comparison section (if triggered)
try:
    from utils.backend_selector import render_backend_comparison
    render_backend_comparison()
except:
    pass

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
