"""
Superspace Anomaly Detection Lab
==================================

Advanced statistical arbitrage using supersymmetry, ghost fields, and topological invariants.
Implements 14D superspace (7 bosonic + 7 fermionic) for enhanced anomaly detection in financial time series.

Mathematical Framework:
- Supermanifolds & Grassmann Algebra
- Ghost Field Dynamics
- Chern-Simons Topological Invariants
- 14-Dimensional Market Modeling
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats
from statsmodels.tsa.stattools import coint
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import shared utilities
from utils.ui_components import apply_custom_css, render_sidebar_navigation, ensure_data_loaded
from utils.data_persistence import load_dataset, list_datasets, save_dataset
from python.data.data_fetcher import fetch_intraday_data

# Import Rust bindings
try:
    import hft_py
    superspace_rs = hft_py.superspace  # type: ignore  # Dynamic submodule
    RUST_AVAILABLE = True
except (ImportError, AttributeError) as e:
    st.error(f"⚠️ Rust superspace bindings not available: {e}")
    st.info("Please rebuild: `cd rust_python_bindings && maturin develop --release`")
    RUST_AVAILABLE = False
    superspace_rs = None

# Page config
st.set_page_config(
    page_title="Superspace Anomaly Lab",
    page_icon="🌌",
    layout="wide"
)

apply_custom_css()
render_sidebar_navigation(current_page="Superspace Anomaly Lab")

# Ensure data is loaded (will auto-load most recent dataset if needed)
data_available = ensure_data_loaded()

# Session state initialization
if 'superspace_params' not in st.session_state:
    st.session_state.superspace_params = {
        'threshold': 2.5,
        'alpha': 0.5,
        'beta': 0.5,
        'cs_window': 30,
        'ghost_noise': 0.1,
        'window_14d': 20,
    }

if 'superspace_results' not in st.session_state:
    st.session_state.superspace_results = None

if 'selected_assets' not in st.session_state:
    st.session_state.selected_assets = []

# Title and description
st.markdown('<h1 class="main-header">🌌 Superspace Anomaly Detection Lab</h1>', unsafe_allow_html=True)
st.markdown("""
<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;'>
    <h3 style='color: white; margin: 0 0 1rem 0;'>⚛️ Advanced Statistical Arbitrage using Theoretical Physics</h3>
    <p style='color: white; margin: 0; line-height: 1.6;'>
        Detect market anomalies using <strong>supersymmetry</strong>, <strong>ghost fields</strong>, 
        and <strong>topological invariants</strong> from quantum field theory.
        <br>
        <strong>14-Dimensional Superspace:</strong> 7 bosonic (price, volume, volatility, trend, momentum, liquidity, sentiment) 
        + 7 fermionic (ghost fields encoding hidden dynamics)
    </p>
</div>
""", unsafe_allow_html=True)

# Initialize tab state in session
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = 0

# Main tabs with programmatic selection
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📚 Theory",
    "📊 Stock Selection", 
    "⚙️ Parameters",
    "🔬 Analysis & Visualization",
    "📈 Portfolio Optimization",
    "🎯 Backtesting",
    "📖 Documentation"
])

# Helper function to switch tabs
def switch_to_tab(tab_index):
    """Switch to a specific tab by storing in session state"""
    st.session_state.current_tab = tab_index
    # Force scroll to top
    st.markdown(
        """
        <script>
        window.parent.document.querySelector('section.main').scrollTo(0, 0);
        </script>
        """,
        unsafe_allow_html=True
    )

# ============================================================================
# TAB 1: MATHEMATICAL THEORY
# ============================================================================
with tab1:
    st.header("📚 Mathematical Theory")
    
    theory_subtab1, theory_subtab2, theory_subtab3, theory_subtab4 = st.tabs([
        "Supermanifolds & Grassmann Algebra",
        "Ghost Fields",
        "Chern-Simons Theory",
        "14D Market Modeling"
    ])
    
    with theory_subtab1:
        st.subheader("🌐 Supermanifolds and Grassmann Algebra")
        
        st.markdown(r"""
        ### What is a Supermanifold?
        
        A **supermanifold** $\mathcal{M}$ extends classical manifolds by including both commuting 
        and anti-commuting coordinates:
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(r"""
            **Bosonic coordinates** $x^{\mu}$:
            
            $$x^{\mu} x^{\nu} = x^{\nu} x^{\mu}$$
            
            *(commuting - like classical variables)*
            """)
        
        with col2:
            st.markdown(r"""
            **Fermionic coordinates** $\theta^{\alpha}$:
            
            $$\theta^{\alpha} \theta^{\beta} = -\theta^{\beta} \theta^{\alpha}$$
            
            *(anti-commuting - quantum-like)*
            """)
        
        st.markdown(r"""
        ---
        
        **Point in Superspace:**
        
        $$\mathcal{P} = \left(x^0, x^1, \ldots, x^{d_b-1}, \theta^1, \theta^2, \ldots, \theta^{d_f}\right)$$
        
        where $d_b =$ bosonic dimensions, $d_f =$ fermionic dimensions.
        
        ### Grassmann Numbers
        
        Fermionic coordinates $\theta^{\alpha}$ are **Grassmann numbers** with special properties:
        
        **1. Anti-commutation:**
        
        $$\theta^{\alpha} \theta^{\beta} = -\theta^{\beta} \theta^{\alpha}$$
        
        **2. Nilpotency:**
        
        $$\left(\theta^{\alpha}\right)^2 = 0$$
        
        **3. Anticommutator vanishes:**
        
        $$\left\{\theta^{\alpha}, \theta^{\beta}\right\} = \theta^{\alpha} \theta^{\beta} + \theta^{\beta} \theta^{\alpha} = 0$$
        
        **Key Properties:**
        1. Any function of Grassmann variables terminates at finite order
        2. Integration and differentiation coincide: $\int d\theta^\alpha \, \theta^\alpha = 1$
        3. Encode fermionic degrees of freedom (spin-like)
        
        ### Superfields
        
        A **superfield** $\Phi(x,\theta)$ on the supermanifold:
        
        $$\Phi(x,\theta) = \phi(x) + \theta \psi(x) + \frac{1}{2}\theta^2 F(x)$$
        
        **Financial Interpretation:**
        - $\phi(t)$ = asset price at time $t$
        - $\psi(t)$ = encoded momentum/volatility information  
        - $F(t)$ = market stress/regime indicator
        
        ### Why Superspace for Finance?
        
        **Traditional:** Time series lives in $\mathbb{R}^n$  
        **Superspace:** Time series lives in $\mathbb{R}^{d_b} \times \mathbb{G}^{d_f}$ (with Grassmann manifold $\mathbb{G}$)
        
        **Advantages:**
        1. Captures **hidden correlations** through fermionic sector
        2. Natural **regime encoding** in ghost field dynamics
        3. **Topological invariants** detect structural changes
        4. **Supersymmetry** relates different market observables
        """)
        
        # Interactive Grassmann demo
        st.markdown("---")
        st.subheader("🧮 Interactive Grassmann Algebra")
        
        col1, col2 = st.columns(2)
        with col1:
            g1_scalar = st.number_input("θ₁ scalar part", value=1.0, key="g1_s")
            g1_grass = st.number_input("θ₁ Grassmann part", value=1.0, key="g1_g")
        with col2:
            g2_scalar = st.number_input("θ₂ scalar part", value=2.0, key="g2_s")
            g2_grass = st.number_input("θ₂ Grassmann part", value=1.0, key="g2_g")
        
        if RUST_AVAILABLE and superspace_rs is not None:
            assert superspace_rs is not None  # Type narrowing for PyLance
            # GrassmannNumber expects a Vec<f64> for grassmann parameter
            g1 = superspace_rs.PyGrassmannNumber(g1_scalar, [g1_grass])
            g2 = superspace_rs.PyGrassmannNumber(g2_scalar, [g2_grass])
            
            st.markdown("**Results:**")
            st.write(f"θ₁ = {g1}")
            st.write(f"θ₂ = {g2}")
            
            g_sum = g1 + g2
            g_product = g1 * g2
            g_anticomm = g1 * g2 + g2 * g1
            
            st.write(f"θ₁ + θ₂ = {g_sum}")
            st.write(f"θ₁ · θ₂ = {g_product}")
            st.write(f"{{θ₁, θ₂}} (anticommutator) = {g_anticomm}")
            st.write(f"Nilpotency: θ₁² = {g1 * g1}")
    
    with theory_subtab2:
        st.subheader("👻 Ghost Fields")
        
        st.markdown("""
        Ghost fields are auxiliary variables from **quantum field theory** that encode hidden market dynamics.
        They represent non-physical degrees of freedom that nevertheless affect observable quantities.
        """)
        
        st.markdown(r"""
        ### What are Ghost Fields?
        
        In **BRST quantization** and gauge theory, we introduce:
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(r"""
            **Ghost field** $c_i(t)$:
            - Encodes gauge redundancies
            - Captures regime transitions
            - Fermionic statistics
            """)
        
        with col2:
            st.markdown(r"""
            **Anti-ghost field** $\bar{c}_i(t)$:
            - Conjugate to ghost field
            - Ensures gauge invariance
            - Completes BRST symmetry
            """)
        
        st.markdown(r"""
        ---
        
        ### Ghost Field Dynamics (Langevin Equation)
        
        Ghost fields evolve according to stochastic dynamics:
        
        $$\frac{dc_i}{dt} = -\gamma c_i - \frac{\partial H}{\partial \bar{c}_i} + \xi_i(t)$$
        
        **Parameters:**
        - $\gamma$ = friction coefficient (damping)
        - $H$ = market Hamiltonian (energy function)
        - $\xi_i(t)$ = Gaussian white noise
        
        ### Market Hamiltonian
        
        The total energy combines bosonic (observable) and fermionic (ghost) sectors:
        
        $$H(p, q, c, \bar{c}) = \underbrace{\sum_i \left[\frac{p_i^2}{2m_i} + \frac{k}{2}q_i^2\right]}_{\text{Bosonic}} + \underbrace{\sum_{ij} \bar{c}_i M_{ij} c_j}_{\text{Fermionic}}$$
        
        **Bosonic variables:**
        - $p_i$ = price momentum (returns)
        - $q_i$ = normalized price deviation
        - $m_i$ = effective mass (inertia parameter)
        - $k$ = spring constant (mean-reversion)
        
        **Fermionic variables:**
        - $c_i, \bar{c}_i$ = ghost/anti-ghost fields
        - $M_{ij}$ = interaction matrix
        
        ---
        
        ### Ghost Field Observables
        
        **1. Divergence (Expansion/Contraction):**
        
        $$\nabla \cdot \mathbf{c}(t) = \sum_{i=1}^{7} \frac{\partial c_i}{\partial x^i}$$
        
        - $\nabla \cdot \mathbf{c} > 0$ → Expansion (overbought)
        - $\nabla \cdot \mathbf{c} < 0$ → Contraction (oversold)
        - $\nabla \cdot \mathbf{c} \approx 0$ → Equilibrium
        
        **2. Curl (Rotation):**
        
        $$\nabla \times \mathbf{c} = \begin{vmatrix} \mathbf{e}_1 & \mathbf{e}_2 & \mathbf{e}_3 \\ \frac{\partial}{\partial x^1} & \frac{\partial}{\partial x^2} & \frac{\partial}{\partial x^3} \\ c_1 & c_2 & c_3 \end{vmatrix}$$
        
        - High $|\nabla \times \mathbf{c}|$ → Rotational dynamics
        - Low $|\nabla \times \mathbf{c}|$ → Laminar flow
        
        ---
        
        ### BRST Symmetry
        
        Ghost fields respect **BRST symmetry** (gauge invariance):
        
        **BRST charge:**
        
        $$Q = \sum_{i=1}^{7} \bar{c}_i p_i$$
        
        **Nilpotency:**
        
        $$Q^2 = 0$$
        
        Ensures physical consistency and gauge invariance.
        
        ### Financial Interpretation
        
        - **Ghost fields** encode information not visible in prices alone
        - Capture **hidden order flow**, **sentiment shifts**, **regime changes**
        - **Divergence** signals market stress and potential reversals
        - **BRST invariance** ensures mathematical consistency
        """)
        
        st.markdown("---")
        st.subheader("🔬 Interactive Ghost Field Evolution")
        
        if RUST_AVAILABLE and superspace_rs is not None:
            assert superspace_rs is not None  # Type narrowing for PyLance
            st.markdown("""
            This demo shows how ghost fields track hidden market dynamics. 
            **High divergence** indicates non-equilibrium states and potential regime changes.
            """)
            
            col1, col2 = st.columns([2, 1])
            with col2:
                st.markdown("**Parameters:**")
                demo_gamma = st.slider("Friction (γ)", 0.05, 0.5, 0.1, 0.05, key="demo_gamma")
                demo_noise = st.slider("Noise amplitude", 0.05, 0.3, 0.1, 0.05, key="demo_noise")
                add_shock = st.checkbox("Add market shock at t=50", value=True)
            
            with col1:
                # Generate demo data
                np.random.seed(42)
                demo_price = np.cumsum(np.random.randn(100) * 0.5) + 100
                if add_shock:
                    demo_price[50:] += 5 + np.cumsum(np.random.randn(50) * 0.7)
                
                demo_params = superspace_rs.PyGhostFieldParams(
                    n_modes=7,
                    damping=demo_gamma,
                    noise_level=demo_noise,
                    spring_constant=1.0
                )
                
                # from_bosonic_coords expects Vec<Vec<f64>> (time series of vectors)
                # Create a simple 2D structure: each timestep has a single value
                bosonic_2d = [[float(p)] for p in demo_price[:7]]
                ghost_system = superspace_rs.PyGhostFieldSystem.from_bosonic_coords(
                    bosonic_2d,
                    demo_params
                )
                
                # Ghost system is pre-computed from bosonic coords
                # Access divergence and curl using time indices
                divergences = []
                curl_magnitudes = []
                
                n_times = min(len(demo_price), len(ghost_system.divergence))
                for i in range(n_times):
                    divergences.append(ghost_system.get_divergence_at(i))
                    curl_magnitudes.append(abs(ghost_system.get_curl_at(i)))
                
                # Create subplots
                fig = make_subplots(
                    rows=3, cols=1,
                    shared_xaxes=True,
                    subplot_titles=(
                        "Market Price",
                        "Ghost Field Divergence (∇·c) - Measures Expansion/Contraction",
                        "Ghost Field Curl (∇×c) - Measures Rotation/Circulation"
                    ),
                    vertical_spacing=0.08,
                    row_heights=[0.35, 0.35, 0.30]
                )
                
                # Price
                fig.add_trace(
                    go.Scatter(y=demo_price, name="Price", line=dict(color='blue', width=2)),
                    row=1, col=1
                )
                
                # Divergence with threshold
                fig.add_trace(
                    go.Scatter(y=divergences, name="Divergence", 
                              line=dict(color='red', width=2)),
                    row=2, col=1
                )
                div_threshold = np.mean(np.abs(divergences)) + np.std(divergences)
                # Add threshold lines (horizontal lines don't support row/col in subplots, use shapes instead)
                fig.add_hline(y=div_threshold, line_dash="dash", line_color="orange",
                             annotation_text="Alert threshold")
                fig.add_hline(y=-div_threshold, line_dash="dash", line_color="orange")
                
                # Curl magnitude
                fig.add_trace(
                    go.Scatter(y=curl_magnitudes, name="Curl Magnitude",
                              line=dict(color='purple', width=2)),
                    row=3, col=1
                )
                
                fig.update_layout(
                    height=700,
                    showlegend=True,
                    hovermode='x unified'
                )
                fig.update_xaxes(title_text="Time", row=3, col=1)
                fig.update_yaxes(title_text="Price ($)", row=1, col=1)
                fig.update_yaxes(title_text="∇·c", row=2, col=1)
                fig.update_yaxes(title_text="|∇×c|", row=3, col=1)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Insights
                max_div_idx = np.argmax(np.abs(divergences))
                st.info(f"""
                **Key Insights:**
                - Maximum divergence at t={max_div_idx} (price: ${demo_price[max_div_idx]:.2f})
                - Divergence > 0: Expansion (potential overbought)
                - Divergence < 0: Contraction (potential oversold)
                - High curl: Rotational dynamics (momentum shifts)
                """)
    
    with theory_subtab3:
        st.subheader("🌀 Chern-Simons Theory")
        
        st.markdown(r"""
        ### Topological Field Theory
        
        **Chern-Simons (CS) theory** is a topological quantum field theory in (2+1) dimensions.
        Unlike standard field theories, CS theory is **metric-independent** and captures 
        **topological invariants**.
        
        ### CS Action
        
        $$S_{CS}[A] = \frac{k}{4\pi} \int_M \text{Tr}\left(A \wedge dA + \frac{2}{3} A \wedge A \wedge A\right)$$
        
        where:
        - $A$ = gauge field (constructed from normalized price deviations)
        - $k$ = coupling constant
        - $M$ = spacetime manifold
        
        ### Discrete CS for Finance
        
        For discrete time series, the CS invariant is approximated:
        
        $$CS[A](t) \approx \sum_{i,j} \epsilon_{ijk} A_i(t) \left(\frac{\partial A_j}{\partial t} - \frac{\partial A_k}{\partial t}\right) + \frac{2}{3} A_i A_j A_k$$
        
        ### Why CS Theory for Markets?
        
        1. **Topological Stability:** Small noise doesn't affect the invariant
        2. **Regime Detection:** $|\Delta CS(t)|$ spikes at structural breaks
        3. **Non-local Information:** Captures long-range correlations
        4. **Gauge Independence:** Results don't depend on arbitrary normalization choices
        
        ### CS Change Detection
        
        We detect **topological transitions** when:
        
        $$|\Delta CS(t)| = |CS(t) - CS(t-1)| > \text{threshold}$$
        
        **High $|\Delta CS|$** → Regime change → Trading opportunity or risk
        
        ### Wilson Loops
        
        The **Wilson loop** for a closed path $C$:
        
        $$W[C] = \text{Tr}\left[\mathcal{P} \exp\left(\oint_C A\right)\right]$$
        
        provides another topological invariant encoding path-dependent behavior.
        
        ### Connection to Knot Theory
        
        CS theory is deeply connected to **knot invariants** (Jones polynomial).
        Market trajectories in superspace can be viewed as **knots** in configuration space.
        """)
        
        st.markdown("---")
        st.subheader("🎨 CS Invariant Visualization")
        
        if RUST_AVAILABLE and superspace_rs is not None:
            assert superspace_rs is not None  # Type narrowing for PyLance
            # Demo CS calculation
            demo_prices_cs = 100 + np.cumsum(np.random.randn(200) * 0.5)
            # Add a regime change
            demo_prices_cs[100:] += 10 + np.cumsum(np.random.randn(100) * 0.3)
            
            cs_calc = superspace_rs.PyChernSimonsCalculator(coupling=1.0, window=20)
            cs_series = cs_calc.calculate(demo_prices_cs.tolist())
            # Compute changes manually
            cs_changes = np.abs(np.diff(cs_series, prepend=cs_series[0]))
            
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                              subplot_titles=("Price with Regime Change", "Chern-Simons Invariant"))
            
            fig.add_trace(go.Scatter(y=demo_prices_cs, name="Price", line=dict(color='blue')), 
                         row=1, col=1)
            fig.add_trace(go.Scatter(y=cs_series, name="CS Invariant", line=dict(color='purple')), 
                         row=2, col=1)
            
            # Highlight transitions
            threshold = float(np.std(cs_changes) * 2)
            transitions = cs_calc.detect_transitions(cs_series, threshold)
            
            if len(transitions) > 0:
                for trans_idx in transitions:
                    if trans_idx < len(demo_prices_cs):
                        # Add vertical lines at transition points using vline (without row/col)
                        fig.add_vline(
                            x=trans_idx, 
                            line_dash="dash", 
                            line_color="red",
                            annotation_text="Transition"
                        )
            
            fig.update_layout(height=600, showlegend=True)
            fig.update_yaxes(title_text="Price", row=1, col=1)
            fig.update_yaxes(title_text="CS Value", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.success(f"✅ Detected {len(transitions)} topological transitions")
    
    with theory_subtab4:
        st.subheader("🔢 14-Dimensional Market Modeling")
        
        st.markdown(r"""
        ### Why 14 Dimensions?
        
        Our superspace has **14 dimensions**: 7 bosonic + 7 fermionic
        
        This choice is motivated by:
        1. **Completeness:** Captures all major market observables
        2. **Supersymmetry:** Equal bosonic and fermionic dimensions
        3. **Mathematical elegance:** 7 is the maximum dimension for division algebras (octonions)
        
        ### Bosonic Coordinates (Observable)
        
        $$\mathbf{X} = (x^0, x^1, x^2, x^3, x^4, x^5, x^6)$$
        
        1. $x^0$ = **Log Price:** $\ln(p_t)$
        2. $x^1$ = **Log Volume:** $\ln(V_t)$
        3. $x^2$ = **Volatility:** Rolling std of returns
        4. $x^3$ = **Trend:** Rolling regression slope
        5. $x^4$ = **Momentum:** Rate of change
        6. $x^5$ = **Liquidity:** $1/V_t$ (inverse volume)
        7. $x^6$ = **Sentiment:** RSI-like indicator
        
        ### Fermionic Coordinates (Ghost Fields)
        
        $$\mathbf{\Theta} = (\theta^1, \theta^2, \theta^3, \theta^4, \theta^5, \theta^6, \theta^7)$$
        
        Each $\theta^i$ is the **gradient** (time derivative) of the corresponding bosonic coordinate:
        
        $$\theta^i(t) = \frac{\partial x^i}{\partial t}$$
        
        **Interpretation:**
        - $\theta^1$ = Price momentum ghost
        - $\theta^2$ = Volume flow ghost
        - $\theta^3$ = Volatility ghost (rate of vol change)
        - $\theta^4$ = Trend acceleration
        - $\theta^5$ = Momentum ghost
        - $\theta^6$ = Liquidity shock
        - $\theta^7$ = Sentiment shift
        
        ### Full Superspace Point
        
        $$\mathcal{P}(t) = (\mathbf{X}(t), \mathbf{\Theta}(t)) \in \mathbb{R}^7 \times \mathbb{G}^7$$
        
        ### Distance Metric
        
        Euclidean distance in superspace:
        
        $$d(\mathcal{P}_1, \mathcal{P}_2) = \sqrt{\sum_{i=1}^{14} (x^i_1 - x^i_2)^2}$$
        
        (Grassmann components contribute via their coefficients)
        
        ### Dimensionality Reduction
        
        For visualization, we use **PCA** to project 14D → 2D/3D while preserving maximum variance.
        
        ### Anomaly Score
        
        The **unified anomaly score** combines ghost divergence and CS changes:
        
        $$\mathcal{A}(t) = \alpha \cdot Z[\nabla \cdot \mathbf{c}(t)] + \beta \cdot Z[|\Delta CS(t)|]$$
        
        where $Z[\cdot]$ is z-score normalization, and $\alpha, \beta$ are weights (default 0.5 each).
        
        **Threshold:** $\mathcal{A}(t) > \tau$ (typically $\tau = 2.5\sigma$)
        """)
    
    # Navigation button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col3:
        if st.button("➡️ Next: Select Stocks", type="primary", use_container_width=True, key="theory_next"):
            st.session_state.show_stock_selection_hint = True
            st.balloons()
            st.rerun()

# ============================================================================
# TAB 2: STOCK SELECTION
# ============================================================================
with tab2:
    st.header("📊 Stock Selection")
    
    # Show hint if coming from Theory tab
    if st.session_state.get('show_stock_selection_hint'):
        st.success("✅ Welcome! Select your assets below to continue.")
        st.session_state.show_stock_selection_hint = False
    
    # Show data status
    if data_available and st.session_state.historical_data is not None:
        df = st.session_state.historical_data
        if isinstance(df.columns, pd.MultiIndex):
            available_symbols = list(df.columns.get_level_values(0).unique())
        else:
            available_symbols = st.session_state.get('symbols', [])
        
        st.success(f"✅ Data loaded: {len(available_symbols)} symbols available")
        
        # Check if we have enough data points
        if len(df) < 200:
            st.warning(f"⚠️ Only {len(df)} data points available. Superspace analysis works best with 200+ data points.")
            st.info("💡 **Tip:** Load more historical data in the Data Loader (use 1-2 years of daily data or several months of hourly data)")
    else:
        st.error("⚠️ **No data loaded!** The Superspace Anomaly Lab requires historical price data.")
        
        st.markdown("""
        ### 📊 How to Load Data:
        
        **Option 1: Use Data Loader (Recommended)**
        1. Click the button below to go to Data Loader
        2. Enter stock symbols (e.g., `AAPL,NVDA,GOOGL,MSFT,AMZN`)
        3. Select date range (recommend at least **1 year** for daily data)
        4. Click "Fetch Data"
        5. Return to this page
        
        **Option 2: Load from File**
        1. Go to Data Loader
        2. Upload a CSV file with OHLCV data
        3. The file should have columns: Date, Open, High, Low, Close, Volume
        
        **Minimum Requirements:**
        - At least **200 data points** per symbol (preferably 500+)
        - OHLCV data (Open, High, Low, Close, Volume)
        - Multiple symbols recommended (3-10 works well)
        
        **Recommended Symbols for Testing:**
        - Tech: `AAPL,MSFT,GOOGL,NVDA,META`
        - Finance: `JPM,BAC,GS,MS,C`
        - Mixed: `AAPL,JPM,TSLA,AMZN,XOM`
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Go to Data Loader", type="primary", use_container_width=True):
                st.switch_page("pages/data_loader.py")
        with col2:
            if st.button("📖 View Documentation", use_container_width=True):
                st.info("See the 'Documentation' tab for detailed instructions")
        
        st.stop()
    
    st.markdown("""
    Select stocks for superspace anomaly detection. You can:
    1. **Manual Selection:** Choose specific stocks
    2. **Pair Selection:** Find cointegrated pairs
    3. **Basket Selection:** Multi-asset portfolio
    
    💡 **New Feature:** If you select symbols without data, you can automatically fetch the last 7 days with one click!
    """)
    
    # Get available symbols from loaded data
    if 'historical_data' in st.session_state and st.session_state.historical_data is not None:
        df = st.session_state.historical_data
        if isinstance(df.columns, pd.MultiIndex):
            available_symbols = list(df.columns.get_level_values(0).unique())
        else:
            available_symbols = st.session_state.get('symbols', [])
    else:
        available_symbols = []
    
    if not available_symbols:
        st.error("⚠️ No symbols available. Please load data from the Data Loader page.")
        st.stop()
    
    # Use available data
    data = st.session_state.historical_data
    symbols = available_symbols
    
    selection_mode = st.radio("Selection Mode:", 
                              ["Auto-Detect", "Manual", "Auto-Pair", "Basket"],
                              horizontal=True)
    
    if selection_mode == "Auto-Detect":
        st.subheader("🤖 Automatic Stock Detection & Selection")
        
        st.markdown("""
        **AI-Powered Stock Selection** - Automatically identify suitable stocks for arbitrage:
        - ✓ Cointegrated pairs (mean-reverting relationships)
        - ✓ Correlation clusters (sector baskets)
        - ✓ High volatility candidates (anomaly detection)
        - ✓ Adequate data coverage (200+ points)
        """)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            detection_method = st.selectbox(
                "Detection Method:",
                ["Best Pairs", "Best Basket", "High Volatility Singles", "All Suitable"]
            )
        with col2:
            min_correlation = st.slider("Min Correlation:", 0.5, 0.95, 0.7, 0.05)
        with col3:
            max_assets = st.number_input("Max Assets:", 2, 20, 5, 1)
        
        if st.button("🔍 Auto-Detect Suitable Assets", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            eta_text = st.empty()
            
            # Initialize Rust-based parallel analyzer
            import time
            try:
                import hft_py
                RustAnalyzer = hft_py.statistical_analyzer.StatisticalAnalyzer
                use_rust = True
                analyzer = RustAnalyzer()
                st.success("🚀 Using Rust parallel processing (high-performance mode)")
            except (ImportError, AttributeError) as e:
                st.warning(f"⚠️ Rust accelerator unavailable, using Python (slower): {e}")
                use_rust = False
            
            status_text.text("📊 Extracting price data...")
            progress_bar.progress(0.05)
            start_time = time.time()
            
            # Extract close prices as lists for Rust compatibility
            close_prices = {}
            for sym in symbols:
                try:
                    if isinstance(data.columns, pd.MultiIndex):
                        prices = data[sym]['close'].dropna()
                    else:
                        prices = data['close'].dropna()
                    close_prices[sym] = prices.values.tolist()
                except:
                    pass
            
            progress_bar.progress(0.1)
            
            # Filter symbols with sufficient data
            if use_rust:
                status_text.text("🔍 Filtering valid symbols (Rust parallel)...")
                filter_start = time.time()
                valid_symbols = analyzer.filter_valid_symbols(close_prices)
                filter_time = time.time() - filter_start
                eta_text.text(f"⚡ Validation: {filter_time:.2f}s")
            else:
                valid_symbols = {k: v for k, v in close_prices.items() if len(v) >= 200}
            
            if len(valid_symbols) < 2:
                st.error(f"⚠️ Insufficient data! Need at least 2 symbols with 200+ data points. Found: {len(valid_symbols)}")
                st.info("💡 Load more data using the Data Loader")
                if st.button("📥 Download More Data", key="download_more"):
                    st.switch_page("pages/data_loader.py")
                st.stop()
            
            status_text.text(f"✓ {len(valid_symbols)} symbols validated")
            progress_bar.progress(0.3)
            
            # Compute correlation matrix
            if use_rust:
                status_text.text("🔄 Computing correlations (Rust parallel)...")
                corr_start = time.time()
                
                # Create progress callback for ETA
                def progress_callback(current, total, eta):
                    percent = 0.3 + (0.2 * current / total)
                    progress_bar.progress(percent)
                    eta_text.text(f"⏱️ ETA: {int(eta)}s remaining ({current}/{total} pairs)")
                
                symbols_list, corr_matrix_list = analyzer.compute_correlation_matrix(valid_symbols)
                corr_matrix = pd.DataFrame(corr_matrix_list, index=symbols_list, columns=symbols_list)
                
                corr_time = time.time() - corr_start
                eta_text.text(f"⚡ Correlation matrix: {corr_time:.2f}s")
            else:
                status_text.text("🔄 Computing correlations...")
                close_df = pd.DataFrame(close_prices).dropna()
                corr_matrix = close_df.corr()
            
            progress_bar.progress(0.5)
            
            # Compute volatility
            if use_rust:
                status_text.text("📈 Computing volatility (Rust parallel)...")
                vol_start = time.time()
                volatility_rankings = analyzer.compute_volatility_rankings(valid_symbols)
                volatility = pd.Series({r.symbol: r.volatility for r in volatility_rankings})
                vol_time = time.time() - vol_start
                eta_text.text(f"⚡ Volatility: {vol_time:.2f}s")
            else:
                returns = pd.DataFrame(close_prices).pct_change().dropna()
                volatility = returns.std()
            
            progress_bar.progress(0.6)
            
            selected_assets = []
            
            if detection_method == "Best Pairs":
                if use_rust:
                    status_text.text("🔗 Finding best cointegrated pairs (Rust parallel)...")
                    pairs_start = time.time()
                    
                    # Create progress callback with ETA
                    def pairs_progress(current, total, eta):
                        percent = 0.6 + (0.3 * current / total)
                        progress_bar.progress(percent)
                        eta_text.text(f"⏱️ ETA: {int(eta)}s | Testing pairs: {current}/{total}")
                    
                    # Get symbols list for Rust call
                    symbols_list = list(corr_matrix.columns)
                    corr_list = corr_matrix.values.tolist()
                    
                    cointegrated = analyzer.find_cointegrated_pairs(
                        valid_symbols, corr_list, symbols_list, pairs_progress
                    )
                    
                    pairs_time = time.time() - pairs_start
                    eta_text.text(f"⚡ Cointegration testing: {pairs_time:.2f}s ({len(cointegrated)} pairs found)")
                    
                    if cointegrated:
                        # Sort by p-value
                        cointegrated.sort(key=lambda x: x.p_value)
                        best = cointegrated[0]
                        selected_assets = [best.symbol1, best.symbol2]
                        
                        st.success(f"✅ Found {len(cointegrated)} cointegrated pairs in {pairs_time:.1f}s!")
                        st.info(f"**Best Pair:** {best.symbol1} - {best.symbol2}")
                        st.write(f"- Cointegration p-value: {best.p_value:.4f}")
                        st.write(f"- Correlation: {best.correlation:.3f}")
                        st.write(f"- Test statistic: {best.test_statistic:.3f}")
                    else:
                        st.warning("No cointegrated pairs found. Try lowering the correlation threshold.")
                else:
                    # Python fallback
                    status_text.text("🔗 Finding best cointegrated pairs...")
                    pairs_scores = []
                    
                    for i, sym1 in enumerate(valid_symbols):
                        for sym2 in list(valid_symbols.keys())[i+1:]:
                            corr = float(corr_matrix.at[sym1, sym2])  # type: ignore[arg-type]
                            if corr >= min_correlation:
                                try:
                                    _, p_value, _ = coint(close_prices[sym1], close_prices[sym2])
                                    if p_value < 0.05:
                                        pairs_scores.append((sym1, sym2, p_value, corr))
                                except:
                                    pass
                    
                    progress_bar.progress(0.8)
                    
                    if pairs_scores:
                        pairs_scores.sort(key=lambda x: x[2])
                        best_pair = pairs_scores[0]
                        selected_assets = [best_pair[0], best_pair[1]]
                        
                        st.success(f"✅ Found {len(pairs_scores)} cointegrated pairs!")
                        st.info(f"**Best Pair:** {best_pair[0]} - {best_pair[1]}")
                        st.write(f"- Cointegration p-value: {best_pair[2]:.4f}")
                        st.write(f"- Correlation: {best_pair[3]:.3f}")
                    else:
                        st.warning("No cointegrated pairs found. Try lowering the correlation threshold.")
                    
            elif detection_method == "Best Basket":
                if use_rust:
                    status_text.text("🗂️ Building optimal basket (Rust parallel)...")
                    basket_start = time.time()
                    
                    # Progress callback for basket building
                    def basket_progress(current, total, eta):
                        percent = 0.6 + (0.3 * current / total)
                        progress_bar.progress(percent)
                        eta_text.text(f"⏱️ ETA: {int(eta)}s | Building basket: {current}/{total} assets")
                    
                    symbols_list = list(corr_matrix.columns)
                    corr_list = corr_matrix.values.tolist()
                    vol_list = [volatility[s] for s in symbols_list]
                    
                    basket_result = analyzer.build_optimal_basket(
                        corr_list, vol_list, symbols_list, max_assets, basket_progress
                    )
                    
                    selected_assets = basket_result.symbols
                    basket_time = time.time() - basket_start
                    eta_text.text(f"⚡ Basket built: {basket_time:.2f}s")
                    
                    if len(selected_assets) >= 3:
                        st.success(f"✅ Built basket with {len(selected_assets)} assets in {basket_time:.1f}s!")
                        st.info(f"**Average intra-basket correlation:** {basket_result.avg_correlation:.3f}")
                        st.info(f"**Average volatility:** {basket_result.avg_volatility*100:.2f}%")
                    else:
                        st.warning(f"Only found {len(selected_assets)} suitable assets. Try lowering correlation threshold.")
                else:
                    # Python fallback
                    status_text.text("🗂️ Building optimal basket...")
                    
                    cluster_center = volatility.idxmax()
                    basket = [cluster_center]
                    
                    remaining = [s for s in valid_symbols if s != cluster_center]
                    
                    while len(basket) < max_assets and remaining:
                        best_score = -1
                        best_sym = None
                        
                        for sym in remaining:
                            avg_corr_raw = corr_matrix.loc[sym, basket].mean()
                            avg_corr = float(avg_corr_raw) if isinstance(avg_corr_raw, (int, float)) else float(avg_corr_raw.iloc[0]) if hasattr(avg_corr_raw, 'iloc') else float(list(avg_corr_raw)[0])
                            if avg_corr > best_score and avg_corr >= min_correlation:
                                best_score = avg_corr
                                best_sym = sym
                        
                        if best_sym:
                            basket.append(best_sym)
                            remaining.remove(best_sym)
                        else:
                            break
                
                selected_assets = basket
                progress_bar.progress(0.9)
                
                if len(selected_assets) >= 3:
                    st.success(f"✅ Built basket with {len(selected_assets)} assets!")
                    avg_correlation = corr_matrix.loc[selected_assets, selected_assets].values[np.triu_indices(len(selected_assets), k=1)].mean()
                    st.info(f"**Average intra-basket correlation:** {avg_correlation:.3f}")
                else:
                    st.warning(f"Only found {len(selected_assets)} suitable assets. Try lowering correlation threshold.")
                    
            elif detection_method == "High Volatility Singles":
                status_text.text("📈 Identifying high volatility candidates...")
                
                if use_rust:
                    # Already computed in parallel earlier
                    selected_assets = [r.symbol for r in volatility_rankings[:max_assets]]
                    progress_bar.progress(0.9)
                    eta_text.text(f"⚡ Instant selection from pre-computed rankings")
                    
                    st.success(f"✅ Selected {len(selected_assets)} high-volatility assets!")
                    st.write("**Volatility Rankings:**")
                    for i, sym in enumerate(selected_assets):
                        rank_info = next((r for r in volatility_rankings if r.symbol == sym), None)
                        if rank_info:
                            st.write(f"#{i+1}. {sym}: {rank_info.volatility*100:.2f}% daily std")
                else:
                    # Python fallback
                    top_volatile = volatility.nlargest(min(max_assets, len(volatility)))
                    selected_assets = top_volatile.index.tolist()
                    progress_bar.progress(0.9)
                    
                    st.success(f"✅ Selected {len(selected_assets)} high-volatility assets!")
                    st.write("**Volatility Rankings:**")
                    for sym in selected_assets:
                        vol_value = float(volatility[sym])
                        st.write(f"- {sym}: {vol_value*100:.2f}% daily std")
                    
            else:  # All Suitable
                if use_rust:
                    status_text.text("🎯 Finding all suitable combinations (Rust parallel)...")
                    all_start = time.time()
                    
                    # Progress callback
                    def all_progress(current, total, eta):
                        percent = 0.6 + (0.3 * current / total)
                        progress_bar.progress(percent)
                        eta_text.text(f"⏱️ ETA: {int(eta)}s | Analyzed: {current}/{total}")
                    
                    symbols_list = list(corr_matrix.columns)
                    corr_list = corr_matrix.values.tolist()
                    
                    selected_assets = analyzer.find_all_suitable(
                        valid_symbols, corr_list, symbols_list, max_assets, all_progress
                    )
                    
                    all_time = time.time() - all_start
                    progress_bar.progress(0.9)
                    eta_text.text(f"⚡ Complete analysis: {all_time:.2f}s")
                    
                    st.success(f"✅ Found {len(selected_assets)} suitable assets in {all_time:.1f}s!")
                else:
                    # Python fallback
                    status_text.text("🎯 Finding all suitable combinations...")
                    
                    all_candidates = set()
                    
                    # Add cointegrated pairs
                    for i, sym1 in enumerate(valid_symbols):
                        for sym2 in list(valid_symbols.keys())[i+1:]:
                            if float(corr_matrix.at[sym1, sym2]) >= min_correlation:  # type: ignore[arg-type]
                                try:
                                    _, p_value, _ = coint(close_prices[sym1], close_prices[sym2])
                                    if p_value < 0.05:
                                        all_candidates.add(sym1)
                                        all_candidates.add(sym2)
                                except:
                                    pass
                    
                    # Add top volatile
                    top_vol = volatility.nlargest(5).index
                    all_candidates.update(top_vol)
                    
                    selected_assets = list(all_candidates)[:max_assets]
                    progress_bar.progress(0.9)
                    
                    st.success(f"✅ Found {len(selected_assets)} suitable assets!")
            
            # Display results
            if selected_assets:
                st.session_state.selected_assets = selected_assets
                progress_bar.progress(1.0)
                
                total_time = time.time() - start_time
                if use_rust:
                    status_text.success(f"✓ Analysis complete in {total_time:.1f}s (Rust accelerated)")
                    eta_text.success(f"🚀 Performance: {len(selected_assets)} assets analyzed with parallel processing")
                else:
                    status_text.success(f"✓ Analysis complete in {total_time:.1f}s")
                
                st.markdown("---")
                st.subheader("📋 Selected Assets")
                
                cols = st.columns(min(len(selected_assets), 5))
                for idx, sym in enumerate(selected_assets):
                    with cols[idx % 5]:
                        vol_value = float(volatility.get(sym, volatility.get(str(sym), 0.0)))
                        st.metric(str(sym), f"{vol_value*100:.1f}%", "volatility")
                
                # Check for missing data
                missing_symbols = [str(s) for s in selected_assets if s not in valid_symbols]
                if missing_symbols:
                    st.warning(f"⚠️ Some symbols have insufficient data: {', '.join(missing_symbols)}")
                    if st.button("📥 Download Missing Data", key="download_missing"):
                        st.session_state.download_symbols = missing_symbols
                        st.switch_page("pages/data_loader.py")
                
                st.markdown("---")
                st.info("✅ Assets selected! Proceed to the **Parameters** tab to configure and run analysis.")
            else:
                progress_bar.progress(1.0)
                status_text.warning("No suitable assets found with current criteria.")
                st.info("💡 Try: Lower correlation threshold or fetch more diverse symbols")
                
                if st.button("📥 Load More Symbols"):
                    st.switch_page("pages/data_loader.py")
    
    elif selection_mode == "Manual":
            st.subheader("🎯 Manual Stock Selection")
            
            # Use session state as default and key for persistence
            if 'manual_selection' not in st.session_state:
                st.session_state.manual_selection = st.session_state.selected_assets if st.session_state.selected_assets else []
            
            selected = st.multiselect(
                "Select stocks (1 for single analysis, 2 for pairs, 3+ for portfolio):",
                options=symbols,
                default=st.session_state.manual_selection,
                key="manual_stock_selector"
            )
            
            # Update both manual selection and selected_assets
            if selected != st.session_state.manual_selection:
                st.session_state.manual_selection = selected
                st.session_state.selected_assets = selected
            
            if selected:
                st.success(f"✅ Selected {len(selected)} asset(s): {', '.join(selected)}")
                
                # Show preview
                st.subheader("Data Preview")
                for symbol in selected:
                    with st.expander(f"📈 {symbol}"):
                        try:
                            if isinstance(data.columns, pd.MultiIndex):
                                symbol_data = data[symbol].copy()
                            else:
                                symbol_data = data.copy()
                            
                            # Reset index to avoid duplicate column issues
                            if isinstance(symbol_data.index, pd.DatetimeIndex):
                                symbol_data = symbol_data.reset_index()
                            
                            st.write(f"**Shape:** {symbol_data.shape}")
                            if len(symbol_data) > 0:
                                st.write(f"**Records:** {len(symbol_data)}")
                                # Show first and last date if timestamp column exists
                                if 'timestamp' in symbol_data.columns:
                                    st.write(f"**Date range:** {symbol_data['timestamp'].iloc[0]} to {symbol_data['timestamp'].iloc[-1]}")
                            
                            # Display only first 10 rows to avoid issues
                            st.dataframe(symbol_data.head(10))
                        except Exception as e:
                            st.error(f"Error displaying preview for {symbol}: {str(e)}")
                            st.write("Data shape:", data.shape if hasattr(data, 'shape') else 'N/A')
    
    elif selection_mode == "Auto-Pair":
            st.subheader("🔗 Automatic Pair Selection")
            
            st.markdown("""
            Find cointegrated pairs using Engle-Granger test.
            **Cointegration** means two non-stationary series share a common stochastic trend.
            """)
            
            min_p_value = st.slider("Maximum p-value (significance level):", 
                                   0.001, 0.1, 0.05, 0.001,
                                   help="Lower = stricter cointegration requirement")
            
            if st.button("🔍 Find Cointegrated Pairs"):
                with st.spinner("Analyzing pairs..."):
                    pairs = []
                    
                    for i, sym1 in enumerate(symbols):
                        for sym2 in symbols[i+1:]:
                            try:
                                # Handle multi-level columns
                                if isinstance(data.columns, pd.MultiIndex):
                                    df1 = data[sym1]
                                    df2 = data[sym2]
                                else:
                                    continue  # Skip if not multi-level
                                
                                # Align data
                                common_idx = df1.index.intersection(df2.index)
                                if len(common_idx) < 50:
                                    continue
                                
                                p1 = df1.loc[common_idx, 'close'].values
                                p2 = df2.loc[common_idx, 'close'].values
                                
                                # Cointegration test
                                score, p_value, _ = coint(p1, p2)
                                
                                if p_value < min_p_value:
                                    # Compute correlation for additional info
                                    corr = np.corrcoef(p1, p2)[0, 1]
                                    pairs.append({
                                        'Pair': f"{sym1}/{sym2}",
                                        'P-Value': p_value,
                                        'Score': score,
                                        'Correlation': corr,
                                        'Symbol1': sym1,
                                        'Symbol2': sym2
                                    })
                            except:
                                continue
                    
                    if pairs:
                        pairs_df = pd.DataFrame(pairs).sort_values('P-Value')
                        st.success(f"✅ Found {len(pairs)} cointegrated pairs")
                        st.dataframe(pairs_df.style.format({
                            'P-Value': '{:.4f}',
                            'Score': '{:.2f}',
                            'Correlation': '{:.3f}'
                        }))
                        
                        # Select pair
                        selected_pair_idx = st.selectbox(
                            "Select a pair:",
                            range(len(pairs_df)),
                            format_func=lambda x: pairs_df.iloc[x]['Pair']
                        )
                        
                        if st.button("Use Selected Pair"):
                            pair_row = pairs_df.iloc[selected_pair_idx]
                            st.session_state.selected_assets = [pair_row['Symbol1'], pair_row['Symbol2']]
                            st.success(f"✅ Selected pair: {pair_row['Pair']}")
                    else:
                        st.warning(f"No cointegrated pairs found with p-value < {min_p_value}")
        
    else:  # Basket mode
        st.subheader("🗂️ Basket Selection")
        
        st.markdown("Create a basket of assets based on sector, correlation, or custom criteria.")
        
        basket_mode = st.radio("Basket Mode:", ["By Sector", "By Correlation", "Custom"], horizontal=True)
        
        if basket_mode == "By Sector":
            st.markdown("""
            **Sector-based selection** groups assets by industry classification.
            This helps capture sector-specific dynamics and correlations.
            """)
            
            # Extract unique sectors from symbols (basic heuristic)
            # In production, you'd have proper sector metadata
            sector_keywords = {
                    'Tech': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AMD', 'INTC', 'TSLA'],
                    'Finance': ['JPM', 'BAC', 'GS', 'MS', 'C', 'WFC', 'BLK', 'V', 'MA'],
                    'Healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'LLY', 'TMO', 'MRK', 'DHR'],
                    'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO'],
                    'Consumer': ['AMZN', 'WMT', 'HD', 'NKE', 'MCD', 'SBUX', 'TGT', 'COST']
                }
                
            # Classify available symbols
            sector_assets = {}
            for sector, keywords in sector_keywords.items():
                matches = [s for s in symbols if any(k in s for k in keywords)]
                if matches:
                    sector_assets[sector] = matches
            
            # Add "Other" category for unclassified
            classified = set()
            for assets in sector_assets.values():
                classified.update(assets)
            other = [s for s in symbols if s not in classified]
            if other:
                sector_assets['Other'] = other
            
            if sector_assets:
                selected_sector = st.selectbox(
                    "Select sector:",
                    options=list(sector_assets.keys())
                )
                
                assets_in_sector = sector_assets[selected_sector]
                st.info(f"📊 {len(assets_in_sector)} assets available in {selected_sector}")
                
                n_assets = st.slider(
                    "Number of assets to select:",
                    min_value=3,
                    max_value=min(10, len(assets_in_sector)),
                    value=min(5, len(assets_in_sector))
                )
                
                # Auto-select top N by liquidity/volume if available
                selected_basket = assets_in_sector[:n_assets]
                
                st.write(f"**Selected assets:** {', '.join(selected_basket)}")
                
            if st.button("✅ Create Sector Basket"):
                    st.session_state.selected_assets = selected_basket
                    st.success(f"✅ Created {selected_sector} sector basket with {len(selected_basket)} assets")
                    st.rerun()
            else:
                st.warning("No sector classifications available. Please use Custom mode.")
        
        elif basket_mode == "By Correlation":
                st.markdown("""
                **Correlation-based selection** finds assets with similar or contrarian price movements.
                - **High correlation (>0.7):** Similar movements (sector ETFs, related stocks)
                - **Low correlation (<0.3):** Diversification candidates
                - **Negative correlation (<-0.3):** Hedging pairs
                """)
                
                corr_type = st.radio(
                    "Correlation preference:",
                    ["High Correlation (>0.7)", "Medium Correlation (0.3-0.7)", "Low Correlation (<0.3)"],
                    horizontal=False
                )
                
                n_basket = st.slider("Basket size:", 3, min(10, len(symbols)), 5)
                
                if st.button("🔍 Find Correlated Assets", type="primary"):
                    with st.spinner("Computing correlations..."):
                        # Build correlation matrix
                        price_data = {}
                        common_dates = None
                        
                        for sym in symbols:
                            try:
                                if isinstance(data.columns, pd.MultiIndex):
                                    symbol_data = data[sym]
                                    if 'close' in symbol_data.columns:
                                        price_data[sym] = symbol_data['close']
                                        if common_dates is None:
                                            common_dates = symbol_data.index
                                        else:
                                            common_dates = common_dates.intersection(symbol_data.index)
                            except Exception:
                                continue
                        
                        if len(price_data) < 3:
                            st.error("Insufficient data for correlation analysis")
                        else:
                            # Align all series to common dates
                            aligned_data = pd.DataFrame({
                                sym: series.loc[common_dates] 
                                for sym, series in price_data.items()
                            })
                            
                            # Compute returns and correlation
                            returns = aligned_data.pct_change().dropna()
                            corr_matrix = returns.corr()
                            
                            # Find assets based on correlation type
                            if "High" in corr_type:
                                threshold = 0.7
                                condition = corr_matrix > threshold
                            elif "Medium" in corr_type:
                                threshold_low, threshold_high = 0.3, 0.7
                                condition = (corr_matrix > threshold_low) & (corr_matrix < threshold_high)
                            else:  # Low
                                threshold = 0.3
                                condition = corr_matrix < threshold
                            
                            # Find most connected assets
                            connection_counts = condition.sum(axis=1) - 1  # Subtract self
                            top_assets = connection_counts.nlargest(n_basket).index.tolist()
                            
                            # Show results
                            st.success(f"✅ Found {len(top_assets)} assets matching criteria")
                            
                            # Show correlation heatmap for selected assets
                            selected_corr = corr_matrix.loc[top_assets, top_assets]
                            
                            fig = go.Figure(data=go.Heatmap(
                                z=selected_corr.values,
                                x=selected_corr.columns,
                                y=selected_corr.index,
                                colorscale='RdBu',
                                zmid=0,
                                text=selected_corr.values,
                                texttemplate='%{text:.2f}',
                                textfont={"size": 10},
                                colorbar=dict(title="Correlation")
                            ))
                            
                            fig.update_layout(
                                title="Correlation Matrix of Selected Assets",
                                height=500,
                                xaxis={'side': 'bottom'},
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Summary stats
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Avg Correlation", f"{selected_corr.values.mean():.3f}")
                            with col2:
                                st.metric("Max Correlation", f"{selected_corr.values.max():.3f}")
                            with col3:
                                st.metric("Min Correlation", f"{selected_corr.values.min():.3f}")
                            
                            st.write("**Selected assets:**", ', '.join(top_assets))
                            
                        if st.button("✅ Use This Basket"):
                            st.session_state.selected_assets = top_assets
                            st.success(f"✅ Created correlation basket with {len(top_assets)} assets")
                            st.rerun()
        
        else:  # Custom mode
            st.markdown("""
            **Custom selection** allows you to manually choose any combination of assets.
            """)
            
            n_assets = st.slider("Number of assets:", 3, min(10, len(symbols)), 5)
            selected_basket = st.multiselect(
                "Select assets for basket:",
                options=symbols,
                max_selections=n_assets
            )
            
            if len(selected_basket) >= 3:
                st.write(f"**Selected:** {', '.join(selected_basket)}")
                
                if st.button("✅ Create Custom Basket"):
                    st.session_state.selected_assets = selected_basket
                    st.success(f"✅ Created custom basket with {len(selected_basket)} assets")
                    st.rerun()
            elif len(selected_basket) > 0:
                st.warning(f"Please select at least 3 assets (currently {len(selected_basket)})")
    
    # Navigation button
    st.markdown("---")
    if st.session_state.selected_assets:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col3:
            if st.button("➡️ Next: Configure Parameters", type="primary", use_container_width=True, key="stock_next"):
                st.session_state.show_parameters_hint = True
                st.success(f"✅ {len(st.session_state.selected_assets)} assets selected!")
                st.balloons()
                st.rerun()
    else:
        st.info("💡 Select assets above to continue")

# ============================================================================
# TAB 3: PARAMETER CONFIGURATION
# ============================================================================
with tab3:
    st.header("⚙️ Parameter Configuration")
    
    # Show hint if coming from Stock Selection tab
    if st.session_state.get('show_parameters_hint'):
        st.success(f"✅ Great! Configure parameters for {len(st.session_state.selected_assets)} assets.")
        st.session_state.show_parameters_hint = False
    
    st.markdown("""
    Configure superspace analysis parameters. These control:
    - Anomaly detection sensitivity
    - Ghost field dynamics
    - Chern-Simons coupling
    - 14D feature engineering
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Anomaly Detection")
        
        threshold = st.slider(
            "Anomaly Threshold (σ)",
            min_value=1.5,
            max_value=4.0,
            value=st.session_state.superspace_params['threshold'],
            step=0.1,
            help="Z-score threshold for anomaly detection. Higher = fewer but stronger signals"
        )
        
        alpha = st.slider(
            "Ghost Divergence Weight (α)",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.superspace_params['alpha'],
            step=0.05,
            help="Weight for ghost field divergence in combined score"
        )
        
        beta = st.slider(
            "Chern-Simons Weight (β)",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.superspace_params['beta'],
            step=0.05,
            help="Weight for CS topological invariant in combined score"
        )
        
        st.info(f"Combined weights: α + β = {alpha + beta:.2f} (recommended: 1.0)")
    
    with col2:
        st.subheader("👻 Ghost Field Parameters")
        
        ghost_noise = st.slider(
            "Ghost Noise Amplitude",
            min_value=0.01,
            max_value=0.5,
            value=st.session_state.superspace_params['ghost_noise'],
            step=0.01,
            help="Stochastic noise in ghost field evolution"
        )
        
        gamma = st.slider(
            "Friction Coefficient (γ)",
            min_value=0.01,
            max_value=0.5,
            value=0.1,
            step=0.01,
            help="Damping in ghost field Langevin dynamics"
        )
        
        spring_k = st.slider(
            "Spring Constant (k)",
            min_value=0.5,
            max_value=2.0,
            value=1.0,
            step=0.1,
            help="Restoring force in ghost field Hamiltonian"
        )
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("🌀 Chern-Simons")
        
        cs_coupling = st.slider(
            "CS Coupling Constant",
            min_value=0.5,
            max_value=3.0,
            value=1.0,
            step=0.1,
            help="Coupling constant in Chern-Simons action"
        )
        
        cs_window = st.slider(
            "CS Window Size",
            min_value=10,
            max_value=60,
            value=st.session_state.superspace_params['cs_window'],
            step=5,
            help="Rolling window for CS invariant computation"
        )
    
    with col4:
        st.subheader("🔢 14D Features")
        
        window_14d = st.slider(
            "Feature Window",
            min_value=10,
            max_value=50,
            value=st.session_state.superspace_params['window_14d'],
            step=5,
            help="Rolling window for computing volatility, trend, etc."
        )
        
        use_pca = st.checkbox(
            "Use PCA Projection",
            value=True,
            help="Project 14D space to 2D/3D for visualization"
        )
    
    # Advanced parameters
    with st.expander("🔧 Advanced Parameters"):
        st.markdown("**BRST & Gauge Theory**")
        
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            brst_check = st.checkbox("Enforce BRST Symmetry", value=True)
            gauge_inv = st.checkbox("Check Gauge Invariance", value=False)
        
        with col_a2:
            dt_ghost = st.number_input("Ghost Time Step (dt)", value=0.01, step=0.001, format="%.3f")
            n_ghost_modes = st.number_input("Number of Ghost Modes", value=7, min_value=1, max_value=10)
    
    # Save parameters
    if st.button("💾 Save Parameters", type="primary"):
        st.session_state.superspace_params.update({
            'threshold': threshold,
            'alpha': alpha,
            'beta': beta,
            'cs_window': cs_window,
            'ghost_noise': ghost_noise,
            'window_14d': window_14d,
            'gamma': gamma,
            'spring_k': spring_k,
            'cs_coupling': cs_coupling,
            'use_pca': use_pca,
            'brst_check': brst_check,
            'dt_ghost': dt_ghost,
            'n_ghost_modes': n_ghost_modes
        })
        st.success("✅ Parameters saved!")
    
    # Navigation buttons
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col3:
        if st.session_state.selected_assets and st.session_state.get('superspace_params'):
            if st.button("➡️ Next: Run Analysis", type="primary", use_container_width=True, key="params_next"):
                st.session_state.show_analysis_hint = True
                st.success("✅ Parameters configured!")
                st.balloons()
                st.rerun()
        else:
            st.button("➡️ Next: Run Analysis", disabled=True, use_container_width=True)
            if not st.session_state.selected_assets:
                st.warning("⚠️ Select assets first in Stock Selection tab")
            else:
                st.info("💡 Configure parameters above, then proceed to analysis")

# ============================================================================
# TAB 4: ANALYSIS & VISUALIZATION
# ============================================================================
with tab4:
    st.header("🔬 Superspace Analysis & Visualization")
    
    # Show hint if coming from Parameters tab
    if st.session_state.get('show_analysis_hint'):
        st.success("✅ Ready to analyze! Click 'Run Superspace Analysis' below.")
        st.session_state.show_analysis_hint = False
    
    if not st.session_state.selected_assets:
        st.warning("⚠️ Please select assets in the Stock Selection tab")
        st.info("👆 Go to **📊 Stock Selection** tab to choose assets")
    elif 'historical_data' not in st.session_state or st.session_state.historical_data is None:
        st.warning("⚠️ Please load historical data from the Data Loader page")
        st.info("💡 Use the **Data Loader** page in the sidebar to fetch data")
    else:
        data = st.session_state.historical_data
        selected = st.session_state.selected_assets
        params = st.session_state.superspace_params
        
        st.markdown(f"""
        **Analyzing:** {', '.join(str(s) for s in selected)}  
        **Mode:** {'Single Asset' if len(selected) == 1 else 'Pairs' if len(selected) == 2 else 'Portfolio'}
        """)
        
        # ============================================================================
        # SMART DATA VALIDATION AND CONVERSION
        # ============================================================================
        st.subheader("📊 Data Validation")
        
        # Create expander for data processing details
        with st.expander("🔍 Data Processing Details", expanded=True):
            st.write("**Raw data info:**")
            st.write(f"- Type: {type(data)}")
            st.write(f"- Shape: {data.shape if isinstance(data, pd.DataFrame) else 'N/A'}")
            if isinstance(data, pd.DataFrame):
                st.write(f"- Index type: {type(data.index).__name__}")
                st.write(f"- Index names: {data.index.names if hasattr(data.index, 'names') else data.index.name}")
                st.write(f"- Columns type: {type(data.columns).__name__}")
                st.write(f"- Columns: {list(data.columns)[:20]}")  # Show actual column names
                if isinstance(data.columns, pd.MultiIndex):
                    st.write(f"- Column levels: {data.columns.names}")
                    st.write(f"- Level 0 values: {list(data.columns.get_level_values(0).unique())}")
                st.write(f"- Has 'symbol' column: {'symbol' in data.columns if hasattr(data, 'columns') else 'N/A'}")
                st.write(f"- Has 'timestamp' column: {'timestamp' in data.columns if hasattr(data, 'columns') else 'N/A'}")
                # Show first few rows for debugging
                st.write("**First 3 rows:**")
                st.dataframe(data.head(3), use_container_width=True)
        
        # Convert data to proper format: Dict[symbol -> DataFrame]
        data_dict = {}
        conversion_needed = False
        conversion_error = None
        
        if isinstance(data, dict):
            # Already in dict format
            data_dict = data
            st.success("✅ Data already in dictionary format")
            
        elif isinstance(data, pd.DataFrame):
            # Check format and convert as needed
            
            # Case 1: Row-based MultiIndex (timestamp, symbol) - NEED TO CONVERT
            if isinstance(data.index, pd.MultiIndex) and 'symbol' in data.index.names:
                st.info("🔄 Converting from row-based MultiIndex format...")
                conversion_needed = True
                try:
                    # Unstack to get symbols as columns
                    data_unstacked = data.unstack(level='symbol')
                    # Swap levels to (symbol, ohlcv)
                    data_unstacked.columns = data_unstacked.columns.swaplevel(0, 1)
                    data_unstacked = data_unstacked.sort_index(axis=1)
                    # Extract to dict
                    for symbol in selected:
                        if symbol in data_unstacked.columns.get_level_values(0):
                            data_dict[symbol] = data_unstacked[symbol].copy()
                    st.success(f"✅ Converted to column format: {len(data_dict)} symbols found")
                    # Update session state
                    st.session_state.historical_data = data_unstacked
                except Exception as e:
                    st.error(f"❌ Conversion failed: {e}")
                    conversion_error = str(e)
            
            # Case 2: Flat format with 'symbol' column - NEED TO CONVERT
            elif 'symbol' in data.columns:
                st.info("🔄 Converting from flat format with symbol column...")
                conversion_needed = True
                try:
                    # Check if timestamp is in columns or index
                    if 'timestamp' in data.columns:
                        data_indexed = data.set_index(['timestamp', 'symbol'])
                    elif data.index.name == 'timestamp' or pd.api.types.is_datetime64_any_dtype(data.index):
                        data_indexed = data.set_index('symbol', append=True)
                    else:
                        raise ValueError("No timestamp found in columns or index")
                    
                    # Unstack to get symbols as columns
                    data_unstacked = data_indexed.unstack(level='symbol')
                    data_unstacked.columns = data_unstacked.columns.swaplevel(0, 1)
                    data_unstacked = data_unstacked.sort_index(axis=1)
                    
                    # Extract to dict
                    for symbol in selected:
                        if symbol in data_unstacked.columns.get_level_values(0):
                            data_dict[symbol] = data_unstacked[symbol].copy()
                    st.success(f"✅ Converted from flat format: {len(data_dict)} symbols found")
                    # Update session state
                    st.session_state.historical_data = data_unstacked
                except Exception as e:
                    st.error(f"❌ Conversion failed: {e}")
                    conversion_error = str(e)
            
            # Case 3: Column-based MultiIndex (symbol, ohlcv) - ALREADY CORRECT
            elif isinstance(data.columns, pd.MultiIndex):
                st.success("✅ Data already in column-based MultiIndex format")
                try:
                    for symbol in selected:
                        if symbol in data.columns.get_level_values(0):
                            data_dict[symbol] = data[symbol].copy()
                except Exception as e:
                    st.error(f"❌ Failed to extract symbols: {e}")
                    conversion_error = str(e)
            
            # Case 4: Single symbol data (no MultiIndex)
            elif len(selected) == 1:
                st.info("ℹ️ Single symbol data detected")
                data_dict[selected[0]] = data.copy()
            
            # Case 5: Data with no symbol info (corrupted/lost during load)
            else:
                st.error("❌ **DATA CORRUPTED - No Symbol Information**")
                st.warning(f"""
                **Current data state:**
                - Shape: {data.shape}
                - Columns: {list(data.columns)[:10]}
                - Index: {type(data.index).__name__}
                - **Missing**: timestamp and symbol information
                
                **This data cannot be used for analysis.**
                """)
                
                st.info("""
                **🔧 How to Fix:**
                
                **Option 1: Load from Saved Dataset (Recommended)**
                1. Go to **Data Loader** page (left sidebar)
                2. Click on **"📂 Saved Datasets"** tab
                3. Find your dataset and click **"📤 Load"**
                4. Come back to this tab
                
                **Option 2: Fetch Fresh Data**
                1. Go to **Data Loader** page (left sidebar)  
                2. Click on **"🔄 Fetch Data"** tab
                3. Enter symbols: `{', '.join(str(s) for s in selected)}`
                4. Select date range and click **"Fetch Data"**
                5. Come back to this tab
                
                **Option 3: Clear and Restart**
                1. Click hamburger menu (☰) → "Clear cache"
                2. Refresh browser (Cmd+R)
                3. Start fresh from Data Loader
                """)
                
                # Add quick action button
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🗑️ Clear Corrupted Data", type="secondary", use_container_width=True):
                        st.session_state.historical_data = None
                        st.success("✅ Cleared! Now go to Data Loader to load fresh data.")
                        st.rerun()
                with col2:
                    if st.button("📂 Go to Data Loader", type="primary", use_container_width=True):
                        st.switch_page("pages/data_loader.py")
                
                conversion_error = "No symbol information found in data structure"
        
        # Show what we found
        st.write("---")
        st.write("**📋 Data Extraction Results:**")
        st.write("")
        
        # If conversion error occurred, show it prominently
        if conversion_error:
            st.error(f"🚫 **Data Processing Failed:** {conversion_error}")
            st.write("")
        
        # Display results per symbol
        for symbol in selected:
            if symbol in data_dict:
                count = len(data_dict[symbol])
                status = "✅" if count >= 300 else "⚠️"
                st.write(f"{status} **{symbol}**: {count:,} data points")
            else:
                st.write(f"❌ **{symbol}**: Not found in data")
        
        # If no data was extracted at all, provide clear guidance
        if not data_dict:
            st.write("")
            st.error("🚫 **NO DATA EXTRACTED**")
            st.write("""
            **Next steps:**
            1. 🔍 Check the "Data Processing Details" expander above
            2. 📥 Go to **Data Loader** page (sidebar) to reload data
            3. ✅ Ensure data is loaded with symbol information preserved
            4. 🔄 Come back to this tab after loading
            """)
            st.stop()  # Stop processing here
        
        # Identify missing and insufficient data
        missing_symbols = [s for s in selected if s not in data_dict]
        insufficient_symbols = [(s, len(data_dict[s])) for s in selected if s in data_dict and len(data_dict[s]) < 300]
        
        # Handle missing symbols
        if missing_symbols:
            st.error(f"❌ Missing data for: **{', '.join(str(m) for m in missing_symbols)}**")
            st.warning("🔄 Load Missing Data will fetch enough history to meet the 300 data point minimum")
            st.info("📖 **Automatic fetch amounts:**\n- 1min frequency: 7 days (~10,000 points)\n- 1h frequency: 30 days (~720 points)\n- 1d frequency: 1 year (~365 points)")
            
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                fetch_frequency = st.selectbox(
                    "Frequency",
                    options=["1h", "1min"],
                    index=0,
                    key="fetch_frequency_selector"
                )
            
            with col2:
                if st.button("📥 Load Missing Data", type="primary", key="load_missing_data_main"):
                    st.session_state['fetch_missing_symbols'] = missing_symbols
                    st.session_state['fetch_frequency'] = fetch_frequency
                    st.toast(f"🔄 Starting fetch for {len(missing_symbols)} symbols ({fetch_frequency})...", icon="📡")
                    st.rerun()
            
            with col3:
                st.info("💡 Or load data manually from the **Data Loader** page in the sidebar")
            
            st.stop()  # Don't proceed until data is loaded
        
        # Handle insufficient data
        if insufficient_symbols:
            st.error("❌ Insufficient data for analysis")
            for sym, count in insufficient_symbols:
                st.warning(f"⚠️ {sym}: {count} rows (need minimum 300 for reliable analysis)")
            st.info("📖 **Recommendation:** Go to **Data Loader** page and fetch more historical data")
            st.info("💡 **Minimum requirements:**\n- Daily data: 1-2 years (300-500 points)\n- Hourly data: 2-3 months (300-2000 points)\n- Minute data: 5-7 days (300-10000 points)")
            st.stop()  # Don't proceed with insufficient data
        
        # If we got here, all data is present and sufficient!
        st.success(f"✅ All {len(selected)} symbols have sufficient data (300+ points each)")
        
        # ============================================================================
        # AUTO-FETCH MISSING DATA (runs independently of analysis button)
        # ============================================================================
        # Check if we should fetch data (after Load Missing Data button was clicked)
        if 'fetch_missing_symbols' in st.session_state:
            missing_to_fetch = st.session_state['fetch_missing_symbols']
            del st.session_state['fetch_missing_symbols']  # Clear flag immediately
            
            # Show what we're about to fetch
            st.info(f"📡 Preparing to fetch: **{', '.join(str(m) for m in missing_to_fetch)}**")
            
            try:
                from datetime import datetime, timedelta
                import time
                
                # Calculate date range based on frequency to ensure 300+ data points
                end_date = datetime.now()
                fetch_frequency = st.session_state.get('fetch_frequency', '1h')
                
                # Calculate appropriate history based on frequency
                if fetch_frequency == '1min':
                    # 1 minute: need ~5 days for 300+ points (300 points = 5 hours)
                    days_back = 7  # Get 7 days to be safe
                    expected_points = 7 * 24 * 60  # ~10,080 points
                elif fetch_frequency == '1h':
                    # 1 hour: need ~13 days for 300+ points
                    days_back = 30  # Get 30 days (720 hours)
                    expected_points = 30 * 24  # ~720 points
                else:
                    # Daily: need 300+ days
                    days_back = 365  # Get 1 year
                    expected_points = 365
                
                start_date = end_date - timedelta(days=days_back)
                
                # Format dates
                start_str = start_date.strftime('%Y-%m-%d')
                end_str = end_date.strftime('%Y-%m-%d')
                
                st.write(f"📅 **Date range:** {start_str} to {end_str} ({days_back} days)")
                st.write(f"⏱️ **Interval:** {fetch_frequency}")
                st.write(f"📊 **Expected data points:** ~{expected_points} per symbol")
                st.write(f"📊 **Source:** Auto-detect (Yahoo Finance for stocks)")
                st.write(f"🔢 **Symbols to fetch:** {len(missing_to_fetch)}")
                
                # Show a manual progress indicator
                progress_placeholder = st.empty()
                status_placeholder = st.empty()
                
                progress_placeholder.progress(0.0)
                status_placeholder.write("🚀 Initializing data fetch...")
                
                # Create a placeholder for fetch progress
                fetch_container = st.container()
                
                with fetch_container:
                    # Fetch missing data - progress bars will show here
                    new_data = fetch_intraday_data(
                        symbols=[str(m) for m in missing_to_fetch],
                        start=start_str,
                        end=end_str,
                        interval=fetch_frequency,
                        source="auto"
                    )
                
                progress_placeholder.progress(1.0)
                status_placeholder.success("✅ Data fetch complete!")
                
                if new_data is not None and not new_data.empty:
                    st.success(f"✅ Fetched {len(missing_to_fetch)} symbol(s) successfully!")
                    st.write(f"📊 **Fetched data shape:** {new_data.shape}")
                    st.write(f"📋 **Index levels:** {new_data.index.names if hasattr(new_data.index, 'names') else 'Single level'}")
                    
                    try:
                        # The fetched data has MultiIndex on rows: (timestamp, symbol)
                        # We need to unstack it to get MultiIndex on columns: (symbol, ohlcv)
                        
                        # Unstack the 'symbol' level to move it to columns
                        new_data_unstacked = new_data.unstack(level='symbol')
                        
                        # Swap levels so it's (symbol, column) instead of (column, symbol)
                        new_data_unstacked.columns = new_data_unstacked.columns.swaplevel(0, 1)
                        new_data_unstacked = new_data_unstacked.sort_index(axis=1)
                        
                        st.write(f"✅ **Reformatted data columns:** {list(new_data_unstacked.columns.get_level_values(0).unique())}")
                        
                        # Now merge with existing data
                        if isinstance(data, pd.DataFrame) and isinstance(data.columns, pd.MultiIndex):
                            # Both have MultiIndex on columns - concatenate
                            # Use outer join to keep all timestamps
                            combined = pd.concat([data, new_data_unstacked], axis=1)
                            # Remove duplicate columns if any
                            combined = combined.loc[:, ~combined.columns.duplicated()]
                            st.session_state.historical_data = combined
                            st.success(f"✅ Merged with existing data! Total symbols: {list(combined.columns.get_level_values(0).unique())}")
                        else:
                            # Just use the new data
                            st.session_state.historical_data = new_data_unstacked
                            st.success(f"✅ Set as new dataset! Symbols: {list(new_data_unstacked.columns.get_level_values(0).unique())}")
                        
                        st.balloons()
                        st.info("🔄 Page will reload with updated data...")
                        time.sleep(2)
                        st.rerun()
                    except Exception as merge_error:
                        st.error(f"❌ Error processing data: {str(merge_error)}")
                        st.exception(merge_error)
                        st.write("**Debug info:**")
                        st.write(f"- New data type: {type(new_data)}")
                        st.write(f"- Index: {new_data.index}")
                        st.write(f"- Columns: {new_data.columns}")
                else:
                    st.error("❌ Failed to fetch data. No data returned.")
                    st.info("💡 Try loading data manually from the **Data Loader** page")
            except Exception as e:
                st.error(f"❌ Error fetching data: {str(e)}")
                st.exception(e)
                st.info("💡 Try loading data manually from the **Data Loader** page")
        
        # ============================================================================
        # RUN ANALYSIS BUTTON
        # ============================================================================
        if st.button("🚀 Run Superspace Analysis", type="primary"):
            if not RUST_AVAILABLE:
                st.error("Rust bindings not available. Cannot run analysis.")
            else:
                with st.spinner("Computing 14D superspace..."):
                    try:
                        results = {}
                        
                        # Use the already validated data_dict from above
                        # No need to re-convert or re-validate
                        
                        # Process each symbol
                        window_size = params.get('window_14d', 20)
                        
                        st.info(f"📊 Analyzing {len(data_dict)} symbols with window size {window_size}")
                        
                        for symbol in selected:
                            # Data already validated to have 300+ points
                            df = data_dict[symbol].copy()
                            
                            # ===== 1. Construct 14D Superspace =====
                            st.text(f"Building 14D superspace for {symbol}...")
                            
                            # Bosonic coordinates
                            prices = df['close'].values
                            volumes = df['volume'].values
                            
                            # Compute features with proper window
                            window = params.get('window_14d', 20)
                            
                            log_price = np.log(prices)
                            log_volume = np.log(np.array(volumes) + 1)  # +1 to avoid log(0), convert to numpy
                            
                            # Volatility (rolling std of returns)
                            returns = np.diff(log_price, prepend=log_price[0])
                            volatility = pd.Series(returns).rolling(window).std().fillna(0).values
                            
                            # Trend (rolling regression slope)
                            trend = np.zeros(len(prices))
                            for i in range(window, len(prices)):
                                x = np.arange(window)
                                y = log_price[i-window:i]
                                trend[i] = np.polyfit(x, y, 1)[0]
                            
                            # Momentum (rate of change)
                            momentum = pd.Series(log_price).pct_change(window).fillna(0).values
                            
                            # Liquidity proxy (inverse volume)
                            liquidity = 1.0 / (np.array(volumes) + 1)
                            
                            # Sentiment (RSI-like)
                            delta = pd.Series(prices).diff()
                            gain = delta.where(delta > 0, 0).rolling(window).mean()
                            loss = -delta.where(delta < 0, 0).rolling(window).mean()
                            rs = gain / (loss + 1e-10)
                            sentiment = (100 - (100 / (1 + rs))).fillna(50).values
                            
                            # Stack bosonic coordinates (ensure all are numpy arrays)
                            bosonic = np.column_stack([
                                np.asarray(log_price), np.asarray(log_volume), np.asarray(volatility), 
                                np.asarray(trend), np.asarray(momentum), np.asarray(liquidity), np.asarray(sentiment)
                            ])
                            
                            # Fermionic coordinates (time derivatives)
                            fermionic = np.diff(bosonic, axis=0, prepend=bosonic[0:1])
                            
                            # Normalize
                            scaler = StandardScaler()
                            bosonic_norm = scaler.fit_transform(bosonic)
                            fermionic_norm = scaler.fit_transform(fermionic)
                            
                            superspace_14d = np.hstack([bosonic_norm, fermionic_norm])
                            
                            # ===== 2. Ghost Field Evolution =====
                            st.text("Computing ghost field dynamics...")
                            
                            assert superspace_rs is not None, "Rust bindings not available"
                                
                            ghost_params = superspace_rs.PyGhostFieldParams(
                                n_modes=params.get('n_ghost_modes', 7),
                                damping=params.get('gamma', 0.1),
                                noise_level=params.get('ghost_noise', 0.1),
                                spring_constant=params.get('spring_k', 1.0)
                            )
                            
                            # Convert to 2D list for from_bosonic_coords
                            bosonic_2d = [row[:7].tolist() for row in bosonic_norm]
                            ghost_system = superspace_rs.PyGhostFieldSystem.from_bosonic_coords(
                                bosonic_2d,
                                ghost_params
                            )
                            
                            # Access pre-computed divergences from the system
                            ghost_divergences = np.array(ghost_system.divergence)
                            
                            # ===== 3. Chern-Simons Invariants =====
                            st.text("Computing Chern-Simons topological invariants...")
                            
                            assert superspace_rs is not None, "Rust bindings not available"
                                
                            cs_calc = superspace_rs.PyChernSimonsCalculator(
                                coupling=params.get('cs_coupling', 1.0),
                                window=params.get('cs_window', 30)
                            )
                            
                            cs_series = cs_calc.calculate(prices.tolist())
                            
                            # Compute changes manually
                            cs_changes = np.abs(np.diff(cs_series, prepend=cs_series[0]))
                            
                            # ===== 4. Anomaly Detection =====
                            st.text("Detecting anomalies...")
                            
                            # Z-score normalization
                            ghost_z = (ghost_divergences - np.mean(ghost_divergences)) / (np.std(ghost_divergences) + 1e-10)
                            cs_z = (cs_changes - np.mean(cs_changes)) / (np.std(cs_changes) + 1e-10)
                            
                            # Combined anomaly score
                            alpha = params.get('alpha', 0.5)
                            beta = params.get('beta', 0.5)
                            anomaly_score = alpha * np.abs(ghost_z) + beta * cs_z
                            
                            # Detect anomalies
                            threshold = params.get('threshold', 2.5)
                            anomalies = anomaly_score > threshold
                            anomaly_indices = np.where(anomalies)[0]
                            
                            # ===== 5. PCA Projection =====
                            if params.get('use_pca', True):
                                pca = PCA(n_components=3)
                                superspace_3d = pca.fit_transform(superspace_14d)
                                explained_var = pca.explained_variance_ratio_
                            else:
                                superspace_3d = superspace_14d[:, :3]
                                explained_var = [0.33, 0.33, 0.33]
                            
                            # Store results
                            results[symbol] = {
                                'prices': prices,
                                'dates': df.index,
                                'superspace_14d': superspace_14d,
                                'superspace_3d': superspace_3d,
                                'ghost_divergence': ghost_divergences,
                                'cs_series': cs_series,
                                'cs_changes': cs_changes,
                                'anomaly_score': anomaly_score,
                                'anomalies': anomalies,
                                'anomaly_indices': anomaly_indices,
                                'explained_variance': explained_var,
                                'bosonic': bosonic_norm,
                                'fermionic': fermionic_norm
                            }
                        
                        st.session_state.superspace_results = results
                        st.success(f"✅ Analysis complete! Detected {sum(len(r['anomaly_indices']) for r in results.values())} total anomalies")
                        
                    except Exception as e:
                        st.error(f"Error during analysis: {e}")
                        import traceback
                        st.code(traceback.format_exc())
        
        # Visualization
        if st.session_state.superspace_results:
            st.markdown("---")
            st.subheader("📊 Visualization")
            
            results = st.session_state.superspace_results
            viz_symbol = st.selectbox("Select asset to visualize:", list(results.keys()))
            
            if viz_symbol:
                res = results[viz_symbol]
                
                viz_tabs = st.tabs([
                    "Price & Anomalies",
                    "Ghost Fields",
                    "Chern-Simons",
                    "14D Superspace",
                    "Anomaly Details"
                ])
                
                with viz_tabs[0]:
                    # Price with anomaly markers
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=res['dates'],
                        y=res['prices'],
                        name="Price",
                        line=dict(color='blue', width=2)
                    ))
                    
                    # Mark anomalies
                    if len(res['anomaly_indices']) > 0:
                        fig.add_trace(go.Scatter(
                            x=res['dates'][res['anomaly_indices']],
                            y=res['prices'][res['anomaly_indices']],
                            mode='markers',
                            name='Anomalies',
                            marker=dict(color='red', size=10, symbol='x')
                        ))
                    
                    fig.update_layout(
                        title=f"{viz_symbol} Price with Detected Anomalies",
                        xaxis_title="Date",
                        yaxis_title="Price",
                        height=500,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.metric("Total Anomalies", len(res['anomaly_indices']))
                
                with viz_tabs[1]:
                    # Ghost field divergence
                    fig = make_subplots(
                        rows=2, cols=1,
                        shared_xaxes=True,
                        subplot_titles=("Ghost Field Divergence", "Price"),
                        vertical_spacing=0.1
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=res['dates'], y=res['ghost_divergence'], 
                                  name="Ghost Divergence", line=dict(color='purple')),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=res['dates'], y=res['prices'], 
                                  name="Price", line=dict(color='blue')),
                        row=2, col=1
                    )
                    
                    # Mark high divergence regions
                    threshold_ghost = np.mean(res['ghost_divergence']) + 2 * np.std(res['ghost_divergence'])
                    fig.add_hline(y=threshold_ghost, line_dash="dash", line_color="red", 
                                 annotation_text="Threshold")
                    
                    fig.update_layout(height=600, showlegend=True)
                    fig.update_xaxes(title_text="Date", row=2, col=1)
                    fig.update_yaxes(title_text="Divergence", row=1, col=1)
                    fig.update_yaxes(title_text="Price", row=2, col=1)
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with viz_tabs[2]:
                    # Chern-Simons invariant
                    fig = make_subplots(
                        rows=3, cols=1,
                        shared_xaxes=True,
                        subplot_titles=("CS Invariant", "CS Changes", "Price"),
                        vertical_spacing=0.08
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=res['dates'], y=res['cs_series'], 
                                  name="CS Invariant", line=dict(color='green')),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=res['dates'], y=res['cs_changes'], 
                                  name="CS Changes", line=dict(color='orange')),
                        row=2, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=res['dates'], y=res['prices'], 
                                  name="Price", line=dict(color='blue')),
                        row=3, col=1
                    )
                    
                    fig.update_layout(height=700, showlegend=True)
                    st.plotly_chart(fig, use_container_width=True)
                
                with viz_tabs[3]:
                    # 14D Superspace (PCA projection)
                    fig = go.Figure()
                    
                    # Color by anomaly score
                    fig.add_trace(go.Scatter3d(
                        x=res['superspace_3d'][:, 0],
                        y=res['superspace_3d'][:, 1],
                        z=res['superspace_3d'][:, 2],
                        mode='markers',
                        marker=dict(
                            size=3,
                            color=res['anomaly_score'],
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title="Anomaly<br>Score")
                        ),
                        text=[f"Date: {d}<br>Score: {s:.2f}" 
                              for d, s in zip(res['dates'], res['anomaly_score'])],
                        hovertemplate='%{text}<extra></extra>'
                    ))
                    
                    # Highlight anomalies
                    if len(res['anomaly_indices']) > 0:
                        anom_idx = res['anomaly_indices']
                        fig.add_trace(go.Scatter3d(
                            x=res['superspace_3d'][anom_idx, 0],
                            y=res['superspace_3d'][anom_idx, 1],
                            z=res['superspace_3d'][anom_idx, 2],
                            mode='markers',
                            marker=dict(size=6, color='red', symbol='x'),
                            name='Anomalies'
                        ))
                    
                    fig.update_layout(
                        title=f"14D Superspace (PCA Projection)<br>Explained Variance: {sum(res['explained_variance'][:3])*100:.1f}%",
                        scene=dict(
                            xaxis_title=f"PC1 ({res['explained_variance'][0]*100:.1f}%)",
                            yaxis_title=f"PC2 ({res['explained_variance'][1]*100:.1f}%)",
                            zaxis_title=f"PC3 ({res['explained_variance'][2]*100:.1f}%)"
                        ),
                        height=600
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with viz_tabs[4]:
                    # Anomaly details table
                    if len(res['anomaly_indices']) > 0:
                        anom_data = []
                        for idx in res['anomaly_indices']:
                            anom_data.append({
                                'Date': res['dates'][idx],
                                'Price': res['prices'][idx],
                                'Anomaly Score': res['anomaly_score'][idx],
                                'Ghost Divergence': res['ghost_divergence'][idx],
                                'CS Change': res['cs_changes'][idx]
                            })
                        
                        anom_df = pd.DataFrame(anom_data)
                        st.dataframe(
                            anom_df.style.format({
                                'Price': '{:.2f}',
                                'Anomaly Score': '{:.3f}',
                                'Ghost Divergence': '{:.3f}',
                                'CS Change': '{:.3f}'
                            }),
                            use_container_width=True
                        )
                        
                        # Download button
                        csv = anom_df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Anomaly Report",
                            csv,
                            f"anomalies_{viz_symbol}.csv",
                            "text/csv"
                        )
                    else:
                        st.info("No anomalies detected with current parameters")

# ============================================================================
# TAB 5: PORTFOLIO OPTIMIZATION
# ============================================================================
with tab5:
    st.header("📈 Portfolio Optimization")
    
    if not st.session_state.superspace_results:
        st.warning("⚠️ Please run superspace analysis first (Tab 4)")
    else:
        st.markdown("""
        Use detected anomalies to optimize portfolio allocation:
        - **Long** assets with high negative anomaly divergence (oversold)
        - **Short** assets with high positive anomaly divergence (overbought)
        - Risk-adjusted position sizing
        """)
        
        results = st.session_state.superspace_results
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("⚙️ Portfolio Parameters")
            
            portfolio_size = st.number_input(
                "Portfolio Size ($)",
                min_value=10000,
                max_value=10000000,
                value=100000,
                step=10000
            )
            
            max_position = st.slider(
                "Max Position Size (%)",
                min_value=5,
                max_value=50,
                value=20,
                step=5
            )
            
            risk_free_rate = st.number_input(
                "Risk-Free Rate (%)",
                min_value=0.0,
                max_value=10.0,
                value=2.0,
                step=0.1
            ) / 100
            
            rebalance_threshold = st.slider(
                "Rebalance Threshold",
                min_value=1.0,
                max_value=3.0,
                value=2.0,
                step=0.1,
                help="Anomaly score threshold for rebalancing"
            )
        
        with col2:
            st.subheader("📊 Current Signals")
            
            # Get latest signals
            signals = []
            for symbol, res in results.items():
                latest_score = res['anomaly_score'][-1]
                latest_price = res['prices'][-1]
                latest_ghost = res['ghost_divergence'][-1]
                
                signal = "HOLD"
                if latest_score > rebalance_threshold:
                    signal = "SHORT" if latest_ghost > 0 else "LONG"
                
                signals.append({
                    'Symbol': symbol,
                    'Signal': signal,
                    'Anomaly Score': latest_score,
                    'Price': latest_price,
                    'Ghost Div': latest_ghost
                })
            
            signals_df = pd.DataFrame(signals)
            st.dataframe(
                signals_df.style.format({
                    'Anomaly Score': '{:.3f}',
                    'Price': '{:.2f}',
                    'Ghost Div': '{:.3f}'
                }).map(  # Changed from applymap (deprecated in pandas 2.1.0+)
                    lambda x: 'background-color: lightgreen' if x == 'LONG' 
                    else 'background-color: lightcoral' if x == 'SHORT' 
                    else '',
                    subset=['Signal']
                ),
                use_container_width=True
            )
        
        if st.button("🎯 Generate Portfolio Allocation"):
            with st.spinner("Optimizing portfolio..."):
                # Simple equal-weight allocation based on signals
                long_positions = signals_df[signals_df['Signal'] == 'LONG']
                short_positions = signals_df[signals_df['Signal'] == 'SHORT']
                
                n_long = len(long_positions)
                n_short = len(short_positions)
                total_positions = n_long + n_short
                
                if total_positions == 0:
                    st.info("No active signals. Portfolio remains in cash.")
                else:
                    allocations = []
                    
                    # Calculate position sizes
                    position_size = min(max_position / 100, 1.0 / total_positions)
                    
                    for _, row in long_positions.iterrows():
                        allocations.append({
                            'Symbol': row['Symbol'],
                            'Position': 'LONG',
                            'Weight': position_size,
                            'Allocation ($)': portfolio_size * position_size,
                            'Shares': int((portfolio_size * position_size) / row['Price']),
                            'Price': row['Price']
                        })
                    
                    for _, row in short_positions.iterrows():
                        allocations.append({
                            'Symbol': row['Symbol'],
                            'Position': 'SHORT',
                            'Weight': -position_size,
                            'Allocation ($)': portfolio_size * position_size,
                            'Shares': int((portfolio_size * position_size) / row['Price']),
                            'Price': row['Price']
                        })
                    
                    alloc_df = pd.DataFrame(allocations)
                    
                    st.success("✅ Portfolio Allocation Generated")
                    st.dataframe(
                        alloc_df.style.format({
                            'Weight': '{:.2%}',
                            'Allocation ($)': '${:,.2f}',
                            'Shares': '{:,}',
                            'Price': '${:.2f}'
                        }),
                        use_container_width=True
                    )
                    
                    # Portfolio metrics
                    st.subheader("📊 Portfolio Metrics")
                    
                    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                    
                    with col_m1:
                        total_invested = alloc_df['Allocation ($)'].sum()
                        st.metric("Total Invested", f"${total_invested:,.0f}")
                    
                    with col_m2:
                        cash_remaining = portfolio_size - total_invested
                        st.metric("Cash Remaining", f"${cash_remaining:,.0f}")
                    
                    with col_m3:
                        leverage = total_invested / portfolio_size
                        st.metric("Leverage", f"{leverage:.2f}x")
                    
                    with col_m4:
                        net_exposure = alloc_df['Weight'].sum()
                        st.metric("Net Exposure", f"{net_exposure:.1%}")

# ============================================================================
# TAB 6: BACKTESTING
# ============================================================================
with tab6:
    st.header("🎯 Backtesting")
    
    if not st.session_state.superspace_results:
        st.warning("⚠️ Please run superspace analysis first (Tab 4)")
    else:
        st.markdown("""
        Backtest the superspace anomaly strategy:
        - Trade on detected anomalies
        - Compare with baseline (buy-and-hold)
        - Compute performance metrics
        """)
        
        results = st.session_state.superspace_results
        
        # Backtesting parameters
        col1, col2 = st.columns(2)
        
        with col1:
            initial_capital = st.number_input(
                "Initial Capital ($)",
                min_value=10000,
                max_value=1000000,
                value=100000,
                step=10000
            )
            
            entry_threshold = st.slider(
                "Entry Threshold (σ)",
                min_value=1.5,
                max_value=4.0,
                value=2.5,
                step=0.1
            )
            
            exit_threshold = st.slider(
                "Exit Threshold (σ)",
                min_value=0.5,
                max_value=2.0,
                value=1.0,
                step=0.1
            )
        
        with col2:
            position_size_pct = st.slider(
                "Position Size (%)",
                min_value=10,
                max_value=100,
                value=50,
                step=10
            ) / 100
            
            transaction_cost = st.number_input(
                "Transaction Cost (bps)",
                min_value=0,
                max_value=100,
                value=5,
                step=1
            ) / 10000
            
            stop_loss = st.number_input(
                "Stop Loss (%)",
                min_value=1.0,
                max_value=20.0,
                value=5.0,
                step=0.5
            ) / 100
        
        if st.button("▶️ Run Backtest", type="primary"):
            with st.spinner("Running backtest..."):
                # Simple backtest for first asset
                symbol = list(results.keys())[0]
                res = results[symbol]
                
                prices = res['prices']
                anomaly_score = res['anomaly_score']
                ghost_div = res['ghost_divergence']
                dates = res['dates']
                
                # Initialize
                capital = initial_capital
                position = 0  # Current position size
                position_price = 0  # Entry price
                equity_curve: list[float] = [float(initial_capital)]
                trades = []
                
                # Baseline (buy and hold)
                baseline_shares = initial_capital / prices[0]
                baseline_curve = baseline_shares * prices
                
                # Strategy
                for i in range(1, len(prices)):
                    current_price = prices[i]
                    score = anomaly_score[i]
                    ghost = ghost_div[i]
                    
                    # Check for entry signal
                    if position == 0 and score > entry_threshold:
                        # Determine direction based on ghost divergence
                        direction = -1 if ghost > 0 else 1  # Contrarian
                        
                        # Enter position
                        position_value = capital * position_size_pct
                        shares = position_value / current_price
                        position = direction * shares
                        position_price = current_price
                        
                        # Apply transaction costs
                        cost = position_value * transaction_cost
                        capital -= cost
                        
                        trades.append({
                            'Date': dates[i],
                            'Type': 'ENTRY',
                            'Direction': 'LONG' if direction > 0 else 'SHORT',
                            'Price': current_price,
                            'Shares': shares,
                            'Capital': capital
                        })
                    
                    # Check for exit signal
                    elif position != 0:
                        exit_signal = False
                        
                        # Exit if anomaly score drops
                        if score < exit_threshold:
                            exit_signal = True
                        
                        # Stop loss
                        if position > 0:  # Long position
                            if current_price < position_price * (1 - stop_loss):
                                exit_signal = True
                        else:  # Short position
                            if current_price > position_price * (1 + stop_loss):
                                exit_signal = True
                        
                        if exit_signal:
                            # Close position
                            position_value = abs(position) * current_price
                            pnl = position * (current_price - position_price)
                            capital += pnl
                            
                            # Apply transaction costs
                            cost = position_value * transaction_cost
                            capital -= cost
                            
                            trades.append({
                                'Date': dates[i],
                                'Type': 'EXIT',
                                'Direction': 'CLOSE',
                                'Price': current_price,
                                'PnL': pnl - cost,
                                'Capital': capital
                            })
                            
                            position = 0
                            position_price = 0
                    
                    # Update equity
                    if position != 0:
                        unrealized_pnl = position * (current_price - position_price)
                        total_equity = capital + unrealized_pnl
                    else:
                        total_equity = capital
                    
                    equity_curve.append(float(total_equity))
                
                # Final close if still in position
                if position != 0:
                    pnl = position * (prices[-1] - position_price)
                    capital += pnl
                    equity_curve[-1] = capital
                
                # Store results
                st.session_state.backtest_results = {
                    'equity_curve': np.array(equity_curve),
                    'baseline_curve': baseline_curve,
                    'trades': trades,
                    'dates': dates,
                    'final_capital': capital
                }
                
                st.success("✅ Backtest complete!")
        
        # Display results
        if 'backtest_results' in st.session_state:
            bt_results = st.session_state.backtest_results
            
            st.markdown("---")
            st.subheader("📊 Performance Results")
            
            # Metrics
            equity = bt_results['equity_curve']
            baseline = bt_results['baseline_curve']
            
            strategy_return = (equity[-1] - equity[0]) / equity[0]
            baseline_return = (baseline[-1] - baseline[0]) / baseline[0]
            
            # Sharpe ratio (annualized)
            strategy_returns = np.diff(equity) / equity[:-1]
            sharpe = np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-10) * np.sqrt(252)
            
            # Max drawdown
            cummax = np.maximum.accumulate(equity)
            drawdown = (equity - cummax) / cummax
            max_dd = np.min(drawdown)
            
            col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
            
            with col_m1:
                st.metric("Total Return", f"{strategy_return*100:.2f}%", 
                         f"{(strategy_return - baseline_return)*100:.2f}% vs BH")
            
            with col_m2:
                st.metric("Final Capital", f"${bt_results['final_capital']:,.0f}")
            
            with col_m3:
                st.metric("Sharpe Ratio", f"{sharpe:.2f}")
            
            with col_m4:
                st.metric("Max Drawdown", f"{max_dd*100:.2f}%")
            
            with col_m5:
                st.metric("# Trades", len(bt_results['trades']))
            
            # Equity curve
            st.subheader("💰 Equity Curve")
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=bt_results['dates'],
                y=equity,
                name="Superspace Strategy",
                line=dict(color='green', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=bt_results['dates'],
                y=baseline,
                name="Buy & Hold",
                line=dict(color='blue', width=2, dash='dash')
            ))
            
            fig.update_layout(
                title="Strategy vs Buy-and-Hold",
                xaxis_title="Date",
                yaxis_title="Portfolio Value ($)",
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Trade log
            st.subheader("📝 Trade Log")
            
            if bt_results['trades']:
                trades_df = pd.DataFrame(bt_results['trades'])
                st.dataframe(
                    trades_df.style.format({
                        'Price': '${:.2f}',
                        'Shares': '{:.2f}',
                        'PnL': '${:,.2f}',
                        'Capital': '${:,.2f}'
                    }),
                    use_container_width=True
                )

# ============================================================================
# TAB 7: DOCUMENTATION
# ============================================================================
with tab7:
    st.header("📖 Documentation")
    
    doc_tabs = st.tabs([
        "Quick Start",
        "Mathematical Background",
        "Code Examples",
        "Troubleshooting"
    ])
    
    with doc_tabs[0]:
        st.markdown("""
        ## 🚀 Quick Start Guide
        
        ### 1. Load Data ⭐ IMPORTANT
        
        **Navigate to Data Loader page:**
        - Click "💾 Data Loader" in the sidebar navigation
        
        **Download Historical Data:**
        - **Symbols:** Enter tickers separated by commas
          - Example: `AAPL,NVDA,GOOGL,MSFT,AMZN,IBM,CSCO`
        - **Date Range:** Select at least 1-2 years
          - Daily data: 1-2 years (250-500 data points)
          - Hourly data: 3-6 months (500-1000 data points)
        - **Data Source:** Yahoo Finance (default)
        - Click **"Fetch Data"** and wait for download
        
        **Alternative: Upload CSV**
        - Format: Date, Open, High, Low, Close, Volume
        - Minimum 200 rows per symbol
        
        **Verify Data:**
        - Check that data loaded successfully
        - Ensure you have **at least 200+ data points per asset**
        - More data = better analysis (500+ recommended)
        
        ### 2. Select Assets
        - Go to **Stock Selection** tab
        - Choose 1-3 assets for analysis
        - For pairs trading, use **Auto-Pair** mode
        
        ### 3. Configure Parameters
        - **Anomaly Threshold:** Start with 2.5σ
        - **Weights:** α = β = 0.5 (balanced)
        - **Ghost Noise:** 0.1 (default)
        - **CS Window:** 30 periods
        
        ### 4. Run Analysis
        - Click **Run Superspace Analysis**
        - Wait for 14D space construction (30-60 seconds)
        - Review detected anomalies
        
        ### 5. Optimize & Backtest
        - Generate portfolio allocation
        - Run backtest with your parameters
        - Compare with buy-and-hold baseline
        
        ---
        
        ## 📐 Key Concepts
        
        **Superspace:** 14-dimensional space combining:
        - 7 bosonic coordinates (observable market features)
        - 7 fermionic coordinates (ghost fields, hidden dynamics)
        
        **Ghost Fields:** Auxiliary variables encoding:
        - Non-equilibrium dynamics
        - Regime changes
        - Hidden correlations
        
        **Chern-Simons Invariant:** Topological measure detecting:
        - Structural breaks
        - Regime transitions
        - Market phase changes
        
        **Anomaly Score:** Combined metric:
        $$\\mathcal{A}(t) = \\alpha \\cdot Z[\\nabla \\cdot \\mathbf{c}] + \\beta \\cdot Z[|\\Delta CS|]$$
        
        ---
        
        ## ⚙️ Recommended Settings
        
        | Scenario | Threshold | α | β | Ghost Noise |
        |----------|-----------|---|---|-------------|
        | Conservative | 3.0σ | 0.6 | 0.4 | 0.05 |
        | Balanced | 2.5σ | 0.5 | 0.5 | 0.10 |
        | Aggressive | 2.0σ | 0.4 | 0.6 | 0.15 |
        
        ---
        
        ## 🎯 Trading Signals
        
        **LONG Signal:** High anomaly score + negative ghost divergence  
        → Market oversold, expect reversion
        
        **SHORT Signal:** High anomaly score + positive ghost divergence  
        → Market overbought, expect correction
        
        **HOLD:** Anomaly score below threshold  
        → No clear signal
        """)
    
    with doc_tabs[1]:
        st.markdown(r"""
        ## 📚 Mathematical Background
        
        ### Supermanifolds
        
        A supermanifold $\mathcal{M}$ is a manifold with both commuting (bosonic) and anti-commuting (fermionic) coordinates:
        
        $$\mathcal{M} = \mathbb{R}^{d_b} \times \mathbb{G}^{d_f}$$
        
        where $\mathbb{G}$ is the Grassmann manifold.
        
        **Point representation:**
        $$\mathcal{P} = (x^\mu, \theta^\alpha) \quad \mu = 0, \ldots, d_b-1, \quad \alpha = 1, \ldots, d_f$$
        
        ---
        
        ### Grassmann Algebra
        
        Fermionic coordinates satisfy:
        
        $$\begin{align}
        \theta^\alpha \theta^\beta + \theta^\beta \theta^\alpha &= 0 \quad \text{(anticommutation)} \\
        (\theta^\alpha)^2 &= 0 \quad \text{(nilpotency)}
        \end{align}$$
        
        **Integration:** Berezin integral
        $$\int d\theta \, \theta = 1, \quad \int d\theta \, 1 = 0$$
        
        ---
        
        ### Ghost Field Dynamics
        
        Evolution equation (Langevin):
        
        $$\frac{dc_i}{dt} = -\gamma c_i - \frac{\partial H}{\partial \bar{c}_i} + \xi_i(t)$$
        
        **Hamiltonian:**
        $$H = \sum_i \left[\frac{p_i^2}{2m} + \frac{k}{2}q_i^2\right] + \sum_{ij} \bar{c}_i M_{ij} c_j$$
        
        **Ghost divergence:**
        $$\nabla \cdot \mathbf{c} = \sum_i \frac{\partial c_i}{\partial x^i}$$
        
        ---
        
        ### Chern-Simons Theory
        
        **Action functional:**
        $$S_{CS}[A] = \frac{k}{4\pi} \int_M \text{Tr}\left(A \wedge dA + \frac{2}{3} A \wedge A \wedge A\right)$$
        
        **Discrete approximation:**
        $$CS[A](t) \approx \sum_{ijk} \epsilon_{ijk} A_i \left(\partial_t A_j - \partial_t A_k\right) + \frac{2}{3} A_i A_j A_k$$
        
        **Gauge field construction:** From normalized price deviations
        $$A_i(t) = \frac{p_i(t) - \bar{p}_i}{\sigma_i}$$
        
        ---
        
        ### BRST Symmetry
        
        Quantum field theory requires gauge fixing → introduce ghosts
        
        **BRST charge:**
        $$Q = \sum_i \bar{c}_i \left(-\gamma c_i - \frac{\partial H}{\partial \bar{c}_i}\right)$$
        
        **Nilpotency:** $Q^2 = 0$ (ensures consistency)
        
        **Physical states:** $Q|\psi\rangle = 0$
        
        ---
        
        ### 14D Market Model
        
        **Bosonic sector** (observable):
        1. Log price: $x^0 = \ln(p)$
        2. Log volume: $x^1 = \ln(V)$
        3. Volatility: $x^2 = \sigma_{\text{rolling}}$
        4. Trend: $x^3 = \beta_{\text{regression}}$
        5. Momentum: $x^4 = \text{ROC}$
        6. Liquidity: $x^5 = 1/V$
        7. Sentiment: $x^6 = \text{RSI}$
        
        **Fermionic sector** (ghost):
        $$\theta^i = \frac{\partial x^i}{\partial t}$$
        
        ---
        
        ### Anomaly Detection
        
        **Combined score:**
        $$\mathcal{A}(t) = \alpha \cdot Z[\nabla \cdot \mathbf{c}(t)] + \beta \cdot Z[|\Delta CS(t)|]$$
        
        where $Z[\cdot]$ is z-score normalization.
        
        **Threshold:** $\mathcal{A}(t) > \tau$ (typically $\tau \in [2, 3]$)
        
        **Signal direction:** From sign of ghost divergence
        - $\nabla \cdot \mathbf{c} > 0$ → SHORT (overbought)
        - $\nabla \cdot \mathbf{c} < 0$ → LONG (oversold)
        
        ---
        
        ### References
        
        1. **Supersymmetry:** Wess & Bagger, "Supersymmetry and Supergravity" (1992)
        2. **BRST:** Becchi, Rouet, Stora; Tyutin (1975)
        3. **Chern-Simons:** Witten, "Quantum Field Theory and the Jones Polynomial" (1989)
        4. **Finance Applications:** "Anomaly on Superspace of Time Series Data" (2020)
        """)
    
    with doc_tabs[2]:
        st.markdown("""
        ## 💻 Code Examples
        
        ### Example 1: Basic Grassmann Numbers
        
        ```python
        import hft_py.superspace as sp
        
        # Create Grassmann numbers
        g1 = sp.PyGrassmannNumber(1.0, 1.0)  # scalar=1.0, grass=1.0
        g2 = sp.PyGrassmannNumber(2.0, 1.0)
        
        # Operations
        g_sum = g1 + g2
        g_product = g1 * g2
        
        # Anticommutator (should be 0)
        anticomm = g1 * g2 + g2 * g1
        print(f"Anticommutator: {anticomm}")  # ~0
        
        # Nilpotency
        g1_squared = g1 * g1
        print(f"θ² = {g1_squared}")  # 0
        ```
        
        ---
        
        ### Example 2: Ghost Field Evolution
        
        ```python
        # Setup parameters
        params = sp.PyGhostFieldParams(
            n_modes=7,
            dt=0.01,
            gamma=0.1,
            noise_amplitude=0.1,
            spring_constant=1.0
        )
        
        # Initialize ghost system (needs 2D array)
        initial_coords_2d = [[1.0], [0.5], [0.3], [-0.2], [0.1], [0.4], [-0.1]]
        ghost_system = sp.PyGhostFieldSystem.from_bosonic_coords(
            initial_coords_2d,
            params
        )
        
        # Access pre-computed divergences
        divergences = ghost_system.divergence
        ```
        
        ---
        
        ### Example 3: Chern-Simons Invariant
        
        ```python
        # Create calculator (with window parameter)
        cs_calc = sp.PyChernSimonsCalculator(coupling=1.0, window=30)
        
        # Compute CS series
        prices = load_price_data()
        cs_series = cs_calc.calculate(prices.tolist())
        
        # Compute changes manually
        cs_changes = np.abs(np.diff(cs_series, prepend=cs_series[0]))
        
        # Detect transitions
        threshold = 2.0 * np.std(cs_changes)
        transitions = cs_calc.detect_transitions(cs_series, threshold)
        
        print(f"Detected {len(transitions)} topological transitions")
        ```
        
        ---
        
        ### Example 4: Full Analysis Pipeline
        
        ```python
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        
        # Load data
        prices = load_price_data()
        
        # Build 14D superspace
        bosonic = construct_bosonic_features(prices)  # 7 features
        fermionic = np.diff(bosonic, axis=0, prepend=bosonic[0:1])
        
        # Normalize
        scaler = StandardScaler()
        bosonic_norm = scaler.fit_transform(bosonic)
        fermionic_norm = scaler.fit_transform(fermionic)
        
        superspace_14d = np.hstack([bosonic_norm, fermionic_norm])
        
        # Ghost field analysis
        bosonic_2d = [row.tolist() for row in bosonic_norm]
        ghost_params = sp.PyGhostFieldParams()
        ghost_system = sp.PyGhostFieldSystem.from_bosonic_coords(bosonic_2d, ghost_params)
        ghost_divergences = np.array(ghost_system.divergence)
        
        # Chern-Simons analysis
        cs_calc = sp.PyChernSimonsCalculator(coupling=1.0, window=30)
        cs_series = cs_calc.calculate(prices.tolist())
        cs_changes = np.abs(np.diff(cs_series, prepend=cs_series[0]))
        
        # Combined anomaly score
        ghost_z = zscore(ghost_divergences)
        cs_z = zscore(cs_changes)
        
        alpha, beta = 0.5, 0.5
        anomaly_score = alpha * np.abs(ghost_z) + beta * cs_z
        
        # Detect anomalies
        threshold = 2.5
        anomalies = anomaly_score > threshold
        
        print(f"Detected {np.sum(anomalies)} anomalies")
        ```
        
        ---
        
        ### Example 5: Backtesting
        
        ```python
        # Initialize
        capital = 100000
        position = 0
        equity_curve = [capital]
        
        # Backtest loop
        for i in range(1, len(prices)):
            score = anomaly_score[i]
            ghost_div = ghost_divergences[i]
            price = prices[i]
            
            # Entry logic
            if position == 0 and score > 2.5:
                direction = -1 if ghost_div > 0 else 1  # Contrarian
                position = direction * (capital * 0.5) / price
                entry_price = price
            
            # Exit logic
            elif position != 0 and score < 1.0:
                pnl = position * (price - entry_price)
                capital += pnl
                position = 0
            
            # Update equity
            if position != 0:
                unrealized = position * (price - entry_price)
                equity = capital + unrealized
            else:
                equity = capital
            
            equity_curve.append(equity)
        
        # Compute metrics
        total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
        print(f"Total return: {total_return*100:.2f}%")
        ```
        """)
    
    with doc_tabs[3]:
        st.markdown("""
        ## 🔧 Troubleshooting
        
        ### Rust Bindings Not Available
        
        **Error:** `ImportError: No module named 'hft_py.superspace'`
        
        **Solution:**
        ```bash
        cd rust_python_bindings
        maturin develop --release
        ```
        
        Or rebuild with:
        ```bash
        make build-python
        ```
        
        ---
        
        ### Insufficient Data
        
        **Error:** `ValueError: Not enough data points`
        
        **Cause:** Need at least 200+ data points for reliable analysis
        
        **Solution:**
        - Load longer historical data
        - Reduce window sizes in parameters
        - Use higher frequency data (hourly vs daily)
        
        ---
        
        ### No Anomalies Detected
        
        **Issue:** Analysis runs but finds 0 anomalies
        
        **Possible causes:**
        1. Threshold too high → Lower to 2.0σ
        2. Market too stable → Increase ghost noise
        3. Wrong weights → Try α=0.6, β=0.4
        
        ---
        
        ### Numerical Instabilities
        
        **Issue:** NaN or Inf values in results
        
        **Solutions:**
        - Check for zero/missing data in inputs
        - Ensure log(0) doesn't occur → use log(x + 1)
        - Reduce ghost noise amplitude
        - Increase friction coefficient γ
        
        ---
        
        ### Slow Performance
        
        **Issue:** Analysis takes > 2 minutes
        
        **Optimizations:**
        - Reduce window sizes
        - Use fewer ghost modes (n_modes=5)
        - Downsample data before analysis
        - Ensure Rust bindings compiled with --release
        
        ---
        
        ### Memory Issues
        
        **Issue:** Out of memory errors
        
        **Solutions:**
        - Analyze fewer assets at once
        - Reduce data length
        - Don't store full 14D history
        - Use incremental processing
        
        ---
        
        ### Unexpected Trading Signals
        
        **Issue:** Too many or contradictory signals
        
        **Solutions:**
        - Increase entry threshold
        - Add cooldown period between trades
        - Use stricter exit conditions
        - Filter by CS transition confirmation
        
        ---
        
        ### Installation Issues
        
        **Rust not found:**
        ```bash
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
        source $HOME/.cargo/env
        ```
        
        **Maturin not found:**
        ```bash
        pip install maturin
        ```
        
        **Compilation errors:**
        ```bash
        rustup update
        cargo clean
        maturin develop --release
        ```
        
        ---
        
        ### Getting Help
        
        1. **Check logs:** Look for detailed error messages
        2. **Documentation:** Review mathematical background
        3. **Examples:** Run provided code examples
        4. **Issues:** File bug report with:
           - Error message
           - Data characteristics
           - Parameter settings
           - System info
        """)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>🌌 <strong>Superspace Anomaly Detection Lab</strong> | Built with theoretical physics & Rust</p>
    <p>Implements concepts from quantum field theory, topology, and supersymmetry for financial markets</p>
</div>
""", unsafe_allow_html=True)
