"""
Signature Methods Lab
Path signature analysis for time series and trading strategies
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import shared UI components
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.ui_components import render_sidebar_navigation, apply_custom_css

st.set_page_config(page_title="Signature Methods Lab", page_icon="✍️", layout="wide")

# Render sidebar navigation and apply CSS
render_sidebar_navigation(current_page="Signature Methods Lab")
apply_custom_css()

st.markdown('<h1 class="lab-header">✍️ Signature Methods Lab</h1>', unsafe_allow_html=True)
st.markdown("### Path signature analysis for feature extraction and classification")
st.markdown("---")

# Main content
tab1, tab2, tab3 = st.tabs(["📚 Introduction", "🔬 Analysis", "⚡ Trading Signals"])

with tab1:
    st.markdown("### What are Path Signatures?")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        Path signatures are a powerful mathematical tool for analyzing time series data. 
        They provide a coordinate-free description of paths that captures:
        
        - **All statistical moments** of the path
        - **Non-linear interactions** between coordinates
        - **Order of events** (not just values)
        - **Scale-invariant features**
        
        #### Mathematical Foundation
        
        For a path $X: [0,T] \\to \\mathbb{R}^d$, the signature is defined as:
        
        $$S(X)_{0,T} = (1, S^1, S^2, S^3, ...)$$
        
        where $S^k$ are iterated integrals:
        
        $$S^k_{i_1,...,i_k} = \\int_{0<t_1<...<t_k<T} dX^{i_1}_{t_1} ... dX^{i_k}_{t_k}$$
        
        #### Applications in Finance
        
        - **Feature extraction** from price paths
        - **Regime classification** 
        - **Optimal execution** strategies
        - **Signature trading** strategies
        - **Model-free pricing** of path-dependent options
        """)
    
    with col2:
        st.markdown("""
        ### Key Properties
        
        ✅ **Universal**: Characterizes paths uniquely
        
        ✅ **Efficient**: Low-dimensional representation
        
        ✅ **Robust**: Insensitive to noise
        
        ✅ **Interpretable**: Each term has meaning
        
        ✅ **Composable**: Signatures multiply along paths
        """)
        
        st.info("""
        **References:**
        - Lyons (1998): Rough paths theory
        - Levin et al. (2013): Signatures in ML
        - Cochrane & Lyons (2019): Signature methods in finance
        """)

with tab2:
    st.markdown("### Signature Analysis")
    
    if 'historical_data' not in st.session_state or st.session_state.historical_data is None:
        st.warning("⚠️ Please load data first from the Data Loader page")
        if st.button("💾 Go to Data Loader"):
            st.switch_page("pages/data_loader.py")
    else:
        st.info("🚧 Coming soon: Interactive signature computation and visualization")
        
        st.markdown("""
        ### Planned Features:
        
        1. **Signature Computation**
           - Truncated signature up to level N
           - Log-signature for efficient computation
           - Signature kernel for comparison
        
        2. **Path Classification**
           - Bullish vs bearish pattern recognition
           - Volatility regime detection
           - Trend strength estimation
        
        3. **Visualization**
           - Signature coordinates over time
           - Path space visualization
           - Feature importance analysis
        
        4. **Trading Strategy**
           - Signature-based entry/exit signals
           - Pattern matching for similar historical paths
           - Expected return prediction
        """)

with tab3:
    st.markdown("### Signature Trading Strategies")
    
    st.info("🚧 Coming soon: Signature-based trading signals")
    
    st.markdown("""
    ### Strategy Components:
    
    #### 1. Pattern Recognition
    - Compute signatures of historical profitable patterns
    - Match current market conditions to historical signatures
    - Generate signals when patterns are similar
    
    #### 2. Optimal Execution
    - Use signatures to predict short-term price moves
    - Optimize order placement based on path features
    - Minimize market impact using signature dynamics
    
    #### 3. Risk Management
    - Detect regime changes from signature evolution
    - Early warning signals from signature divergence
    - Position sizing based on path uncertainty
    
    #### Example: Signature Trading Rule
    
    ```python
    # Compute signature of recent price path
    sig_current = compute_signature(prices[-window:], level=3)
    
    # Compare to profitable historical patterns
    for hist_pattern in profitable_patterns:
        similarity = signature_kernel(sig_current, hist_pattern)
        if similarity > threshold:
            # Generate trading signal
            signal = generate_signal(hist_pattern.label)
    ```
    """)

# Sidebar
with st.sidebar:
    st.markdown("### 🎛️ Settings")
    
    truncation_level = st.slider("Signature Level", 2, 5, 3)
    st.info(f"Computing signature up to level {truncation_level}")
    
    st.markdown("---")
    st.markdown("### 📚 Resources")
    st.markdown("""
    - [Rough Paths Theory](https://en.wikipedia.org/wiki/Rough_path)
    - [esig Library](https://esig.readthedocs.io/)
    - [Signatures in ML](https://arxiv.org/abs/1603.03788)
    """)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>✍️ Signature Methods Lab | Part of HFT Arbitrage Lab</p>
</div>
""", unsafe_allow_html=True)
