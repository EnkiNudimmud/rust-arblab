"""
Shared UI Components
Common navigation and styling for all pages
"""

import streamlit as st


def render_sidebar_navigation(current_page="Home"):
    """
    Render sidebar navigation with collapsible sections
    
    Args:
        current_page: Name of current page to highlight
    """
    with st.sidebar:
        st.markdown("### 🗺️ Navigation")
        st.markdown("---")
        
        # Home
        if current_page == "Home":
            st.success("🏠 **Home**")
        else:
            if st.button("🏠 Home", use_container_width=True, key="nav_home"):
                st.switch_page("HFT_Arbitrage_Lab.py")
        
        st.markdown("---")
        
        # Data & Market Section
        with st.expander("📊 **Data & Market**", expanded=current_page in ["Data Loader", "Portfolio View"]):
            if current_page == "Data Loader":
                st.success("💾 **Data Loader**")
            else:
                if st.button("💾 Data Loader", use_container_width=True, key="nav_data"):
                    st.switch_page("pages/data_loader.py")
            
            if current_page == "Portfolio View":
                st.success("💼 **Portfolio View**")
            else:
                if st.button("💼 Portfolio View", use_container_width=True, key="nav_portfolio"):
                    st.switch_page("pages/portfolio_view.py")
        
        # Research Labs Section
        with st.expander("🔬 **Research Labs**", expanded=current_page in [
            "Mean Reversion Lab", "Rough Heston Lab", "Chiarella Model Lab", 
            "Signature Methods Lab", "Portfolio Analytics Lab"
        ]):
            if current_page == "Mean Reversion Lab":
                st.success("📉 **Mean Reversion Lab**")
            else:
                if st.button("📉 Mean Reversion Lab", use_container_width=True, key="nav_meanrev"):
                    st.switch_page("pages/lab_mean_reversion.py")
            
            if current_page == "Rough Heston Lab":
                st.success("📈 **Rough Heston Lab**")
            else:
                if st.button("📈 Rough Heston Lab", use_container_width=True, key="nav_heston"):
                    st.switch_page("pages/lab_rough_heston.py")
            
            if current_page == "Chiarella Model Lab":
                st.success("🌀 **Chiarella Model Lab**")
            else:
                if st.button("🌀 Chiarella Model Lab", use_container_width=True, key="nav_chiarella"):
                    st.switch_page("pages/lab_chiarella.py")
            
            if current_page == "Signature Methods Lab":
                st.success("✍️ **Signature Methods Lab**")
            else:
                if st.button("✍️ Signature Methods Lab", use_container_width=True, key="nav_signature"):
                    st.switch_page("pages/lab_signature_methods.py")
            
            if current_page == "Portfolio Analytics Lab":
                st.success("📊 **Portfolio Analytics Lab**")
            else:
                if st.button("📊 Portfolio Analytics Lab", use_container_width=True, key="nav_analytics"):
                    st.switch_page("pages/lab_portfolio_analytics.py")
        
        # Trading Strategies Section
        with st.expander("⚡ **Trading Strategies**", expanded=current_page in [
            "Strategy Backtest", "Arbitrage Analysis", "Derivatives Strategies", "Options Strategies"
        ]):
            if current_page == "Strategy Backtest":
                st.success("📈 **Strategy Backtest**")
            else:
                if st.button("📈 Strategy Backtest", use_container_width=True, key="nav_backtest"):
                    st.switch_page("pages/strategy_backtest.py")
            
            if current_page == "Arbitrage Analysis":
                st.success("🔄 **Arbitrage Analysis**")
            else:
                if st.button("🔄 Arbitrage Analysis", use_container_width=True, key="nav_arbitrage"):
                    st.switch_page("pages/arbitrage_analysis.py")
            
            if current_page == "Derivatives Strategies":
                st.success("📊 **Derivatives Strategies**")
            else:
                if st.button("📊 Derivatives Strategies", use_container_width=True, key="nav_derivatives"):
                    st.switch_page("pages/derivatives_strategies.py")
            
            if current_page == "Options Strategies":
                st.success("🎯 **Options Strategies**")
            else:
                if st.button("🎯 Options Strategies", use_container_width=True, key="nav_options"):
                    st.switch_page("pages/options_strategies.py")
        
        # Live Trading Section
        with st.expander("🔴 **Live Trading**", expanded=current_page in [
            "Live Trading", "Live Derivatives", "Affine Models"
        ]):
            if current_page == "Live Trading":
                st.success("⚡ **Live Trading**")
            else:
                if st.button("⚡ Live Trading", use_container_width=True, key="nav_live"):
                    st.switch_page("pages/live_trading.py")
            
            if current_page == "Live Derivatives":
                st.success("🎲 **Live Derivatives**")
            else:
                if st.button("🎲 Live Derivatives", use_container_width=True, key="nav_live_deriv"):
                    st.switch_page("pages/derivatives.py")
            
            if current_page == "Affine Models":
                st.success("📐 **Affine Models**")
            else:
                if st.button("📐 Affine Models", use_container_width=True, key="nav_affine"):
                    st.switch_page("pages/affine_models.py")


def apply_custom_css():
    """Apply consistent custom CSS styling across all pages"""
    st.markdown("""
    <style>
        .lab-header {
            font-size: 2.5rem;
            font-weight: bold;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 1rem;
        }
        
        .main-header {
            font-size: 3.5rem;
            font-weight: 900;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            padding: 2rem 0 1rem 0;
            margin-bottom: 0.5rem;
        }
        
        .subtitle {
            text-align: center;
            color: #666;
            font-size: 1.2rem;
            margin-bottom: 2rem;
        }
        
        .feature-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 15px;
            color: white;
            margin: 1rem 0;
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
            transition: all 0.3s ease;
            height: 100%;
        }
        
        .feature-card:hover {
            transform: translateY(-8px);
            box-shadow: 0 15px 40px rgba(102, 126, 234, 0.5);
        }
        
        .status-card {
            background: #f8f9fa;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 5px solid #667eea;
            margin: 0.5rem 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        .info-card {
            background: #f0f7ff;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 5px solid #667eea;
            margin: 1rem 0;
        }
        
        .metric-card {
            background: #f0f7ff;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 5px solid #667eea;
            margin: 0.5rem 0;
        }
        
        .status-online {
            color: #10b981;
            font-weight: bold;
            font-size: 1.1rem;
        }
        
        .status-offline {
            color: #ef4444;
            font-weight: bold;
            font-size: 1.1rem;
        }
        
        .metric-big {
            font-size: 2.5rem;
            font-weight: bold;
            color: #667eea;
        }
        
        /* Hide native Streamlit page navigation */
        [data-testid="stSidebarNav"] {
            display: none;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
        }
        
        [data-testid="stSidebar"] .stButton button {
            border-radius: 8px;
            transition: all 0.2s;
        }
        
        [data-testid="stSidebar"] .stButton button:hover {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            transform: translateX(5px);
        }
    </style>
    """, unsafe_allow_html=True)
