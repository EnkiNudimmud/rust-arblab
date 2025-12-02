"""
Shared UI Components
Common navigation and styling for all pages
"""

import streamlit as st


def toggle_theme():
    """Toggle between light and dark mode"""
    if 'theme_mode' not in st.session_state:
        st.session_state.theme_mode = 'dark'  # Default to dark
    
    # Toggle
    st.session_state.theme_mode = 'light' if st.session_state.theme_mode == 'dark' else 'dark'


def get_theme_colors():
    """Get color scheme based on current theme"""
    if 'theme_mode' not in st.session_state:
        st.session_state.theme_mode = 'dark'
    
    if st.session_state.theme_mode == 'light':
        return {
            'bg_primary': '#ffffff',
            'bg_secondary': '#f8f9fa',
            'bg_card': '#ffffff',
            'bg_feature': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            'text_primary': '#1a1a1a',
            'text_secondary': '#4a4a4a',
            'text_muted': '#666666',
            'border': '#e0e0e0',
            'shadow': 'rgba(0, 0, 0, 0.1)',
            'accent': '#667eea',
            'success': '#10b981',
            'error': '#ef4444',
            'sidebar_bg': 'linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%)',
            'code_bg': '#f5f5f5',
            'code_text': '#2d2d2d',
        }
    else:  # dark mode
        return {
            'bg_primary': '#1a1a1a',
            'bg_secondary': '#2d2d2d',
            'bg_card': '#242424',
            'bg_feature': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            'text_primary': '#e8e8e8',
            'text_secondary': '#c0c0c0',
            'text_muted': '#888888',
            'border': '#404040',
            'shadow': 'rgba(0, 0, 0, 0.5)',
            'accent': '#7c92ff',
            'success': '#10b981',
            'error': '#ef4444',
            'sidebar_bg': 'linear-gradient(180deg, #242424 0%, #1a1a1a 100%)',
            'code_bg': '#2d2d2d',
            'code_text': '#e8e8e8',
        }


def render_sidebar_navigation(current_page="Home"):
    """
    Render sidebar navigation with collapsible sections
    
    Args:
        current_page: Name of current page to highlight
    """
    with st.sidebar:
        # Theme toggle button at the top
        if 'theme_mode' not in st.session_state:
            st.session_state.theme_mode = 'dark'
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("### 🗺️ Navigation")
        with col2:
            theme_icon = "☀️" if st.session_state.theme_mode == 'dark' else "🌙"
            if st.button(theme_icon, key="theme_toggle", help="Toggle light/dark mode"):
                toggle_theme()
                st.rerun()
        
        st.markdown("---")
        
        # Home
        if current_page == "Home":
            st.success("🏠 **Home**")
        else:
            if st.button("🏠 Home", use_container_width=True, key="nav_home"):
                # Navigate to main app - Streamlit resolves from run location
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
            "Signature Methods Lab", "Portfolio Analytics Lab", "PCA Arbitrage Lab",
            "Momentum Trading Lab", "Market Making Lab"
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
            
            if current_page == "PCA Arbitrage Lab":
                st.success("🎯 **PCA Arbitrage Lab**")
            else:
                if st.button("🎯 PCA Arbitrage Lab", use_container_width=True, key="nav_pca"):
                    st.switch_page("pages/lab_pca_arbitrage.py")
            
            if current_page == "Momentum Trading Lab":
                st.success("📈 **Momentum Trading Lab**")
            else:
                if st.button("📈 Momentum Trading Lab", use_container_width=True, key="nav_momentum"):
                    st.switch_page("pages/lab_momentum.py")
            
            if current_page == "Market Making Lab":
                st.success("🌊 **Market Making Lab**")
            else:
                if st.button("🌊 Market Making Lab", use_container_width=True, key="nav_mm"):
                    st.switch_page("pages/lab_market_making.py")
        
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
    """Apply consistent custom CSS styling across all pages with theme support"""
    colors = get_theme_colors()
    
    st.markdown(f"""
    <style>
        /* Main App Background */
        .stApp {{
            background-color: {colors['bg_primary']};
            color: {colors['text_primary']};
        }}
        
        /* Main content area */
        .main .block-container {{
            background-color: {colors['bg_primary']};
            color: {colors['text_primary']};
        }}
        
        /* Headers */
        .lab-header {{
            font-size: 2.5rem;
            font-weight: bold;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 1rem;
        }}
        
        .main-header {{
            font-size: 3.5rem;
            font-weight: 900;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            padding: 2rem 0 1rem 0;
            margin-bottom: 0.5rem;
        }}
        
        .subtitle {{
            text-align: center;
            color: {colors['text_muted']};
            font-size: 1.2rem;
            margin-bottom: 2rem;
        }}
        
        /* Text elements */
        h1, h2, h3, h4, h5, h6 {{
            color: {colors['text_primary']} !important;
        }}
        
        p, span, div {{
            color: {colors['text_primary']};
        }}
        
        label {{
            color: {colors['text_primary']} !important;
        }}
        
        /* Cards */
        .feature-card {{
            background: {colors['bg_feature']};
            padding: 2rem;
            border-radius: 15px;
            color: white;
            margin: 1rem 0;
            box-shadow: 0 10px 30px {colors['shadow']};
            transition: all 0.3s ease;
            height: 100%;
        }}
        
        .feature-card:hover {{
            transform: translateY(-8px);
            box-shadow: 0 15px 40px rgba(102, 126, 234, 0.5);
        }}
        
        .feature-card h3 {{
            color: white !important;
        }}
        
        .feature-card ul, .feature-card li {{
            color: white !important;
        }}
        
        .status-card {{
            background: {colors['bg_card']};
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 5px solid {colors['accent']};
            margin: 0.5rem 0;
            box-shadow: 0 2px 8px {colors['shadow']};
            color: {colors['text_primary']};
        }}
        
        .status-card strong {{
            color: {colors['text_primary']};
        }}
        
        .info-card {{
            background: {colors['bg_secondary']};
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 5px solid {colors['accent']};
            margin: 1rem 0;
            color: {colors['text_primary']};
        }}
        
        .metric-card {{
            background: {colors['bg_secondary']};
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 5px solid {colors['accent']};
            margin: 0.5rem 0;
            color: {colors['text_primary']};
        }}
        
        /* Status indicators */
        .status-online {{
            color: {colors['success']};
            font-weight: bold;
            font-size: 1.1rem;
        }}
        
        .status-offline {{
            color: {colors['error']};
            font-weight: bold;
            font-size: 1.1rem;
        }}
        
        .metric-big {{
            font-size: 2.5rem;
            font-weight: bold;
            color: {colors['accent']};
        }}
        
        /* Input fields and text areas */
        .stTextInput input, .stTextArea textarea, .stNumberInput input {{
            background-color: {colors['bg_secondary']} !important;
            color: {colors['text_primary']} !important;
            border: 1px solid {colors['border']} !important;
        }}
        
        .stSelectbox select, .stMultiSelect select {{
            background-color: {colors['bg_secondary']} !important;
            color: {colors['text_primary']} !important;
            border: 1px solid {colors['border']} !important;
        }}
        
        /* Dataframes */
        .dataframe {{
            background-color: {colors['bg_card']} !important;
            color: {colors['text_primary']} !important;
        }}
        
        .dataframe th {{
            background-color: {colors['bg_secondary']} !important;
            color: {colors['text_primary']} !important;
        }}
        
        .dataframe td {{
            background-color: {colors['bg_card']} !important;
            color: {colors['text_primary']} !important;
            border: 1px solid {colors['border']} !important;
        }}
        
        /* Code blocks */
        code {{
            background-color: {colors['code_bg']} !important;
            color: {colors['code_text']} !important;
            padding: 0.2rem 0.4rem;
            border-radius: 4px;
        }}
        
        pre {{
            background-color: {colors['code_bg']} !important;
            color: {colors['code_text']} !important;
            border: 1px solid {colors['border']};
            padding: 1rem;
            border-radius: 8px;
        }}
        
        /* Expanders */
        .streamlit-expanderHeader {{
            background-color: {colors['bg_secondary']} !important;
            color: {colors['text_primary']} !important;
            border: 1px solid {colors['border']};
        }}
        
        .streamlit-expanderContent {{
            background-color: {colors['bg_card']} !important;
            border: 1px solid {colors['border']};
            color: {colors['text_primary']};
        }}
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {{
            background-color: {colors['bg_secondary']};
            border-bottom: 2px solid {colors['border']};
        }}
        
        .stTabs [data-baseweb="tab"] {{
            color: {colors['text_secondary']} !important;
            background-color: transparent;
        }}
        
        .stTabs [aria-selected="true"] {{
            color: {colors['accent']} !important;
            border-bottom: 2px solid {colors['accent']};
        }}
        
        /* Metrics */
        [data-testid="stMetricValue"] {{
            color: {colors['text_primary']} !important;
        }}
        
        [data-testid="stMetricLabel"] {{
            color: {colors['text_secondary']} !important;
        }}
        
        /* Info/Warning/Error boxes */
        .stAlert {{
            background-color: {colors['bg_card']};
            color: {colors['text_primary']};
            border: 1px solid {colors['border']};
        }}
        
        /* Hide native Streamlit page navigation */
        [data-testid="stSidebarNav"] {{
            display: none;
        }}
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {{
            background: {colors['sidebar_bg']};
        }}
        
        [data-testid="stSidebar"] > div:first-child {{
            background: {colors['sidebar_bg']};
        }}
        
        [data-testid="stSidebar"] * {{
            color: {colors['text_primary']} !important;
        }}
        
        [data-testid="stSidebar"] .stButton button {{
            border-radius: 8px;
            transition: all 0.2s;
            background-color: {colors['bg_card']};
            border: 1px solid {colors['border']};
            color: {colors['text_primary']} !important;
        }}
        
        [data-testid="stSidebar"] .stButton button:hover {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white !important;
            transform: translateX(5px);
        }}
        
        /* Main content buttons - improved visibility */
        .stButton button {{
            border-radius: 8px;
            transition: all 0.3s;
            font-weight: 600;
            padding: 0.5rem 1rem;
        }}
        
        .stButton button[kind="primary"] {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white !important;
            border: none;
        }}
        
        .stButton button[kind="primary"]:hover {{
            background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }}
        
        .stButton button[kind="secondary"] {{
            background-color: {colors['bg_secondary']};
            color: {colors['text_primary']} !important;
            border: 1px solid {colors['border']};
        }}
        
        .stButton button[kind="secondary"]:hover {{
            background-color: {colors['accent']};
            color: white !important;
            border-color: {colors['accent']};
        }}
        
        /* Default buttons */
        .stButton button:not([kind]) {{
            background-color: {colors['bg_secondary']};
            color: {colors['text_primary']} !important;
            border: 1px solid {colors['border']};
        }}
        
        .stButton button:not([kind]):hover {{
            background-color: {colors['accent']};
            color: white !important;
            transform: translateY(-2px);
        }}
        
        /* Markdown in sidebar */
        [data-testid="stSidebar"] .stMarkdown {{
            color: {colors['text_primary']};
        }}
        
        /* Dividers */
        hr {{
            border-color: {colors['border']};
        }}
        
        /* Success boxes */
        [data-testid="stSidebar"] .element-container div[data-testid="stMarkdownContainer"] {{
            color: {colors['text_primary']};
        }}
        
        /* Download buttons */
        .stDownloadButton button {{
            background-color: {colors['bg_secondary']};
            color: {colors['text_primary']};
            border: 1px solid {colors['border']};
        }}
        
        .stDownloadButton button:hover {{
            background-color: {colors['accent']};
            color: white;
        }}
        
        /* Select box (dropdown) styling - ENHANCED FIX FOR DARK MODE */
        .stSelectbox div[data-baseweb="select"] {{
            background-color: {colors['bg_card']} !important;
            border: 1px solid {colors['border']} !important;
        }}
        
        .stSelectbox div[data-baseweb="select"] > div {{
            background-color: {colors['bg_card']} !important;
            border: 1px solid {colors['border']} !important;
            color: {colors['text_primary']} !important;
        }}
        
        .stSelectbox div[data-baseweb="select"] > div:hover {{
            border-color: {colors['accent']} !important;
        }}
        
        /* Dropdown menu container */
        .stSelectbox [data-baseweb="popover"] {{
            background-color: {colors['bg_card']} !important;
        }}
        
        /* Dropdown menu options list */
        .stSelectbox [role="listbox"] {{
            background-color: {colors['bg_card']} !important;
            border: 1px solid {colors['border']} !important;
        }}
        
        .stSelectbox ul[role="listbox"] {{
            background-color: {colors['bg_card']} !important;
        }}
        
        /* Individual dropdown options - NUCLEAR OPTION WITH INLINE STYLE OVERRIDE */
        .stSelectbox [role="option"] {{
            background-color: white !important;
            color: #000000 !important;
            padding: 8px 12px !important;
            -webkit-text-fill-color: #000000 !important;
        }}
        
        .stSelectbox [role="option"] * {{
            color: #000000 !important;
            -webkit-text-fill-color: #000000 !important;
        }}
        
        .stSelectbox [role="option"] span {{
            color: #000000 !important;
            -webkit-text-fill-color: #000000 !important;
        }}
        
        .stSelectbox [role="option"] div {{
            color: #000000 !important;
            -webkit-text-fill-color: #000000 !important;
        }}
        
        .stSelectbox [role="option"]:hover {{
            background-color: {colors['accent']} !important;
            color: white !important;
            -webkit-text-fill-color: white !important;
        }}
        
        .stSelectbox [role="option"]:hover * {{
            color: white !important;
            -webkit-text-fill-color: white !important;
        }}
        
        .stSelectbox [aria-selected="true"] {{
            background-color: #f0f0f0 !important;
            color: #000000 !important;
            -webkit-text-fill-color: #000000 !important;
            font-weight: bold !important;
        }}
        
        .stSelectbox [aria-selected="true"] * {{
            color: #000000 !important;
            -webkit-text-fill-color: #000000 !important;
        }}
        
        /* Selected option text in dropdown button */
        .stSelectbox div[data-baseweb="select"] span {{
            color: {colors['text_primary']} !important;
        }}
        
        .stSelectbox div[data-baseweb="select"] div {{
            color: {colors['text_primary']} !important;
        }}
        
        /* Dropdown list items - ABSOLUTE BLACK TEXT */
        .stSelectbox li {{
            background-color: white !important;
            color: #000000 !important;
            -webkit-text-fill-color: #000000 !important;
        }}
        
        .stSelectbox li * {{
            color: #000000 !important;
            -webkit-text-fill-color: #000000 !important;
        }}
        
        .stSelectbox li span {{
            color: #000000 !important;
            -webkit-text-fill-color: #000000 !important;
        }}
        
        .stSelectbox li div {{
            color: #000000 !important;
            -webkit-text-fill-color: #000000 !important;
        }}
        
        .stSelectbox li:hover {{
            background-color: {colors['accent']} !important;
            color: white !important;
            -webkit-text-fill-color: white !important;
        }}
        
        .stSelectbox li:hover * {{
            color: white !important;
            -webkit-text-fill-color: white !important;
        }}
        
        /* BaseWeb specific selectors - ABSOLUTE BLACK */
        .stSelectbox [class*="option"] {{
            color: #000000 !important;
            -webkit-text-fill-color: #000000 !important;
        }}
        
        .stSelectbox [class*="option"] * {{
            color: #000000 !important;
            -webkit-text-fill-color: #000000 !important;
        }}
        
        .stSelectbox [class*="ListItem"] {{
            background-color: white !important;
            color: #000000 !important;
            -webkit-text-fill-color: #000000 !important;
        }}
        
        .stSelectbox [class*="ListItem"] * {{
            color: #000000 !important;
            -webkit-text-fill-color: #000000 !important;
        }}
        
        /* Dropdown menu container - WHITE BACKGROUND */
        .stSelectbox [data-baseweb="popover"] {{
            background-color: white !important;
        }}
        
        .stSelectbox [data-baseweb="popover"] * {{
            color: #000000 !important;
            -webkit-text-fill-color: #000000 !important;
        }}
        
        .stSelectbox [role="listbox"] {{
            background-color: white !important;
        }}
        
        .stSelectbox [role="listbox"] * {{
            color: #000000 !important;
            -webkit-text-fill-color: #000000 !important;
        }}
        
        .stSelectbox ul[role="listbox"] {{
            background-color: white !important;
        }}
        
        .stSelectbox ul[role="listbox"] * {{
            color: #000000 !important;
            -webkit-text-fill-color: #000000 !important;
        }}
        
        /* Multi-select - ENHANCED */
        .stMultiSelect div[data-baseweb="select"] {{
            background-color: {colors['bg_card']} !important;
            border: 1px solid {colors['border']} !important;
        }}
        
        .stMultiSelect div[data-baseweb="select"] > div {{
            background-color: {colors['bg_card']} !important;
            border: 1px solid {colors['border']} !important;
        }}
        
        .stMultiSelect [data-baseweb="popover"] {{
            background-color: {colors['bg_card']} !important;
        }}
        
        .stMultiSelect [role="listbox"] {{
            background-color: {colors['bg_card']} !important;
            border: 1px solid {colors['border']} !important;
        }}
        
        .stMultiSelect ul[role="listbox"] {{
            background-color: {colors['bg_card']} !important;
        }}
        
        .stMultiSelect [role="option"] {{
            background-color: {colors['bg_card']} !important;
            color: {colors['text_primary']} !important;
            padding: 8px 12px !important;
        }}
        
        .stMultiSelect [role="option"]:hover {{
            background-color: {colors['accent']} !important;
            color: white !important;
        }}
        
        .stMultiSelect li {{
            background-color: {colors['bg_card']} !important;
            color: {colors['text_primary']} !important;
        }}
        
        .stMultiSelect li:hover {{
            background-color: {colors['accent']} !important;
            color: white !important;
        }}
        
        .stMultiSelect span {{
            color: {colors['text_primary']} !important;
        }}
        
        .stMultiSelect div {{
            color: {colors['text_primary']} !important;
        }}
        
        /* Selected tags in multi-select */
        .stMultiSelect [data-baseweb="tag"] {{
            background-color: {colors['accent']} !important;
            color: white !important;
        }}
        
        /* Number input and text input */
        .stNumberInput input, .stTextInput input {{
            background-color: {colors['bg_card']} !important;
            color: {colors['text_primary']} !important;
            border: 1px solid {colors['border']} !important;
        }}
        
        .stNumberInput input:focus, .stTextInput input:focus {{
            border-color: {colors['accent']} !important;
        }}
        
        /* Text area */
        .stTextArea textarea {{
            background-color: {colors['bg_card']} !important;
            color: {colors['text_primary']} !important;
            border: 1px solid {colors['border']} !important;
        }}
        
        .stTextArea textarea:focus {{
            border-color: {colors['accent']} !important;
        }}
        
        /* Slider */
        .stSlider {{
            color: {colors['text_primary']} !important;
        }}
        
        .stSlider [data-baseweb="slider"] {{
            background-color: {colors['bg_card']} !important;
        }}
    </style>
    
    <script>
        // Force dropdown text color after DOM loads
        function fixDropdownColors() {{
            const observer = new MutationObserver(function(mutations) {{
                // Find all dropdown options
                const options = document.querySelectorAll('.stSelectbox [role="option"], .stSelectbox li, .stSelectbox [class*="option"], .stSelectbox [class*="ListItem"]');
                options.forEach(option => {{
                    option.style.setProperty('color', '#000000', 'important');
                    option.style.setProperty('-webkit-text-fill-color', '#000000', 'important');
                    
                    // Also fix all child elements
                    const children = option.querySelectorAll('*');
                    children.forEach(child => {{
                        child.style.setProperty('color', '#000000', 'important');
                        child.style.setProperty('-webkit-text-fill-color', '#000000', 'important');
                    }});
                }});
                
                // Fix dropdown containers
                const containers = document.querySelectorAll('.stSelectbox [role="listbox"], .stSelectbox ul[role="listbox"], .stSelectbox [data-baseweb="popover"]');
                containers.forEach(container => {{
                    container.style.setProperty('background-color', 'white', 'important');
                    const allElements = container.querySelectorAll('*');
                    allElements.forEach(el => {{
                        el.style.setProperty('color', '#000000', 'important');
                        el.style.setProperty('-webkit-text-fill-color', '#000000', 'important');
                    }});
                }});
            }});
            
            observer.observe(document.body, {{
                childList: true,
                subtree: true
            }});
            
            // Run once immediately
            setTimeout(fixDropdownColors, 100);
        }}
        
        // Execute when page loads
        if (document.readyState === 'loading') {{
            document.addEventListener('DOMContentLoaded', fixDropdownColors);
        }} else {{
            fixDropdownColors();
        }}
    </script>
    """, unsafe_allow_html=True)
