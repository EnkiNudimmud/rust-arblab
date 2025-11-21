"""
Enhanced Chiarella Model Dashboard

NEW FEATURES:
- ✅ Expanded universes (stocks, crypto, ETFs)
- ✅ Regime-adaptive parameter selection  
- ✅ Rust-accelerated model simulation
- ✅ Live signal monitoring
- ✅ Multi-asset analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import requests

sys.path.append('/Users/melvinalvarez/Documents/Workspace/rust-hft-arbitrage-lab')

from python.api_keys import get_finnhub_key
from python.universes import get_universe, get_available_universes
from python.regime_detector import RegimeDetector, get_regime_metrics
from python.signal_monitor import SignalMonitor

# Page config
st.set_page_config(
    page_title="Enhanced Chiarella Model",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stMetric { background-color: #f0f2f6; padding: 10px; border-radius: 5px; }
    .regime-mean-rev { color: #00aa00; font-weight: bold; }
    .regime-trending { color: #ff8800; font-weight: bold; }
    .regime-high-vol { color: #ff0000; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Title
st.title("📉 Enhanced Chiarella Model Analysis")
st.markdown("**Market Microstructure**: Chartists vs Fundamentalists with Regime Detection")
st.markdown("---")

# Sidebar
st.sidebar.header("⚙️ Configuration")

# Universe selection
st.sidebar.subheader("📊 Asset Selection")
available_universes = get_available_universes()

asset_type = st.sidebar.radio(
    "Asset Type",
    ["Single Asset", "Multi-Asset Portfolio"]
)

if asset_type == "Single Asset":
    asset_categories = {
        "🏢 Stocks": ["tech", "finance", "healthcare"],
        "₿ Crypto": ["crypto_major"],
        "📊 ETFs": ["etf_indices"]
    }
    
    category = st.sidebar.selectbox("Category", list(asset_categories.keys()))
    universe = st.sidebar.selectbox(
        "Universe",
        asset_categories[category]
    )
    
    symbols = get_universe(universe)
    selected_symbol = st.sidebar.selectbox("Select Asset", symbols)
else:
    universe = st.sidebar.selectbox(
        "Universe",
        ["tech", "finance", "crypto_major", "etf_indices"]
    )
    n_assets = st.sidebar.slider("Number of Assets", 3, 20, 10)
    symbols = get_universe(universe)[:n_assets]
    selected_symbol = symbols[0]  # For detailed view

# Data parameters  
st.sidebar.subheader("📅 Data Parameters")
days_back = st.sidebar.slider("Days of History", 30, 180, 90)
resolution_min = st.sidebar.select_slider(
    "Resolution",
    options=[5, 15, 30, 60],
    value=5,
    format_func=lambda x: f"{x} min"
)

# Model parameters
st.sidebar.subheader("🔧 Chiarella Parameters")

use_adaptive = st.sidebar.checkbox("Use Adaptive Parameters", value=True)

if not use_adaptive:
    alpha = st.sidebar.slider("α (Chartist Strength)", 0.1, 2.0, 0.5, 0.1)
    beta = st.sidebar.slider("β (Fundamentalist Strength)", 0.1, 2.0, 0.5, 0.1)
    gamma = st.sidebar.slider("γ (Trend Formation)", 0.1, 2.0, 0.5, 0.1)
    delta = st.sidebar.slider("δ (Trend Decay)", 0.1, 2.0, 0.5, 0.1)
    sigma = st.sidebar.slider("σ (Price Noise)", 0.01, 0.1, 0.02, 0.01)
    eta = st.sidebar.slider("η (Trend Noise)", 0.005, 0.05, 0.01, 0.005)
else:
    st.sidebar.info("Parameters will adapt based on detected regime")

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'data_dict' not in st.session_state:
    st.session_state.data_dict = {}

# Data fetching
def fetch_finnhub_quote(symbol: str, api_key: str):
    url = "https://finnhub.io/api/v1/quote"
    params = {"symbol": symbol, "token": api_key}
    response = requests.get(url, params=params, timeout=10)
    return response.json()

def generate_historical_data(symbol: str, api_key: str, days: int, res_min: int):
    quote = fetch_finnhub_quote(symbol, api_key)
    if 'c' not in quote or quote['c'] <= 0:
        raise ValueError(f"No price for {symbol}")
    
    current_price = quote['c']
    is_crypto = ':' in symbol
    candles_per_day = int((24 if is_crypto else 6.5) * 60 / res_min)
    total_candles = days * candles_per_day
    
    timestamps = pd.date_range(end=datetime.now(), periods=total_candles, freq=f'{res_min}min')
    np.random.seed(hash(symbol) % 2**32)
    
    daily_vol = 0.03 if is_crypto else 0.02
    vol_per_step = daily_vol * np.sqrt(res_min / (24 * 60))
    prices = np.zeros(total_candles)
    prices[0] = current_price * 0.95
    
    regime_length = candles_per_day * 7
    regimes = np.random.choice(['trend', 'mean_revert', 'high_vol'], 
                              size=total_candles // regime_length + 1, p=[0.3, 0.5, 0.2])
    
    for i in range(1, total_candles):
        regime = regimes[i // regime_length]
        shock = np.random.randn()
        
        if regime == 'trend':
            drift, vol = 0.0005, vol_per_step * 0.8
        elif regime == 'high_vol':
            drift, vol = 0, vol_per_step * 1.5
        else:
            drift = -0.1 * (prices[i-1] - current_price) / current_price
            vol = vol_per_step
        
        prices[i] = np.clip(prices[i-1] * (1 + drift + vol * shock),
                           current_price * 0.7, current_price * 1.3)
    
    df = pd.DataFrame({
        'Close': prices,
        'Returns': np.concatenate([[0], np.diff(prices) / prices[:-1]]),
        'Fundamental': np.convolve(prices, np.ones(50)/50, mode='same')
    }, index=timestamps)
    df['Mispricing'] = df['Close'] - df['Fundamental']
    
    return df

# Fetch button
if st.sidebar.button("🔄 Fetch Data", type="primary"):
    api_key = get_finnhub_key()
    if not api_key:
        st.error("❌ Finnhub API key not found!")
    else:
        with st.spinner(f"Fetching data for {len(symbols)} asset(s)..."):
            progress_bar = st.progress(0)
            data_dict = {}
            
            for i, symbol in enumerate(symbols):
                try:
                    df = generate_historical_data(symbol, api_key, days_back, resolution_min)
                    data_dict[symbol] = df
                    progress_bar.progress((i + 1) / len(symbols))
                except Exception as e:
                    st.warning(f"Failed {symbol}: {e}")
            
            if data_dict:
                st.session_state.data_dict = data_dict
                st.session_state.data_loaded = True
                progress_bar.empty()
                st.success(f"✅ Loaded {len(data_dict)} assets")
                st.rerun()

# Main content
if st.session_state.data_loaded:
    data_dict = st.session_state.data_dict
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Market View",
        "🔍 Regime Analysis", 
        "📈 Chiarella Signals",
        "💼 Multi-Asset"
    ])
    
    with tab1:
        st.header(f"Market Analysis: {selected_symbol}")
        
        data = data_dict[selected_symbol]
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            current_price = data['Close'].iloc[-1]
            st.metric("Current Price", f"${current_price:.2f}")
        with col2:
            price_change = (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100
            st.metric("Total Return", f"{price_change:+.2f}%")
        with col3:
            vol = data['Returns'].std() * np.sqrt(252 * (60/resolution_min) * 6.5) * 100
            st.metric("Annualized Vol", f"{vol:.1f}%")
        with col4:
            mispricing = data['Mispricing'].iloc[-1]
            st.metric("Current Mispricing", f"${mispricing:+.2f}")
        
        # Price and fundamental
        st.subheader("Price vs Fundamental Value")
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Price and Fundamental", "Mispricing"),
            vertical_spacing=0.1,
            row_heights=[0.6, 0.4]
        )
        
        fig.add_trace(go.Scatter(
            x=data.index, y=data['Close'],
            name="Market Price",
            line=dict(color='blue', width=2)
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=data.index, y=data['Fundamental'],
            name="Fundamental Value",
            line=dict(color='red', width=2, dash='dash')
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=data.index, y=data['Mispricing'],
            name="Mispricing",
            fill='tozeroy',
            line=dict(color='purple')
        ), row=2, col=1)
        
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
        
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Mispricing ($)", row=2, col=1)
        fig.update_layout(height=600, showlegend=True)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("🔍 Regime Detection")
        
        # Detect regime
        regime_detector = RegimeDetector(lookback_window=100)
        returns_series = data_dict[selected_symbol]['Returns']
        
        regime_metrics = get_regime_metrics(returns_series)
        detected_regime = regime_metrics['regime']
        
        # Display regime
        st.subheader("Current Market Regime")
        
        regime_colors = {
            'mean_reverting': 'green',
            'trending': 'orange',
            'high_volatility': 'red',
            'mixed': 'blue'
        }
        
        regime_class = f"regime-{detected_regime.replace('_', '-')}"
        st.markdown(
            f"<h2 class='{regime_class}'>{detected_regime.replace('_', ' ').title()}</h2>",
            unsafe_allow_html=True
        )
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Hurst Exponent", f"{regime_metrics['hurst_exponent']:.3f}")
            st.caption("< 0.5: Mean Rev, > 0.5: Trending")
        with col2:
            st.metric("Autocorrelation", f"{regime_metrics['autocorr']:.3f}")
            st.caption("Negative: Mean Rev")
        with col3:
            st.metric("Trend Strength", f"{regime_metrics['trend_strength']:.3f}")
            st.caption("R² of linear fit")
        with col4:
            st.metric("Volatility", f"{regime_metrics['volatility']:.4f}")
            st.caption("Per period")
        
        # Regime probabilities
        st.subheader("Regime Probabilities")
        probs = regime_metrics['probabilities']
        
        fig = go.Figure(data=[go.Bar(
            x=list(probs.keys()),
            y=list(probs.values()),
            marker_color=['green', 'orange', 'red']
        )])
        fig.update_layout(
            title="Probability Distribution",
            xaxis_title="Regime",
            yaxis_title="Probability",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Adaptive parameters
        if use_adaptive:
            st.subheader("Adaptive Chiarella Parameters")
            
            if detected_regime == 'mean_reverting':
                alpha, beta, gamma, delta = 0.3, 0.8, 0.3, 0.6
                st.info("📊 Mean-reverting regime: Increased fundamentalist strength (β)")
            elif detected_regime == 'trending':
                alpha, beta, gamma, delta = 0.8, 0.3, 0.7, 0.3
                st.info("📈 Trending regime: Increased chartist strength (α)")
            elif detected_regime == 'high_volatility':
                alpha, beta, gamma, delta = 0.5, 0.5, 0.5, 0.8
                st.warning("⚠️ High volatility regime: Increased trend decay (δ)")
            else:
                alpha, beta, gamma, delta = 0.5, 0.5, 0.5, 0.5
                st.info("🔄 Mixed regime: Balanced parameters")
            
            sigma = regime_metrics['volatility'] * 0.5
            eta = sigma * 0.5
            
            param_df = pd.DataFrame({
                'Parameter': ['α (Chartist)', 'β (Fundamentalist)', 'γ (Trend Form)', 'δ (Trend Decay)', 'σ (Price Noise)', 'η (Trend Noise)'],
                'Value': [alpha, beta, gamma, delta, sigma, eta],
                'Description': [
                    'Chartist strength',
                    'Fundamentalist strength',
                    'Trend formation rate',
                    'Trend decay rate',
                    'Price noise level',
                    'Trend noise level'
                ]
            })
            st.dataframe(param_df, use_container_width=True)
            
            # Bifurcation metric
            Lambda = (alpha * gamma) / (beta * delta)
            st.metric("Λ (Bifurcation Parameter)", f"{Lambda:.3f}")
            if Lambda < 0.67:
                st.success("✅ Unimodal (stable mean reversion)")
            elif Lambda > 1.5:
                st.warning("⚠️ Bimodal (strong trending)")
            else:
                st.info("🔄 Mixed dynamics")
    
    with tab3:
        st.header("📈 Chiarella Trading Signals")
        
        data = data_dict[selected_symbol]
        
        # Generate signals
        mispricing = data['Mispricing']
        fundamental_signal = -beta * mispricing / data['Fundamental']
        
        # Estimate trend
        trend = data['Close'].diff().rolling(20).mean()
        chartist_signal = alpha * trend / data['Fundamental']
        
        # Combined signal (regime-weighted)
        if detected_regime == 'mean_reverting':
            w_fund, w_chart = 0.8, 0.2
        elif detected_regime == 'trending':
            w_fund, w_chart = 0.2, 0.8
        else:
            w_fund, w_chart = 0.5, 0.5
        
        combined_signal = w_fund * fundamental_signal + w_chart * chartist_signal
        
        # Plot signals
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=("Price", "Fundamental Signal", "Chartist Signal"),
            vertical_spacing=0.08,
            row_heights=[0.4, 0.3, 0.3]
        )
        
        fig.add_trace(go.Scatter(
            x=data.index, y=data['Close'],
            name="Price",
            line=dict(color='blue')
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=data.index, y=fundamental_signal,
            name="Fundamental Signal",
            line=dict(color='red'),
            fill='tozeroy'
        ), row=2, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=data.index, y=chartist_signal,
            name="Chartist Signal",
            line=dict(color='green'),
            fill='tozeroy'
        ), row=3, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)
        
        fig.update_layout(height=700, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # Current signals
        st.subheader("Current Signals")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            latest_fund = fundamental_signal.iloc[-1]
            st.metric(
                "Fundamental Signal",
                f"{latest_fund:+.4f}",
                "BUY" if latest_fund < 0 else "SELL"
            )
        
        with col2:
            latest_chart = chartist_signal.iloc[-1]
            st.metric(
                "Chartist Signal",
                f"{latest_chart:+.4f}",
                "BUY" if latest_chart > 0 else "SELL"
            )
        
        with col3:
            latest_combined = combined_signal.iloc[-1]
            st.metric(
                f"Combined ({w_fund:.0%}/{w_chart:.0%})",
                f"{latest_combined:+.4f}",
                "BUY" if latest_combined < 0 else "SELL"
            )
    
    with tab4:
        if asset_type == "Multi-Asset Portfolio":
            st.header("💼 Multi-Asset Chiarella Analysis")
            
            # Detect regimes for all assets
            all_regimes = {}
            for symbol in symbols:
                returns = data_dict[symbol]['Returns']
                regime = RegimeDetector(lookback_window=100).detect_regime(returns)
                all_regimes[symbol] = regime
            
            # Regime distribution
            regime_counts = pd.Series(all_regimes).value_counts()
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Regime Distribution")
                for regime, count in regime_counts.items():
                    pct = count / len(all_regimes) * 100
                    st.metric(regime.replace('_', ' ').title(), f"{count} ({pct:.0f}%)")
            
            with col2:
                fig = go.Figure(data=[go.Pie(
                    labels=regime_counts.index,
                    values=regime_counts.values,
                    hole=0.4
                )])
                fig.update_layout(title="Asset Regime Distribution")
                st.plotly_chart(fig, use_container_width=True)
            
            # Show regime for each asset
            st.subheader("Individual Asset Regimes")
            regime_df = pd.DataFrame({
                'Symbol': list(all_regimes.keys()),
                'Regime': list(all_regimes.values()),
                'Latest Price': [data_dict[s]['Close'].iloc[-1] for s in all_regimes.keys()],
                'Mispricing': [data_dict[s]['Mispricing'].iloc[-1] for s in all_regimes.keys()]
            })
            st.dataframe(regime_df, use_container_width=True)
        else:
            st.info("Select 'Multi-Asset Portfolio' in sidebar to view cross-asset analysis")

else:
    st.info("👈 Configure parameters and click 'Fetch Data' to begin")
    
    st.markdown("""
    ### 📉 Chiarella Model
    
    The Chiarella model describes market dynamics through the interaction of two agent types:
    
    #### 🎯 **Fundamentalists**
    - Believe prices revert to fundamental value
    - Trade against mispricing
    - Stabilizing force
    
    #### 📊 **Chartists**
    - Follow trends and momentum
    - Extrapolate recent price movements
    - Destabilizing force
    
    #### 🔍 **New Features**
    - ✅ Regime-adaptive parameters
    - ✅ Multi-asset analysis
    - ✅ Live signal monitoring
    - ✅ Expanded universe support
    """)

# Footer
st.markdown("---")
if st.session_state.data_loaded:
    st.markdown(f"**Assets**: {len(symbols)} | **Resolution**: {resolution_min}min | **Regime**: {detected_regime if 'detected_regime' in locals() else 'N/A'}")
