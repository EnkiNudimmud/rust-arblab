"""
Derivatives Module
==================

Options and Futures data visualization and analysis:
- Options chains with Greeks
- Futures quotes and term structure
- Implied volatility surface
- Options strategies builder
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from scipy.stats import norm
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import strategy rendering functions
from app.pages.derivatives_strategies import (
    render_straddle, render_strangle, render_butterfly,
    render_iron_condor, render_iron_butterfly,
    render_single_option, render_vertical_spread,
    render_calendar_spread, render_covered_call,
    render_cash_secured_put, render_ratio_spread
)

def render():
    """Render the derivatives page"""
    st.title("📈 Options & Futures")
    st.markdown("Derivatives data, Greeks calculation, and strategy analysis")
    
    # Tabs for different derivative types
    tab1, tab2, tab3 = st.tabs([
        "📊 Options Chain",
        "📉 Futures",
        "🎯 Strategy Builder"
    ])
    
    with tab1:
        render_options_chain()
    
    with tab2:
        render_futures()
    
    with tab3:
        render_strategy_builder()

def render_options_chain():
    """Render options chain viewer"""
    
    st.markdown("### Options Chain Viewer")
    
    # Configuration
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### Configuration")
        
        underlying = st.text_input(
            "Underlying Symbol",
            value="AAPL",
            help="Enter stock symbol"
        )
        
        spot_price = st.number_input(
            "Spot Price ($)",
            value=150.0,
            min_value=0.01,
            step=1.0
        )
        
        expiry_days = st.number_input(
            "Days to Expiration",
            value=30,
            min_value=1,
            max_value=365,
            step=1
        )
        
        risk_free_rate = st.number_input(
            "Risk-Free Rate (%)",
            value=5.0,
            min_value=0.0,
            max_value=20.0,
            step=0.1
        ) / 100
        
        implied_vol = st.number_input(
            "Implied Volatility (%)",
            value=30.0,
            min_value=1.0,
            max_value=200.0,
            step=1.0
        ) / 100
        
        if st.button("🔄 Generate Options Chain", use_container_width=True):
            generate_options_chain(
                underlying, spot_price, expiry_days,
                risk_free_rate, implied_vol
            )
    
    with col2:
        st.markdown("#### Options Chain")
        
        if 'options_chain' in st.session_state.derivatives_data:
            display_options_chain()
        else:
            st.info("Configure parameters and click 'Generate Options Chain' to view options data")
            
            st.markdown("""
            **Features:**
            - Complete options chain with Calls and Puts
            - Greeks calculation (Delta, Gamma, Theta, Vega, Rho)
            - Implied volatility analysis
            - Interactive visualizations
            """)

def generate_options_chain(
    underlying: str,
    spot: float,
    days_to_expiry: int,
    risk_free: float,
    iv: float
):
    """Generate synthetic options chain with Greeks"""
    
    with st.spinner("Generating options chain..."):
        # Generate strikes around spot
        strike_range = spot * 0.3  # ±30% from spot
        n_strikes = 21
        strikes = np.linspace(spot - strike_range, spot + strike_range, n_strikes)
        
        # Time to expiration in years
        T = days_to_expiry / 365.0
        
        options_data = []
        
        for strike in strikes:
            # Calculate Call Greeks
            call_price, call_greeks = black_scholes_call(
                spot, strike, T, risk_free, iv
            )
            
            # Calculate Put Greeks
            put_price, put_greeks = black_scholes_put(
                spot, strike, T, risk_free, iv
            )
            
            options_data.append({
                'Strike': strike,
                'Call Price': call_price,
                'Call Delta': call_greeks['delta'],
                'Call Gamma': call_greeks['gamma'],
                'Call Theta': call_greeks['theta'],
                'Call Vega': call_greeks['vega'],
                'Call Rho': call_greeks['rho'],
                'Put Price': put_price,
                'Put Delta': put_greeks['delta'],
                'Put Gamma': put_greeks['gamma'],
                'Put Theta': put_greeks['theta'],
                'Put Vega': put_greeks['vega'],
                'Put Rho': put_greeks['rho'],
            })
        
        df = pd.DataFrame(options_data)
        
        st.session_state.derivatives_data['options_chain'] = {
            'data': df,
            'underlying': underlying,
            'spot': spot,
            'expiry_days': days_to_expiry,
            'risk_free': risk_free,
            'iv': iv
        }
        
        st.success(f"✅ Generated options chain for {underlying}")
        st.rerun()

def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> tuple:
    """Calculate Black-Scholes call option price and Greeks"""
    
    if T <= 0 or sigma <= 0:
        return 0.0, {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Price
    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
    # Greeks
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
             - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    
    greeks = {
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega,
        'rho': rho
    }
    
    return price, greeks

def black_scholes_put(S: float, K: float, T: float, r: float, sigma: float) -> tuple:
    """Calculate Black-Scholes put option price and Greeks"""
    
    if T <= 0 or sigma <= 0:
        return 0.0, {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Price
    price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    # Greeks
    delta = -norm.cdf(-d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
             + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    
    greeks = {
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega,
        'rho': rho
    }
    
    return price, greeks

def display_options_chain():
    """Display options chain data"""
    
    chain_data = st.session_state.derivatives_data['options_chain']
    df = chain_data['data']
    spot = chain_data['spot']
    underlying = chain_data['underlying']
    
    # Display metadata
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Underlying", underlying)
    with col2:
        st.metric("Spot Price", f"${spot:.2f}")
    with col3:
        st.metric("Days to Expiry", chain_data['expiry_days'])
    with col4:
        st.metric("IV", f"{chain_data['iv']*100:.1f}%")
    
    st.markdown("---")
    
    # Options chain table
    st.markdown("#### Options Chain Data")
    
    # Split into Calls and Puts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Calls**")
        calls_df = df[['Strike', 'Call Price', 'Call Delta', 'Call Gamma', 'Call Theta', 'Call Vega']].copy()
        calls_df.columns = ['Strike', 'Price', 'Delta', 'Gamma', 'Theta', 'Vega']
        
        # Highlight ATM
        def highlight_atm(row):
            return ['background-color: #2a3f5f' if abs(row['Strike'] - spot) < spot * 0.05 else '' for _ in row]
        
        st.dataframe(
            calls_df.style.format({
                'Strike': '${:.2f}',
                'Price': '${:.2f}',
                'Delta': '{:.3f}',
                'Gamma': '{:.4f}',
                'Theta': '{:.4f}',
                'Vega': '{:.4f}'
            }).apply(highlight_atm, axis=1),
            use_container_width=True,
            height=400
        )
    
    with col2:
        st.markdown("**Puts**")
        puts_df = df[['Strike', 'Put Price', 'Put Delta', 'Put Gamma', 'Put Theta', 'Put Vega']].copy()
        puts_df.columns = ['Strike', 'Price', 'Delta', 'Gamma', 'Theta', 'Vega']
        
        st.dataframe(
            puts_df.style.format({
                'Strike': '${:.2f}',
                'Price': '${:.2f}',
                'Delta': '{:.3f}',
                'Gamma': '{:.4f}',
                'Theta': '{:.4f}',
                'Vega': '{:.4f}'
            }).apply(highlight_atm, axis=1),
            use_container_width=True,
            height=400
        )
    
    st.markdown("---")
    
    # Visualizations
    st.markdown("#### Greeks Visualization")
    
    # Select Greek to visualize
    greek = st.selectbox(
        "Select Greek",
        ["Delta", "Gamma", "Theta", "Vega", "Rho"]
    )
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['Strike'],
        y=df[f'Call {greek}'],
        mode='lines+markers',
        name=f'Call {greek}',
        line={'color': 'green', 'width': 2}
    ))
    
    fig.add_trace(go.Scatter(
        x=df['Strike'],
        y=df[f'Put {greek}'],
        mode='lines+markers',
        name=f'Put {greek}',
        line={'color': 'red', 'width': 2}
    ))
    
    # Add spot price line
    fig.add_vline(
        x=spot,
        line_dash="dash",
        line_color="cyan",
        annotation_text="Spot"
    )
    
    fig.update_layout(
        title=f"{greek} across Strikes",
        xaxis_title="Strike Price ($)",
        yaxis_title=greek,
        template="plotly_dark",
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Payoff diagram
    st.markdown("#### Options Payoff")
    
    selected_strike = st.select_slider(
        "Select Strike",
        options=df['Strike'].tolist(),
        value=spot
    )
    
    display_payoff_diagram(df, selected_strike, spot)

def display_payoff_diagram(df: pd.DataFrame, strike: float, spot: float):
    """Display options payoff diagram"""
    
    # Get option prices for selected strike
    option_row = df[df['Strike'] == strike].iloc[0]
    call_price = option_row['Call Price']
    put_price = option_row['Put Price']
    
    # Generate spot range for payoff
    spot_range = np.linspace(spot * 0.5, spot * 1.5, 100)
    
    # Calculate payoffs at expiration
    call_payoff = np.maximum(spot_range - strike, 0) - call_price
    put_payoff = np.maximum(strike - spot_range, 0) - put_price
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=spot_range,
        y=call_payoff,
        mode='lines',
        name=f'Long Call @ ${strike:.2f}',
        line={'color': 'green', 'width': 2}
    ))
    
    fig.add_trace(go.Scatter(
        x=spot_range,
        y=put_payoff,
        mode='lines',
        name=f'Long Put @ ${strike:.2f}',
        line={'color': 'red', 'width': 2}
    ))
    
    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    
    # Current spot
    fig.add_vline(x=spot, line_dash="dot", line_color="cyan", annotation_text="Current Spot")
    
    fig.update_layout(
        title="Options Payoff at Expiration",
        xaxis_title="Underlying Price ($)",
        yaxis_title="P&L ($)",
        template="plotly_dark",
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_futures():
    """Render futures data viewer"""
    
    st.markdown("### Futures Quotes")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### Configuration")
        
        underlying = st.text_input(
            "Underlying",
            value="ES",
            help="E.g., ES (S&P 500 E-mini), CL (Crude Oil), GC (Gold)"
        )
        
        spot_price = st.number_input(
            "Spot Price",
            value=4500.0,
            step=10.0
        )
        
        n_contracts = st.number_input(
            "Number of Contracts",
            value=12,
            min_value=1,
            max_value=24,
            step=1
        )
        
        cost_of_carry = st.number_input(
            "Cost of Carry (% p.a.)",
            value=5.0,
            step=0.1
        ) / 100
        
        if st.button("🔄 Generate Futures Curve", use_container_width=True):
            generate_futures_curve(underlying, spot_price, n_contracts, cost_of_carry)
    
    with col2:
        st.markdown("#### Futures Term Structure")
        
        if 'futures_curve' in st.session_state.derivatives_data:
            display_futures_curve()
        else:
            st.info("Configure parameters and click 'Generate Futures Curve'")
            
            st.markdown("""
            **Features:**
            - Futures term structure
            - Contango/Backwardation analysis
            - Roll yield calculation
            - Basis analysis
            """)

def generate_futures_curve(underlying: str, spot: float, n_contracts: int, carry_rate: float):
    """Generate synthetic futures curve"""
    
    with st.spinner("Generating futures curve..."):
        # Generate monthly contracts
        contracts = []
        base_date = datetime.now()
        
        for i in range(n_contracts):
            expiry_date = base_date + timedelta(days=30 * (i + 1))
            T = (expiry_date - base_date).days / 365.0
            
            # Simple cost-of-carry model
            futures_price = spot * np.exp(carry_rate * T)
            
            # Add some realistic noise
            noise = np.random.normal(0, spot * 0.002)
            futures_price += noise
            
            basis = futures_price - spot
            basis_pct = (basis / spot) * 100
            
            contracts.append({
                'Contract': f'{underlying} {expiry_date.strftime("%b%y")}',
                'Expiry': expiry_date,
                'Days to Expiry': (expiry_date - base_date).days,
                'Futures Price': futures_price,
                'Spot Price': spot,
                'Basis': basis,
                'Basis %': basis_pct,
                'Time (Years)': T
            })
        
        df = pd.DataFrame(contracts)
        
        st.session_state.derivatives_data['futures_curve'] = {
            'data': df,
            'underlying': underlying,
            'spot': spot,
            'carry_rate': carry_rate
        }
        
        st.success(f"✅ Generated futures curve for {underlying}")
        st.rerun()

def display_futures_curve():
    """Display futures curve"""
    
    curve_data = st.session_state.derivatives_data['futures_curve']
    df = curve_data['data']
    underlying = curve_data['underlying']
    spot = curve_data['spot']
    
    # Display metadata
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Underlying", underlying)
    with col2:
        st.metric("Spot Price", f"${spot:.2f}")
    with col3:
        market_structure = "Contango" if df['Basis'].iloc[-1] > 0 else "Backwardation"
        st.metric("Market Structure", market_structure)
    
    st.markdown("---")
    
    # Futures table
    st.dataframe(
        df[['Contract', 'Days to Expiry', 'Futures Price', 'Basis', 'Basis %']].style.format({
            'Days to Expiry': '{:.0f}',
            'Futures Price': '${:.2f}',
            'Basis': '${:.2f}',
            'Basis %': '{:+.2f}%'
        }),
        use_container_width=True,
        height=300
    )
    
    st.markdown("---")
    
    # Term structure chart
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
        subplot_titles=("Futures Term Structure", "Basis")
    )
    
    # Futures prices
    fig.add_trace(
        go.Scatter(
            x=df['Time (Years)'],
            y=df['Futures Price'],
            mode='lines+markers',
            name='Futures Price',
            line={'color': 'cyan', 'width': 2},
            marker={'size': 8}
        ),
        row=1, col=1
    )
    
    # Spot price line
    fig.add_hline(
        y=spot,
        line_dash="dash",
        line_color="yellow",
        annotation_text="Spot",
        row=1, col=1
    )
    
    # Basis
    fig.add_trace(
        go.Bar(
            x=df['Time (Years)'],
            y=df['Basis'],
            name='Basis',
            marker_color=['green' if b > 0 else 'red' for b in df['Basis']]
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        template="plotly_dark",
        height=600,
        showlegend=True,
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="Time to Expiry (Years)", row=2, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Basis ($)", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)

def render_strategy_builder():
    """Render options strategy builder with arbitrage strategies"""
    
    st.markdown("### 🎯 Options Arbitrage Strategies")
    
    st.markdown("""
    Explore common options strategies with interactive payoff diagrams, Greeks analysis, 
    and detailed explanations. Each strategy shows profit/loss curves and key metrics.
    """)
    
    # Sidebar for common parameters
    with st.sidebar:
        st.markdown("### Strategy Parameters")
        
        spot_price = st.number_input(
            "Current Stock Price ($)",
            value=100.0,
            min_value=1.0,
            step=1.0,
            key="strat_spot"
        )
        
        volatility = st.number_input(
            "Implied Volatility (%)",
            value=25.0,
            min_value=1.0,
            max_value=200.0,
            step=1.0,
            key="strat_vol"
        ) / 100
        
        risk_free_rate = st.number_input(
            "Risk-Free Rate (%)",
            value=5.0,
            min_value=0.0,
            max_value=20.0,
            step=0.1,
            key="strat_rf"
        ) / 100
        
        days_to_expiry = st.number_input(
            "Days to Expiration",
            value=30,
            min_value=1,
            max_value=365,
            step=1,
            key="strat_dte"
        )
    
    # Strategy selection
    strategy_categories = {
        "Directional": ["Long Call", "Long Put", "Bull Call Spread", "Bear Put Spread"],
        "Volatility": ["Long Straddle", "Long Strangle", "Short Straddle", "Short Strangle"],
        "Income": ["Covered Call", "Cash-Secured Put", "Iron Condor", "Iron Butterfly"],
        "Advanced": ["Long Butterfly", "Short Butterfly", "Calendar Spread", "Ratio Spread"]
    }
    
    selected_category = st.selectbox(
        "Strategy Category",
        list(strategy_categories.keys()),
        help="Choose a category of options strategies"
    )
    
    selected_strategy = st.selectbox(
        "Select Strategy",
        strategy_categories[selected_category],
        help="Select specific strategy to analyze"
    )
    
    st.markdown("---")
    
    # Render selected strategy
    T = days_to_expiry / 365.0
    
    if selected_strategy == "Long Straddle":
        render_straddle(spot_price, T, risk_free_rate, volatility, is_long=True)
    elif selected_strategy == "Short Straddle":
        render_straddle(spot_price, T, risk_free_rate, volatility, is_long=False)
    elif selected_strategy == "Long Strangle":
        render_strangle(spot_price, T, risk_free_rate, volatility, is_long=True)
    elif selected_strategy == "Short Strangle":
        render_strangle(spot_price, T, risk_free_rate, volatility, is_long=False)
    elif selected_strategy == "Long Butterfly":
        render_butterfly(spot_price, T, risk_free_rate, volatility, is_long=True)
    elif selected_strategy == "Short Butterfly":
        render_butterfly(spot_price, T, risk_free_rate, volatility, is_long=False)
    elif selected_strategy == "Iron Condor":
        render_iron_condor(spot_price, T, risk_free_rate, volatility)
    elif selected_strategy == "Iron Butterfly":
        render_iron_butterfly(spot_price, T, risk_free_rate, volatility)
    elif selected_strategy == "Long Call":
        render_single_option(spot_price, T, risk_free_rate, volatility, "call", is_long=True)
    elif selected_strategy == "Long Put":
        render_single_option(spot_price, T, risk_free_rate, volatility, "put", is_long=True)
    elif selected_strategy == "Bull Call Spread":
        render_vertical_spread(spot_price, T, risk_free_rate, volatility, "call", is_bull=True)
    elif selected_strategy == "Bear Put Spread":
        render_vertical_spread(spot_price, T, risk_free_rate, volatility, "put", is_bull=False)
    elif selected_strategy == "Calendar Spread":
        render_calendar_spread(spot_price, risk_free_rate, volatility)
    elif selected_strategy == "Covered Call":
        render_covered_call(spot_price, T, risk_free_rate, volatility)
    elif selected_strategy == "Cash-Secured Put":
        render_cash_secured_put(spot_price, T, risk_free_rate, volatility)
    elif selected_strategy == "Ratio Spread":
        render_ratio_spread(spot_price, T, risk_free_rate, volatility)

# Execute the render function when page is loaded
if __name__ == "__main__":
    render()
