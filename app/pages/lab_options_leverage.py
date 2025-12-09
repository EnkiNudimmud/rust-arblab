"""
Options Leverage Lab
Real options data fetching with leveraged strategies and risk hedging
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
from datetime import datetime, timedelta
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import shared UI components
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.ui_components import render_sidebar_navigation, apply_custom_css

st.set_page_config(page_title="Options Leverage Lab", page_icon="⚡", layout="wide")

# Render sidebar navigation and apply CSS
render_sidebar_navigation(current_page="Options Leverage Lab")
apply_custom_css()

st.markdown('<h1 class="lab-header">⚡ Options Leverage Lab</h1>', unsafe_allow_html=True)
st.markdown("### Leveraged strategies with real options data and risk hedging")
st.markdown("---")

# Helper Functions

def fetch_options_yfinance(symbol, expiration_date=None):
    """Fetch real options data using yfinance"""
    try:
        import yfinance as yf
        
        ticker = yf.Ticker(symbol)
        
        # Get available expiration dates
        expirations = ticker.options
        
        if not expirations:
            return None, None, None
        
        # Use provided date or nearest expiration
        if expiration_date:
            exp_date = expiration_date
        else:
            exp_date = expirations[0] if len(expirations) > 0 else None
        
        if exp_date is None:
            return None, None, None
        
        # Fetch options chain
        options = ticker.option_chain(exp_date)
        calls = options.calls
        puts = options.puts
        
        # Get current stock price
        stock_info = ticker.history(period="1d")
        current_price = stock_info['Close'].iloc[-1] if len(stock_info) > 0 else None
        
        return calls, puts, current_price
        
    except Exception as e:
        st.error(f"Error fetching options for {symbol}: {str(e)}")
        return None, None, None

def calculate_implied_volatility_approx(option_price, S, K, T, r, option_type='call'):
    """
    Approximate implied volatility using Newton-Raphson
    """
    if T <= 0 or option_price <= 0:
        return 0.0
    
    # Initial guess
    sigma = 0.3
    
    for _ in range(50):  # Max iterations
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            vega = S * norm.pdf(d1) * np.sqrt(T)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            vega = S * norm.pdf(d1) * np.sqrt(T)
        
        if vega < 1e-10:
            break
        
        diff = option_price - price
        
        if abs(diff) < 0.01:
            break
        
        sigma = sigma + diff / vega
        sigma = max(0.01, min(sigma, 3.0))  # Bounds
    
    return sigma

def calculate_greeks(S, K, T, r, sigma, option_type='call'):
    """Calculate option Greeks"""
    if T <= 0:
        return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        delta = norm.cdf(d1)
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - 
                r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        delta = norm.cdf(d1) - 1
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + 
                r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    
    return {
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega
    }

def synthetic_long_with_collar(stock_signal, S, calls_df, puts_df, capital, risk_tolerance=0.15):
    """
    Leveraged long with protective collar
    - Buy ATM call (leverage)
    - Buy OTM put (downside protection)
    - Sell OTM call (finance the put, cap upside)
    """
    # Find ATM call for leverage
    atm_call = calls_df.iloc[(calls_df['strike'] - S).abs().argsort()[:1]]
    
    if len(atm_call) == 0:
        return None
    
    # Find OTM put for protection (5-10% below)
    put_strike_target = S * (1 - risk_tolerance)
    protective_put = puts_df.iloc[(puts_df['strike'] - put_strike_target).abs().argsort()[:1]]
    
    # Find OTM call to sell (10-15% above)
    call_strike_target = S * 1.12
    covered_call = calls_df.iloc[(calls_df['strike'] - call_strike_target).abs().argsort()[:1]]
    
    if len(protective_put) == 0 or len(covered_call) == 0:
        return None
    
    # Calculate costs
    long_call_cost = atm_call['lastPrice'].iloc[0] if 'lastPrice' in atm_call else atm_call['ask'].iloc[0]
    put_cost = protective_put['lastPrice'].iloc[0] if 'lastPrice' in protective_put else protective_put['ask'].iloc[0]
    call_credit = covered_call['lastPrice'].iloc[0] if 'lastPrice' in covered_call else covered_call['bid'].iloc[0]
    
    net_debit = long_call_cost + put_cost - call_credit
    
    # Number of contracts
    contracts = int(capital / (net_debit * 100))
    
    if contracts < 1:
        return None
    
    total_cost = contracts * net_debit * 100
    
    # Calculate leverage
    notional_exposure = contracts * 100 * S
    leverage_ratio = notional_exposure / capital
    
    # Risk metrics
    max_loss = total_cost  # Limited to premium paid
    max_gain = contracts * 100 * (covered_call['strike'].iloc[0] - atm_call['strike'].iloc[0] - net_debit)
    breakeven = atm_call['strike'].iloc[0] + net_debit
    
    return {
        'strategy': 'Leveraged Long with Collar',
        'signal': stock_signal,
        'contracts': contracts,
        'total_cost': total_cost,
        'notional_exposure': notional_exposure,
        'leverage_ratio': leverage_ratio,
        'max_loss': max_loss,
        'max_gain': max_gain,
        'breakeven': breakeven,
        'components': {
            'long_call': {
                'strike': atm_call['strike'].iloc[0],
                'premium': long_call_cost,
                'contracts': contracts
            },
            'protective_put': {
                'strike': protective_put['strike'].iloc[0],
                'premium': put_cost,
                'contracts': contracts
            },
            'covered_call': {
                'strike': covered_call['strike'].iloc[0],
                'premium': call_credit,
                'contracts': -contracts  # Short
            }
        }
    }

def bull_call_spread(stock_signal, S, calls_df, capital):
    """
    Bull call spread for moderate bullish signals
    - Buy ATM call
    - Sell OTM call
    """
    # Buy ATM call
    long_call = calls_df.iloc[(calls_df['strike'] - S).abs().argsort()[:1]]
    
    # Sell call 5-10% OTM
    short_strike_target = S * 1.07
    short_call = calls_df.iloc[(calls_df['strike'] - short_strike_target).abs().argsort()[:1]]
    
    if len(long_call) == 0 or len(short_call) == 0:
        return None
    
    long_premium = long_call['lastPrice'].iloc[0] if 'lastPrice' in long_call else long_call['ask'].iloc[0]
    short_premium = short_call['lastPrice'].iloc[0] if 'lastPrice' in short_call else short_call['bid'].iloc[0]
    
    net_debit = long_premium - short_premium
    
    contracts = int(capital / (net_debit * 100))
    
    if contracts < 1:
        return None
    
    total_cost = contracts * net_debit * 100
    
    max_gain = contracts * 100 * (short_call['strike'].iloc[0] - long_call['strike'].iloc[0] - net_debit)
    max_loss = total_cost
    breakeven = long_call['strike'].iloc[0] + net_debit
    
    leverage_ratio = (contracts * 100 * S) / capital
    
    return {
        'strategy': 'Bull Call Spread',
        'signal': stock_signal,
        'contracts': contracts,
        'total_cost': total_cost,
        'notional_exposure': contracts * 100 * S,
        'leverage_ratio': leverage_ratio,
        'max_loss': max_loss,
        'max_gain': max_gain,
        'breakeven': breakeven,
        'components': {
            'long_call': {
                'strike': long_call['strike'].iloc[0],
                'premium': long_premium,
                'contracts': contracts
            },
            'short_call': {
                'strike': short_call['strike'].iloc[0],
                'premium': short_premium,
                'contracts': -contracts
            }
        }
    }

def bear_put_spread(stock_signal, S, puts_df, capital):
    """
    Bear put spread for bearish/bubble signals
    - Buy ATM put
    - Sell OTM put
    """
    # Buy ATM put
    long_put = puts_df.iloc[(puts_df['strike'] - S).abs().argsort()[:1]]
    
    # Sell put 5-10% OTM
    short_strike_target = S * 0.93
    short_put = puts_df.iloc[(puts_df['strike'] - short_strike_target).abs().argsort()[:1]]
    
    if len(long_put) == 0 or len(short_put) == 0:
        return None
    
    long_premium = long_put['lastPrice'].iloc[0] if 'lastPrice' in long_put else long_put['ask'].iloc[0]
    short_premium = short_put['lastPrice'].iloc[0] if 'lastPrice' in short_put else short_put['bid'].iloc[0]
    
    net_debit = long_premium - short_premium
    
    contracts = int(capital / (net_debit * 100))
    
    if contracts < 1:
        return None
    
    total_cost = contracts * net_debit * 100
    
    max_gain = contracts * 100 * (long_put['strike'].iloc[0] - short_put['strike'].iloc[0] - net_debit)
    max_loss = total_cost
    breakeven = long_put['strike'].iloc[0] - net_debit
    
    leverage_ratio = (contracts * 100 * S) / capital
    
    return {
        'strategy': 'Bear Put Spread',
        'signal': stock_signal,
        'contracts': contracts,
        'total_cost': total_cost,
        'notional_exposure': contracts * 100 * S,
        'leverage_ratio': leverage_ratio,
        'max_loss': max_loss,
        'max_gain': max_gain,
        'breakeven': breakeven,
        'components': {
            'long_put': {
                'strike': long_put['strike'].iloc[0],
                'premium': long_premium,
                'contracts': contracts
            },
            'short_put': {
                'strike': short_put['strike'].iloc[0],
                'premium': short_premium,
                'contracts': -contracts
            }
        }
    }

def iron_condor(S, calls_df, puts_df, capital):
    """
    Iron condor for neutral/range-bound outlook
    - Sell OTM call spread
    - Sell OTM put spread
    """
    # Short call spread (above current price)
    short_call_strike = S * 1.05
    long_call_strike = S * 1.10
    
    short_call = calls_df.iloc[(calls_df['strike'] - short_call_strike).abs().argsort()[:1]]
    long_call = calls_df.iloc[(calls_df['strike'] - long_call_strike).abs().argsort()[:1]]
    
    # Short put spread (below current price)
    short_put_strike = S * 0.95
    long_put_strike = S * 0.90
    
    short_put = puts_df.iloc[(puts_df['strike'] - short_put_strike).abs().argsort()[:1]]
    long_put = puts_df.iloc[(puts_df['strike'] - long_put_strike).abs().argsort()[:1]]
    
    if any(len(df) == 0 for df in [short_call, long_call, short_put, long_put]):
        return None
    
    # Premiums
    short_call_prem = short_call['lastPrice'].iloc[0] if 'lastPrice' in short_call else short_call['bid'].iloc[0]
    long_call_prem = long_call['lastPrice'].iloc[0] if 'lastPrice' in long_call else long_call['ask'].iloc[0]
    short_put_prem = short_put['lastPrice'].iloc[0] if 'lastPrice' in short_put else short_put['bid'].iloc[0]
    long_put_prem = long_put['lastPrice'].iloc[0] if 'lastPrice' in long_put else long_put['ask'].iloc[0]
    
    net_credit = (short_call_prem + short_put_prem) - (long_call_prem + long_put_prem)
    
    if net_credit <= 0:
        return None
    
    contracts = int(capital / (5 * 100))  # Assume max risk of $500 per spread
    
    if contracts < 1:
        return None
    
    max_gain = contracts * net_credit * 100
    max_loss = contracts * 100 * (long_call['strike'].iloc[0] - short_call['strike'].iloc[0]) - max_gain
    
    return {
        'strategy': 'Iron Condor',
        'signal': 'Neutral/Range-bound',
        'contracts': contracts,
        'total_cost': -max_gain,  # Credit received
        'notional_exposure': contracts * 100 * S,
        'leverage_ratio': (contracts * 100 * S) / capital,
        'max_loss': max_loss,
        'max_gain': max_gain,
        'breakeven_upper': short_call['strike'].iloc[0] + net_credit,
        'breakeven_lower': short_put['strike'].iloc[0] - net_credit,
        'components': {
            'short_call': {'strike': short_call['strike'].iloc[0], 'premium': short_call_prem, 'contracts': -contracts},
            'long_call': {'strike': long_call['strike'].iloc[0], 'premium': long_call_prem, 'contracts': contracts},
            'short_put': {'strike': short_put['strike'].iloc[0], 'premium': short_put_prem, 'contracts': -contracts},
            'long_put': {'strike': long_put['strike'].iloc[0], 'premium': long_put_prem, 'contracts': contracts}
        }
    }

# Main content
tab1, tab2, tab3 = st.tabs(["📊 Fetch Options Data", "⚡ Leverage Strategies", "🛡️ Hedging Analysis"])

with tab1:
    st.markdown("### Real-Time Options Data")
    
    st.markdown("""
    Fetch live options chains from Yahoo Finance for any stock symbol.
    View calls, puts, and implied volatility surface.
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        symbol_input = st.text_input("Stock Symbol", value="AAPL", 
                                     help="Enter ticker symbol (e.g., AAPL, TSLA, SPY)")
    
    with col2:
        risk_free_rate = st.number_input("Risk-Free Rate", value=0.045, step=0.005, format="%.3f")
    
    if st.button("📡 Fetch Options Chain", type="primary"):
        with st.spinner(f"Fetching options data for {symbol_input}..."):
            calls, puts, current_price = fetch_options_yfinance(symbol_input)
            
            if calls is None or puts is None:
                st.error("Could not fetch options data. Check symbol or try again.")
            else:
                # Store in session state
                st.session_state['options_data'] = {
                    'symbol': symbol_input,
                    'calls': calls,
                    'puts': puts,
                    'current_price': current_price,
                    'risk_free_rate': risk_free_rate
                }
                
                st.success(f"✅ Fetched options for {symbol_input}")
                
                # Display current price
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.metric("Current Price", f"${current_price:.2f}")
                with col_b:
                    st.metric("Calls Available", len(calls))
                with col_c:
                    st.metric("Puts Available", len(puts))
                
                # Display options chains
                st.markdown("### 📞 Call Options")
                
                calls_display = calls[['strike', 'lastPrice', 'bid', 'ask', 'volume', 
                                      'openInterest', 'impliedVolatility']].copy()
                calls_display['type'] = 'CALL'
                
                # Highlight ATM
                calls_display['ATM'] = calls_display['strike'].apply(
                    lambda x: '✓' if abs(x - current_price) / current_price < 0.02 else ''
                )
                
                st.dataframe(calls_display.head(10), use_container_width=True)
                
                st.markdown("### 📉 Put Options")
                
                puts_display = puts[['strike', 'lastPrice', 'bid', 'ask', 'volume', 
                                    'openInterest', 'impliedVolatility']].copy()
                puts_display['type'] = 'PUT'
                puts_display['ATM'] = puts_display['strike'].apply(
                    lambda x: '✓' if abs(x - current_price) / current_price < 0.02 else ''
                )
                
                st.dataframe(puts_display.head(10), use_container_width=True)
                
                # IV Surface visualization
                st.markdown("### 📊 Implied Volatility Surface")
                
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('Call IV', 'Put IV'),
                    specs=[[{'type': 'scatter'}, {'type': 'scatter'}]]
                )
                
                # Call IV
                fig.add_trace(
                    go.Scatter(
                        x=calls['strike'],
                        y=calls['impliedVolatility'] * 100,
                        mode='lines+markers',
                        name='Call IV',
                        line={'color': 'blue'}
                    ),
                    row=1, col=1
                )
                
                # Put IV
                fig.add_trace(
                    go.Scatter(
                        x=puts['strike'],
                        y=puts['impliedVolatility'] * 100,
                        mode='lines+markers',
                        name='Put IV',
                        line={'color': 'red'}
                    ),
                    row=1, col=2
                )
                
                # Mark ATM
                fig.add_vline(x=current_price, line_dash="dash", line_color="green", 
                            row=1, col=1, annotation_text="ATM")
                fig.add_vline(x=current_price, line_dash="dash", line_color="green",
                            row=1, col=2, annotation_text="ATM")
                
                fig.update_xaxes(title_text="Strike", row=1, col=1)
                fig.update_xaxes(title_text="Strike", row=1, col=2)
                fig.update_yaxes(title_text="IV (%)", row=1, col=1)
                fig.update_yaxes(title_text="IV (%)", row=1, col=2)
                fig.update_layout(height=500, showlegend=False)
                
                st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("### Leveraged Options Strategies")
    
    if 'options_data' not in st.session_state:
        st.info("💡 Fetch options data in the first tab")
    else:
        options_data = st.session_state['options_data']
        symbol = options_data['symbol']
        calls = options_data['calls']
        puts = options_data['puts']
        S = options_data['current_price']
        
        st.markdown(f"**{symbol}** @ ${S:.2f}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            signal_type = st.selectbox(
                "Market Signal",
                ["Strong Bullish (High MR, Low Bubble)", 
                 "Moderate Bullish",
                 "Bearish/Bubble Risk",
                 "Neutral/Range-bound"]
            )
            
            capital = st.number_input("Capital to Deploy ($)", value=10000.0, step=1000.0)
        
        with col2:
            risk_tolerance = st.slider("Risk Tolerance", 0.05, 0.30, 0.15, 0.05,
                                      help="Maximum acceptable loss as % of stock price")
        
        if st.button("⚡ Generate Leverage Strategies", type="primary"):
            with st.spinner("Analyzing leverage opportunities..."):
                strategies = []
                
                # Select strategy based on signal
                if "Strong Bullish" in signal_type:
                    # Leveraged long with collar
                    strategy1 = synthetic_long_with_collar(signal_type, S, calls, puts, 
                                                          capital, risk_tolerance)
                    if strategy1:
                        strategies.append(strategy1)
                    
                    # Bull call spread
                    strategy2 = bull_call_spread(signal_type, S, calls, capital)
                    if strategy2:
                        strategies.append(strategy2)
                
                elif "Moderate Bullish" in signal_type:
                    # Bull call spread (primary)
                    strategy1 = bull_call_spread(signal_type, S, calls, capital)
                    if strategy1:
                        strategies.append(strategy1)
                
                elif "Bearish" in signal_type:
                    # Bear put spread
                    strategy1 = bear_put_spread(signal_type, S, puts, capital)
                    if strategy1:
                        strategies.append(strategy1)
                
                else:  # Neutral
                    # Iron condor
                    strategy1 = iron_condor(S, calls, puts, capital)
                    if strategy1:
                        strategies.append(strategy1)
                
                if not strategies:
                    st.error("Could not construct strategies with available options. Try different parameters.")
                else:
                    # Store strategies
                    st.session_state['leverage_strategies'] = strategies
                    
                    # Display strategies
                    for idx, strategy in enumerate(strategies):
                        st.markdown(f"### 🎯 Strategy {idx + 1}: {strategy['strategy']}")
                        
                        # Key metrics
                        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                        
                        with metric_col1:
                            st.metric("Leverage Ratio", f"{strategy['leverage_ratio']:.2f}x")
                        with metric_col2:
                            st.metric("Capital Required", f"${strategy['total_cost']:.2f}")
                        with metric_col3:
                            st.metric("Max Gain", f"${strategy['max_gain']:.2f}")
                        with metric_col4:
                            st.metric("Max Loss", f"${strategy['max_loss']:.2f}")
                        
                        # Risk-reward
                        risk_reward = strategy['max_gain'] / strategy['max_loss'] if strategy['max_loss'] > 0 else 0
                        roi_potential = (strategy['max_gain'] / strategy['total_cost'] * 100) if strategy['total_cost'] > 0 else 0
                        
                        st.info(f"""
                        **Risk/Reward:** {risk_reward:.2f} | **ROI Potential:** {roi_potential:.1f}% | 
                        **Breakeven:** ${strategy.get('breakeven', 0):.2f}
                        """)
                        
                        # Components breakdown
                        with st.expander("📋 Strategy Components"):
                            components_df = []
                            for comp_name, comp_data in strategy['components'].items():
                                components_df.append({
                                    'Component': comp_name.replace('_', ' ').title(),
                                    'Position': 'LONG' if comp_data['contracts'] > 0 else 'SHORT',
                                    'Contracts': abs(comp_data['contracts']),
                                    'Strike': f"${comp_data['strike']:.2f}",
                                    'Premium': f"${comp_data['premium']:.2f}",
                                    'Total': f"${abs(comp_data['contracts']) * comp_data['premium'] * 100:.2f}"
                                })
                            
                            st.table(pd.DataFrame(components_df))
                        
                        # P&L diagram
                        st.markdown("#### 💹 Profit/Loss Diagram")
                        
                        price_range = np.linspace(S * 0.8, S * 1.2, 100)
                        pnl = []
                        
                        for price in price_range:
                            total_pnl = 0
                            
                            for comp_name, comp in strategy['components'].items():
                                K = comp['strike']
                                premium = comp['premium']
                                contracts = comp['contracts']
                                
                                if 'call' in comp_name:
                                    intrinsic = max(price - K, 0)
                                    if contracts > 0:  # Long
                                        total_pnl += contracts * (intrinsic - premium) * 100
                                    else:  # Short
                                        total_pnl += contracts * (intrinsic - premium) * 100
                                else:  # put
                                    intrinsic = max(K - price, 0)
                                    if contracts > 0:  # Long
                                        total_pnl += contracts * (intrinsic - premium) * 100
                                    else:  # Short
                                        total_pnl += contracts * (intrinsic - premium) * 100
                            
                            pnl.append(total_pnl)
                        
                        fig_pnl = go.Figure()
                        
                        fig_pnl.add_trace(go.Scatter(
                            x=price_range,
                            y=pnl,
                            mode='lines',
                            name='P&L',
                            line={'color': 'purple', 'width': 3},
                            fill='tozeroy',
                            fillcolor='rgba(128,0,128,0.1)'
                        ))
                        
                        fig_pnl.add_vline(x=S, line_dash="dash", line_color="gray",
                                        annotation_text="Current Price")
                        fig_pnl.add_hline(y=0, line_color="black", line_width=1)
                        
                        if 'breakeven' in strategy:
                            fig_pnl.add_vline(x=strategy['breakeven'], line_dash="dot", 
                                            line_color="green", annotation_text="Breakeven")
                        
                        fig_pnl.update_layout(
                            xaxis_title='Stock Price at Expiration',
                            yaxis_title='Profit/Loss ($)',
                            height=400
                        )
                        
                        st.plotly_chart(fig_pnl, use_container_width=True)
                        
                        st.markdown("---")

with tab3:
    st.markdown("### Hedging & Risk Analysis")
    
    if 'leverage_strategies' not in st.session_state:
        st.info("💡 Generate leverage strategies in the previous tab")
    else:
        strategies = st.session_state['leverage_strategies']
        options_data = st.session_state['options_data']
        S = options_data['current_price']
        
        st.markdown("#### 🛡️ Portfolio Hedging Recommendations")
        
        # Portfolio integration
        if 'optimal_portfolio' in st.session_state:
            portfolio = st.session_state['optimal_portfolio']
            
            st.success("✅ Portfolio detected from Portfolio Optimizer Lab")
            
            st.markdown("**Portfolio Positions:**")
            for symbol, weight in zip(portfolio['symbols'], portfolio['weights']):
                st.write(f"- {symbol}: {weight*100:.1f}%")
            
            st.markdown("---")
            st.markdown("#### 🔒 Hedging Strategies")
            
            hedge_type = st.selectbox(
                "Hedge Type",
                ["Portfolio Protection (Puts)", "Income Generation (Covered Calls)", 
                 "Volatility Hedge (Straddle)", "Tail Risk Hedge (OTM Puts)"]
            )
            
            hedge_capital = st.number_input("Hedge Budget ($)", value=5000.0, step=500.0)
            
            if st.button("🛡️ Generate Hedge", type="primary"):
                st.info(f"**Selected Hedge:** {hedge_type}")
                
                if "Protection" in hedge_type:
                    st.markdown("""
                    ### Protective Put Strategy
                    
                    **Structure:**
                    - Buy OTM puts on each position (or index proxy)
                    - Strike selection: 5-10% below current prices
                    - Provides downside protection while maintaining upside
                    
                    **Cost:** ~1-3% of portfolio value per quarter
                    
                    **When to use:** Expecting volatility or drawdowns but want to stay invested
                    """)
                
                elif "Income" in hedge_type:
                    st.markdown("""
                    ### Covered Call Strategy
                    
                    **Structure:**
                    - Sell OTM calls on existing positions
                    - Strike selection: 5-10% above current prices
                    - Generates premium income, caps upside
                    
                    **Income:** ~1-3% of position value per month
                    
                    **When to use:** Neutral to moderately bullish, want to enhance returns
                    """)
                
                elif "Volatility" in hedge_type:
                    st.markdown("""
                    ### Long Straddle Hedge
                    
                    **Structure:**
                    - Buy ATM call + ATM put
                    - Profits from large moves in either direction
                    - Protects against unexpected events
                    
                    **Cost:** ~3-5% of portfolio value
                    
                    **When to use:** Expecting large move but uncertain of direction
                    """)
                
                else:  # Tail risk
                    st.markdown("""
                    ### Tail Risk Hedge (Black Swan Protection)
                    
                    **Structure:**
                    - Buy far OTM puts (10-20% below)
                    - Low cost, high payoff in crash scenarios
                    - Insurance against extreme events
                    
                    **Cost:** ~0.5-1% of portfolio value per quarter
                    
                    **When to use:** Want protection against market crashes at minimal cost
                    """)
                
                # Generic metrics
                st.markdown("#### 📊 Hedge Effectiveness Metrics")
                
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    hedge_ratio = hedge_capital / 100000 * 100  # Assume $100k portfolio
                    st.metric("Hedge Ratio", f"{hedge_ratio:.2f}%")
                with col_b:
                    protection_level = 85 if "Protection" in hedge_type else 95
                    st.metric("Protection Level", f"{protection_level}%")
                with col_c:
                    annual_cost = hedge_capital * 4 if "Protection" in hedge_type else hedge_capital * 12
                    st.metric("Annualized Cost", f"${annual_cost:.0f}")
        
        else:
            st.warning("No portfolio detected. Showing general hedging principles.")
        
        # Hedging best practices
        st.markdown("---")
        st.markdown("### 💡 Hedging Best Practices")
        
        st.info("""
        **1. Cost-Benefit Analysis:**
        - Hedging costs money - evaluate if protection justifies expense
        - Consider rolling hedges (monthly/quarterly) vs one-time
        
        **2. Dynamic Hedging:**
        - Adjust hedge ratios based on market conditions
        - Increase protection when volatility rises
        - Reduce hedges in low-volatility environments
        
        **3. Correlation Considerations:**
        - Ensure hedges are negatively correlated with positions
        - Test hedge effectiveness in historical scenarios
        
        **4. Leverage + Hedge = Risk Management:**
        - Leverage amplifies gains but also losses
        - Appropriate hedging allows responsible leverage use
        - Monitor Greek exposures (delta, gamma, vega)
        
        **5. Expiration Management:**
        - Don't let hedges expire worthless
        - Roll hedges before expiration
        - Consider calendar spreads for continuous protection
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>⚡ Options Leverage Lab | Real Data, Real Strategies</p>
</div>
""", unsafe_allow_html=True)
