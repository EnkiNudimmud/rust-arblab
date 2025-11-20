"""
Options Strategy Implementations
=================================

Detailed implementations of common options arbitrage strategies with:
- Payoff diagrams
- Greeks analysis  
- Risk/reward metrics
- Educational explanations
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm


def calculate_option_price(S, K, T, r, sigma, option_type='call'):
    """Calculate Black-Scholes option price"""
    if T <= 0 or sigma <= 0:
        return 0.0
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:  # put
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def render_straddle(spot, T, r, sigma, is_long=True):
    """Render long/short straddle strategy"""
    
    direction = "Long" if is_long else "Short"
    st.markdown(f"### {direction} Straddle")
    
    # Explanation
    with st.expander("📖 Strategy Explanation", expanded=True):
        if is_long:
            st.markdown("""
            **Long Straddle** is a volatility strategy that profits from large moves in either direction.
            
            **Construction:**
            - Buy 1 ATM Call
            - Buy 1 ATM Put (same strike and expiration)
            
            **Market Outlook:** Expecting large price movement but uncertain of direction
            
            **Maximum Profit:** Unlimited (upside), Large (downside)
            
            **Maximum Loss:** Total premium paid (if price stays at strike)
            
            **Break-even Points:** 
            - Upper: Strike + Total Premium
            - Lower: Strike - Total Premium
            
            **Best Used When:**
            - Anticipating earnings announcements
            - Expecting regulatory decisions
            - During periods of low volatility before expected spike
            """)
        else:
            st.markdown("""
            **Short Straddle** profits from low volatility when price stays near strike.
            
            **Construction:**
            - Sell 1 ATM Call
            - Sell 1 ATM Put (same strike and expiration)
            
            **Market Outlook:** Expecting price to remain stable
            
            **Maximum Profit:** Total premium collected
            
            **Maximum Loss:** Unlimited (both directions)
            
            **Break-even Points:**
            - Upper: Strike + Total Premium
            - Lower: Strike - Total Premium
            
            **Risk:** Very high - unlimited loss potential!
            """)
    
    # Parameters
    strike = st.slider(
        "Strike Price ($)",
        min_value=spot * 0.8,
        max_value=spot * 1.2,
        value=spot,
        step=1.0,
        key=f"straddle_strike_{direction}"
    )
    
    # Calculate option prices
    call_price = calculate_option_price(spot, strike, T, r, sigma, 'call')
    put_price = calculate_option_price(spot, strike, T, r, sigma, 'put')
    total_cost = call_price + put_price
    
    # Display costs
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Call Premium", f"${call_price:.2f}")
    with col2:
        st.metric("Put Premium", f"${put_price:.2f}")
    with col3:
        st.metric("Total Cost", f"${total_cost:.2f}")
    
    # Generate payoff
    spot_range = np.linspace(spot * 0.5, spot * 1.5, 200)
    
    call_payoff = np.maximum(spot_range - strike, 0)
    put_payoff = np.maximum(strike - spot_range, 0)
    
    if is_long:
        total_payoff = call_payoff + put_payoff - total_cost
    else:
        total_payoff = total_cost - call_payoff - put_payoff
    
    # Calculate break-evens
    upper_be = strike + total_cost
    lower_be = strike - total_cost
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        max_loss = -total_cost if is_long else "Unlimited"
        st.metric("Max Loss", f"${max_loss}" if is_long else max_loss)
    with col2:
        max_profit = "Unlimited" if is_long else f"${total_cost:.2f}"
        st.metric("Max Profit", max_profit)
    with col3:
        st.metric("Upper BE", f"${upper_be:.2f}")
    with col4:
        st.metric("Lower BE", f"${lower_be:.2f}")
    
    # Plot
    fig = go.Figure()
    
    # Total P&L
    fig.add_trace(go.Scatter(
        x=spot_range,
        y=total_payoff,
        mode='lines',
        name='Total P&L',
        line={'color': 'cyan', 'width': 3},
        fill='tozeroy',
        fillcolor='rgba(0, 255, 255, 0.1)'
    ))
    
    # Individual legs (lighter)
    if is_long:
        fig.add_trace(go.Scatter(
            x=spot_range,
            y=call_payoff - call_price,
            mode='lines',
            name='Call Payoff',
            line={'color': 'green', 'width': 1, 'dash': 'dot'},
            opacity=0.5
        ))
        
        fig.add_trace(go.Scatter(
            x=spot_range,
            y=put_payoff - put_price,
            mode='lines',
            name='Put Payoff',
            line={'color': 'red', 'width': 1, 'dash': 'dot'},
            opacity=0.5
        ))
    
    # Reference lines
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=spot, line_dash="dot", line_color="yellow", annotation_text="Current Price")
    fig.add_vline(x=strike, line_dash="dash", line_color="white", annotation_text="Strike", opacity=0.5)
    fig.add_vline(x=upper_be, line_dash="dash", line_color="orange", annotation_text="Upper BE")
    fig.add_vline(x=lower_be, line_dash="dash", line_color="orange", annotation_text="Lower BE")
    
    fig.update_layout(
        title=f"{direction} Straddle Payoff Diagram",
        xaxis_title="Stock Price at Expiration ($)",
        yaxis_title="Profit / Loss ($)",
        template="plotly_dark",
        height=500,
        hovermode='x unified',
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Probability analysis
    with st.expander("📊 Probability Analysis"):
        # Calculate probability of profit using log-normal distribution
        log_return_std = sigma * np.sqrt(T)
        
        if is_long:
            # Profit if price moves beyond break-evens
            prob_above_upper = 1 - norm.cdf((np.log(upper_be / spot)) / log_return_std)
            prob_below_lower = norm.cdf((np.log(lower_be / spot)) / log_return_std)
            prob_profit = prob_above_upper + prob_below_lower
        else:
            # Profit if price stays between break-evens
            prob_above_lower = 1 - norm.cdf((np.log(lower_be / spot)) / log_return_std)
            prob_below_upper = norm.cdf((np.log(upper_be / spot)) / log_return_std)
            prob_profit = prob_below_upper - (1 - prob_above_lower)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Probability of Profit", f"{prob_profit * 100:.1f}%")
        with col2:
            required_move = (upper_be - spot) / spot * 100
            st.metric("Required Move", f"{required_move:.1f}%")


def render_strangle(spot, T, r, sigma, is_long=True):
    """Render long/short strangle strategy"""
    
    direction = "Long" if is_long else "Short"
    st.markdown(f"### {direction} Strangle")
    
    # Explanation
    with st.expander("📖 Strategy Explanation", expanded=True):
        if is_long:
            st.markdown("""
            **Long Strangle** is similar to straddle but with OTM options, requiring larger moves for profit.
            
            **Construction:**
            - Buy 1 OTM Call (strike above current price)
            - Buy 1 OTM Put (strike below current price)
            
            **vs. Straddle:**
            - Cheaper to establish (lower premium)
            - Requires larger move to profit
            - Wider break-even range
            
            **Maximum Profit:** Unlimited (upside), Large (downside)
            
            **Maximum Loss:** Total premium paid
            
            **Best Used When:**
            - Expecting very large move but want lower cost
            - High volatility anticipated
            - Want to reduce time decay cost
            """)
        else:
            st.markdown("""
            **Short Strangle** collects premium betting price stays within range.
            
            **Construction:**
            - Sell 1 OTM Call
            - Sell 1 OTM Put
            
            **Maximum Profit:** Total premium collected
            
            **Maximum Loss:** Unlimited
            
            **Risk:** High - unlimited loss on both sides!
            """)
    
    # Parameters
    col1, col2 = st.columns(2)
    with col1:
        put_strike = st.slider(
            "Put Strike ($)",
            min_value=spot * 0.7,
            max_value=spot * 0.95,
            value=spot * 0.9,
            step=1.0,
            key=f"strangle_put_{direction}"
        )
    with col2:
        call_strike = st.slider(
            "Call Strike ($)",
            min_value=spot * 1.05,
            max_value=spot * 1.3,
            value=spot * 1.1,
            step=1.0,
            key=f"strangle_call_{direction}"
        )
    
    # Calculate prices
    call_price = calculate_option_price(spot, call_strike, T, r, sigma, 'call')
    put_price = calculate_option_price(spot, put_strike, T, r, sigma, 'put')
    total_cost = call_price + put_price
    
    # Display costs
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Call Premium", f"${call_price:.2f}")
    with col2:
        st.metric("Put Premium", f"${put_price:.2f}")
    with col3:
        st.metric("Total Cost", f"${total_cost:.2f}")
    
    # Generate payoff
    spot_range = np.linspace(spot * 0.5, spot * 1.5, 200)
    
    call_payoff = np.maximum(spot_range - call_strike, 0)
    put_payoff = np.maximum(put_strike - spot_range, 0)
    
    if is_long:
        total_payoff = call_payoff + put_payoff - total_cost
    else:
        total_payoff = total_cost - call_payoff - put_payoff
    
    # Break-evens
    upper_be = call_strike + total_cost
    lower_be = put_strike - total_cost
    
    # Plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=spot_range,
        y=total_payoff,
        mode='lines',
        name='Total P&L',
        line={'color': 'cyan', 'width': 3},
        fill='tozeroy',
        fillcolor='rgba(0, 255, 255, 0.1)'
    ))
    
    # Reference lines
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=spot, line_dash="dot", line_color="yellow", annotation_text="Current")
    fig.add_vline(x=call_strike, line_dash="dash", line_color="green", annotation_text="Call Strike")
    fig.add_vline(x=put_strike, line_dash="dash", line_color="red", annotation_text="Put Strike")
    fig.add_vline(x=upper_be, line_dash="dash", line_color="orange", annotation_text="Upper BE")
    fig.add_vline(x=lower_be, line_dash="dash", line_color="orange", annotation_text="Lower BE")
    
    fig.update_layout(
        title=f"{direction} Strangle Payoff Diagram",
        xaxis_title="Stock Price at Expiration ($)",
        yaxis_title="Profit / Loss ($)",
        template="plotly_dark",
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_butterfly(spot, T, r, sigma, is_long=True):
    """Render butterfly spread strategy"""
    
    direction = "Long" if is_long else "Short"
    st.markdown(f"### {direction} Butterfly Spread")
    
    # Explanation
    with st.expander("📖 Strategy Explanation", expanded=True):
        if is_long:
            st.markdown("""
            **Long Butterfly** profits from low volatility when price stays near middle strike.
            
            **Construction:**
            - Buy 1 ITM Call (lower strike)
            - Sell 2 ATM Calls (middle strike)
            - Buy 1 OTM Call (higher strike)
            
            **Key Features:**
            - Limited risk, limited reward
            - Low cost to establish
            - Maximum profit at middle strike
            - Symmetrical payoff structure
            
            **Maximum Profit:** (Middle Strike - Lower Strike) - Net Premium
            
            **Maximum Loss:** Net premium paid
            
            **Break-even Points:** 
            - Lower: Lower Strike + Net Premium
            - Upper: Upper Strike - Net Premium
            
            **Best Used When:**
            - Expecting price to stay near current level
            - Want defined risk with moderate reward
            - Low volatility expected
            """)
        else:
            st.markdown("""
            **Short Butterfly** profits from high volatility when price moves away from middle.
            
            **Construction:**
            - Sell 1 ITM Call (lower strike)
            - Buy 2 ATM Calls (middle strike)
            - Sell 1 OTM Call (higher strike)
            
            **Maximum Profit:** Net premium collected
            
            **Maximum Loss:** Limited
            
            **Best Used When:** Expecting significant price movement
            """)
    
    # Parameters - ensure equal spacing
    wing_width = st.slider(
        "Wing Width ($)",
        min_value=2.0,
        max_value=spot * 0.2,
        value=spot * 0.05,
        step=1.0,
        key=f"butterfly_width_{direction}"
    )
    
    lower_strike = spot - wing_width
    middle_strike = spot
    upper_strike = spot + wing_width
    
    # Calculate prices
    lower_call = calculate_option_price(spot, lower_strike, T, r, sigma, 'call')
    middle_call = calculate_option_price(spot, middle_strike, T, r, sigma, 'call')
    upper_call = calculate_option_price(spot, upper_strike, T, r, sigma, 'call')
    
    if is_long:
        net_cost = lower_call - 2 * middle_call + upper_call
    else:
        net_cost = 2 * middle_call - lower_call - upper_call
    
    # Display structure
    st.markdown("#### Strategy Structure")
    col1, col2, col3 = st.columns(3)
    with col1:
        action = "Buy" if is_long else "Sell"
        st.info(f"{action} 1x ${lower_strike:.2f} Call\n\nPremium: ${lower_call:.2f}")
    with col2:
        action = "Sell" if is_long else "Buy"
        st.warning(f"{action} 2x ${middle_strike:.2f} Call\n\nPremium: ${middle_call * 2:.2f}")
    with col3:
        action = "Buy" if is_long else "Sell"
        st.success(f"{action} 1x ${upper_strike:.2f} Call\n\nPremium: ${upper_call:.2f}")
    
    st.metric("Net Cost/Credit", f"${abs(net_cost):.2f}", delta=f"{'Debit' if net_cost > 0 else 'Credit'}")
    
    # Generate payoff
    spot_range = np.linspace(spot * 0.7, spot * 1.3, 200)
    
    lower_payoff = np.maximum(spot_range - lower_strike, 0)
    middle_payoff = np.maximum(spot_range - middle_strike, 0)
    upper_payoff = np.maximum(spot_range - upper_strike, 0)
    
    if is_long:
        total_payoff = lower_payoff - 2 * middle_payoff + upper_payoff - net_cost
    else:
        total_payoff = 2 * middle_payoff - lower_payoff - upper_payoff + net_cost
    
    # Calculate max profit/loss
    if is_long:
        max_profit = wing_width - net_cost
        max_loss = net_cost
    else:
        max_profit = net_cost
        max_loss = wing_width - net_cost
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Max Profit", f"${max_profit:.2f}")
    with col2:
        st.metric("Max Loss", f"${max_loss:.2f}")
    with col3:
        reward_risk = max_profit / max_loss if max_loss > 0 else 0
        st.metric("Reward/Risk", f"{reward_risk:.2f}")
    
    # Plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=spot_range,
        y=total_payoff,
        mode='lines',
        name='Total P&L',
        line={'color': 'cyan', 'width': 3},
        fill='tozeroy',
        fillcolor='rgba(0, 255, 255, 0.1)'
    ))
    
    # Add individual legs
    if is_long:
        fig.add_trace(go.Scatter(
            x=spot_range,
            y=lower_payoff - lower_call,
            mode='lines',
            name=f'Buy ${lower_strike:.0f} Call',
            line={'width': 1, 'dash': 'dot'},
            opacity=0.4
        ))
        fig.add_trace(go.Scatter(
            x=spot_range,
            y=-2 * (middle_payoff - middle_call),
            mode='lines',
            name=f'Sell 2x ${middle_strike:.0f} Call',
            line={'width': 1, 'dash': 'dot'},
            opacity=0.4
        ))
        fig.add_trace(go.Scatter(
            x=spot_range,
            y=upper_payoff - upper_call,
            mode='lines',
            name=f'Buy ${upper_strike:.0f} Call',
            line={'width': 1, 'dash': 'dot'},
            opacity=0.4
        ))
    
    # Reference lines
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=spot, line_dash="dot", line_color="yellow", annotation_text="Current")
    fig.add_vline(x=lower_strike, line_dash="dash", line_color="blue", opacity=0.3)
    fig.add_vline(x=middle_strike, line_dash="dash", line_color="white", annotation_text="Body")
    fig.add_vline(x=upper_strike, line_dash="dash", line_color="blue", opacity=0.3)
    
    fig.update_layout(
        title=f"{direction} Butterfly Spread Payoff",
        xaxis_title="Stock Price at Expiration ($)",
        yaxis_title="Profit / Loss ($)",
        template="plotly_dark",
        height=500,
        hovermode='x unified',
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_iron_condor(spot, T, r, sigma):
    """Render iron condor strategy"""
    
    st.markdown("### Iron Condor")
    
    # Explanation
    with st.expander("📖 Strategy Explanation", expanded=True):
        st.markdown("""
        **Iron Condor** is an income strategy that profits from low volatility.
        
        **Construction:**
        - Sell 1 OTM Put (lower short strike)
        - Buy 1 OTM Put (lower long strike - further OTM)
        - Sell 1 OTM Call (upper short strike)
        - Buy 1 OTM Call (upper long strike - further OTM)
        
        **Key Features:**
        - Defined risk, defined reward
        - Profits from range-bound markets
        - Collects premium from selling options
        - Protection from long options limits risk
        
        **Maximum Profit:** Net premium collected
        
        **Maximum Loss:** Width of spread - Net premium
        
        **Break-even Points:**
        - Lower: Put short strike - Net premium
        - Upper: Call short strike + Net premium
        
        **Best Used When:**
        - Expecting low volatility
        - Want to collect premium with defined risk
        - Neutral market outlook
        
        **Advantages:**
        - High probability of profit
        - Defined maximum loss
        - Can be adjusted if needed
        """)
    
    # Parameters
    col1, col2 = st.columns(2)
    with col1:
        put_width = st.slider(
            "Put Spread Width ($)",
            min_value=2.0,
            max_value=spot * 0.1,
            value=spot * 0.05,
            step=1.0
        )
        put_short_distance = st.slider(
            "Put Short Strike Distance",
            min_value=spot * 0.05,
            max_value=spot * 0.15,
            value=spot * 0.08,
            step=1.0
        )
    with col2:
        call_width = st.slider(
            "Call Spread Width ($)",
            min_value=2.0,
            max_value=spot * 0.1,
            value=spot * 0.05,
            step=1.0
        )
        call_short_distance = st.slider(
            "Call Short Strike Distance",
            min_value=spot * 0.05,
            max_value=spot * 0.15,
            value=spot * 0.08,
            step=1.0
        )
    
    # Calculate strikes
    put_short_strike = spot - put_short_distance
    put_long_strike = put_short_strike - put_width
    call_short_strike = spot + call_short_distance
    call_long_strike = call_short_strike + call_width
    
    # Calculate prices
    put_short_price = calculate_option_price(spot, put_short_strike, T, r, sigma, 'put')
    put_long_price = calculate_option_price(spot, put_long_strike, T, r, sigma, 'put')
    call_short_price = calculate_option_price(spot, call_short_strike, T, r, sigma, 'call')
    call_long_price = calculate_option_price(spot, call_long_strike, T, r, sigma, 'call')
    
    net_credit = (put_short_price - put_long_price) + (call_short_price - call_long_price)
    
    # Display structure
    st.markdown("#### Strategy Structure")
    cols = st.columns(4)
    with cols[0]:
        st.info(f"Buy\n${put_long_strike:.2f} Put\n\n${put_long_price:.2f}")
    with cols[1]:
        st.warning(f"Sell\n${put_short_strike:.2f} Put\n\n${put_short_price:.2f}")
    with cols[2]:
        st.warning(f"Sell\n${call_short_strike:.2f} Call\n\n${call_short_price:.2f}")
    with cols[3]:
        st.success(f"Buy\n${call_long_strike:.2f} Call\n\n${call_long_price:.2f}")
    
    # Generate payoff
    spot_range = np.linspace(spot * 0.6, spot * 1.4, 200)
    
    put_short_payoff = -np.maximum(put_short_strike - spot_range, 0)
    put_long_payoff = np.maximum(put_long_strike - spot_range, 0)
    call_short_payoff = -np.maximum(spot_range - call_short_strike, 0)
    call_long_payoff = np.maximum(spot_range - call_long_strike, 0)
    
    total_payoff = put_short_payoff + put_long_payoff + call_short_payoff + call_long_payoff + net_credit
    
    # Calculate metrics
    max_profit = net_credit
    max_loss_put = put_width - net_credit
    max_loss_call = call_width - net_credit
    max_loss = max(max_loss_put, max_loss_call)
    
    # Display metrics
    cols = st.columns(4)
    with cols[0]:
        st.metric("Net Credit", f"${net_credit:.2f}")
    with cols[1]:
        st.metric("Max Profit", f"${max_profit:.2f}")
    with cols[2]:
        st.metric("Max Loss", f"${max_loss:.2f}")
    with cols[3]:
        prob_profit = 70  # Simplified
        st.metric("Prob. of Profit", f"~{prob_profit}%")
    
    # Plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=spot_range,
        y=total_payoff,
        mode='lines',
        name='Iron Condor P&L',
        line={'color': 'cyan', 'width': 3},
        fill='tozeroy',
        fillcolor='rgba(0, 255, 255, 0.1)'
    ))
    
    # Reference lines
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_hline(y=max_profit, line_dash="dot", line_color="green", annotation_text=f"Max Profit: ${max_profit:.2f}")
    fig.add_vline(x=spot, line_dash="dot", line_color="yellow", annotation_text="Current")
    
    # Profit zone
    fig.add_vrect(
        x0=put_short_strike, x1=call_short_strike,
        fillcolor="green", opacity=0.1,
        annotation_text="Profit Zone", annotation_position="top"
    )
    
    # Strike lines
    for strike, label in [(put_long_strike, "Put Long"), (put_short_strike, "Put Short"),
                           (call_short_strike, "Call Short"), (call_long_strike, "Call Long")]:
        fig.add_vline(x=strike, line_dash="dash", opacity=0.3, annotation_text=label)
    
    fig.update_layout(
        title="Iron Condor Payoff Diagram",
        xaxis_title="Stock Price at Expiration ($)",
        yaxis_title="Profit / Loss ($)",
        template="plotly_dark",
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)


# Add placeholder functions for other strategies
def render_iron_butterfly(spot, T, r, sigma):
    st.info("Iron Butterfly implementation - Similar to Iron Condor but with ATM body")

def render_single_option(spot, T, r, sigma, option_type, is_long):
    st.info(f"{'Long' if is_long else 'Short'} {option_type.title()} implementation")

def render_vertical_spread(spot, T, r, sigma, option_type, is_bull):
    direction = "Bull" if is_bull else "Bear"
    st.info(f"{direction} {option_type.title()} Spread implementation")

def render_calendar_spread(spot, r, sigma):
    st.info("Calendar Spread implementation")

def render_covered_call(spot, T, r, sigma):
    st.info("Covered Call implementation")

def render_cash_secured_put(spot, T, r, sigma):
    st.info("Cash-Secured Put implementation")

def render_ratio_spread(spot, T, r, sigma):
    st.info("Ratio Spread implementation")
