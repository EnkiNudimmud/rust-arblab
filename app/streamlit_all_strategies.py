"""
Multi-Strategy Trading System Dashboard
========================================

A comprehensive Streamlit app showcasing ALL trading strategies with:
- Mathematical equations and theory
- Rich interactive visualizations
- Multi-strategy backtesting comparison
- Real-world intraday data support

Strategies Included:
1. Mean Reversion (CARA, Sharpe, Multi-Period)
2. Statistical Arbitrage (Pairs Trading)
3. Triangular Arbitrage
4. Market Making
5. Signature-based Optimal Stopping
6. Hawkes Process Modeling

Author: Rust HFT Arbitrage Lab
"""

import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import time

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to path
sys.path.append('/Users/melvinalvarez/Documents/Workspace/rust-hft-arbitrage-lab')

# Import modules
from python import meanrev
from python.data_fetcher import fetch_intraday_data, get_close_prices, get_universe_symbols

# Check Rust availability
try:
    import rust_connector
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Multi-Strategy Trading Lab",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .strategy-header {
        font-size: 2rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .equation {
        font-size: 1.1rem;
        background-color: #f8f9fa;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# STRATEGY IMPLEMENTATIONS
# ============================================================================

class MeanReversionStrategy:
    """Mean-reverting portfolio strategies (PCA, CARA, Sharpe)"""
    
    @staticmethod
    def theory() -> str:
        return r"""
        ### Mean Reversion Theory
        
        **Ornstein-Uhlenbeck Process:**
        $$dS_t = \theta(\mu - S_t)dt + \sigma dW_t$$
        
        **CARA Utility:**
        $$w^* = \frac{1}{\gamma} \Sigma^{-1} \mu$$
        
        **Sharpe Optimization:**
        $$w^* = \frac{\Sigma^{-1}(\mu - r_f \mathbf{1})}{\mathbf{1}^T \Sigma^{-1}(\mu - r_f \mathbf{1})}$$
        """
    
    @staticmethod
    def compute(prices: pd.DataFrame, params: Dict) -> Dict:
        """Compute mean-reverting portfolios"""
        results = {}
        
        # Compute log returns
        rets = np.log(prices).diff().dropna()
        
        # 1. PC1 Mean Reversion
        try:
            # pca_portfolios expects a DataFrame, not numpy array
            pcs, pca_info = meanrev.pca_portfolios(rets, n_components=min(5, rets.shape[1]))
            # pcs is shape (n_components, n_assets), we want first component
            pc1_weights = pcs[0, :]  # Use first row (first principal component)
            pc1_weights = pc1_weights / (np.sum(np.abs(pc1_weights)) + 1e-12)
            # Align prices with returns index
            aligned_prices = prices.loc[rets.index]
            pc1_series = (aligned_prices.values @ pc1_weights)
            pc1_series = pd.Series(pc1_series, index=rets.index)
            
            back = meanrev.backtest_with_costs(
                pc1_series, 
                entry_z=params.get('entry_z', 2.0),
                exit_z=params.get('exit_z', 0.5),
                transaction_cost=params.get('transaction_cost', 0.001)
            )
            results['PC1'] = {'series': pc1_series, 'weights': pc1_weights, 'backtest': back}
        except Exception as e:
            st.warning(f"PC1 failed: {e}")
        
        # 2. CARA Optimal
        try:
            # cara_optimal_weights needs expected_returns AND covariance as separate arguments
            expected_returns = rets.mean().values
            covariance = rets.cov().values
            
            cara_result = meanrev.cara_optimal_weights(expected_returns, covariance, gamma=params.get('gamma', 2.0))
            cara_w = np.array(cara_result['weights'], dtype=float)
            cara_w = cara_w / (np.sum(np.abs(cara_w)) + 1e-12)
            cara_series = prices.iloc[len(prices) - len(rets):].values @ cara_w
            cara_series = pd.Series(cara_series, index=rets.index)
            
            cara_back = meanrev.backtest_with_costs(
                cara_series,
                entry_z=params.get('entry_z', 2.0),
                exit_z=params.get('exit_z', 0.5),
                transaction_cost=params.get('transaction_cost', 0.001)
            )
            results['CARA'] = {'series': cara_series, 'weights': cara_w, 'backtest': cara_back}
        except Exception as e:
            st.warning(f"CARA failed: {e}")
        
        # 3. Sharpe Optimal
        try:
            # sharpe_optimal_weights expects 'risk_free_rate' not 'risk_free'
            expected_returns = rets.mean().values
            covariance = rets.cov().values
            
            sharpe_result = meanrev.sharpe_optimal_weights(expected_returns, covariance, risk_free_rate=params.get('risk_free', 0.02))
            sharpe_w = np.array(sharpe_result['weights'], dtype=float)
            sharpe_w = sharpe_w / (np.sum(np.abs(sharpe_w)) + 1e-12)
            sharpe_series = prices.iloc[len(prices) - len(rets):].values @ sharpe_w
            sharpe_series = pd.Series(sharpe_series, index=rets.index)
            
            sharpe_back = meanrev.backtest_with_costs(
                sharpe_series,
                entry_z=params.get('entry_z', 2.0),
                exit_z=params.get('exit_z', 0.5),
                transaction_cost=params.get('transaction_cost', 0.001)
            )
            results['Sharpe'] = {'series': sharpe_series, 'weights': sharpe_w, 'backtest': sharpe_back}
        except Exception as e:
            st.warning(f"Sharpe failed: {e}")
        
        return results


class PairsTradingStrategy:
    """Statistical arbitrage pairs trading"""
    
    @staticmethod
    def theory() -> str:
        return r"""
        ### Pairs Trading Theory
        
        **OLS Hedge Ratio:**
        $$y_t = \beta x_t + c + \epsilon_t$$
        
        **Spread:**
        $$s_t = y_t - \beta x_t$$
        
        **Z-Score:**
        $$z_t = \frac{s_t - \mu_s}{\sigma_s}$$
        
        **Trading Signal:** Long when $z < -2$, Short when $z > 2$
        """
    
    @staticmethod
    def compute(prices: pd.DataFrame, params: Dict) -> Dict:
        """Compute pairs trading strategy"""
        if prices.shape[1] < 2:
            return {}
        
        results = {}
        
        # Use first two assets as pair
        x = prices.iloc[:, 0].values
        y = prices.iloc[:, 1].values
        
        # OLS regression
        X = np.vstack([x, np.ones_like(x)]).T
        beta, c = np.linalg.lstsq(X, y, rcond=None)[0]
        
        # Compute spread
        spread = y - (beta * x + c)
        spread_series = pd.Series(spread, index=prices.index)
        
        # Rolling z-score
        window = params.get('window', 50)
        mu = spread_series.rolling(window).mean()
        sig = spread_series.rolling(window).std().replace(0, 1e-9)
        z = (spread_series - mu) / sig
        
        # Trading signals
        entry_z = params.get('entry_z', 2.0)
        pos = (z < -entry_z).astype(int) - (z > entry_z).astype(int)
        
        # Compute PnL
        ret = spread_series.diff().fillna(0)
        pnl = (pos.shift(1).fillna(0) * ret).cumsum()
        
        results['Pairs'] = {
            'spread': spread_series,
            'zscore': z,
            'positions': pos,
            'pnl': pnl,
            'beta': beta,
            'symbols': (prices.columns[0], prices.columns[1])
        }
        
        return results


class TriangularArbStrategy:
    """Triangular arbitrage detection"""
    
    @staticmethod
    def theory() -> str:
        return r"""
        ### Triangular Arbitrage Theory
        
        **Three currency pairs:** A/B, B/C, C/A
        
        **Forward path:**
        $$P_{\text{forward}} = P_{AB} \times P_{BC} \times P_{CA}$$
        
        **Arbitrage opportunity when:**
        $$P_{\text{forward}} \neq 1$$
        
        **Profit:**
        $$\pi = |1 - P_{\text{forward}}| - \text{costs}$$
        """
    
    @staticmethod
    def compute(prices: pd.DataFrame, params: Dict) -> Dict:
        """Detect triangular arbitrage opportunities"""
        if prices.shape[1] < 3:
            return {}
        
        results = {}
        
        # Use first three assets to form triangle
        p1 = prices.iloc[:, 0]
        p2 = prices.iloc[:, 1]
        p3 = prices.iloc[:, 2]
        
        # Synthetic cross rates
        forward = (p1 / p2) * (p2 / p3) * (p3 / p1)
        
        # Arbitrage signal
        arb_signal = np.abs(forward - 1.0)
        threshold = params.get('threshold', 0.001)
        
        opportunities = arb_signal > threshold
        
        # Compute theoretical PnL (before costs)
        pnl = (arb_signal * opportunities).cumsum() * 10000  # Scale for visibility
        
        results['Triangular'] = {
            'forward': forward,
            'arb_signal': arb_signal,
            'opportunities': opportunities,
            'pnl': pnl,
            'symbols': (prices.columns[0], prices.columns[1], prices.columns[2])
        }
        
        return results


class MarketMakingStrategy:
    """Market making with inventory control"""
    
    @staticmethod
    def theory() -> str:
        return r"""
        ### Market Making Theory
        
        **Quote Prices:**
        $$P_{\text{bid}} = P_{\text{mid}} - \frac{s}{2} - \gamma \cdot I$$
        $$P_{\text{ask}} = P_{\text{mid}} + \frac{s}{2} - \gamma \cdot I$$
        
        Where:
        - $s$ = spread
        - $I$ = inventory
        - $\gamma$ = inventory aversion
        
        **PnL:**
        $$\text{PnL}_t = \text{Cash}_t + I_t \times P_{\text{mid},t}$$
        """
    
    @staticmethod
    def compute(prices: pd.DataFrame, params: Dict) -> Dict:
        """Simulate market making strategy"""
        if prices.shape[1] < 1:
            return {}
        
        results = {}
        
        # Use first asset
        mid = prices.iloc[:, 0].values
        T = len(mid)
        
        spread = params.get('spread', 0.002) * mid  # 20 bps
        inv_aversion = params.get('inv_aversion', 0.1)
        
        inventory = np.zeros(T)
        cash = 0.0
        pnl = np.zeros(T)
        
        for t in range(1, T):
            # Current inventory affects quotes
            bid = mid[t] - spread[t] / 2 - inv_aversion * inventory[t-1]
            ask = mid[t] + spread[t] / 2 - inv_aversion * inventory[t-1]
            
            # Simulate fills (simple random model)
            if np.random.rand() < 0.3:  # 30% fill probability
                side = np.random.choice([-1, 1])  # -1=buy, 1=sell
                
                inventory[t] = inventory[t-1] + side
                
                if side == -1:  # We buy (pay ask)
                    cash -= ask
                else:  # We sell (receive bid)
                    cash += bid
            else:
                inventory[t] = inventory[t-1]
            
            # Mark to market
            pnl[t] = cash + inventory[t] * mid[t]
        
        results['MarketMaking'] = {
            'mid': pd.Series(mid, index=prices.index),
            'inventory': pd.Series(inventory, index=prices.index),
            'pnl': pd.Series(pnl, index=prices.index),
            'symbol': prices.columns[0]
        }
        
        return results


# ============================================================================
# SIDEBAR CONTROLS
# ============================================================================

def sidebar_controls():
    st.sidebar.title("🎛️ Strategy Controls")
    
    # Data selection
    st.sidebar.header("📊 Data Configuration")
    market = st.sidebar.selectbox("Market", ["crypto", "stocks"])
    
    default_symbols = get_universe_symbols(market)[:30]
    symbols = st.sidebar.multiselect(
        "Symbols (max 30)",
        default_symbols,
        default=default_symbols[:10]
    )
    
    interval = st.sidebar.selectbox("Interval", ["1min", "5min", "15min", "1h"], index=3)
    
    days_back = st.sidebar.slider("Days of history", 1, 30, 7)
    end_date = datetime.utcnow().date()
    start_date = (datetime.utcnow() - timedelta(days=days_back)).date()
    
    # Strategy selection
    st.sidebar.header("🎯 Strategy Selection")
    strategies = st.sidebar.multiselect(
        "Active Strategies",
        ["Mean Reversion", "Pairs Trading", "Triangular Arb", "Market Making"],
        default=["Mean Reversion", "Pairs Trading"]
    )
    
    # Common parameters
    st.sidebar.header("⚙️ Parameters")
    
    params = {}
    
    if "Mean Reversion" in strategies:
        st.sidebar.subheader("Mean Reversion")
        params['meanrev_entry_z'] = st.sidebar.slider("Entry Z-score", 0.5, 4.0, 2.0)
        params['meanrev_exit_z'] = st.sidebar.slider("Exit Z-score", 0.0, 2.0, 0.5)
        params['meanrev_gamma'] = st.sidebar.number_input("CARA γ", 0.0, 10.0, 2.0)
        params['meanrev_risk_free'] = st.sidebar.number_input("Risk-free rate", 0.0, 0.1, 0.02)
    
    if "Pairs Trading" in strategies:
        st.sidebar.subheader("Pairs Trading")
        params['pairs_window'] = st.sidebar.slider("Rolling window", 20, 200, 50)
        params['pairs_entry_z'] = st.sidebar.slider("Pairs entry Z", 1.0, 3.0, 2.0)
    
    if "Triangular Arb" in strategies:
        st.sidebar.subheader("Triangular Arbitrage")
        params['tri_threshold'] = st.sidebar.number_input("Arb threshold", 0.0001, 0.01, 0.001, format="%.4f")
    
    if "Market Making" in strategies:
        st.sidebar.subheader("Market Making")
        params['mm_spread'] = st.sidebar.number_input("Spread (bps)", 1.0, 100.0, 20.0) / 10000
        params['mm_inv_aversion'] = st.sidebar.slider("Inventory aversion", 0.0, 1.0, 0.1)
    
    params['transaction_cost'] = st.sidebar.number_input("Transaction cost (bps)", 0.0, 50.0, 10.0) / 10000
    
    return {
        'market': market,
        'symbols': symbols,
        'interval': interval,
        'start': start_date.strftime('%Y-%m-%d'),
        'end': end_date.strftime('%Y-%m-%d'),
        'strategies': strategies,
        **params
    }


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_strategy_comparison(results: Dict[str, Dict]):
    """Plot all strategy equity curves"""
    fig = go.Figure()
    
    for strategy_name, strategy_data in results.items():
        if 'pnl' in strategy_data:
            pnl = strategy_data['pnl']
            if isinstance(pnl, pd.Series):
                pnl = pnl.values
            
            fig.add_trace(go.Scatter(
                y=pnl,
                name=strategy_name,
                mode='lines',
                line=dict(width=2)
            ))
    
    fig.update_layout(
        title="📈 Multi-Strategy PnL Comparison",
        xaxis_title="Time Period",
        yaxis_title="Cumulative PnL",
        height=500,
        hovermode='x unified',
        legend=dict(x=0.01, y=0.99)
    )
    
    return fig


def plot_weights_heatmap(weights_dict: Dict[str, np.ndarray], symbols: List[str]):
    """Plot portfolio weights as heatmap"""
    strategies = list(weights_dict.keys())
    weights_matrix = np.array([weights_dict[s] for s in strategies])
    
    fig = go.Figure(data=go.Heatmap(
        z=weights_matrix,
        x=symbols,
        y=strategies,
        colorscale='RdBu',
        zmid=0,
        text=weights_matrix,
        texttemplate='%{text:.3f}',
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title="Portfolio Weights by Strategy",
        xaxis_title="Symbol",
        yaxis_title="Strategy",
        height=400
    )
    
    return fig


def compute_metrics(pnl_series) -> Dict:
    """Compute performance metrics from PnL series"""
    if isinstance(pnl_series, pd.Series):
        pnl = pnl_series.values
    else:
        pnl = np.array(pnl_series)
    
    returns = np.diff(pnl, prepend=pnl[0])
    
    total_return = pnl[-1] if len(pnl) > 0 else 0.0
    
    if len(returns) > 1 and returns.std() > 0:
        sharpe = returns.mean() / returns.std() * np.sqrt(252)
    else:
        sharpe = 0.0
    
    # Max drawdown
    cummax = np.maximum.accumulate(pnl)
    drawdown = pnl - cummax
    max_dd = drawdown.min() if len(drawdown) > 0 else 0.0
    
    return {
        'Total PnL': total_return,
        'Sharpe': sharpe,
        'Max Drawdown': max_dd,
        'Volatility': returns.std() if len(returns) > 1 else 0.0
    }


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    st.markdown('<h1 class="main-header">🚀 Multi-Strategy Trading Lab</h1>', unsafe_allow_html=True)
    
    # Info banner
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"🔧 Rust Backend: {'✅ Active' if RUST_AVAILABLE else '⚠️ Python Fallback'}")
    with col2:
        st.info("📚 Strategies: 4+ implementations")
    with col3:
        st.info("📊 Data: Real-time & Historical")
    
    # Sidebar
    controls = sidebar_controls()
    
    if not controls['symbols']:
        st.warning("⚠️ Please select at least one symbol from the sidebar.")
        return
    
    # Fetch data button
    if st.sidebar.button("🚀 Run Backtest", type="primary"):
        with st.spinner("Fetching data and computing strategies..."):
            try:
                # Fetch prices
                prices = fetch_intraday_data(
                    symbols=controls['symbols'],
                    start=controls['start'],
                    end=controls['end'],
                    interval=controls['interval'],
                    source='synthetic'  # Change to 'finnhub' for real data
                )
                
                prices = get_close_prices(prices)
                prices = prices.fillna(method='ffill').fillna(method='bfill').dropna()
                
                st.success(f"✅ Data loaded: {prices.shape[0]} periods × {prices.shape[1]} symbols")
                
                # Store in session state
                st.session_state['prices'] = prices
                st.session_state['controls'] = controls
                
            except Exception as e:
                st.error(f"❌ Error loading data: {e}")
                return
    
    # Main content
    if 'prices' not in st.session_state:
        st.info("👈 Configure parameters and click 'Run Backtest' to start")
        
        # Show theory sections
        st.markdown("---")
        st.header("📚 Strategy Theory")
        
        tabs = st.tabs(["Mean Reversion", "Pairs Trading", "Triangular Arb", "Market Making"])
        
        with tabs[0]:
            st.markdown(MeanReversionStrategy.theory())
        with tabs[1]:
            st.markdown(PairsTradingStrategy.theory())
        with tabs[2]:
            st.markdown(TriangularArbStrategy.theory())
        with tabs[3]:
            st.markdown(MarketMakingStrategy.theory())
        
        return
    
    # Get data from session
    prices = st.session_state['prices']
    controls = st.session_state['controls']
    
    # Compute strategies
    st.header("📊 Strategy Results")
    
    all_results = {}
    all_metrics = {}
    weights_dict = {}
    
    # Mean Reversion
    if "Mean Reversion" in controls['strategies']:
        st.markdown('<div class="strategy-header">📉 Mean Reversion Strategies</div>', unsafe_allow_html=True)
        st.markdown(MeanReversionStrategy.theory())
        
        with st.spinner("Computing mean reversion..."):
            meanrev_params = {
                'entry_z': controls.get('meanrev_entry_z', 2.0),
                'exit_z': controls.get('meanrev_exit_z', 0.5),
                'gamma': controls.get('meanrev_gamma', 2.0),
                'risk_free': controls.get('meanrev_risk_free', 0.02),
                'transaction_cost': controls.get('transaction_cost', 0.001)
            }
            
            meanrev_results = MeanReversionStrategy.compute(prices, meanrev_params)
            
            for name, data in meanrev_results.items():
                key = f"MeanRev_{name}"
                all_results[key] = {'pnl': data['backtest']['pnl']}
                all_metrics[key] = compute_metrics(data['backtest']['pnl'])
                weights_dict[key] = data['weights']
            
            # Show mean reversion specific plots
            if meanrev_results:
                fig = go.Figure()
                for name, data in meanrev_results.items():
                    fig.add_trace(go.Scatter(
                        y=data['backtest']['pnl'],
                        name=name,
                        mode='lines'
                    ))
                fig.update_layout(title="Mean Reversion PnL", height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    # Pairs Trading
    if "Pairs Trading" in controls['strategies']:
        st.markdown('<div class="strategy-header">📊 Pairs Trading</div>', unsafe_allow_html=True)
        st.markdown(PairsTradingStrategy.theory())
        
        with st.spinner("Computing pairs trading..."):
            pairs_params = {
                'window': controls.get('pairs_window', 50),
                'entry_z': controls.get('pairs_entry_z', 2.0)
            }
            
            pairs_results = PairsTradingStrategy.compute(prices, pairs_params)
            
            if 'Pairs' in pairs_results:
                data = pairs_results['Pairs']
                all_results['Pairs'] = {'pnl': data['pnl']}
                all_metrics['Pairs'] = compute_metrics(data['pnl'])
                
                # Show pairs-specific plots
                fig = make_subplots(
                    rows=3, cols=1,
                    subplot_titles=("Spread", "Z-Score", "PnL"),
                    vertical_spacing=0.1
                )
                
                fig.add_trace(go.Scatter(y=data['spread'].values, name="Spread"), row=1, col=1)
                fig.add_trace(go.Scatter(y=data['zscore'].values, name="Z-Score"), row=2, col=1)
                fig.add_hline(y=2, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=-2, line_dash="dash", line_color="green", row=2, col=1)
                fig.add_trace(go.Scatter(y=data['pnl'].values, name="PnL"), row=3, col=1)
                
                fig.update_layout(height=800, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                st.info(f"📌 Pair: {data['symbols'][0]} vs {data['symbols'][1]} | Hedge Ratio β = {data['beta']:.4f}")
    
    # Triangular Arbitrage
    if "Triangular Arb" in controls['strategies']:
        st.markdown('<div class="strategy-header">🔺 Triangular Arbitrage</div>', unsafe_allow_html=True)
        st.markdown(TriangularArbStrategy.theory())
        
        with st.spinner("Detecting triangular arbitrage..."):
            tri_params = {
                'threshold': controls.get('tri_threshold', 0.001)
            }
            
            tri_results = TriangularArbStrategy.compute(prices, tri_params)
            
            if 'Triangular' in tri_results:
                data = tri_results['Triangular']
                all_results['Triangular'] = {'pnl': data['pnl']}
                all_metrics['Triangular'] = compute_metrics(data['pnl'])
                
                # Show triangular arb plots
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=("Arbitrage Signal", "Cumulative PnL")
                )
                
                fig.add_trace(go.Scatter(y=data['arb_signal'].values, name="Arb Signal"), row=1, col=1)
                fig.add_hline(y=tri_params['threshold'], line_dash="dash", line_color="red", row=1, col=1)
                fig.add_trace(go.Scatter(y=data['pnl'].values, name="PnL"), row=2, col=1)
                
                fig.update_layout(height=600, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                n_opps = data['opportunities'].sum()
                st.info(f"📌 Triangle: {data['symbols']} | Opportunities: {n_opps}")
    
    # Market Making
    if "Market Making" in controls['strategies']:
        st.markdown('<div class="strategy-header">💹 Market Making</div>', unsafe_allow_html=True)
        st.markdown(MarketMakingStrategy.theory())
        
        with st.spinner("Simulating market making..."):
            mm_params = {
                'spread': controls.get('mm_spread', 0.002),
                'inv_aversion': controls.get('mm_inv_aversion', 0.1)
            }
            
            mm_results = MarketMakingStrategy.compute(prices, mm_params)
            
            if 'MarketMaking' in mm_results:
                data = mm_results['MarketMaking']
                all_results['MarketMaking'] = {'pnl': data['pnl']}
                all_metrics['MarketMaking'] = compute_metrics(data['pnl'])
                
                # Show market making plots
                fig = make_subplots(
                    rows=3, cols=1,
                    subplot_titles=("Mid Price", "Inventory", "PnL"),
                    vertical_spacing=0.1
                )
                
                fig.add_trace(go.Scatter(y=data['mid'].values, name="Mid"), row=1, col=1)
                fig.add_trace(go.Scatter(y=data['inventory'].values, name="Inventory"), row=2, col=1)
                fig.add_hline(y=0, line_dash="dash", row=2, col=1)
                fig.add_trace(go.Scatter(y=data['pnl'].values, name="PnL"), row=3, col=1)
                
                fig.update_layout(height=800, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                st.info(f"📌 Symbol: {data['symbol']} | Final Inventory: {data['inventory'].iloc[-1]:.2f}")
    
    # Multi-strategy comparison
    if len(all_results) > 1:
        st.markdown("---")
        st.header("🎯 Multi-Strategy Comparison")
        
        fig = plot_strategy_comparison(all_results)
        st.plotly_chart(fig, use_container_width=True)
        
        # Metrics table
        metrics_df = pd.DataFrame(all_metrics).T
        metrics_df['Total PnL'] = metrics_df['Total PnL'].apply(lambda x: f"${x:,.2f}")
        metrics_df['Sharpe'] = metrics_df['Sharpe'].apply(lambda x: f"{x:.3f}")
        metrics_df['Max Drawdown'] = metrics_df['Max Drawdown'].apply(lambda x: f"${x:,.2f}")
        metrics_df['Volatility'] = metrics_df['Volatility'].apply(lambda x: f"{x:.4f}")
        
        st.subheader("📊 Performance Metrics")
        st.dataframe(metrics_df, use_container_width=True)
        
        # Weights heatmap (if available)
        if weights_dict:
            st.subheader("⚖️ Portfolio Weights")
            fig_weights = plot_weights_heatmap(weights_dict, prices.columns.tolist())
            st.plotly_chart(fig_weights, use_container_width=True)
    
    # Detailed Trade Analysis Section
    if len(all_results) > 0:
        st.markdown("---")
        st.header("🔍 Detailed Strategy Analysis")
        
        # Strategy selector
        selected_strategy = st.selectbox(
            "Select a strategy for detailed analysis:",
            list(all_results.keys())
        )
        
        if selected_strategy:
            st.markdown(f'<div class="strategy-header">📊 {selected_strategy} - Detailed View</div>', unsafe_allow_html=True)
            
            # Get strategy data
            strategy_data = all_results[selected_strategy]
            pnl_series = strategy_data['pnl']
            if isinstance(pnl_series, pd.Series):
                pnl_array = pnl_series.values
                pnl_index = pnl_series.index
            else:
                pnl_array = np.array(pnl_series)
                pnl_index = np.arange(len(pnl_array))
            
            # User-defined initial capital
            col1, col2 = st.columns(2)
            with col1:
                initial_capital = st.number_input(
                    "💰 Initial Capital ($)",
                    min_value=1000.0,
                    max_value=10000000.0,
                    value=100000.0,
                    step=10000.0
                )
            with col2:
                rebalance_freq = st.selectbox(
                    "🔄 Rebalancing Frequency",
                    ["No rebalancing", "Daily", "Weekly", "Monthly"],
                    index=0
                )
            
            # Calculate portfolio value over time
            returns = np.diff(pnl_array, prepend=pnl_array[0]) if len(pnl_array) > 0 else np.array([0])
            portfolio_value = initial_capital + pnl_array
            
            # Portfolio weights (if available)
            if selected_strategy in weights_dict:
                weights = weights_dict[selected_strategy]
                
                st.subheader("📊 Portfolio Composition")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Show top 10 weights
                    weights_df = pd.DataFrame({
                        'Symbol': prices.columns[:len(weights)],
                        'Weight': weights,
                        'Position': ['Long' if w > 0 else 'Short' if w < 0 else 'Neutral' for w in weights]
                    })
                    weights_df = weights_df.sort_values('Weight', key=abs, ascending=False).head(10)
                    st.dataframe(weights_df, use_container_width=True)
                
                with col2:
                    # Pie chart of absolute weights
                    top_weights = weights_df.copy()
                    top_weights['Abs Weight'] = top_weights['Weight'].abs()
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=top_weights['Symbol'],
                        values=top_weights['Abs Weight'],
                        hole=0.3
                    )])
                    fig_pie.update_layout(title="Top 10 Holdings (by absolute weight)", height=400)
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                # Portfolio statistics
                st.subheader("📈 Portfolio Statistics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    long_weight = weights[weights > 0].sum()
                    st.metric("Long Exposure", f"{long_weight:.2%}")
                
                with col2:
                    short_weight = abs(weights[weights < 0].sum())
                    st.metric("Short Exposure", f"{short_weight:.2%}")
                
                with col3:
                    net_exposure = long_weight - short_weight
                    st.metric("Net Exposure", f"{net_exposure:.2%}")
                
                with col4:
                    gross_exposure = long_weight + short_weight
                    st.metric("Gross Exposure", f"{gross_exposure:.2%}")
            
            # Performance Analysis
            st.subheader("💹 Performance Analysis")
            
            # Key metrics in columns
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                final_value = portfolio_value[-1] if len(portfolio_value) > 0 else initial_capital
                total_return = (final_value - initial_capital) / initial_capital
                st.metric("Total Return", f"{total_return:.2%}", f"${final_value - initial_capital:,.2f}")
            
            with col2:
                if len(returns) > 1 and returns.std() > 0:
                    sharpe = returns.mean() / returns.std() * np.sqrt(252)
                else:
                    sharpe = 0.0
                st.metric("Sharpe Ratio", f"{sharpe:.3f}")
            
            with col3:
                if len(portfolio_value) > 0:
                    cummax = np.maximum.accumulate(portfolio_value)
                    drawdown = (portfolio_value - cummax) / cummax
                    max_dd = drawdown.min()
                else:
                    max_dd = 0.0
                st.metric("Max Drawdown", f"{max_dd:.2%}", f"${max_dd * initial_capital:,.2f}")
            
            with col4:
                if len(returns) > 1:
                    volatility = returns.std() * np.sqrt(252)
                else:
                    volatility = 0.0
                st.metric("Volatility (Ann.)", f"{volatility:.2%}")
            
            with col5:
                if len(portfolio_value) > 1:
                    days = len(portfolio_value)
                    cagr = (final_value / initial_capital) ** (252 / days) - 1
                else:
                    cagr = 0.0
                st.metric("CAGR", f"{cagr:.2%}")
            
            # Portfolio value evolution
            st.subheader("📊 Portfolio Value Evolution")
            
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=("Portfolio Value", "Daily Returns", "Drawdown"),
                vertical_spacing=0.08,
                row_heights=[0.5, 0.25, 0.25]
            )
            
            # Portfolio value
            fig.add_trace(go.Scatter(
                x=pnl_index,
                y=portfolio_value,
                name="Portfolio Value",
                line=dict(color='blue', width=2),
                fill='tonexty'
            ), row=1, col=1)
            
            fig.add_hline(y=initial_capital, line_dash="dash", line_color="gray",
                         annotation_text="Initial Capital", row=1, col=1)
            
            # Daily returns
            fig.add_trace(go.Bar(
                x=pnl_index[1:] if len(pnl_index) > 1 else pnl_index,
                y=returns[1:] if len(returns) > 1 else returns,
                name="Daily Returns",
                marker_color=['green' if r > 0 else 'red' for r in (returns[1:] if len(returns) > 1 else returns)]
            ), row=2, col=1)
            
            # Drawdown
            if len(portfolio_value) > 0:
                cummax = np.maximum.accumulate(portfolio_value)
                drawdown_pct = (portfolio_value - cummax) / cummax * 100
                fig.add_trace(go.Scatter(
                    x=pnl_index,
                    y=drawdown_pct,
                    name="Drawdown",
                    fill='tozeroy',
                    line=dict(color='red', width=1)
                ), row=3, col=1)
            
            fig.update_xaxes(title_text="Time", row=3, col=1)
            fig.update_yaxes(title_text="Value ($)", row=1, col=1)
            fig.update_yaxes(title_text="Return ($)", row=2, col=1)
            fig.update_yaxes(title_text="Drawdown (%)", row=3, col=1)
            
            fig.update_layout(height=800, showlegend=False, hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)
            
            # Rolling statistics
            st.subheader("📊 Rolling Statistics (50-period window)")
            
            if len(returns) > 50:
                window = 50
                rolling_mean = pd.Series(returns).rolling(window).mean()
                rolling_std = pd.Series(returns).rolling(window).std()
                rolling_sharpe = rolling_mean / rolling_std * np.sqrt(252)
                
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=("Rolling Mean Return", "Rolling Sharpe Ratio"),
                    vertical_spacing=0.1
                )
                
                fig.add_trace(go.Scatter(
                    x=pnl_index,
                    y=rolling_mean,
                    name="Rolling Mean",
                    line=dict(color='blue')
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(
                    x=pnl_index,
                    y=rolling_sharpe,
                    name="Rolling Sharpe",
                    line=dict(color='green')
                ), row=2, col=1)
                
                fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
                fig.add_hline(y=1, line_dash="dash", line_color="orange",
                             annotation_text="Sharpe=1.0", row=2, col=1)
                
                fig.update_xaxes(title_text="Time")
                fig.update_yaxes(title_text="Mean Return", row=1, col=1)
                fig.update_yaxes(title_text="Sharpe Ratio", row=2, col=1)
                
                fig.update_layout(height=500, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            # Trade statistics (if positions available)
            if selected_strategy in ['PC1', 'CARA', 'Sharpe']:
                st.subheader("📊 Rebalancing Analysis")
                
                # Simulate rebalancing costs
                rebal_cost_bps = st.slider("Rebalancing cost (bps)", 1, 50, 10)
                rebal_cost = rebal_cost_bps / 10000
                
                if rebalance_freq == "Daily":
                    rebal_periods = list(range(0, len(pnl_array), 1))
                elif rebalance_freq == "Weekly":
                    rebal_periods = list(range(0, len(pnl_array), 5))
                elif rebalance_freq == "Monthly":
                    rebal_periods = list(range(0, len(pnl_array), 20))
                else:
                    rebal_periods = [0]  # No rebalancing
                
                n_rebalances = len(rebal_periods) - 1
                total_rebal_cost = n_rebalances * rebal_cost * portfolio_value[0] if n_rebalances > 0 else 0
                
                adjusted_final = final_value - total_rebal_cost
                adjusted_return = (adjusted_final - initial_capital) / initial_capital
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rebalancing Events", f"{n_rebalances}")
                with col2:
                    st.metric("Total Rebalancing Cost", f"${total_rebal_cost:,.2f}")
                with col3:
                    st.metric("Net Return (after costs)", f"{adjusted_return:.2%}",
                             delta=f"{(adjusted_return - total_return):.2%}")
                
                # Show rebalancing schedule
                if n_rebalances > 0 and n_rebalances < 20:
                    st.write("**Rebalancing Schedule:**")
                    rebal_df = pd.DataFrame({
                        'Period': rebal_periods,
                        'Portfolio Value': [portfolio_value[min(i, len(portfolio_value)-1)] for i in rebal_periods],
                        'Cost': [rebal_cost * portfolio_value[min(i, len(portfolio_value)-1)] for i in rebal_periods]
                    })
                    st.dataframe(rebal_df, use_container_width=True)
            
            # Risk analysis
            st.subheader("⚠️ Risk Analysis")
            
            if len(returns) > 1:
                # VaR and CVaR
                var_95 = np.percentile(returns, 5)
                cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("VaR (95%)", f"${var_95:,.2f}")
                
                with col2:
                    st.metric("CVaR (95%)", f"${cvar_95:,.2f}")
                
                with col3:
                    positive_returns = returns[returns > 0]
                    win_rate = len(positive_returns) / len(returns) * 100 if len(returns) > 0 else 0
                    st.metric("Win Rate", f"{win_rate:.1f}%")
                
                with col4:
                    if len(positive_returns) > 0 and len(returns[returns < 0]) > 0:
                        profit_factor = positive_returns.sum() / abs(returns[returns < 0].sum())
                    else:
                        profit_factor = 0.0
                    st.metric("Profit Factor", f"{profit_factor:.2f}")
                
                # Return distribution
                st.write("**Return Distribution:**")
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=returns,
                    nbinsx=50,
                    name="Returns",
                    marker_color='lightblue'
                ))
                fig.add_vline(x=returns.mean(), line_dash="dash", line_color="green",
                             annotation_text=f"Mean: ${returns.mean():.2f}")
                fig.add_vline(x=var_95, line_dash="dash", line_color="red",
                             annotation_text=f"VaR 95%: ${var_95:.2f}")
                
                fig.update_layout(
                    title="Daily Return Distribution",
                    xaxis_title="Return ($)",
                    yaxis_title="Frequency",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p><strong>Rust HFT Arbitrage Lab</strong> | Multi-Strategy Trading System</p>
        <p>Powered by Rust 🦀 + Python 🐍 + Streamlit ⚡</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
