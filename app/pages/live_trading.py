"""
Live Trading Module
===================

Real-time strategy execution with WebSocket data feeds:
- Live market data streaming
- Real-time strategy execution
- Trade logging and monitoring
- Risk management
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time
import threading
import queue
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from python.rust_bridge import list_connectors, get_connector
from python.strategies.definitions import AVAILABLE_STRATEGIES
from python.strategies.executor import StrategyExecutor, StrategyConfig

def render():
    """Render the live trading page"""
    st.title("🔴 Live Trading")
    st.markdown("Execute strategies on real-time market data via WebSocket")
    
    # Safety warning
    if not st.session_state.get('live_trading_acknowledged', False):
        st.warning("⚠️ **Important:** This is for paper trading / simulation only. No real trades will be executed.")
        if st.checkbox("I understand this is simulated trading"):
            st.session_state.live_trading_acknowledged = True
            st.rerun()
        return
    
    # Main layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        configure_live_trading()
    
    with col2:
        display_live_feed()
    
    # Bottom section - trade log and analytics
    st.markdown("---")
    display_live_analytics()

def configure_live_trading():
    """Configure live trading parameters"""
    
    st.markdown("### Configuration")
    
    # Connector selection
    connectors = list_connectors()
    default_idx = connectors.index("finnhub") if "finnhub" in connectors else 0
    connector_name = st.selectbox(
        "Data Source",
        connectors,
        index=default_idx,
        help="Select market data connector"
    )
    
    st.info("📝 Credentials loaded from `api_keys.properties`")
    
    # Connection mode
    connection_mode = st.radio(
        "Connection Mode",
        ["Polling (REST)", "Streaming (WebSocket)"],
        help="Polling: Fetch periodically. WebSocket: Real-time stream"
    )
    
    use_websocket = connection_mode == "Streaming (WebSocket)"
    
    if connection_mode == "Polling (REST)":
        update_interval = st.slider(
            "Update Interval (ms)",
            200, 5000, 500, step=100
        )
    else:
        update_interval = 200  # UI update frequency
        st.info("🔴 WebSocket provides real-time streaming updates")
    
    # Symbol selection
    try:
        connector = get_connector(connector_name)
        symbols = connector.list_symbols() if hasattr(connector, "list_symbols") else []
        
        if len(symbols) > 0:
            selected_symbols = st.multiselect(
                "Symbols",
                symbols,
                default=symbols[:min(3, len(symbols))],
                help="Select symbols to monitor"
            )
        else:
            selected_symbols = st.text_input(
                "Symbols (comma-separated)",
                value="BTC/USD,ETH/USD"
            ).split(',')
            selected_symbols = [s.strip() for s in selected_symbols if s.strip()]
    except Exception as e:
        st.error(f"Failed to get connector: {e}")
        return
    
    st.markdown("---")
    
    # Strategy selection for live execution
    st.markdown("### Strategy")
    
    enable_strategy = st.checkbox(
        "Enable Strategy Execution",
        value=False,
        help="Execute strategy on live data"
    )
    
    if enable_strategy:
        strategy_names = list(AVAILABLE_STRATEGIES.keys())
        strategy_names.extend(["Mean Reversion (PCA)", "Pairs Trading"])
        
        selected_strategy = st.selectbox(
            "Strategy",
            strategy_names
        )
        
        # Quick parameters
        with st.expander("Strategy Parameters"):
            if "Mean Reversion" in selected_strategy:
                entry_z = st.number_input("Entry Z-Score", value=2.0)
                exit_z = st.number_input("Exit Z-Score", value=0.5)
                lookback = st.number_input("Lookback Period", value=60, step=10)
                strategy_params = {'entry_z': entry_z, 'exit_z': exit_z, 'lookback': lookback}
            elif "Pairs Trading" in selected_strategy:
                entry_z = st.number_input("Entry Z-Score", value=2.0)
                exit_z = st.number_input("Exit Z-Score", value=0.5)
                strategy_params = {'entry_z': entry_z, 'exit_z': exit_z}
            else:
                strategy_params = {}
        
        st.session_state.live_strategy = selected_strategy
        st.session_state.live_strategy_params = strategy_params
    else:
        st.session_state.live_strategy = None
    
    st.markdown("---")
    
    # Control buttons
    is_active = st.session_state.get('live_trading_active', False)
    
    if not is_active:
        if st.button("▶️ Start Live Feed", type="primary", use_container_width=True):
            start_live_trading(
                connector_name=connector_name,
                symbols=selected_symbols,
                use_websocket=use_websocket,
                update_interval=update_interval
            )
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("⏸️ Pause", use_container_width=True):
                st.session_state.live_trading_active = False
                st.rerun()
        with col_b:
            if st.button("⏹️ Stop & Reset", use_container_width=True):
                stop_live_trading()
        
        # Show status
        buffer_size = len(st.session_state.live_data_buffer)
        use_ws = st.session_state.get('live_use_websocket', False)
        mode_text = "WebSocket" if use_ws else "REST Polling"
        st.success(f"🔴 LIVE ({mode_text}) - {buffer_size} updates")
        
        # Show connection info for WebSocket
        if use_ws:
            ws_status = st.session_state.get('live_ws_status', {})
            active_connections = sum(1 for s in ws_status.values() if s.get('connected'))
            total_updates = sum(s.get('update_count', 0) for s in ws_status.values())
            st.caption(f"🔌 {active_connections}/{len(ws_status)} connections active • {total_updates} total updates")
        
        # Auto-refresh
        time.sleep(update_interval / 1000)
        st.rerun()

def start_live_trading(connector_name: str, symbols: List[str], use_websocket: bool, update_interval: int):
    """Start live trading session"""
    
    # Initialize session state
    st.session_state.live_trading_active = True
    st.session_state.live_data_buffer = []
    st.session_state.live_connector_name = connector_name
    st.session_state.live_symbols = symbols
    st.session_state.live_use_websocket = use_websocket
    st.session_state.live_start_time = datetime.now()
    st.session_state.live_ws_status = {}  # Track WebSocket status per symbol
    
    # Initialize trade log if needed
    if 'trade_log' not in st.session_state:
        st.session_state.trade_log = []
    
    # Initialize strategy executor if enabled
    if st.session_state.get('live_strategy'):
        config = StrategyConfig(
            strategy_name=st.session_state.live_strategy,
            parameters=st.session_state.get('live_strategy_params', {}),
            mode='live',
            initial_capital=st.session_state.portfolio['cash']
        )
        st.session_state.live_executor = StrategyExecutor(config)
    
    # Start WebSocket connections if enabled
    if use_websocket:
        start_websocket_streams(connector_name, symbols)
    
    st.rerun()

def start_websocket_streams(connector_name: str, symbols: List[str]):
    """Start WebSocket streams for specified symbols"""
    
    try:
        connector = get_connector(connector_name)
        
        # Check if connector supports streaming
        if not hasattr(connector, 'start_stream'):
            st.error(f"{connector_name} connector does not support WebSocket streaming")
            st.session_state.live_use_websocket = False
            return
        
        # Create a thread-safe queue for WebSocket data
        if 'ws_data_queue' not in st.session_state:
            st.session_state.ws_data_queue = queue.Queue()
        
        # Get the queue reference (DO NOT access session_state inside callbacks)
        data_queue = st.session_state.ws_data_queue
        
        # Define callback for WebSocket updates
        def create_callback(symbol):
            def callback(orderbook):
                try:
                    # Extract top of book
                    bid, ask = extract_top_of_book(orderbook)
                    
                    if bid > 0 and ask > 0:
                        data_point = {
                            'timestamp': datetime.now(),
                            'symbol': symbol,
                            'bid': bid,
                            'ask': ask,
                            'mid': (bid + ask) / 2
                        }
                        
                        # Put data in thread-safe queue (using captured reference, NOT session_state)
                        data_queue.put(data_point)
                        
                except Exception as e:
                    # Queue error status update
                    error_status = {
                        'type': 'error',
                        'symbol': symbol,
                        'error': str(e),
                        'timestamp': datetime.now()
                    }
                    data_queue.put(error_status)
            return callback
        
        # Start streams for each symbol
        for symbol in symbols:
            try:
                connector.start_stream(symbol, create_callback(symbol))
                st.session_state.live_ws_status[symbol] = {
                    'connected': True,
                    'last_update': datetime.now(),
                    'update_count': 0
                }
            except Exception as e:
                st.session_state.live_ws_status[symbol] = {
                    'connected': False,
                    'error': str(e),
                    'last_update': datetime.now()
                }
        
        # Store connector reference
        st.session_state.live_connector = connector
        
    except Exception as e:
        st.error(f"Failed to start WebSocket streams: {e}")
        st.session_state.live_use_websocket = False

def stop_live_trading():
    """Stop live trading and reset"""
    
    # Stop WebSocket streams if active
    if st.session_state.get('live_connector') and hasattr(st.session_state.live_connector, 'stop_stream'):
        try:
            # Attempt to stop all active streams
            for symbol in st.session_state.get('live_ws_status', {}).keys():
                try:
                    st.session_state.live_connector.stop_stream(symbol)
                except Exception as e:
                    pass  # Continue stopping other streams
        except Exception:
            pass  # Connector cleanup failed, continue anyway
    
    st.session_state.live_trading_active = False
    st.session_state.live_data_buffer = []
    st.session_state.live_executor = None
    st.session_state.live_ws_status = {}
    st.session_state.live_connector = None
    
    # Clear the WebSocket queue
    if 'ws_data_queue' in st.session_state:
        try:
            while not st.session_state.ws_data_queue.empty():
                st.session_state.ws_data_queue.get_nowait()
        except Exception:
            pass
        st.session_state.ws_data_queue = None
    
    st.rerun()

def display_websocket_status():
    """Display WebSocket connection status"""
    
    ws_status = st.session_state.get('live_ws_status', {})
    ws_queue = st.session_state.get('ws_data_queue')
    
    if not ws_status:
        st.info("🔌 Initializing WebSocket connections...")
        if ws_queue:
            queue_size = ws_queue.qsize() if hasattr(ws_queue, 'qsize') else 0
            if queue_size > 0:
                st.caption(f"📥 {queue_size} updates queued")
        return
    
    st.markdown("#### 🔴 WebSocket Status")
    
    # Show queue status
    if ws_queue:
        queue_size = ws_queue.qsize() if hasattr(ws_queue, 'qsize') else 0
        if queue_size > 0:
            st.info(f"📥 Processing {queue_size} queued updates...")
    
    cols = st.columns(len(ws_status))
    
    for idx, (symbol, status) in enumerate(ws_status.items()):
        with cols[idx]:
            if status.get('connected'):
                update_count = status.get('update_count', 0)
                last_update = status.get('last_update')
                
                if last_update:
                    time_since = (datetime.now() - last_update).total_seconds()
                    if time_since < 5:
                        st.success(f"✅ {symbol}")
                        st.caption(f"{update_count} updates")
                    else:
                        st.warning(f"⚠️ {symbol}")
                        st.caption(f"Stale ({int(time_since)}s)")
                else:
                    st.info(f"🔌 {symbol}")
                    st.caption("Connecting...")
            else:
                st.error(f"❌ {symbol}")
                error = status.get('error', 'Connection failed')
                st.caption(f"{error[:30]}...")
    
    st.markdown("---")

def display_live_feed():
    """Display live market data feed"""
    
    st.markdown("### Live Market Feed")
    
    if not st.session_state.get('live_trading_active', False):
        st.info("Click 'Start Live Feed' to begin receiving market data")
        
        # Show example visualization
        st.markdown("#### Preview")
        st.caption("Live data will appear here once started")
        
        # Placeholder chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0], y=[0], mode='lines', name='Bid'))
        fig.add_trace(go.Scatter(x=[0], y=[0], mode='lines', name='Ask'))
        fig.update_layout(
            template="plotly_dark",
            height=400,
            xaxis_title="Time",
            yaxis_title="Price"
        )
        st.plotly_chart(fig, use_container_width=True)
        return
    
    # Show WebSocket connection status if using WebSocket
    use_websocket = st.session_state.get('live_use_websocket', False)
    if use_websocket:
        display_websocket_status()
    
    # Fetch live data
    fetch_live_data()
    
    # Display current data
    buffer = st.session_state.live_data_buffer
    
    if len(buffer) == 0:
        st.info("⏳ Waiting for data...")
        if use_websocket:
            st.caption("WebSocket is connecting... First update should arrive within a few seconds.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(buffer)
    
    # Show latest quotes
    st.markdown("#### Current Quotes")
    
    latest = df.groupby('symbol').tail(1)
    
    for _, row in latest.iterrows():
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Symbol", row['symbol'])
        with col2:
            st.metric("Bid", f"${row['bid']:.2f}")
        with col3:
            st.metric("Ask", f"${row['ask']:.2f}")
        with col4:
            spread = row['ask'] - row['bid']
            spread_bps = (spread / row['bid']) * 10000 if row['bid'] > 0 else 0
            st.metric("Spread", f"{spread_bps:.1f} bps")
    
    st.markdown("---")
    
    # Chart for each symbol
    for symbol in df['symbol'].unique():
        symbol_df = df[df['symbol'] == symbol].tail(100)
        
        with st.expander(f"📈 {symbol} Chart", expanded=True):
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=symbol_df['timestamp'],
                y=symbol_df['bid'],
                mode='lines',
                name='Bid',
                line={'color': 'green', 'width': 2}
            ))
            
            fig.add_trace(go.Scatter(
                x=symbol_df['timestamp'],
                y=symbol_df['ask'],
                mode='lines',
                name='Ask',
                line={'color': 'red', 'width': 2}
            ))
            
            # Add mid price
            mid = (symbol_df['bid'] + symbol_df['ask']) / 2
            fig.add_trace(go.Scatter(
                x=symbol_df['timestamp'],
                y=mid,
                mode='lines',
                name='Mid',
                line={'color': 'cyan', 'width': 1, 'dash': 'dash'}
            ))
            
            fig.update_layout(
                template="plotly_dark",
                height=300,
                showlegend=True,
                hovermode='x unified',
                xaxis_title="Time",
                yaxis_title="Price ($)"
            )
            
            st.plotly_chart(fig, use_container_width=True)

def fetch_live_data():
    """Fetch live market data - handles both REST polling and WebSocket modes"""
    
    use_websocket = st.session_state.get('live_use_websocket', False)
    
    # In WebSocket mode, data is pushed via callbacks, just check snapshots
    if use_websocket:
        fetch_websocket_snapshots()
    else:
        fetch_rest_polling()

def fetch_websocket_snapshots():
    """Process WebSocket data from the thread-safe queue"""
    
    ws_queue = st.session_state.get('ws_data_queue')
    
    if not ws_queue:
        return
    
    # Process all pending data from queue
    items_processed = 0
    max_items = 100  # Limit processing per update to avoid blocking
    
    try:
        while items_processed < max_items:
            try:
                # Non-blocking get from queue
                data = ws_queue.get_nowait()
                
                if data.get('type') == 'error':
                    # Handle error status
                    symbol = data.get('symbol', 'unknown')
                    error = data.get('error', 'Unknown error')
                    st.session_state.live_ws_status[symbol] = {
                        'connected': False,
                        'error': error,
                        'last_update': data.get('timestamp', datetime.now())
                    }
                else:
                    # Handle data point
                    st.session_state.live_data_buffer.append(data)
                    
                    # Keep buffer size manageable
                    if len(st.session_state.live_data_buffer) > 1000:
                        st.session_state.live_data_buffer = st.session_state.live_data_buffer[-1000:]
                    
                    # Update WebSocket status
                    symbol = data.get('symbol', 'unknown')
                    current_status = st.session_state.live_ws_status.get(symbol, {})
                    st.session_state.live_ws_status[symbol] = {
                        'connected': True,
                        'last_update': data.get('timestamp', datetime.now()),
                        'update_count': current_status.get('update_count', 0) + 1
                    }
                    
                    # Execute strategy if enabled
                    if st.session_state.get('live_executor'):
                        execute_strategy_on_tick(data)
                
                items_processed += 1
                
            except queue.Empty:
                # No more data in queue
                break
                
    except Exception as e:
        st.warning(f"Error processing WebSocket data: {e}")

def fetch_rest_polling():
    """Fetch data via REST API polling"""
    
    connector_name = st.session_state.get('live_connector_name')
    symbols = st.session_state.get('live_symbols', [])
    
    try:
        connector = get_connector(connector_name)
        
        for symbol in symbols:
            # Fetch orderbook
            if hasattr(connector, 'fetch_orderbook_sync'):
                ob = connector.fetch_orderbook_sync(symbol)
            elif hasattr(connector, 'fetch_orderbook'):
                ob = connector.fetch_orderbook(symbol)
            else:
                continue
            
            # Extract top of book
            bid, ask = extract_top_of_book(ob)
            
            if bid > 0 and ask > 0:
                data_point = {
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'bid': bid,
                    'ask': ask,
                    'mid': (bid + ask) / 2
                }
                
                st.session_state.live_data_buffer.append(data_point)
                
                # Keep buffer size manageable
                if len(st.session_state.live_data_buffer) > 1000:
                    st.session_state.live_data_buffer = st.session_state.live_data_buffer[-1000:]
                
                # Execute strategy if enabled
                if st.session_state.get('live_executor'):
                    execute_strategy_on_tick(data_point)
                
    except Exception as e:
        st.error(f"Error fetching live data: {e}")

def extract_top_of_book(ob) -> tuple:
    """Extract bid/ask from orderbook"""
    try:
        if isinstance(ob, dict):
            if ob.get("bids") and ob.get("asks"):
                bid = float(ob["bids"][0][0])
                ask = float(ob["asks"][0][0])
                return bid, ask
        else:
            # Rust OrderBook object
            if hasattr(ob, 'bids') and hasattr(ob, 'asks'):
                if ob.bids and ob.asks:
                    bid = float(ob.bids[0][0])
                    ask = float(ob.asks[0][0])
                    return bid, ask
    except Exception:
        pass
    
    return 0.0, 0.0

def execute_strategy_on_tick(data_point: Dict):
    """Execute strategy logic on new market data"""
    
    executor = st.session_state.get('live_executor')
    if not executor:
        return
    
    # Simple strategy execution placeholder
    # In production, this would implement full strategy logic
    
    symbol = data_point['symbol']
    mid_price = data_point['mid']
    
    # Example: Simple mean reversion check
    buffer = st.session_state.live_data_buffer
    symbol_data = [d for d in buffer if d['symbol'] == symbol]
    
    if len(symbol_data) > 60:  # Need enough data
        prices = [d['mid'] for d in symbol_data[-60:]]
        mean = np.mean(prices)
        std = np.std(prices)
        
        if std > 0:
            z_score = (mid_price - mean) / std
            
            # Simple signal logic
            if abs(z_score) > 2.0 and symbol not in executor.positions:
                # Generate signal
                side = 'sell' if z_score > 0 else 'buy'
                qty = 1.0
                
                # Log trade (not executed, just logged)
                trade = {
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'side': side,
                    'qty': qty,
                    'price': mid_price,
                    'z_score': z_score,
                    'status': 'signal'
                }
                
                st.session_state.trade_log.append(trade)

def display_live_analytics():
    """Display live trading analytics"""
    
    st.markdown("### Live Analytics")
    
    tab1, tab2, tab3 = st.tabs(["📊 Statistics", "📝 Trade Log", "⚡ Signals"])
    
    with tab1:
        display_live_statistics()
    
    with tab2:
        display_trade_log()
    
    with tab3:
        display_live_signals()

def display_live_statistics():
    """Display live statistics"""
    
    buffer = st.session_state.get('live_data_buffer', [])
    
    if len(buffer) == 0:
        st.info("No data yet")
        return
    
    df = pd.DataFrame(buffer)
    
    # Per-symbol statistics
    for symbol in df['symbol'].unique():
        symbol_df = df[df['symbol'] == symbol]
        
        with st.expander(f"📊 {symbol} Statistics", expanded=False):
            cols = st.columns(4)
            
            with cols[0]:
                avg_bid = symbol_df['bid'].mean()
                st.metric("Avg Bid", f"${avg_bid:.2f}")
            
            with cols[1]:
                avg_ask = symbol_df['ask'].mean()
                st.metric("Avg Ask", f"${avg_ask:.2f}")
            
            with cols[2]:
                avg_spread = (symbol_df['ask'] - symbol_df['bid']).mean()
                st.metric("Avg Spread", f"${avg_spread:.4f}")
            
            with cols[3]:
                volatility = symbol_df['mid'].std()
                st.metric("Volatility", f"${volatility:.4f}")
            
            # Price range
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Min Mid", f"${symbol_df['mid'].min():.2f}")
            with col2:
                st.metric("Max Mid", f"${symbol_df['mid'].max():.2f}")

def display_trade_log():
    """Display trade log"""
    
    trade_log = st.session_state.get('trade_log', [])
    
    if len(trade_log) == 0:
        st.info("No trades/signals yet")
    else:
        df = pd.DataFrame(trade_log)
        st.dataframe(df.tail(50), use_container_width=True, height=400)
        
        # Download trade log
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Trade Log",
            data=csv,
            file_name=f"trade_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

def display_live_signals():
    """Display live trading signals using Chiarella Model"""
    
    buffer = st.session_state.get('live_data_buffer', [])
    
    if len(buffer) == 0:
        st.info("⏳ Waiting for market data to generate signals...")
        return
    
    # Lazy import to avoid startup issues
    try:
        from python.strategies.chiarella_signals import ChiarellaSignalGenerator, estimate_fundamental_price
    except ImportError:
        st.error("Chiarella signal generator not found. Please ensure python/strategies/chiarella_signals.py exists.")
        return
    
    df = pd.DataFrame(buffer)
    
    # Algorithm explanation
    with st.expander("📚 About the Chiarella Model", expanded=False):
        st.markdown("""
        ### Mode-Switching Chiarella Model
        
        **Based on:** Kurth & Bouchaud (2025), arXiv:2511.13277
        
        The model describes markets as competition between:
        - **Fundamentalists**: Traders who believe prices revert to fair value
        - **Chartists**: Momentum/trend-following traders
        
        #### Core Equations
        
        $$
        \\frac{dp}{dt} = \\alpha \\cdot \\text{trend}(t) - \\beta \\cdot \\text{mispricing}(t) + \\text{noise}
        $$
        
        $$
        \\frac{d\\text{trend}}{dt} = \\gamma \\cdot \\Delta p(t) - \\delta \\cdot \\text{trend}(t) + \\text{noise}
        $$
        
        Where:
        - $\\alpha$: Chartist strength (trend feedback)
        - $\\beta$: Fundamentalist strength (mean reversion)
        - $\\gamma$: Trend formation speed
        - $\\delta$: Trend decay rate
        
        #### Regime Detection
        
        Bifurcation parameter: $\\Lambda = \\frac{\\alpha \\cdot \\gamma}{\\beta \\cdot \\delta}$
        
        - $\\Lambda < 0.67$: **Mean-Reverting** (fundamentalists dominate)
        - $0.67 \\leq \\Lambda \\leq 1.5$: **Mixed** regime
        - $\\Lambda > 1.5$: **Trending** (chartists dominate, bubbles possible)
        """)
    
    # Generate signals for each symbol
    for symbol in df['symbol'].unique():
        symbol_df = df[df['symbol'] == symbol].copy()
        
        if len(symbol_df) < 2:
            continue
        
        st.markdown(f"### 🎯 {symbol} Trading Signals")
        
        # Initialize or get Chiarella model for this symbol
        model_key = f'chiarella_model_{symbol}'
        
        prices = symbol_df['mid'].values
        
        # Estimate fundamental price
        fundamental_price = estimate_fundamental_price(prices.tolist(), method='ema')
        
        # Initialize model if needed
        if model_key not in st.session_state:
            st.session_state[model_key] = ChiarellaSignalGenerator(
                fundamental_price=fundamental_price,
                alpha=0.3,  # Moderate chartist influence
                beta=0.5,   # Stronger fundamentalist influence
                gamma=0.4,  # Moderate trend formation
                delta=0.2   # Slow trend decay
            )
        
        model = st.session_state[model_key]
        
        # Update model with latest prices
        for price in prices[-20:]:  # Update with recent prices
            model.update(price)
        
        # Update fundamental estimate
        model.update_fundamental(fundamental_price)
        
        # Generate signal
        signal = model.generate_signal()
        
        # Display signal dashboard
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            signal_strength = signal['strength']
            signal_color = 'green' if signal_strength > 0 else 'red' if signal_strength < 0 else 'gray'
            st.metric(
                "Signal Strength",
                f"{signal_strength:.3f}",
                delta=f"{'BUY' if signal_strength > 0.3 else 'SELL' if signal_strength < -0.3 else 'NEUTRAL'}",
                delta_color="normal" if abs(signal_strength) > 0.3 else "off"
            )
        
        with col2:
            regime_emoji = {
                'mean_reverting': '↩️',
                'trending': '📈',
                'mixed': '⚖️'
            }
            regime_name = signal['regime'].replace('_', ' ').title()
            st.metric(
                "Market Regime",
                f"{regime_emoji.get(signal['regime'], '❓')} {regime_name}",
                delta=f"Λ = {signal['bifurcation_parameter']:.2f}"
            )
        
        with col3:
            st.metric(
                "Position Size",
                f"{signal['position_size']:.1%}",
                delta=f"Confidence: {signal['confidence']:.1%}"
            )
        
        with col4:
            st.metric(
                "Mispricing",
                f"{signal['mispricing_pct']:+.2f}%",
                delta=f"Risk: {signal['risk']:.2%}"
            )
        
        # Detailed breakdown
        with st.expander("📊 Detailed Signal Analysis", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Signal Components")
                
                # Component breakdown
                fig_components = go.Figure()
                
                components = ['Fundamental', 'Chartist', 'Combined']
                values = [
                    signal['signal_fundamental'],
                    signal['signal_chartist'],
                    signal['strength']
                ]
                colors = ['blue', 'orange', 'green' if signal['strength'] > 0 else 'red']
                
                fig_components.add_trace(go.Bar(
                    x=components,
                    y=values,
                    marker_color=colors,
                    text=[f"{v:.3f}" for v in values],
                    textposition='outside'
                ))
                
                fig_components.update_layout(
                    title="Signal Decomposition",
                    yaxis_title="Signal Strength",
                    template="plotly_dark",
                    height=300,
                    showlegend=False
                )
                
                st.plotly_chart(fig_components, use_container_width=True)
                
                # Regime weights
                st.markdown("**Regime Weights:**")
                st.progress(signal['weight_fundamental'], text=f"Fundamentalist: {signal['weight_fundamental']:.0%}")
                st.progress(signal['weight_chartist'], text=f"Chartist: {signal['weight_chartist']:.0%}")
            
            with col2:
                st.markdown("#### Price & Trend Analysis")
                
                # Price vs fundamental
                current_price = prices[-1]
                
                fig_price = go.Figure()
                
                # Add price line
                fig_price.add_trace(go.Scatter(
                    y=prices[-50:],
                    mode='lines',
                    name='Market Price',
                    line=dict(color='white', width=2)
                ))
                
                # Add fundamental line
                fig_price.add_hline(
                    y=fundamental_price,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Fundamental",
                    annotation_position="right"
                )
                
                # Shade mispricing
                if current_price > fundamental_price:
                    fig_price.add_hrect(
                        y0=fundamental_price,
                        y1=current_price,
                        fillcolor="green",
                        opacity=0.2,
                        annotation_text="Overvalued",
                        annotation_position="top right"
                    )
                else:
                    fig_price.add_hrect(
                        y0=current_price,
                        y1=fundamental_price,
                        fillcolor="red",
                        opacity=0.2,
                        annotation_text="Undervalued",
                        annotation_position="bottom right"
                    )
                
                fig_price.update_layout(
                    title=f"Price vs Fundamental (Trend: {signal['trend']:.4f})",
                    yaxis_title="Price ($)",
                    xaxis_title="Time",
                    template="plotly_dark",
                    height=300
                )
                
                st.plotly_chart(fig_price, use_container_width=True)
        
        # Trading recommendation
        st.markdown("#### 💡 Trading Recommendation")
        
        if abs(signal['strength']) > 0.5:
            action = "STRONG BUY" if signal['strength'] > 0.5 else "STRONG SELL"
            color = "green" if signal['strength'] > 0.5 else "red"
            
            st.markdown(f"""
            <div style="padding: 20px; border-radius: 10px; background-color: {'rgba(0,255,0,0.1)' if signal['strength'] > 0.5 else 'rgba(255,0,0,0.1)'}; border-left: 5px solid {color};">
                <h3 style="color: {color}; margin: 0;">🎯 {action}</h3>
                <p style="margin: 10px 0 0 0;">
                    <strong>Recommended Position:</strong> {signal['position_size']:.1%} of capital<br>
                    <strong>Expected Return:</strong> {signal['expected_return']:.2%}<br>
                    <strong>Risk (Volatility):</strong> {signal['risk']:.2%}<br>
                    <strong>Confidence:</strong> {signal['confidence']:.1%}
                </p>
            </div>
            """, unsafe_allow_html=True)
        elif abs(signal['strength']) > 0.3:
            action = "BUY" if signal['strength'] > 0 else "SELL"
            st.info(f"💼 **{action}** signal detected. Position size: {signal['position_size']:.1%}")
        else:
            st.warning("⏸️ **NEUTRAL** - No strong signal. Consider waiting for better setup.")
        
        st.markdown("---")
