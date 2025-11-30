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
from python.lob_recorder import (
    LOBRecorder, OrderBookSnapshot, OrderBookUpdate, LOBAnalytics,
    parse_binance_orderbook, parse_binance_diff_depth, OrderBookLevel
)
from utils.ui_components import render_sidebar_navigation, apply_custom_css

def render():
    """Render the live trading page"""
    render_sidebar_navigation(current_page="Live Trading")
    apply_custom_css()
    
    # Initialize session state
    if 'theme_mode' not in st.session_state:
        st.session_state.theme_mode = 'dark'
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = {
            'positions': {},
            'cash': 100000.0,
            'initial_capital': 100000.0,
            'history': []
        }
    
    st.title("🔴 Live Trading")
    st.markdown("Execute strategies on real-time market data via WebSocket")
    
    # Safety warning
    if not st.session_state.get('live_trading_acknowledged', False):
        st.warning("⚠️ **Important:** This is for paper trading / simulation only. No real trades will be executed.")
        if st.checkbox("I understand this is simulated trading", key="live_trading_acknowledge"):
            st.session_state.live_trading_acknowledged = True
            st.rerun()
        return
    
    # Main layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        configure_live_trading()
    
    with col2:
        display_live_feed()
    
    # Bottom section - trade log, analytics, and LOB
    st.markdown("---")
    display_live_analytics_and_lob()
    
    # Auto-refresh when trading is active
    if st.session_state.get('live_trading_active', False):
        update_interval = st.session_state.get('live_update_interval', 500)
        time.sleep(update_interval / 1000)
        st.rerun()

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
    
    if use_websocket:
        st.info("💡 **WebSocket Tips:**\n- Use crypto symbols: `BINANCE:BTCUSDT`, `BINANCE:ETHUSDT`\n- Stock symbols require Finnhub premium tier\n- Free tier: max 1 connection (shared across all symbols)")
    
    if connection_mode == "Polling (REST)":
        update_interval = st.slider(
            "Update Interval (ms)",
            200, 5000, 500, step=100,
            help="How often to fetch new data"
        )
    else:
        # WebSocket UI refresh rate - slower to prevent chart instability
        update_interval = st.slider(
            "Chart Update Interval (ms)",
            500, 2000, 1000, step=100,
            help="How often to refresh charts (data streams in real-time)"
        )
        st.info("🔴 WebSocket streams data continuously, chart updates at selected interval")
    
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
        help="Execute strategy on live data",
        key="enable_strategy_live"
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

def start_live_trading(connector_name: str, symbols: List[str], use_websocket: bool, update_interval: int):
    """Start live trading session"""
    
    # Initialize session state
    st.session_state.live_trading_active = True
    st.session_state.live_data_buffer = []
    st.session_state.live_connector_name = connector_name
    st.session_state.live_symbols = symbols
    st.session_state.live_use_websocket = use_websocket
    st.session_state.live_update_interval = update_interval  # Store interval for polling
    st.session_state.live_start_time = datetime.now()
    st.session_state.live_ws_status = {}  # Track WebSocket status per symbol
    st.session_state.live_ws_logs = []  # WebSocket debug logs
    
    # Initialize trade log if needed
    if 'trade_log' not in st.session_state:
        st.session_state.trade_log = []
    
    # Initialize LOB recorder
    st.session_state.lob_recorder = LOBRecorder(
        symbols=symbols,
        snapshot_interval=60,  # Full snapshot every minute
        max_levels=20,
        storage_path="data/lob"
    )
    st.session_state.lob_enabled = True
    
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

def log_ws(message: str, level: str = "INFO"):
    """Add log entry to WebSocket log buffer"""
    if 'live_ws_logs' not in st.session_state:
        st.session_state.live_ws_logs = []
    
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    log_entry = f"[{timestamp}] {level}: {message}"
    st.session_state.live_ws_logs.append(log_entry)
    
    # Keep last 100 logs
    if len(st.session_state.live_ws_logs) > 100:
        st.session_state.live_ws_logs = st.session_state.live_ws_logs[-100:]

def start_websocket_streams(connector_name: str, symbols: List[str]):
    """Start WebSocket streams for specified symbols"""
    
    log_ws(f"Starting WebSocket streams for {len(symbols)} symbols: {symbols}")
    
    try:
        log_ws(f"Getting connector: {connector_name}")
        connector = get_connector(connector_name)
        log_ws(f"Connector obtained: {type(connector).__name__}")
        
        # Check if connector supports streaming
        if not hasattr(connector, 'start_stream'):
            log_ws(f"ERROR: Connector {connector_name} does not support WebSocket streaming", "ERROR")
            st.error(f"{connector_name} connector does not support WebSocket streaming")
            st.session_state.live_use_websocket = False
            return
        
        log_ws("Connector supports WebSocket streaming")
        
        # Create a thread-safe queue for WebSocket data
        if 'ws_data_queue' not in st.session_state:
            st.session_state.ws_data_queue = queue.Queue()
            log_ws("Created new WebSocket data queue")
        
        # Get the queue reference (DO NOT access session_state inside callbacks)
        data_queue = st.session_state.ws_data_queue
        log_ws("Queue reference obtained")
        
        # Define callback for WebSocket updates
        def create_callback(symbol):
            callback_count = [0]  # Mutable counter for closure
            
            def callback(orderbook):
                try:
                    callback_count[0] += 1
                    
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
                        
                        # Log every 10th callback to avoid spam
                        if callback_count[0] % 10 == 0:
                            log_ws(f"{symbol}: Received update #{callback_count[0]} - bid=${bid:.2f}, ask=${ask:.2f}")
                    else:
                        log_ws(f"{symbol}: Invalid bid/ask - bid={bid}, ask={ask}", "WARN")
                        
                except Exception as e:
                    log_ws(f"{symbol}: Callback error - {e}", "ERROR")
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
        log_ws(f"Starting streams for {len(symbols)} symbols...")
        
        for symbol in symbols:
            try:
                log_ws(f"{symbol}: Calling connector.start_stream()")
                connector.start_stream(symbol, create_callback(symbol))
                log_ws(f"{symbol}: Stream started successfully")
                
                st.session_state.live_ws_status[symbol] = {
                    'connected': True,
                    'last_update': datetime.now(),
                    'update_count': 0
                }
            except Exception as e:
                log_ws(f"{symbol}: Failed to start stream - {e}", "ERROR")
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
    
    # Clear chart state to reset on next start
    if 'live_chart_ranges' in st.session_state:
        st.session_state.live_chart_ranges = {}
    if 'live_chart_placeholders' in st.session_state:
        st.session_state.live_chart_placeholders = {}
    if 'live_chart_symbols' in st.session_state:
        st.session_state.live_chart_symbols = []
    if 'ws_last_chart_update' in st.session_state:
        st.session_state.ws_last_chart_update = {}
    
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
        st.plotly_chart(fig, use_container_width=True, key="empty_live_chart")
        return
    
    # Show WebSocket connection status if using WebSocket
    use_websocket = st.session_state.get('live_use_websocket', False)
    if use_websocket:
        display_websocket_status()
        
        # Show WebSocket logs for debugging
        with st.expander("🔍 WebSocket Debug Logs", expanded=False):
            logs = st.session_state.get('live_ws_logs', [])
            if logs:
                st.code("\n".join(logs[-30:]), language="log")  # Show last 30 logs
            else:
                st.info("No logs yet. Logs will appear here when WebSocket activity occurs.")
    
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
    
    # Show latest quotes in a stable table
    st.markdown("#### Current Quotes")
    
    latest = df.groupby('symbol').tail(1).copy()
    latest['mid'] = (latest['bid'] + latest['ask']) / 2
    latest['spread_bps'] = ((latest['ask'] - latest['bid']) / latest['bid'] * 10000).round(1)
    
    # Display as a clean table instead of metrics to prevent jumping
    quote_table = latest[['symbol', 'bid', 'ask', 'mid', 'spread_bps']].copy()
    quote_table.columns = ['Symbol', 'Bid ($)', 'Ask ($)', 'Mid ($)', 'Spread (bps)']
    quote_table = quote_table.round(2)
    
    st.dataframe(
        quote_table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Symbol": st.column_config.TextColumn("Symbol", width="small"),
            "Bid ($)": st.column_config.NumberColumn("Bid ($)", format="%.2f"),
            "Ask ($)": st.column_config.NumberColumn("Ask ($)", format="%.2f"),
            "Mid ($)": st.column_config.NumberColumn("Mid ($)", format="%.2f"),
            "Spread (bps)": st.column_config.NumberColumn("Spread (bps)", format="%.1f"),
        }
    )
    
    st.markdown("---")
    st.markdown("#### Live Charts")
    
    # Get current symbols (sorted for stable order)
    symbols = sorted(df['symbol'].unique())
    
    # Initialize chart placeholders if this is the first time or symbols changed
    if 'live_chart_placeholders' not in st.session_state:
        st.session_state.live_chart_placeholders = {}
        st.session_state.live_chart_symbols = []
    
    # Initialize axis ranges
    if 'live_chart_ranges' not in st.session_state:
        st.session_state.live_chart_ranges = {}
    
    # Check if symbols changed - if so, clear and recreate placeholders
    if st.session_state.live_chart_symbols != symbols:
        st.session_state.live_chart_placeholders = {}
        st.session_state.live_chart_symbols = symbols
        
        # Create stable placeholders for each symbol
        for symbol in symbols:
            placeholder = st.empty()
            st.session_state.live_chart_placeholders[symbol] = placeholder
    
    # Update each chart in its stable placeholder
    for symbol in symbols:
        symbol_df = df[df['symbol'] == symbol].tail(100)
        
        if len(symbol_df) == 0:
            continue
        
        # Calculate price range with padding
        all_prices = pd.concat([symbol_df['bid'], symbol_df['ask']])
        price_min = all_prices.min()
        price_max = all_prices.max()
        price_range = price_max - price_min if price_max > price_min else 1.0
        
        # Store or update range (expand only, never shrink during session)
        if symbol not in st.session_state.live_chart_ranges:
            st.session_state.live_chart_ranges[symbol] = {
                'min': price_min - price_range * 0.1,
                'max': price_max + price_range * 0.1
            }
        else:
            # Expand range if needed
            current = st.session_state.live_chart_ranges[symbol]
            st.session_state.live_chart_ranges[symbol] = {
                'min': min(current['min'], price_min - price_range * 0.1),
                'max': max(current['max'], price_max + price_range * 0.1)
            }
        
        y_range = st.session_state.live_chart_ranges[symbol]
        
        # Create chart with fixed y-axis
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=symbol_df['timestamp'],
            y=symbol_df['bid'],
            mode='lines',
            name='Bid',
            line={'color': 'green', 'width': 2},
            hovertemplate='Bid: $%{y:.2f}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=symbol_df['timestamp'],
            y=symbol_df['ask'],
            mode='lines',
            name='Ask',
            line={'color': 'red', 'width': 2},
            hovertemplate='Ask: $%{y:.2f}<extra></extra>'
        ))
        
        # Add mid price
        mid = (symbol_df['bid'] + symbol_df['ask']) / 2
        fig.add_trace(go.Scatter(
            x=symbol_df['timestamp'],
            y=mid,
            mode='lines',
            name='Mid',
            line={'color': 'cyan', 'width': 1, 'dash': 'dash'},
            hovertemplate='Mid: $%{y:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title={'text': f"<b>{symbol}</b>", 'x': 0.5, 'xanchor': 'center'},
            template="plotly_dark",
            height=280,
            margin={'l': 60, 'r': 20, 't': 40, 'b': 40},
            showlegend=True,
            legend={'orientation': 'h', 'yanchor': 'bottom', 'y': 1.02, 'xanchor': 'right', 'x': 1},
            hovermode='x unified',
            xaxis_title="Time",
            yaxis_title="Price ($)",
            yaxis={'range': [y_range['min'], y_range['max']], 'fixedrange': False},
            xaxis={'fixedrange': False},
            uirevision=symbol  # Preserves zoom/pan state per symbol
        )
        
        # Update the chart in its stable placeholder
        with st.session_state.live_chart_placeholders[symbol]:
            st.plotly_chart(fig, use_container_width=True, key=f"chart_{symbol}")

def fetch_live_data():
    """Fetch live market data - handles both REST polling and WebSocket modes"""
    
    use_websocket = st.session_state.get('live_use_websocket', False)
    
    # In WebSocket mode, data is pushed via callbacks, just check snapshots
    if use_websocket:
        fetch_websocket_snapshots()
    else:
        fetch_rest_polling()

def fetch_websocket_snapshots():
    """Process WebSocket data from the thread-safe queue with throttling"""
    
    ws_queue = st.session_state.get('ws_data_queue')
    
    if not ws_queue:
        return
    
    # Initialize last update time per symbol for throttling
    if 'ws_last_chart_update' not in st.session_state:
        st.session_state.ws_last_chart_update = {}
    
    # Minimum time between chart updates per symbol (ms)
    throttle_ms = st.session_state.get('live_update_interval', 1000) * 0.8  # 80% of UI refresh
    
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
                    log_ws(f"{symbol}: Error received - {error}", "ERROR")
                    st.session_state.live_ws_status[symbol] = {
                        'connected': False,
                        'error': error,
                        'last_update': data.get('timestamp', datetime.now())
                    }
                else:
                    # Handle data point with throttling
                    symbol = data.get('symbol', 'unknown')
                    now = datetime.now()
                    
                    # Check if enough time has passed since last update for this symbol
                    last_update = st.session_state.ws_last_chart_update.get(symbol)
                    should_add = True
                    
                    if last_update:
                        time_diff_ms = (now - last_update).total_seconds() * 1000
                        should_add = time_diff_ms >= throttle_ms
                    
                    if should_add:
                        # Add to buffer
                        st.session_state.live_data_buffer.append(data)
                        st.session_state.ws_last_chart_update[symbol] = now
                        
                        # Log buffer growth
                        if len(st.session_state.live_data_buffer) % 50 == 0:
                            log_ws(f"Buffer size: {len(st.session_state.live_data_buffer)} data points")
                        
                        # Keep buffer size manageable
                        if len(st.session_state.live_data_buffer) > 1000:
                            st.session_state.live_data_buffer = st.session_state.live_data_buffer[-1000:]
                    
                    # Always update WebSocket status (even if data was throttled)
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
    
    if not connector_name or not symbols:
        return
    
    try:
        connector = get_connector(connector_name)
        
        for symbol in symbols:
            try:
                # Fetch orderbook
                if hasattr(connector, 'fetch_orderbook_sync'):
                    ob = connector.fetch_orderbook_sync(symbol)
                elif hasattr(connector, 'fetch_orderbook'):
                    ob = connector.fetch_orderbook(symbol)
                else:
                    st.warning(f"Connector {connector_name} doesn't support orderbook fetching")
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
                    
                    # Record full LOB if enabled
                    if st.session_state.get('lob_enabled') and isinstance(ob, dict):
                        try:
                            lob_snapshot = parse_binance_orderbook(ob, symbol, connector_name)
                            st.session_state.lob_recorder.add_snapshot(lob_snapshot)
                        except Exception as e:
                            pass  # Silent fail for LOB recording
                    
                    # Execute strategy if enabled
                    if st.session_state.get('live_executor'):
                        execute_strategy_on_tick(data_point)
                else:
                    st.warning(f"No valid bid/ask for {symbol}: bid={bid}, ask={ask}")
                    
            except Exception as e:
                st.error(f"Error fetching {symbol}: {e}")
                import traceback
                st.code(traceback.format_exc())
                
    except Exception as e:
        st.error(f"Error getting connector {connector_name}: {e}")
        import traceback
        st.code(traceback.format_exc())

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

def display_live_analytics_and_lob():
    """Display live trading analytics and LOB data"""
    
    st.markdown("### Live Analytics & Orderbook")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Statistics", "📝 Trade Log", "⚡ Signals", "📖 Limit Order Book"])
    
    with tab1:
        display_live_statistics()
    
    with tab2:
        display_trade_log()
    
    with tab3:
        display_live_signals()
    
    with tab4:
        display_limit_order_book()

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
                
                st.plotly_chart(fig_components, use_container_width=True, key=f"components_{symbol}")
                
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
                
                st.plotly_chart(fig_price, use_container_width=True, key=f"price_vs_fundamental_{symbol}")
        
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

def display_limit_order_book():
    """Display comprehensive Limit Order Book data and analytics"""
    
    if not st.session_state.get('lob_enabled'):
        st.info("📖 Limit Order Book recording is not active. Start live feed to begin recording.")
        
        st.markdown("""
        ### What is Limit Order Book (LOB) Data?
        
        The **Limit Order Book** shows all pending buy and sell orders at different price levels:
        
        - **Bids** (Buy Orders): Prices buyers are willing to pay
        - **Asks** (Sell Orders): Prices sellers are asking
        - **Depth**: Volume available at each price level
        - **Spread**: Difference between best bid and ask
        
        ### Features:
        - 📊 Multi-level orderbook visualization (20+ levels)
        - 📈 LOB heatmaps and depth charts
        - 📉 Real-time imbalance metrics
        - 💰 Market impact analysis
        - 📥 Export LOB data for analysis
        
        Inspired by: [pfei-sa/binance-LOB](https://github.com/pfei-sa/binance-LOB)
        """)
        return
    
    recorder = st.session_state.get('lob_recorder')
    if not recorder:
        st.warning("LOB recorder not initialized")
        return
    
    symbols = st.session_state.get('live_symbols', [])
    if not symbols:
        st.info("No symbols selected")
        return
    
    # Symbol selector
    selected_symbol = st.selectbox("Select Symbol for LOB View", symbols, key="lob_symbol_selector")
    
    # Get current orderbook
    current_book = recorder.get_current_book(selected_symbol)
    
    if not current_book:
        st.info(f"⏳ Waiting for orderbook data for {selected_symbol}...")
        return
    
    # LOB Tabs
    lob_tab1, lob_tab2, lob_tab3, lob_tab4 = st.tabs([
        "📊 Orderbook Levels",
        "📈 LOB Analytics",
        "🔥 Heatmap",
        "💾 Export Data"
    ])
    
    with lob_tab1:
        display_orderbook_levels(current_book, recorder, selected_symbol)
    
    with lob_tab2:
        display_lob_analytics(recorder, selected_symbol)
    
    with lob_tab3:
        display_lob_heatmap(recorder, selected_symbol)
    
    with lob_tab4:
        display_lob_export(recorder, selected_symbol)


def display_orderbook_levels(book: OrderBookSnapshot, recorder: LOBRecorder, symbol: str):
    """Display current orderbook levels in a table format"""
    
    st.markdown(f"### 📖 {symbol} Orderbook Levels")
    
    # Calculate analytics for current book
    analytics = recorder.calculate_analytics(book)
    
    # Display key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Best Bid", f"${analytics.best_bid:.2f}")
    with col2:
        st.metric("Best Ask", f"${analytics.best_ask:.2f}")
    with col3:
        st.metric("Spread", f"${analytics.spread_abs:.4f}", 
                 delta=f"{analytics.spread_bps:.1f} bps")
    with col4:
        st.metric("Mid Price", f"${analytics.mid_price:.2f}")
    with col5:
        imbalance_pct = analytics.volume_imbalance * 100
        st.metric("Imbalance", f"{imbalance_pct:+.1f}%",
                 delta="Bullish" if imbalance_pct > 5 else "Bearish" if imbalance_pct < -5 else "Neutral")
    
    st.markdown("---")
    
    # Display orderbook as two-column table
    col_asks, col_bids = st.columns(2)
    
    with col_asks:
        st.markdown("#### 🔴 Asks (Sell Orders)")
        
        if book.asks:
            # Reverse asks so highest price is at top
            asks_reversed = list(reversed(book.asks[:15]))
            
            asks_data = []
            cumulative_vol = 0
            for level in asks_reversed:
                cumulative_vol += level.quantity * level.price
                asks_data.append({
                    'Price ($)': level.price,
                    'Quantity': level.quantity,
                    'Total ($)': level.quantity * level.price,
                    'Cumulative ($)': cumulative_vol
                })
            
            asks_df = pd.DataFrame(asks_data)
            
            st.dataframe(
                asks_df.style.format({
                    'Price ($)': '{:.2f}',
                    'Quantity': '{:.4f}',
                    'Total ($)': '{:,.2f}',
                    'Cumulative ($)': '{:,.2f}'
                }).background_gradient(subset=['Total ($)'], cmap='Reds'),
                use_container_width=True,
                hide_index=True,
                height=400
            )
        else:
            st.info("No ask orders")
    
    with col_bids:
        st.markdown("#### 🟢 Bids (Buy Orders)")
        
        if book.bids:
            bids_data = []
            cumulative_vol = 0
            for level in book.bids[:15]:
                cumulative_vol += level.quantity * level.price
                bids_data.append({
                    'Price ($)': level.price,
                    'Quantity': level.quantity,
                    'Total ($)': level.quantity * level.price,
                    'Cumulative ($)': cumulative_vol
                })
            
            bids_df = pd.DataFrame(bids_data)
            
            st.dataframe(
                bids_df.style.format({
                    'Price ($)': '{:.2f}',
                    'Quantity': '{:.4f}',
                    'Total ($)': '{:,.2f}',
                    'Cumulative ($)': '{:,.2f}'
                }).background_gradient(subset=['Total ($)'], cmap='Greens'),
                use_container_width=True,
                hide_index=True,
                height=400
            )
        else:
            st.info("No bid orders")
    
    # Depth visualization
    st.markdown("---")
    st.markdown("#### 📊 Order Book Depth Visualization")
    
    if book.bids and book.asks:
        fig = go.Figure()
        
        # Prepare bid data (cumulative)
        bid_prices = [level.price for level in book.bids]
        bid_volumes = [level.quantity * level.price for level in book.bids]
        bid_cumulative = np.cumsum(bid_volumes)
        
        # Prepare ask data (cumulative)
        ask_prices = [level.price for level in book.asks]
        ask_volumes = [level.quantity * level.price for level in book.asks]
        ask_cumulative = np.cumsum(ask_volumes)
        
        # Plot bids
        fig.add_trace(go.Scatter(
            x=bid_prices,
            y=bid_cumulative,
            mode='lines',
            name='Bid Depth',
            fill='tozeroy',
            fillcolor='rgba(0, 255, 0, 0.2)',
            line=dict(color='green', width=2)
        ))
        
        # Plot asks
        fig.add_trace(go.Scatter(
            x=ask_prices,
            y=ask_cumulative,
            mode='lines',
            name='Ask Depth',
            fill='tozeroy',
            fillcolor='rgba(255, 0, 0, 0.2)',
            line=dict(color='red', width=2)
        ))
        
        # Add mid price line
        fig.add_vline(
            x=analytics.mid_price,
            line_dash="dash",
            line_color="cyan",
            annotation_text=f"Mid: ${analytics.mid_price:.2f}",
            annotation_position="top"
        )
        
        fig.update_layout(
            title="Cumulative Order Book Depth",
            xaxis_title="Price ($)",
            yaxis_title="Cumulative Volume ($)",
            template="plotly_dark",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True, key=f"depth_{symbol}")


def display_lob_analytics(recorder: LOBRecorder, symbol: str):
    """Display LOB analytics over time"""
    
    st.markdown(f"### 📈 {symbol} LOB Analytics")
    
    analytics_list = recorder.get_analytics(symbol, n=500)
    
    if len(analytics_list) < 2:
        st.info("⏳ Collecting analytics data... (need at least 2 data points)")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame([a.to_dict() for a in analytics_list])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Display current metrics
    latest = analytics_list[-1]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Spread (bps)", f"{latest.spread_bps:.2f}")
        st.metric("Bid Depth (0.1%)", f"${latest.bid_depth_1:,.0f}")
    
    with col2:
        st.metric("Volume Imbalance", f"{latest.volume_imbalance*100:+.1f}%")
        st.metric("Ask Depth (0.1%)", f"${latest.ask_depth_1:,.0f}")
    
    with col3:
        st.metric("Market Impact ($10k)", f"{latest.market_impact_10k:.2f} bps")
        st.metric("Bid Depth (1%)", f"${latest.bid_depth_10:,.0f}")
    
    with col4:
        st.metric("Effective Spread", f"{latest.effective_spread_bps:.2f} bps")
        st.metric("Ask Depth (1%)", f"${latest.ask_depth_10:,.0f}")
    
    st.markdown("---")
    
    # Time series plots
    st.markdown("#### 📊 Analytics Over Time")
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Spread (bps)', 'Volume Imbalance',
            'Bid/Ask Depth (0.1%)', 'Market Impact ($10k)',
            'Mid Price', 'Effective Spread (bps)'
        ),
        vertical_spacing=0.1
    )
    
    # Spread
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['spread_bps'], name='Spread', line=dict(color='cyan')),
        row=1, col=1
    )
    
    # Imbalance
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['volume_imbalance']*100, name='Imbalance', line=dict(color='orange')),
        row=1, col=2
    )
    
    # Depth
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['bid_depth_1'], name='Bid Depth', line=dict(color='green')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['ask_depth_1'], name='Ask Depth', line=dict(color='red')),
        row=2, col=1
    )
    
    # Market Impact
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['market_impact_10k'], name='Impact', line=dict(color='purple')),
        row=2, col=2
    )
    
    # Mid Price
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['mid_price'], name='Mid Price', line=dict(color='white')),
        row=3, col=1
    )
    
    # Effective Spread
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['effective_spread_bps'], name='Eff. Spread', line=dict(color='pink')),
        row=3, col=2
    )
    
    fig.update_layout(
        height=900,
        showlegend=False,
        template="plotly_dark",
        hovermode='x unified'
    )
    
    fig.update_yaxes(title_text="bps", row=1, col=1)
    fig.update_yaxes(title_text="%", row=1, col=2)
    fig.update_yaxes(title_text="$ Volume", row=2, col=1)
    fig.update_yaxes(title_text="bps", row=2, col=2)
    fig.update_yaxes(title_text="$", row=3, col=1)
    fig.update_yaxes(title_text="bps", row=3, col=2)
    
    st.plotly_chart(fig, use_container_width=True, key=f"analytics_{symbol}")
    
    # Statistics table
    st.markdown("#### 📋 Summary Statistics")
    
    summary_stats = {
        'Metric': [
            'Avg Spread (bps)',
            'Avg Volume Imbalance (%)',
            'Avg Market Impact ($10k)',
            'Avg Bid Depth (0.1%)',
            'Avg Ask Depth (0.1%)',
            'Total Bid Levels',
            'Total Ask Levels'
        ],
        'Value': [
            f"{df['spread_bps'].mean():.2f}",
            f"{df['volume_imbalance'].mean()*100:+.2f}",
            f"{df['market_impact_10k'].mean():.2f} bps",
            f"${df['bid_depth_1'].mean():,.0f}",
            f"${df['ask_depth_1'].mean():,.0f}",
            f"{df['bid_levels'].mean():.1f}",
            f"{df['ask_levels'].mean():.1f}"
        ]
    }
    
    st.dataframe(pd.DataFrame(summary_stats), use_container_width=True, hide_index=True)


def display_lob_heatmap(recorder: LOBRecorder, symbol: str):
    """Display LOB heatmap visualization"""
    
    st.markdown(f"### 🔥 {symbol} Order Book Heatmap")
    
    st.info("📊 Heatmap shows order book evolution over time. Darker colors = higher volume.")
    
    # Get recent snapshots
    snapshots = list(recorder.snapshots[symbol])[-50:]  # Last 50 snapshots
    
    if len(snapshots) < 2:
        st.warning("Need at least 2 snapshots to generate heatmap. Keep recording...")
        return
    
    # Extract price levels and volumes
    times = [s.timestamp for s in snapshots]
    
    # Find price range
    all_prices = []
    for s in snapshots:
        all_prices.extend([b.price for b in s.bids[:10]])
        all_prices.extend([a.price for a in s.asks[:10]])
    
    if not all_prices:
        st.warning("No price data available")
        return
    
    min_price = min(all_prices)
    max_price = max(all_prices)
    price_range = max_price - min_price
    
    # Create price bins
    n_bins = 50
    price_bins = np.linspace(min_price - price_range*0.1, max_price + price_range*0.1, n_bins)
    
    # Build heatmap matrix
    heatmap_data = np.zeros((n_bins, len(snapshots)))
    
    for t_idx, snapshot in enumerate(snapshots):
        # Aggregate volume at each price bin
        for bid in snapshot.bids[:10]:
            bin_idx = np.digitize(bid.price, price_bins) - 1
            if 0 <= bin_idx < n_bins:
                heatmap_data[bin_idx, t_idx] += bid.quantity * bid.price
        
        for ask in snapshot.asks[:10]:
            bin_idx = np.digitize(ask.price, price_bins) - 1
            if 0 <= bin_idx < n_bins:
                heatmap_data[bin_idx, t_idx] -= ask.quantity * ask.price  # Negative for asks
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=[t.strftime("%H:%M:%S") for t in times],
        y=[f"${p:.2f}" for p in price_bins],
        colorscale='RdYlGn',
        zmid=0,
        colorbar=dict(title="Volume ($)")
    ))
    
    fig.update_layout(
        title="Order Book Heatmap (Green=Bids, Red=Asks)",
        xaxis_title="Time",
        yaxis_title="Price",
        template="plotly_dark",
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True, key=f"heatmap_{symbol}")
    
    st.caption("💡 Tip: Look for patterns like walls, gaps, or shifting liquidity zones")


def display_lob_export(recorder: LOBRecorder, symbol: str):
    """Export LOB data and analytics"""
    
    st.markdown(f"### 💾 Export {symbol} LOB Data")
    
    st.markdown("""
    Export orderbook data and analytics for further analysis:
    - **Analytics CSV**: Time series of LOB metrics
    - **Snapshots JSON**: Full orderbook snapshots
    - **Summary Report**: Statistical summary
    """)
    
    # Export analytics
    analytics_df = recorder.export_to_csv(symbol)
    
    if analytics_df.empty:
        st.warning("No data available to export yet")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Data Points", f"{len(analytics_df):,}")
        st.metric("Time Range", f"{(analytics_df['timestamp'].max() - analytics_df['timestamp'].min()).total_seconds():.0f}s")
    
    with col2:
        st.metric("Avg Spread", f"{analytics_df['spread_bps'].mean():.2f} bps")
        st.metric("Avg Imbalance", f"{analytics_df['volume_imbalance'].mean()*100:+.1f}%")
    
    st.markdown("---")
    
    # Download buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Analytics CSV
        csv_data = analytics_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📊 Download Analytics CSV",
            data=csv_data,
            file_name=f"{symbol}_lob_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Snapshots JSON
        snapshots = list(recorder.snapshots[symbol])
        snapshots_json = json.dumps([s.to_dict() for s in snapshots], indent=2)
        st.download_button(
            label="📖 Download Snapshots JSON",
            data=snapshots_json.encode('utf-8'),
            file_name=f"{symbol}_lob_snapshots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col3:
        # Summary report
        summary = f"""
LOB Analysis Report for {symbol}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

=== SUMMARY STATISTICS ===
Data Points: {len(analytics_df)}
Time Range: {analytics_df['timestamp'].min()} to {analytics_df['timestamp'].max()}

=== SPREAD METRICS ===
Avg Spread: {analytics_df['spread_bps'].mean():.2f} bps
Min Spread: {analytics_df['spread_bps'].min():.2f} bps
Max Spread: {analytics_df['spread_bps'].max():.2f} bps
Std Spread: {analytics_df['spread_bps'].std():.2f} bps

=== IMBALANCE METRICS ===
Avg Volume Imbalance: {analytics_df['volume_imbalance'].mean()*100:+.2f}%
Avg Depth Imbalance (0.1%): {analytics_df['depth_imbalance_1'].mean()*100:+.2f}%
Avg Depth Imbalance (0.5%): {analytics_df['depth_imbalance_5'].mean()*100:+.2f}%

=== LIQUIDITY METRICS ===
Avg Market Impact ($10k): {analytics_df['market_impact_10k'].mean():.2f} bps
Avg Effective Spread: {analytics_df['effective_spread_bps'].mean():.2f} bps
Avg Bid Depth (0.1%): ${analytics_df['bid_depth_1'].mean():,.2f}
Avg Ask Depth (0.1%): ${analytics_df['ask_depth_1'].mean():,.2f}

=== ORDERBOOK STRUCTURE ===
Avg Bid Levels: {analytics_df['bid_levels'].mean():.1f}
Avg Ask Levels: {analytics_df['ask_levels'].mean():.1f}
Avg Total Bid Volume: ${analytics_df['total_bid_volume'].mean():,.2f}
Avg Total Ask Volume: ${analytics_df['total_ask_volume'].mean():,.2f}
"""
        
        st.download_button(
            label="📋 Download Summary Report",
            data=summary.encode('utf-8'),
            file_name=f"{symbol}_lob_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    # Preview data
    st.markdown("---")
    st.markdown("#### 👀 Data Preview")
    
    st.dataframe(
        analytics_df.tail(100).style.format({
            'spread_bps': '{:.2f}',
            'spread_abs': '{:.4f}',
            'mid_price': '{:.2f}',
            'volume_imbalance': '{:+.3f}',
            'market_impact_10k': '{:.2f}'
        }),
        use_container_width=True,
        height=400
    )


# Execute the render function when page is loaded
if __name__ == "__main__":
    render()
