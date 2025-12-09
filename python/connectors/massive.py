"""
Massive.com WebSocket Connector for Live Trading
Provides real-time market data streaming for HFT strategies
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
import json
import threading
import time

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

from python.api_keys import get_massive_key

# Import data recorder if available
try:
    from python.massive_helper import WebSocketDataRecorder
    RECORDER_AVAILABLE = True
except ImportError:
    RECORDER_AVAILABLE = False

logger = logging.getLogger(__name__)


class MassiveConnector:
    """
    Massive.com WebSocket connector for live trading.
    
    Features:
    - WebSocket streaming (10 concurrent connections, 100 messages/min)
    - Real-time quotes, orderbook, and trades
    - Support for stocks, options, futures, forex, crypto
    - Thread-safe callbacks for Streamlit integration
    
    Free Tier Limits:
    - 10 concurrent WebSocket connections
    - 100 messages per minute
    - REST API: 100 calls/day, 10/minute
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Massive connector.
        
        Args:
            api_key: Massive API key (if None, loaded from api_keys.properties)
        """
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError("websockets library required for Massive connector. Install with: pip install websockets")
        
        self.api_key = api_key or get_massive_key()
        if not self.api_key:
            logger.warning("No Massive API key found. WebSocket will use synthetic data fallback.")
        
        self.name = "massive"
        self.ws_url = "wss://api.massive.com/v1/stream"  # Example URL - adjust based on actual API
        
        # Connection management
        self._connections: Dict[str, Any] = {}  # symbol -> websocket connection
        self._callbacks: Dict[str, List[Callable]] = {}  # symbol -> list of callbacks
        self._running = False
        self._threads: Dict[str, threading.Thread] = {}  # symbol -> thread
        self._message_count = 0
        self._message_count_reset_time = time.time()
        self._max_messages_per_minute = 100
        
        # Data recording
        self._recorder: Optional[Any] = None  # WebSocketDataRecorder instance
        self._recording_enabled = False
        
        # Synthetic data for testing without API key
        self._base_prices = {
            "BTCUSDT": 43000.0,
            "ETHUSDT": 2300.0,
            "AAPL": 180.0,
            "GOOGL": 140.0,
            "MSFT": 380.0
        }
    
    def list_symbols(self) -> List[str]:
        """Return list of available symbols."""
        return [
            "BTCUSDT", "ETHUSDT", "BNBUSDT",  # Crypto
            "AAPL", "GOOGL", "MSFT", "TSLA",  # Stocks
            "EURUSD", "GBPUSD"  # Forex
        ]
    
    def enable_recording(self, symbols: List[str], save_dir: str = "data/live_recorded"):
        """
        Enable recording of live WebSocket data to datasets.
        
        Args:
            symbols: Symbols to record
            save_dir: Directory to save recorded data
        """
        if not RECORDER_AVAILABLE:
            logger.warning("WebSocketDataRecorder not available - recording disabled")
            return
        
        self._recorder = WebSocketDataRecorder(symbols=symbols, save_dir=save_dir)
        self._recorder.start()
        self._recording_enabled = True
        logger.info(f"📹 Recording enabled for {len(symbols)} symbols")
    
    def disable_recording(self) -> Optional[str]:
        """
        Disable recording and return path to saved dataset.
        
        Returns:
            Path to saved dataset file
        """
        if not self._recording_enabled or not self._recorder:
            return None
        
        self._recorder.stop()
        
        # Save dataset with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name = f"massive_live_{timestamp}"
        filepath = self._recorder.save_dataset(dataset_name, resample_interval="1min")
        
        self._recording_enabled = False
        logger.info(f"💾 Recording saved to {filepath}")
        
        return str(filepath) if filepath else None
    
    def get_recorded_dataframe(self, resample_interval: str = "1min"):
        """Get currently recorded data as DataFrame."""
        if not self._recording_enabled or not self._recorder:
            return None
        return self._recorder.get_dataframe(resample_interval=resample_interval)
    
    def start_stream(self, symbols: List[str], callback: Callable[[str, Dict[str, Any]], None]):
        """
        Start WebSocket streams for given symbols.
        
        Args:
            symbols: List of symbols to stream
            callback: Function(symbol, orderbook) called on each update
        """
        logger.info(f"Starting Massive WebSocket streams for {len(symbols)} symbols: {symbols}")
        
        self._running = True
        
        for symbol in symbols:
            if symbol not in self._callbacks:
                self._callbacks[symbol] = []
            self._callbacks[symbol].append(callback)
            
            # Start thread for this symbol's WebSocket connection
            if symbol not in self._threads:
                thread = threading.Thread(
                    target=self._run_symbol_stream,
                    args=(symbol,),
                    daemon=True
                )
                thread.start()
                self._threads[symbol] = thread
                logger.info(f"Started WebSocket thread for {symbol}")
    
    def _run_symbol_stream(self, symbol: str):
        """Run WebSocket stream for a single symbol (thread target)."""
        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(self._stream_symbol(symbol))
        except Exception as e:
            logger.error(f"Error in WebSocket stream for {symbol}: {e}")
        finally:
            loop.close()
    
    async def _stream_symbol(self, symbol: str):
        """
        Async WebSocket streaming for a single symbol.
        
        Args:
            symbol: Symbol to stream
        """
        retry_delay = 1
        max_retry_delay = 60
        
        while self._running:
            try:
                # Check rate limit (100 messages/minute)
                if self._check_rate_limit():
                    logger.warning("Approaching Massive free tier rate limit (100 msg/min). Slowing down...")
                    await asyncio.sleep(1)
                
                if not self.api_key:
                    # Synthetic data fallback for testing
                    await self._stream_synthetic(symbol)
                else:
                    # Real WebSocket connection
                    await self._stream_real(symbol)
                
                # Reset retry delay on successful connection
                retry_delay = 1
                
            except Exception as e:
                logger.error(f"WebSocket error for {symbol}: {e}")
                logger.info(f"Reconnecting in {retry_delay}s...")
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, max_retry_delay)
    
    async def _stream_real(self, symbol: str):
        """Stream real market data via Massive WebSocket API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        
        async with websockets.connect(self.ws_url, extra_headers=headers) as websocket:
            # Subscribe to symbol
            subscribe_msg = {
                "action": "subscribe",
                "symbols": [symbol],
                "types": ["quote", "trade"]  # Request quotes and trades
            }
            await websocket.send(json.dumps(subscribe_msg))
            logger.info(f"Subscribed to {symbol} on Massive WebSocket")
            
            # Store connection
            self._connections[symbol] = websocket
            
            # Receive messages
            while self._running:
                message = await websocket.recv()
                self._handle_message(symbol, message)
    
    async def _stream_synthetic(self, symbol: str):
        """Stream synthetic data for testing without API key."""
        logger.info(f"Streaming synthetic data for {symbol} (no API key)")
        
        base_price = self._base_prices.get(symbol, 100.0)
        
        while self._running:
            # Generate synthetic orderbook
            import random
            variation = random.uniform(-0.002, 0.002)
            mid_price = base_price * (1.0 + variation)
            spread = mid_price * 0.0005  # 5 bps spread
            
            bid = mid_price - spread / 2
            ask = mid_price + spread / 2
            
            orderbook = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "bids": [[bid, 10.0], [bid * 0.9999, 5.0], [bid * 0.9998, 2.0]],
                "asks": [[ask, 10.0], [ask * 1.0001, 5.0], [ask * 1.0002, 2.0]]
            }
            
            # Trigger callbacks
            self._trigger_callbacks(symbol, orderbook)
            
            # Update every 100ms for realistic tick rate
            await asyncio.sleep(0.1)
    
    def _handle_message(self, symbol: str, message: str):
        """
        Process incoming WebSocket message.
        
        Args:
            symbol: Symbol this message is for
            message: Raw JSON message from WebSocket
        """
        try:
            data = json.loads(message)
            
            # Increment message counter for rate limiting
            self._message_count += 1
            
            # Convert to orderbook format expected by trading system
            if data.get("type") == "quote":
                orderbook = {
                    "symbol": symbol,
                    "timestamp": data.get("timestamp", datetime.now().isoformat()),
                    "bids": [[data.get("bid", 0), data.get("bid_size", 0)]],
                    "asks": [[data.get("ask", 0), data.get("ask_size", 0)]]
                }
                self._trigger_callbacks(symbol, orderbook)
            
            elif data.get("type") == "trade":
                # Update mid price based on trade
                price = data.get("price", 0)
                size = data.get("size", 0)
                
                # Generate orderbook from trade price
                spread = price * 0.0005
                orderbook = {
                    "symbol": symbol,
                    "timestamp": data.get("timestamp", datetime.now().isoformat()),
                    "bids": [[price - spread / 2, size]],
                    "asks": [[price + spread / 2, size]]
                }
                self._trigger_callbacks(symbol, orderbook)
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse WebSocket message: {e}")
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
    
    def _trigger_callbacks(self, symbol: str, orderbook: Dict[str, Any]):
        """
        Trigger all registered callbacks for a symbol.
        
        Args:
            symbol: Symbol that updated
            orderbook: Orderbook data
        """
        # Record data if recording is enabled
        if self._recording_enabled and self._recorder:
            try:
                # Convert orderbook to quote format for recorder
                quote_data = {
                    'type': 'quote',
                    'symbol': symbol,
                    'timestamp': orderbook.get('timestamp', datetime.now().isoformat()),
                    'bid': orderbook['bids'][0][0] if orderbook.get('bids') else 0.0,
                    'ask': orderbook['asks'][0][0] if orderbook.get('asks') else 0.0,
                    'bid_size': orderbook['bids'][0][1] if orderbook.get('bids') else 0,
                    'ask_size': orderbook['asks'][0][1] if orderbook.get('asks') else 0,
                    'last': (orderbook['bids'][0][0] + orderbook['asks'][0][0]) / 2 if orderbook.get('bids') and orderbook.get('asks') else 0.0,
                    'volume': 0
                }
                self._recorder.on_message(quote_data)
            except Exception as e:
                logger.error(f"Error recording data for {symbol}: {e}")
        
        # Trigger user callbacks
        if symbol in self._callbacks:
            for callback in self._callbacks[symbol]:
                try:
                    callback(symbol, orderbook)
                except Exception as e:
                    logger.error(f"Error in callback for {symbol}: {e}")
    
    def _check_rate_limit(self) -> bool:
        """
        Check if approaching rate limit (100 messages/minute).
        
        Returns:
            True if should slow down, False otherwise
        """
        current_time = time.time()
        elapsed = current_time - self._message_count_reset_time
        
        # Reset counter every minute
        if elapsed >= 60:
            self._message_count = 0
            self._message_count_reset_time = current_time
            return False
        
        # Warn if approaching limit (>80% of quota)
        if self._message_count > self._max_messages_per_minute * 0.8:
            return True
        
        return False
    
    def stop_stream(self, symbols: Optional[List[str]] = None):
        """
        Stop WebSocket streams.
        
        Args:
            symbols: Symbols to stop (if None, stops all)
        """
        if symbols is None:
            logger.info("Stopping all Massive WebSocket streams")
            self._running = False
            
            # Wait for threads to finish
            for symbol, thread in self._threads.items():
                if thread.is_alive():
                    thread.join(timeout=2)
            
            self._threads.clear()
            self._callbacks.clear()
            self._connections.clear()
        else:
            for symbol in symbols:
                logger.info(f"Stopping Massive WebSocket stream for {symbol}")
                if symbol in self._callbacks:
                    del self._callbacks[symbol]
                if symbol in self._connections:
                    # Close WebSocket connection
                    try:
                        asyncio.run(self._connections[symbol].close())
                    except Exception as e:
                        logger.error(f"Error closing WebSocket for {symbol}: {e}")
                    del self._connections[symbol]
                if symbol in self._threads:
                    del self._threads[symbol]
    
    def fetch_orderbook_sync(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch orderbook synchronously (for REST API polling mode).
        
        Args:
            symbol: Symbol to fetch
            
        Returns:
            Orderbook dictionary with bids/asks
        """
        # Use synthetic data for now - in production, call REST API
        base_price = self._base_prices.get(symbol, 100.0)
        
        import random
        variation = random.uniform(-0.002, 0.002)
        mid_price = base_price * (1.0 + variation)
        spread = mid_price * 0.0005
        
        bid = mid_price - spread / 2
        ask = mid_price + spread / 2
        
        return {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "bids": [[bid, 10.0], [bid * 0.9999, 5.0], [bid * 0.9998, 2.0]],
            "asks": [[ask, 10.0], [ask * 1.0001, 5.0], [ask * 1.0002, 2.0]]
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connector statistics and rate limit info."""
        return {
            "active_streams": len(self._connections),
            "max_streams": 10,
            "messages_this_minute": self._message_count,
            "max_messages_per_minute": self._max_messages_per_minute,
            "rate_limit_pct": (self._message_count / self._max_messages_per_minute) * 100
        }
