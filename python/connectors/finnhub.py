# python/connectors/finnhub.py
"""
Finnhub connector for real-time market data via WebSocket.
Supports trades, quotes, and level 1 orderbook data.
"""

import json
import logging
import threading
from typing import Optional, Callable
import websocket
import time
import requests

logger = logging.getLogger(__name__)


class FinnhubConnector:
    """
    Finnhub connector using WebSocket API for real-time market data.
    
    Note: Finnhub provides trades and quotes, not full orderbook depth.
    For orderbook simulation, we maintain a synthetic book from quote updates.
    
    Free tier limits: 60 API calls/minute
    """
    
    def __init__(self, api_key: Optional[str] = None):
        # Auto-load API key from api_keys.properties if not provided
        if api_key is None:
            try:
                from python.api_keys import get_api_key
                api_key = get_api_key("FINNHUB_API_KEY")
            except ImportError:
                pass
        
        if not api_key:
            raise ValueError(
                "Finnhub API key not found. Please:\n"
                "1. Copy api_keys.properties.example to api_keys.properties\n"
                "2. Fill in FINNHUB_API_KEY\n"
                "Or provide the key explicitly to the constructor."
            )
        
        self.api_key = api_key
        self.name = "finnhub"
        self.ws_url = f"wss://ws.finnhub.io?token={api_key}"
        
        self.ws = None
        self.ws_thread = None
        self.running = False
        
        # Support multiple symbols on single connection
        self.callbacks = {}  # symbol -> callback
        self.subscribed_symbols = set()
        self.latest_prices = {}  # symbol -> {bid, ask}
        
        # Lock for thread-safe updates
        self._lock = threading.Lock()
    
    def list_symbols(self):
        """Return common symbols available on Finnhub."""
        return [
            "BINANCE:BTCUSDT", "BINANCE:ETHUSDT", "BINANCE:BNBUSDT",
            "AAPL", "GOOGL", "MSFT", "TSLA", "AMZN",
            "COINBASE:BTC-USD", "COINBASE:ETH-USD"
        ]
    
    def fetch_orderbook_sync(self, symbol: str):
        """
        Fetch real-time quote via REST API and return synthetic orderbook.
        Finnhub doesn't provide full depth, so we create a 1-level book from quote.
        """
        # If WebSocket stream is running, return cached data
        with self._lock:
            if symbol in self.latest_prices:
                prices = self.latest_prices[symbol]
                return {
                    "bids": [[prices['bid'], 1.0]],
                    "asks": [[prices['ask'], 1.0]]
                }
        
        # Otherwise, fetch via REST API
        try:
            # Fetch real-time quote from Finnhub REST API
            url = f"https://finnhub.io/api/v1/quote"
            params = {
                "symbol": symbol,
                "token": self.api_key
            }
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            # Extract current price (c), high (h), low (l), open (o), previous close (pc)
            current_price = data.get("c")
            
            if current_price and current_price > 0:
                # Create synthetic bid/ask with small spread (1 bps)
                spread = current_price * 0.0001
                bid = current_price - spread / 2
                ask = current_price + spread / 2
                
                # Update cached values
                with self._lock:
                    self.latest_bid = bid
                    self.latest_ask = ask
                    self.latest_symbol = symbol
                
                return {
                    "bids": [[bid, 1.0]],
                    "asks": [[ask, 1.0]]
                }
            else:
                logger.warning(f"No valid price data for {symbol}")
                return {"bids": [], "asks": []}
                
        except Exception as e:
            logger.error(f"Failed to fetch quote for {symbol}: {e}")
            return {"bids": [], "asks": []}
    
    def start_stream(self, symbol: str, callback: Callable):
        """
        Start WebSocket stream for the given symbol.
        Uses a single shared WebSocket connection for all symbols.
        Callback receives synthetic OrderBook objects.
        """
        with self._lock:
            # Store callback for this symbol
            self.callbacks[symbol] = callback
            self.subscribed_symbols.add(symbol)
        
        # If connection not running, start it
        if not self.running:
            self.running = True
            self.ws_thread = threading.Thread(
                target=self._ws_run,
                daemon=True
            )
            self.ws_thread.start()
            logger.info(f"Started Finnhub WebSocket connection")
        else:
            # Connection already running, just subscribe to new symbol
            if self.ws:
                try:
                    subscribe_msg = json.dumps({"type": "subscribe", "symbol": symbol})
                    self.ws.send(subscribe_msg)
                    logger.info(f"Subscribed to {symbol} on existing connection")
                except Exception as e:
                    logger.error(f"Failed to subscribe to {symbol}: {e}")
    
    def stop_stream(self, symbol: Optional[str] = None):
        """Stop the WebSocket stream for a symbol or all symbols."""
        if symbol:
            # Unsubscribe specific symbol
            with self._lock:
                if symbol in self.subscribed_symbols:
                    self.subscribed_symbols.discard(symbol)
                    self.callbacks.pop(symbol, None)
                    self.latest_prices.pop(symbol, None)
            
            if self.ws:
                try:
                    unsubscribe_msg = json.dumps({"type": "unsubscribe", "symbol": symbol})
                    self.ws.send(unsubscribe_msg)
                    logger.info(f"Unsubscribed from {symbol}")
                except Exception as e:
                    logger.debug(f"Error unsubscribing (ignoring): {e}")
        else:
            # Stop entire connection
            self.running = False
            if self.ws:
                try:
                    self.ws.close()
                except Exception as e:
                    logger.debug(f"Error closing websocket (ignoring): {e}")
            if self.ws_thread and self.ws_thread.is_alive():
                self.ws_thread.join(timeout=2)
            
            with self._lock:
                self.subscribed_symbols.clear()
                self.callbacks.clear()
                self.latest_prices.clear()
            
            logger.info("Stopped Finnhub WebSocket connection")
    
    def latest_snapshot(self, symbol: Optional[str] = None):
        """Return latest synthetic orderbook for a symbol."""
        with self._lock:
            if symbol and symbol in self.latest_prices:
                prices = self.latest_prices[symbol]
                return {
                    "bids": [[prices['bid'], 1.0]],
                    "asks": [[prices['ask'], 1.0]]
                }
            elif not symbol and self.latest_prices:
                # Return first available symbol's data
                first_symbol = next(iter(self.latest_prices))
                prices = self.latest_prices[first_symbol]
                return {
                    "bids": [[prices['bid'], 1.0]],
                    "asks": [[prices['ask'], 1.0]]
                }
            return None
    
    def _ws_run(self):
        """WebSocket thread main loop - handles all subscribed symbols."""
        def on_message(ws, message):
            try:
                data = json.loads(message)
                msg_type = data.get("type")
                
                if msg_type == "trade":
                    # Trade update: {type: 'trade', data: [{p: price, s: symbol, t: time, v: volume}]}
                    trades = data.get("data", [])
                    logger.info(f"Trade message with {len(trades)} trades")
                    
                    for trade in trades:
                        trade_symbol = trade.get("s")  # Get symbol from trade data
                        price = trade.get("p")
                        
                        if price and trade_symbol:
                            # Update bid/ask with small spread
                            spread = price * 0.0001  # 1 bps spread
                            bid = price - spread / 2
                            ask = price + spread / 2
                            
                            with self._lock:
                                self.latest_prices[trade_symbol] = {'bid': bid, 'ask': ask}
                                callback = self.callbacks.get(trade_symbol)
                            
                            logger.info(f"{trade_symbol}: Updated from trade - bid=${bid:.2f}, ask=${ask:.2f}")
                            
                            # Call callback if registered
                            if callback:
                                try:
                                    ob = self._create_orderbook(bid, ask)
                                    callback(ob)
                                except Exception as e:
                                    logger.error(f"{trade_symbol}: Callback error: {e}")
                
                elif msg_type == "ping":
                    # Respond to ping
                    ws.send(json.dumps({"type": "pong"}))
                    logger.debug("Ping/pong")
                
                else:
                    # Log first unknown message type, then debug level
                    logger.info(f"Message type: {msg_type}, data: {data}")
                
            except Exception as e:
                logger.error(f"Message parsing error: {e}, raw: {message[:200]}")
        
        def on_error(ws, error):
            logger.error(f"WebSocket error: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            logger.info(f"WebSocket closed: {close_status_code} - {close_msg}")
        
        def on_open(ws):
            logger.info(f"WebSocket connection opened")
            # Subscribe to all requested symbols
            with self._lock:
                symbols_to_subscribe = list(self.subscribed_symbols)
            
            for sym in symbols_to_subscribe:
                try:
                    subscribe_msg = json.dumps({"type": "subscribe", "symbol": sym})
                    ws.send(subscribe_msg)
                    logger.info(f"Subscribed to {sym}")
                    time.sleep(0.1)  # Small delay between subscriptions
                except Exception as e:
                    logger.error(f"Failed to subscribe to {sym}: {e}")
        
        # Create WebSocket connection
        self.ws = websocket.WebSocketApp(
            self.ws_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        
        # Run WebSocket (blocks until closed)
        self.ws.run_forever()
    
    def _create_orderbook(self, bid: float, ask: float):
        """Create synthetic OrderBook-like object from bid/ask prices."""
        # Try to import Rust OrderBook if available
        try:
            import rust_connector
            return rust_connector.OrderBook(
                bids=[(bid, 1.0)],  # type: ignore[arg-type]
                asks=[(ask, 1.0)]  # type: ignore[arg-type]
            )
        except Exception:
            # Return dict fallback
            return {
                "bids": [[bid, 1.0]],
                "asks": [[ask, 1.0]]
            }
    
    def set_api_credentials(self, api_key: str, api_secret: str):
        """Update API credentials (for compatibility with auth interface)."""
        self.api_key = api_key
        self.ws_url = f"wss://ws.finnhub.io?token={api_key}"
