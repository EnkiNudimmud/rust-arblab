"""
Limit Order Book (LOB) Recorder and Analyzer
=============================================

Python wrapper for Rust LOB implementation.
Captures, stores, and analyzes full limit order book data from Binance and other exchanges.
Inspired by: https://github.com/pfei-sa/binance-LOB

Features:
- Multi-level orderbook capture (20+ levels) - Rust implementation
- Orderbook snapshots and diff streams - Rust implementation  
- LOB reconstruction from diffs - Rust implementation
- Real-time LOB analytics (spread, depth, imbalance) - Rust implementation
- LOB visualization and heatmaps - Python implementation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Generator, Any
from collections import deque
import time
import json
from pathlib import Path

# Fallback Python implementations for type checking
from dataclasses import dataclass

@dataclass
class OrderBookLevel:
    price: float
    quantity: float

@dataclass
class OrderBookSnapshot:
    timestamp: str
    symbol: str
    last_update_id: int
    exchange: str
    bids: List[Tuple[float, float]]
    asks: List[Tuple[float, float]]

@dataclass
class OrderBookUpdate:
    timestamp: str
    symbol: str
    first_update_id: int
    final_update_id: int
    bids: List[Tuple[float, float]]
    asks: List[Tuple[float, float]]

@dataclass
class LOBAnalytics:
    timestamp: str
    symbol: str
    best_bid: float = 0.0
    best_ask: float = 0.0
    spread_abs: float = 0.0
    spread_bps: float = 0.0
    mid_price: float = 0.0
    bid_depth_1: float = 0.0
    ask_depth_1: float = 0.0
    bid_depth_5: float = 0.0
    ask_depth_5: float = 0.0
    bid_depth_10: float = 0.0
    ask_depth_10: float = 0.0
    volume_imbalance: float = 0.0
    price_imbalance: float = 0.0
    depth_imbalance_1: float = 0.0
    depth_imbalance_5: float = 0.0
    bid_levels: int = 0
    ask_levels: int = 0
    total_bid_volume: float = 0.0
    total_ask_volume: float = 0.0
    effective_spread_bps: float = 0.0
    market_impact_10k: float = 0.0

def calculate_lob_analytics(snapshot: OrderBookSnapshot) -> LOBAnalytics:
    """Calculate LOB analytics - overridden by Rust if available"""
    raise NotImplementedError("Rust module required")

def apply_orderbook_update(snapshot: OrderBookSnapshot, update: OrderBookUpdate, max_levels: int) -> OrderBookSnapshot:
    """Apply orderbook update - overridden by Rust if available"""
    raise NotImplementedError("Rust module required")

def parse_binance_orderbook(data: dict, symbol: str, exchange: str) -> OrderBookSnapshot:
    """Parse Binance orderbook - overridden by Rust if available"""
    raise NotImplementedError("Rust module required")

# Import Rust implementations (overrides above definitions if available)
try:
    from rust_connector import (  # type: ignore[import-not-found,no-redef,assignment]
        OrderBookLevel,  # type: ignore[assignment,misc]
        OrderBookSnapshot,  # type: ignore[assignment,misc]
        OrderBookUpdate,  # type: ignore[assignment,misc]
        LOBAnalytics,  # type: ignore[assignment,misc]
        calculate_lob_analytics,  # type: ignore[assignment,misc]
        apply_orderbook_update,  # type: ignore[assignment,misc]
        parse_binance_orderbook  # type: ignore[assignment,misc]
    )
    RUST_LOB_AVAILABLE = True
except ImportError:
    RUST_LOB_AVAILABLE = False
    print("Warning: Rust LOB module not available. Install with: maturin develop")


# Helper functions to convert Rust objects to Python dicts
def snapshot_to_dict(snapshot: OrderBookSnapshot) -> dict:
    """Convert OrderBookSnapshot to dictionary"""
    return {
        'timestamp': snapshot.timestamp,
        'symbol': snapshot.symbol,
        'last_update_id': snapshot.last_update_id,
        'exchange': snapshot.exchange,
        'bids': snapshot.bids,  # Already tuples from Rust
        'asks': snapshot.asks
    }


def analytics_to_dict(analytics: LOBAnalytics) -> dict:
    """Convert LOBAnalytics to dictionary"""
    return {
        'timestamp': analytics.timestamp,
        'symbol': analytics.symbol,
        'best_bid': analytics.best_bid,
        'best_ask': analytics.best_ask,
        'spread_abs': analytics.spread_abs,
        'spread_bps': analytics.spread_bps,
        'mid_price': analytics.mid_price,
        'bid_depth_1': analytics.bid_depth_1,
        'ask_depth_1': analytics.ask_depth_1,
        'bid_depth_5': analytics.bid_depth_5,
        'ask_depth_5': analytics.ask_depth_5,
        'bid_depth_10': analytics.bid_depth_10,
        'ask_depth_10': analytics.ask_depth_10,
        'volume_imbalance': analytics.volume_imbalance,
        'price_imbalance': analytics.price_imbalance,
        'depth_imbalance_1': analytics.depth_imbalance_1,
        'depth_imbalance_5': analytics.depth_imbalance_5,
        'bid_levels': analytics.bid_levels,
        'ask_levels': analytics.ask_levels,
        'total_bid_volume': analytics.total_bid_volume,
        'total_ask_volume': analytics.total_ask_volume,
        'effective_spread_bps': analytics.effective_spread_bps,
        'market_impact_10k': analytics.market_impact_10k
    }

class LOBRecorder:
    """
    Records and manages limit order book data
    
    Features:
    - Periodic snapshots (full orderbook)
    - Continuous diff updates
    - Local orderbook reconstruction
    - Analytics calculation
    """
    
    def __init__(
        self,
        symbols: List[str],
        snapshot_interval: int = 3600,  # Seconds between full snapshots
        max_levels: int = 20,            # Number of price levels to track
        storage_path: Optional[str] = None
    ):
        self.symbols = symbols
        self.snapshot_interval = snapshot_interval
        self.max_levels = max_levels
        
        # Storage
        self.storage_path = Path(storage_path) if storage_path else Path("data/lob")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory buffers
        self.snapshots: Dict[str, deque] = {s: deque(maxlen=100) for s in symbols}
        self.updates: Dict[str, deque] = {s: deque(maxlen=1000) for s in symbols}
        self.analytics: Dict[str, deque] = {s: deque(maxlen=1000) for s in symbols}
        
        # Current orderbook state
        self.current_books: Dict[str, OrderBookSnapshot] = {}
        
        # Timing
        self.last_snapshot_time: Dict[str, datetime] = {}
    
    def add_snapshot(self, snapshot: OrderBookSnapshot):
        """Add a full orderbook snapshot"""
        symbol = snapshot.symbol
        self.snapshots[symbol].append(snapshot)
        self.current_books[symbol] = snapshot
        
        # Parse timestamp string to datetime
        snapshot_time = pd.to_datetime(snapshot.timestamp)
        self.last_snapshot_time[symbol] = snapshot_time
        
        # Calculate analytics
        analytics = self.calculate_analytics(snapshot)
        self.analytics[symbol].append(analytics)
        
        # Persist to disk periodically
        self._maybe_persist_snapshot(snapshot)
    
    def add_update(self, update: OrderBookUpdate):
        """Add a differential orderbook update"""
        symbol = update.symbol
        self.updates[symbol].append(update)
        
        # Apply update to current book if we have a base snapshot
        if symbol in self.current_books:
            self.apply_update(symbol, update)
    
    def apply_update(self, symbol: str, update: OrderBookUpdate):
        """
        Apply differential update to current orderbook using Rust implementation
        """
        current = self.current_books.get(symbol)
        if not current:
            return
        
        if RUST_LOB_AVAILABLE:
            # Use Rust implementation for efficient update application
            updated = apply_orderbook_update(current, update, self.max_levels)
            self.current_books[symbol] = updated
            
            # Calculate analytics on updated book
            analytics = calculate_lob_analytics(updated)
            self.analytics[symbol].append(analytics)
        else:
            print("Warning: Rust LOB module not available, skipping update")
    
    def calculate_analytics(self, snapshot: OrderBookSnapshot) -> LOBAnalytics:
        """Calculate comprehensive analytics from orderbook using Rust implementation"""
        if RUST_LOB_AVAILABLE:
            return calculate_lob_analytics(snapshot)
        else:
            print("Warning: Rust LOB module not available, returning empty analytics")
            # Return empty analytics as fallback
            return LOBAnalytics(
                timestamp=datetime.now().isoformat(),
                symbol=snapshot.symbol if hasattr(snapshot, 'symbol') else "",
                best_bid=0, best_ask=0, spread_abs=0, spread_bps=0, mid_price=0,
                bid_depth_1=0, ask_depth_1=0, bid_depth_5=0, ask_depth_5=0,
                bid_depth_10=0, ask_depth_10=0, volume_imbalance=0,
                price_imbalance=0, depth_imbalance_1=0, depth_imbalance_5=0,
                bid_levels=0, ask_levels=0, total_bid_volume=0, total_ask_volume=0,
                effective_spread_bps=0, market_impact_10k=0
            )
    
    def get_current_book(self, symbol: str) -> Optional[OrderBookSnapshot]:
        """Get current orderbook state"""
        return self.current_books.get(symbol)
    
    def get_analytics(self, symbol: str, n: int = 100) -> List[LOBAnalytics]:
        """Get recent analytics"""
        return list(self.analytics[symbol])[-n:]
    
    def _maybe_persist_snapshot(self, snapshot: OrderBookSnapshot):
        """Persist snapshot to disk if interval has passed"""
        symbol = snapshot.symbol
        
        # Save every snapshot_interval seconds
        last_time = self.last_snapshot_time.get(symbol)
        
        # Parse timestamp from ISO string
        snapshot_time = pd.to_datetime(snapshot.timestamp)
        
        if last_time and (snapshot_time - last_time).total_seconds() < self.snapshot_interval:
            return
        
        # Create filename with date
        date_str = snapshot_time.strftime("%Y%m%d")
        filepath = self.storage_path / f"{symbol}_{date_str}_snapshots.jsonl"
        
        # Append to file
        with open(filepath, 'a') as f:
            f.write(json.dumps(snapshot_to_dict(snapshot)) + '\n')
        
        self.last_snapshot_time[symbol] = snapshot_time
    
    def export_to_csv(self, symbol: str, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None) -> pd.DataFrame:
        """Export analytics to DataFrame"""
        analytics_list = self.get_analytics(symbol, n=10000)
        
        if not analytics_list:
            return pd.DataFrame()
        
        # Convert Rust LOBAnalytics objects to dicts
        df = pd.DataFrame([analytics_to_dict(a) for a in analytics_list])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        if start_time:
            filtered = df[df['timestamp'] >= start_time]
            df = filtered if isinstance(filtered, pd.DataFrame) else pd.DataFrame()
        if end_time:
            filtered = df[df['timestamp'] <= end_time]
            df = filtered if isinstance(filtered, pd.DataFrame) else pd.DataFrame()
        
        return df


def parse_binance_orderbook_py(data: dict, symbol: str, exchange: str = "binance") -> OrderBookSnapshot:
    """Parse orderbook from Binance API response using Rust implementation"""
    if RUST_LOB_AVAILABLE:
        # Pass dict directly to Rust (PyO3 handles conversion)
        return parse_binance_orderbook(data, symbol, exchange)
    else:
        print("Warning: Rust LOB module not available")
        # Fallback: return minimal snapshot
        return OrderBookSnapshot(
            timestamp=datetime.now().isoformat(),
            symbol=symbol,
            last_update_id=data.get('lastUpdateId', 0),
            exchange=exchange,
            bids=[(float(b[0]), float(b[1])) for b in data.get('bids', [])][:20],
            asks=[(float(a[0]), float(a[1])) for a in data.get('asks', [])][:20]
        )


def parse_binance_diff_depth(data: dict, symbol: str) -> OrderBookUpdate:
    """Parse differential orderbook update from Binance WebSocket"""
    timestamp_iso = datetime.fromtimestamp(data.get('E', 0) / 1000).isoformat()
    return OrderBookUpdate(
        timestamp=timestamp_iso,
        symbol=symbol,
        first_update_id=data.get('U', 0),
        final_update_id=data.get('u', 0),
        bids=[(float(b[0]), float(b[1])) for b in data.get('b', [])],
        asks=[(float(a[0]), float(a[1])) for a in data.get('a', [])]
    )
