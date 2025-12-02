"""
Virtual Portfolio Management System
====================================

Manages virtual trading portfolios with the following features:
- Trade execution and position tracking
- P&L calculation and risk metrics
- Portfolio persistence and merging
- Multi-asset support (stocks, crypto, ETFs, options)
- Integration with lab portfolios
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents a trading position"""
    symbol: str
    asset_type: str  # 'stock', 'crypto', 'etf', 'option', 'future'
    quantity: float
    entry_price: float
    entry_time: datetime
    current_price: float = 0.0
    
    @property
    def market_value(self) -> float:
        """Current market value of position"""
        return self.quantity * self.current_price
    
    @property
    def cost_basis(self) -> float:
        """Original cost of position"""
        return self.quantity * self.entry_price
    
    @property
    def unrealized_pnl(self) -> float:
        """Unrealized profit/loss"""
        return self.market_value - self.cost_basis
    
    @property
    def unrealized_pnl_pct(self) -> float:
        """Unrealized P&L percentage"""
        if self.cost_basis == 0:
            return 0.0
        return (self.unrealized_pnl / abs(self.cost_basis)) * 100


@dataclass
class Trade:
    """Represents a completed trade"""
    timestamp: datetime
    symbol: str
    asset_type: str
    action: str  # 'BUY' or 'SELL'
    quantity: float
    price: float
    value: float
    commission: float = 0.0
    strategy: str = "manual"
    notes: str = ""
    
    def to_dict(self):
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'asset_type': self.asset_type,
            'action': self.action,
            'quantity': self.quantity,
            'price': self.price,
            'value': self.value,
            'commission': self.commission,
            'strategy': self.strategy,
            'notes': self.notes
        }


@dataclass
class PortfolioSnapshot:
    """Portfolio state at a point in time"""
    timestamp: datetime
    total_value: float
    cash: float
    positions_value: float
    unrealized_pnl: float
    realized_pnl: float = 0.0
    positions: Dict[str, Position] = field(default_factory=dict)


class VirtualPortfolio:
    """
    Virtual trading portfolio with position tracking and P&L calculation.
    
    Features:
    - Multi-asset support (stocks, crypto, ETFs, options)
    - Automatic P&L tracking
    - Position management
    - Trade history
    - Persistence to disk
    - Merging with lab portfolios
    """
    
    def __init__(
        self,
        name: str,
        initial_cash: float = 100000.0,
        commission_rate: float = 0.001,
        storage_path: str = "data/portfolios"
    ):
        self.name = name
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.commission_rate = commission_rate
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Portfolio state
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.snapshots: List[PortfolioSnapshot] = []
        self.realized_pnl = 0.0
        
        # Load existing portfolio if available
        self.load()
    
    def execute_trade(
        self,
        symbol: str,
        asset_type: str,
        action: str,
        quantity: float,
        price: float,
        strategy: str = "manual",
        notes: str = ""
    ) -> Optional[Trade]:
        """
        Execute a trade and update portfolio.
        
        Args:
            symbol: Asset symbol
            asset_type: Type of asset ('stock', 'crypto', 'etf', 'option')
            action: 'BUY' or 'SELL'
            quantity: Number of units
            price: Execution price
            strategy: Strategy that triggered the trade
            notes: Additional notes
            
        Returns:
            Trade object if successful, None otherwise
        """
        action = action.upper()
        value = quantity * price
        commission = value * self.commission_rate
        
        # Validate action
        if action not in ['BUY', 'SELL']:
            logger.error(f"Invalid action: {action}")
            return None
        
        # Execute based on action
        if action == 'BUY':
            total_cost = value + commission
            if total_cost > self.cash:
                logger.warning(f"Insufficient cash: need {total_cost}, have {self.cash}")
                return None
            
            # Update cash
            self.cash -= total_cost
            
            # Add or update position
            if symbol in self.positions:
                pos = self.positions[symbol]
                # Average down/up
                new_quantity = pos.quantity + quantity
                new_cost = pos.cost_basis + value
                pos.quantity = new_quantity
                pos.entry_price = new_cost / new_quantity if new_quantity > 0 else price
            else:
                self.positions[symbol] = Position(
                    symbol=symbol,
                    asset_type=asset_type,
                    quantity=quantity,
                    entry_price=price,
                    entry_time=datetime.now(),
                    current_price=price
                )
        
        elif action == 'SELL':
            # Check if position exists
            if symbol not in self.positions:
                logger.warning(f"No position to sell: {symbol}")
                return None
            
            pos = self.positions[symbol]
            
            # Check quantity
            if quantity > pos.quantity:
                logger.warning(f"Insufficient quantity: need {quantity}, have {pos.quantity}")
                return None
            
            # Calculate realized P&L
            cost_basis = pos.entry_price * quantity
            pnl = value - cost_basis - commission
            self.realized_pnl += pnl
            
            # Update cash
            self.cash += value - commission
            
            # Update or remove position
            pos.quantity -= quantity
            if pos.quantity < 1e-8:  # Close position
                del self.positions[symbol]
        
        # Record trade
        trade = Trade(
            timestamp=datetime.now(),
            symbol=symbol,
            asset_type=asset_type,
            action=action,
            quantity=quantity,
            price=price,
            value=value,
            commission=commission,
            strategy=strategy,
            notes=notes
        )
        
        self.trades.append(trade)
        
        # Take snapshot
        self.take_snapshot()
        
        # Save portfolio
        self.save()
        
        logger.info(f"Trade executed: {action} {quantity} {symbol} @ {price}")
        return trade
    
    def update_prices(self, prices: Dict[str, float]):
        """Update current prices for all positions"""
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].current_price = price
    
    def take_snapshot(self):
        """Take a snapshot of current portfolio state"""
        positions_value = sum(pos.market_value for pos in self.positions.values())
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        
        snapshot = PortfolioSnapshot(
            timestamp=datetime.now(),
            total_value=self.cash + positions_value,
            cash=self.cash,
            positions_value=positions_value,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=self.realized_pnl,
            positions=self.positions.copy()
        )
        
        self.snapshots.append(snapshot)
        
        # Keep last 10000 snapshots
        if len(self.snapshots) > 10000:
            self.snapshots = self.snapshots[-10000:]
    
    @property
    def total_value(self) -> float:
        """Total portfolio value (cash + positions)"""
        positions_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash + positions_value
    
    @property
    def total_pnl(self) -> float:
        """Total P&L (realized + unrealized)"""
        unrealized = sum(pos.unrealized_pnl for pos in self.positions.values())
        return self.realized_pnl + unrealized
    
    @property
    def total_pnl_pct(self) -> float:
        """Total P&L percentage"""
        if self.initial_cash == 0:
            return 0.0
        return (self.total_pnl / self.initial_cash) * 100
    
    def get_metrics(self) -> Dict:
        """Calculate portfolio metrics"""
        positions_value = sum(pos.market_value for pos in self.positions.values())
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        
        return {
            'total_value': self.total_value,
            'cash': self.cash,
            'positions_value': positions_value,
            'initial_capital': self.initial_cash,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': unrealized_pnl,
            'total_pnl': self.total_pnl,
            'total_pnl_pct': self.total_pnl_pct,
            'n_positions': len(self.positions),
            'n_trades': len(self.trades)
        }
    
    def get_positions_df(self) -> pd.DataFrame:
        """Get positions as DataFrame"""
        if not self.positions:
            return pd.DataFrame()
        
        data = []
        for symbol, pos in self.positions.items():
            data.append({
                'Symbol': symbol,
                'Type': pos.asset_type,
                'Quantity': pos.quantity,
                'Entry Price': pos.entry_price,
                'Current Price': pos.current_price,
                'Market Value': pos.market_value,
                'Cost Basis': pos.cost_basis,
                'Unrealized P&L': pos.unrealized_pnl,
                'Unrealized P&L %': pos.unrealized_pnl_pct
            })
        
        return pd.DataFrame(data)
    
    def get_trades_df(self) -> pd.DataFrame:
        """Get trade history as DataFrame"""
        if not self.trades:
            return pd.DataFrame()
        
        return pd.DataFrame([trade.to_dict() for trade in self.trades])
    
    def save(self):
        """Save portfolio to disk"""
        filepath = self.storage_path / f"{self.name}.json"
        
        data = {
            'name': self.name,
            'initial_cash': self.initial_cash,
            'cash': self.cash,
            'commission_rate': self.commission_rate,
            'realized_pnl': self.realized_pnl,
            'positions': {
                symbol: {
                    'symbol': pos.symbol,
                    'asset_type': pos.asset_type,
                    'quantity': pos.quantity,
                    'entry_price': pos.entry_price,
                    'entry_time': pos.entry_time.isoformat(),
                    'current_price': pos.current_price
                }
                for symbol, pos in self.positions.items()
            },
            'trades': [trade.to_dict() for trade in self.trades[-1000:]],  # Keep last 1000 trades
            'last_updated': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Portfolio saved to {filepath}")
    
    def load(self):
        """Load portfolio from disk"""
        filepath = self.storage_path / f"{self.name}.json"
        
        if not filepath.exists():
            logger.info(f"No existing portfolio found at {filepath}")
            return
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.initial_cash = data.get('initial_cash', self.initial_cash)
            self.cash = data.get('cash', self.cash)
            self.commission_rate = data.get('commission_rate', self.commission_rate)
            self.realized_pnl = data.get('realized_pnl', 0.0)
            
            # Load positions
            self.positions = {}
            for symbol, pos_data in data.get('positions', {}).items():
                self.positions[symbol] = Position(
                    symbol=pos_data['symbol'],
                    asset_type=pos_data['asset_type'],
                    quantity=pos_data['quantity'],
                    entry_price=pos_data['entry_price'],
                    entry_time=datetime.fromisoformat(pos_data['entry_time']),
                    current_price=pos_data.get('current_price', pos_data['entry_price'])
                )
            
            # Load trades
            self.trades = []
            for trade_data in data.get('trades', []):
                self.trades.append(Trade(
                    timestamp=datetime.fromisoformat(trade_data['timestamp']),
                    symbol=trade_data['symbol'],
                    asset_type=trade_data['asset_type'],
                    action=trade_data['action'],
                    quantity=trade_data['quantity'],
                    price=trade_data['price'],
                    value=trade_data['value'],
                    commission=trade_data.get('commission', 0.0),
                    strategy=trade_data.get('strategy', 'manual'),
                    notes=trade_data.get('notes', '')
                ))
            
            logger.info(f"Portfolio loaded from {filepath}")
        
        except Exception as e:
            logger.error(f"Error loading portfolio: {e}")
    
    def merge_with_lab_portfolio(self, lab_portfolio: Dict) -> bool:
        """
        Merge with a portfolio created in labs.
        
        Args:
            lab_portfolio: Portfolio dict from labs with positions and cash
            
        Returns:
            True if successful
        """
        try:
            # Add cash
            if 'cash' in lab_portfolio:
                self.cash += lab_portfolio['cash']
            
            # Add positions
            if 'positions' in lab_portfolio:
                for symbol, pos_data in lab_portfolio['positions'].items():
                    if symbol in self.positions:
                        # Average positions
                        existing = self.positions[symbol]
                        new_qty = pos_data.get('quantity', 0)
                        new_price = pos_data.get('entry_price', 0)
                        
                        total_qty = existing.quantity + new_qty
                        avg_price = (existing.entry_price * existing.quantity + new_price * new_qty) / total_qty
                        
                        existing.quantity = total_qty
                        existing.entry_price = avg_price
                    else:
                        # Add new position
                        self.positions[symbol] = Position(
                            symbol=symbol,
                            asset_type=pos_data.get('asset_type', 'stock'),
                            quantity=pos_data.get('quantity', 0),
                            entry_price=pos_data.get('entry_price', 0),
                            entry_time=datetime.now(),
                            current_price=pos_data.get('current_price', pos_data.get('entry_price', 0))
                        )
            
            self.save()
            logger.info(f"Merged lab portfolio into {self.name}")
            return True
        
        except Exception as e:
            logger.error(f"Error merging portfolios: {e}")
            return False
    
    def export_to_lab_format(self) -> Dict:
        """Export portfolio in lab-compatible format"""
        return {
            'name': self.name,
            'cash': self.cash,
            'initial_capital': self.initial_cash,
            'positions': {
                symbol: {
                    'quantity': pos.quantity,
                    'entry_price': pos.entry_price,
                    'current_price': pos.current_price,
                    'asset_type': pos.asset_type
                }
                for symbol, pos in self.positions.items()
            },
            'history': [
                {
                    'timestamp': snapshot.timestamp.isoformat(),
                    'total_value': snapshot.total_value,
                    'realized_pnl': snapshot.realized_pnl,
                    'unrealized_pnl': snapshot.unrealized_pnl
                }
                for snapshot in self.snapshots[-100:]  # Last 100 snapshots
            ]
        }


def list_portfolios(storage_path: str = "data/portfolios") -> List[str]:
    """List available portfolios"""
    path = Path(storage_path)
    if not path.exists():
        return []
    
    return [f.stem for f in path.glob("*.json")]


def load_portfolio(name: str, storage_path: str = "data/portfolios") -> Optional[VirtualPortfolio]:
    """Load a portfolio by name"""
    portfolio = VirtualPortfolio(name=name, storage_path=storage_path)
    return portfolio
