"""Order management system with risk checks"""

import logging
from typing import Optional, List, Dict
from datetime import datetime
from pathlib import Path

from .broker_adapter import (
    BrokerAdapter, Order, Position, AccountInfo,
    OrderSide, OrderType, OrderStatus
)

logger = logging.getLogger(__name__)


class OrderManager:
    """Manages order placement, modification, and tracking"""
    
    def __init__(
        self,
        broker: BrokerAdapter,
        max_concurrent_trades: int = 3,
        max_daily_trades: int = 20
    ):
        self.broker = broker
        self.max_concurrent_trades = max_concurrent_trades
        self.max_daily_trades = max_daily_trades
        self.daily_trades = 0
        self.daily_reset_time = datetime.utcnow().replace(hour=0, minute=0, second=0)
        
    def place_trade(
        self,
        signal: Dict,
        lot_size: float,
        stop_loss: float,
        take_profit: float
    ) -> Optional[Order]:
        """Place a trade based on signal"""
        
        # Pre-trade checks
        if not self._can_place_trade():
            logger.warning("Cannot place trade - limits reached")
            return None
            
        # Check concurrent positions
        positions = self.broker.get_positions()
        if len(positions) >= self.max_concurrent_trades:
            logger.warning(f"Max concurrent trades reached ({self.max_concurrent_trades})")
            return None
            
        # Determine order side
        side = OrderSide.BUY if signal['direction'] == 1 else OrderSide.SELL
        
        # Create order
        order = Order(
            id="",
            symbol=signal['symbol'],
            side=side,
            type=OrderType.MARKET,
            quantity=lot_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            comment=f"AI_{signal.get('model_version', 'v1')}"
        )
        
        # Place order
        order = self.broker.place_order(order)
        
        if order.status == OrderStatus.FILLED:
            self.daily_trades += 1
            logger.info(f"Trade placed: {order.id} - {side.value} {lot_size} {signal['symbol']}")
            
        return order
        
    def modify_trade(self, order_id: str, stop_loss: float = None, take_profit: float = None) -> bool:
        """Modify existing trade SL/TP"""
        return self.broker.modify_order(order_id, stop_loss=stop_loss, take_profit=take_profit)
        
    def close_trade(self, position_id: str) -> bool:
        """Close a specific position"""
        return self.broker.close_position(position_id)
        
    def close_all_trades(self) -> int:
        """Close all open positions"""
        positions = self.broker.get_positions()
        closed = 0
        
        for pos in positions:
            if self.broker.close_position(pos.symbol):  # Need position ID
                closed += 1
                
        logger.info(f"Closed {closed} positions")
        return closed
        
    def get_open_positions(self) -> List[Position]:
        """Get all open positions"""
        return self.broker.get_positions()
        
    def get_daily_pnl(self) -> float:
        """Calculate daily P&L"""
        positions = self.broker.get_positions()
        return sum(pos.unrealized_pnl for pos in positions)
        
    def _can_place_trade(self) -> bool:
        """Check if we can place a new trade"""
        
        # Check daily reset
        now = datetime.utcnow()
        if now.date() > self.daily_reset_time.date():
            self.daily_reset_time = now.replace(hour=0, minute=0, second=0)
            self.daily_trades = 0
            
        # Check daily limit
        if self.daily_trades >= self.max_daily_trades:
            logger.warning("Daily trade limit reached")
            return False
            
        # Check connection
        if not self.broker.is_connected():
            logger.error("Broker not connected")
            return False
            
        return True
        
    def get_status(self) -> Dict:
        """Get order manager status"""
        return {
            'connected': self.broker.is_connected(),
            'daily_trades': self.daily_trades,
            'max_daily_trades': self.max_daily_trades,
            'open_positions': len(self.broker.get_positions()),
            'max_concurrent': self.max_concurrent_trades
        }
