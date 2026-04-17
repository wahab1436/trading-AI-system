"""Paper trading simulation for testing before live deployment"""

import logging
from datetime import datetime
from typing import Optional, List, Dict
from dataclasses import dataclass, field

from .broker_adapter import (
    Order, Position, AccountInfo,
    OrderSide, OrderType, OrderStatus
)

logger = logging.getLogger(__name__)


@dataclass
class PaperAccount:
    """Simulated trading account"""
    balance: float = 10000.0
    equity: float = 10000.0
    currency: str = "USD"
    positions: List[Dict] = field(default_factory=list)
    orders: List[Dict] = field(default_factory=list)
    trades: List[Dict] = field(default_factory=list)
    
    def __post_init__(self):
        self.starting_balance = self.balance
        self.daily_pnl = 0.0
        self.daily_reset = datetime.utcnow().date()


class PaperTradingBroker:
    """Simulated broker for paper trading"""
    
    def __init__(self, initial_balance: float = 10000.0, spread_pips: float = 0.2):
        self.account = PaperAccount(balance=initial_balance)
        self.spread_pips = spread_pips
        self.slippage_pips = 0.1
        self.order_counter = 0
        
        # Symbol specs
        self.symbol_specs = {
            "XAUUSD": {"pip_value": 10.0, "min_lot": 0.01, "max_lot": 10.0, "step": 0.01},
            "EURUSD": {"pip_value": 10.0, "min_lot": 0.01, "max_lot": 100.0, "step": 0.01}
        }
        
    def connect(self) -> bool:
        """Simulate connection"""
        logger.info("Paper trading connected")
        return True
        
    def disconnect(self) -> bool:
        """Simulate disconnect"""
        logger.info("Paper trading disconnected")
        return True
        
    def is_connected(self) -> bool:
        """Always connected for paper trading"""
        return True
        
    def place_order(self, order: Order) -> Order:
        """Simulate order placement"""
        
        self.order_counter += 1
        order.id = f"PAPER_{self.order_counter}"
        order.created_at = datetime.utcnow()
        
        # Get current market price
        current_price = self._get_market_price(order.symbol)
        
        if order.type == OrderType.MARKET:
            # Apply spread and slippage
            if order.side == OrderSide.BUY:
                order.average_price = current_price['ask'] + self.slippage_pips * 0.0001
            else:
                order.average_price = current_price['bid'] - self.slippage_pips * 0.0001
                
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.filled_at = datetime.utcnow()
            
            # Open position
            self._open_position(order)
            
        else:
            # Pending order
            order.status = OrderStatus.PENDING
            self.account.orders.append({
                'order': order,
                'created_at': datetime.utcnow()
            })
            
        return order
        
    def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order"""
        for i, entry in enumerate(self.account.orders):
            if entry['order'].id == order_id:
                self.account.orders.pop(i)
                logger.info(f"Cancelled order {order_id}")
                return True
        return False
        
    def modify_order(self, order_id: str, **kwargs) -> bool:
        """Modify pending order"""
        for entry in self.account.orders:
            if entry['order'].id == order_id:
                if 'stop_loss' in kwargs:
                    entry['order'].stop_loss = kwargs['stop_loss']
                if 'take_profit' in kwargs:
                    entry['order'].take_profit = kwargs['take_profit']
                return True
        return False
        
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID"""
        for entry in self.account.orders:
            if entry['order'].id == order_id:
                return entry['order']
        return None
        
    def get_positions(self) -> List[Position]:
        """Get all open positions"""
        positions = []
        
        for pos_data in self.account.positions:
            # Calculate unrealized P&L
            current_price = self._get_market_price(pos_data['symbol'])
            
            if pos_data['side'] == OrderSide.BUY:
                unrealized_pnl = (current_price['bid'] - pos_data['open_price']) * pos_data['quantity'] * 100
            else:
                unrealized_pnl = (pos_data['open_price'] - current_price['ask']) * pos_data['quantity'] * 100
                
            positions.append(Position(
                symbol=pos_data['symbol'],
                side=pos_data['side'],
                quantity=pos_data['quantity'],
                open_price=pos_data['open_price'],
                current_price=current_price['bid'] if pos_data['side'] == OrderSide.BUY else current_price['ask'],
                unrealized_pnl=unrealized_pnl,
                realized_pnl=0,
                stop_loss=pos_data.get('stop_loss'),
                take_profit=pos_data.get('take_profit'),
                open_time=pos_data['open_time']
            ))
            
        return positions
        
    def close_position(self, position_id: str) -> bool:
        """Close a position"""
        for i, pos in enumerate(self.account.positions):
            if pos.get('id') == position_id:
                # Calculate realized P&L
                current_price = self._get_market_price(pos['symbol'])
                
                if pos['side'] == OrderSide.BUY:
                    pnl = (current_price['bid'] - pos['open_price']) * pos['quantity'] * 100
                else:
                    pnl = (pos['open_price'] - current_price['ask']) * pos['quantity'] * 100
                    
                # Update balance
                self.account.balance += pnl
                self.account.equity = self.account.balance
                
                # Record trade
                self.account.trades.append({
                    'id': position_id,
                    'symbol': pos['symbol'],
                    'side': pos['side'],
                    'quantity': pos['quantity'],
                    'open_price': pos['open_price'],
                    'close_price': current_price['bid'] if pos['side'] == OrderSide.BUY else current_price['ask'],
                    'pnl': pnl,
                    'open_time': pos['open_time'],
                    'close_time': datetime.utcnow()
                })
                
                # Remove position
                self.account.positions.pop(i)
                
                logger.info(f"Closed position {position_id} with P&L: ${pnl:.2f}")
                return True
                
        return False
        
    def get_account_info(self) -> AccountInfo:
        """Get simulated account info"""
        
        # Update equity with unrealized P&L
        total_unrealized = 0
        for pos in self.account.positions:
            current_price = self._get_market_price(pos['symbol'])
            if pos['side'] == OrderSide.BUY:
                total_unrealized += (current_price['bid'] - pos['open_price']) * pos['quantity'] * 100
            else:
                total_unrealized += (pos['open_price'] - current_price['ask']) * pos['quantity'] * 100
                
        equity = self.account.balance + total_unrealized
        
        return AccountInfo(
            balance=self.account.balance,
            equity=equity,
            margin=0,  # Paper trading doesn't track margin
            free_margin=equity,
            margin_level=100 if equity > 0 else 0,
            currency=self.account.currency,
            unrealized_pnl=total_unrealized,
            realized_pnl_today=self.account.balance - self.account.starting_balance
        )
        
    def get_ticks(self, symbol: str, count: int = 100) -> List[Dict]:
        """Simulate ticks"""
        return [{'bid': 2000, 'ask': 2000.2, 'time': datetime.now()}] * count
        
    def _get_market_price(self, symbol: str) -> Dict:
        """Get simulated market price"""
        # Simplified - in reality would fetch from actual feed
        base_price = 2000.0 if symbol == "XAUUSD" else 1.1000
        return {
            'bid': base_price,
            'ask': base_price + self.spread_pips * 0.0001
        }
        
    def _open_position(self, order: Order):
        """Open a position from filled order"""
        
        self.account.positions.append({
            'id': order.id,
            'symbol': order.symbol,
            'side': order.side,
            'quantity': order.quantity,
            'open_price': order.average_price,
            'stop_loss': order.stop_loss,
            'take_profit': order.take_profit,
            'open_time': order.filled_at
        })
        
        logger.info(f"Opened position: {order.side.value} {order.quantity} {order.symbol} @ {order.average_price}")
        
    def get_performance_summary(self) -> Dict:
        """Get paper trading performance summary"""
        
        total_trades = len(self.account.trades)
        winning_trades = [t for t in self.account.trades if t['pnl'] > 0]
        losing_trades = [t for t in self.account.trades if t['pnl'] < 0]
        
        total_pnl = sum(t['pnl'] for t in self.account.trades)
        
        return {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / total_trades if total_trades > 0 else 0,
            'total_pnl': total_pnl,
            'net_profit': total_pnl,
            'average_win': sum(t['pnl'] for t in winning_trades) / len(winning_trades) if winning_trades else 0,
            'average_loss': sum(t['pnl'] for t in losing_trades) / len(losing_trades) if losing_trades else 0,
            'profit_factor': abs(sum(t['pnl'] for t in winning_trades) / sum(t['pnl'] for t in losing_trades)) if losing_trades else 0,
            'current_balance': self.account.balance,
            'return_pct': ((self.account.balance - self.account.starting_balance) / self.account.starting_balance) * 100
        }
