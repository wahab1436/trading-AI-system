"""MetaTrader 5 broker adapter implementation"""

import logging
from typing import Optional, List, Dict
from datetime import datetime
import time

from .broker_adapter import (
    BrokerAdapter, Order, Position, AccountInfo,
    OrderSide, OrderType, OrderStatus
)

logger = logging.getLogger(__name__)


class MT5Adapter(BrokerAdapter):
    """MetaTrader 5 implementation"""
    
    def __init__(self, server: str = None, login: int = None, password: str = None):
        self.server = server
        self.login = login
        self.password = password
        self._connected = False
        self.mt5 = None
        
    def connect(self) -> bool:
        """Connect to MT5 terminal"""
        try:
            import MetaTrader5 as mt5
            
            if not mt5.initialize():
                logger.error("MT5 initialization failed")
                return False
                
            # Login if credentials provided
            if self.login and self.password:
                authorized = mt5.login(self.login, password=self.password, server=self.server)
                if not authorized:
                    logger.error(f"MT5 login failed: {mt5.last_error()}")
                    mt5.shutdown()
                    return False
                    
            self.mt5 = mt5
            self._connected = True
            logger.info(f"Connected to MT5 - Account: {self.get_account_info().balance}")
            return True
            
        except ImportError:
            logger.error("MetaTrader5 package not installed")
            return False
        except Exception as e:
            logger.error(f"MT5 connection error: {e}")
            return False
            
    def disconnect(self) -> bool:
        """Disconnect from MT5"""
        if self.mt5:
            self.mt5.shutdown()
            self._connected = False
            logger.info("Disconnected from MT5")
        return True
        
    def is_connected(self) -> bool:
        """Check connection status"""
        return self._connected and self.mt5 is not None
        
    def place_order(self, order: Order) -> Order:
        """Place an order on MT5"""
        
        if not self.is_connected():
            logger.error("Not connected to MT5")
            order.status = OrderStatus.REJECTED
            return order
            
        # Map order type
        order_type_map = {
            (OrderSide.BUY, OrderType.MARKET): self.mt5.ORDER_TYPE_BUY,
            (OrderSide.SELL, OrderType.MARKET): self.mt5.ORDER_TYPE_SELL,
            (OrderSide.BUY, OrderType.LIMIT): self.mt5.ORDER_TYPE_BUY_LIMIT,
            (OrderSide.SELL, OrderType.LIMIT): self.mt5.ORDER_TYPE_SELL_LIMIT,
            (OrderSide.BUY, OrderType.STOP): self.mt5.ORDER_TYPE_BUY_STOP,
            (OrderSide.SELL, OrderType.STOP): self.mt5.ORDER_TYPE_SELL_STOP,
        }
        
        order_type = order_type_map.get((order.side, order.type))
        if order_type is None:
            logger.error(f"Unsupported order type: {order.side}/{order.type}")
            order.status = OrderStatus.REJECTED
            return order
            
        # Prepare request
        request = {
            "action": self.mt5.TRADE_ACTION_DEAL if order.type == OrderType.MARKET else self.mt5.TRADE_ACTION_PENDING,
            "symbol": order.symbol,
            "volume": order.quantity,
            "type": order_type,
            "deviation": 20,
            "comment": order.comment or "AI_Trade",
        }
        
        if order.type != OrderType.MARKET:
            request["price"] = order.price
            
        if order.stop_loss:
            request["sl"] = order.stop_loss
            
        if order.take_profit:
            request["tp"] = order.take_profit
            
        # Send order
        result = self.mt5.order_send(request)
        
        if result.retcode != self.mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order failed: {result.comment} (code: {result.retcode})")
            order.status = OrderStatus.REJECTED
            return order
            
        # Update order with result
        order.id = str(result.order)
        order.status = OrderStatus.FILLED if order.type == OrderType.MARKET else OrderStatus.PENDING
        order.filled_quantity = result.volume if order.type == OrderType.MARKET else 0
        order.average_price = result.price if order.type == OrderType.MARKET else None
        order.filled_at = datetime.utcnow() if order.type == OrderType.MARKET else None
        
        logger.info(f"Order placed: {order.id} - {order.side} {order.quantity} {order.symbol}")
        return order
        
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order"""
        
        if not self.is_connected():
            return False
            
        request = {
            "action": self.mt5.TRADE_ACTION_REMOVE,
            "order": int(order_id)
        }
        
        result = self.mt5.order_send(request)
        success = result.retcode == self.mt5.TRADE_RETCODE_DONE
        
        if success:
            logger.info(f"Order {order_id} cancelled")
        else:
            logger.error(f"Failed to cancel order {order_id}: {result.comment}")
            
        return success
        
    def modify_order(self, order_id: str, **kwargs) -> bool:
        """Modify pending order (SL/TP)"""
        
        if not self.is_connected():
            return False
            
        request = {
            "action": self.mt5.TRADE_ACTION_SLTP,
            "order": int(order_id),
            "sl": kwargs.get('stop_loss', 0),
            "tp": kwargs.get('take_profit', 0)
        }
        
        result = self.mt5.order_send(request)
        return result.retcode == self.mt5.TRADE_RETCODE_DONE
        
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order details"""
        
        if not self.is_connected():
            return None
            
        orders = self.mt5.orders_get(order=int(order_id))
        if not orders:
            return None
            
        mt5_order = orders[0]
        
        return Order(
            id=str(mt5_order.order),
            symbol=mt5_order.symbol,
            side=OrderSide.BUY if mt5_order.type in [0, 2] else OrderSide.SELL,
            type=self._map_mt5_order_type(mt5_order.type),
            quantity=mt5_order.volume_current,
            price=mt5_order.price_open,
            stop_loss=mt5_order.sl,
            take_profit=mt5_order.tp,
            status=self._map_mt5_order_state(mt5_order.state),
            filled_quantity=mt5_order.volume_done,
            average_price=mt5_order.price_current,
            created_at=datetime.fromtimestamp(mt5_order.time_setup),
            comment=mt5_order.comment
        )
        
    def get_positions(self) -> List[Position]:
        """Get all open positions"""
        
        if not self.is_connected():
            return []
            
        positions = self.mt5.positions_get()
        if not positions:
            return []
            
        result = []
        for pos in positions:
            result.append(Position(
                symbol=pos.symbol,
                side=OrderSide.BUY if pos.type == 0 else OrderSide.SELL,
                quantity=pos.volume,
                open_price=pos.price_open,
                current_price=pos.price_current,
                unrealized_pnl=pos.profit,
                realized_pnl=0,
                stop_loss=pos.sl,
                take_profit=pos.tp,
                open_time=datetime.fromtimestamp(pos.time)
            ))
            
        return result
        
    def close_position(self, position_id: str) -> bool:
        """Close a position"""
        
        if not self.is_connected():
            return False
            
        # Get position
        positions = self.mt5.positions_get(ticket=int(position_id))
        if not positions:
            return False
            
        pos = positions[0]
        
        # Create opposite order
        close_side = self.mt5.ORDER_TYPE_SELL if pos.type == 0 else self.mt5.ORDER_TYPE_BUY
        
        request = {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": close_side,
            "position": int(position_id),
            "deviation": 20
        }
        
        result = self.mt5.order_send(request)
        success = result.retcode == self.mt5.TRADE_RETCODE_DONE
        
        if success:
            logger.info(f"Position {position_id} closed")
            
        return success
        
    def get_account_info(self) -> AccountInfo:
        """Get account information"""
        
        if not self.is_connected():
            return AccountInfo(balance=0, equity=0, margin=0, free_margin=0, margin_level=0)
            
        account_info = self.mt5.account_info()
        
        return AccountInfo(
            balance=account_info.balance,
            equity=account_info.equity,
            margin=account_info.margin,
            free_margin=account_info.margin_free,
            margin_level=account_info.margin_level,
            currency=account_info.currency,
            unrealized_pnl=account_info.unrealized_pnl,
            realized_pnl_today=account_info.profit
        )
        
    def get_ticks(self, symbol: str, count: int = 100) -> List[Dict]:
        """Get recent ticks"""
        
        if not self.is_connected():
            return []
            
        ticks = self.mt5.copy_ticks_from(symbol, datetime.now(), count, self.mt5.COPY_TICKS_ALL)
        
        if ticks is None:
            return []
            
        return [{
            'time': tick[0],
            'bid': tick[1],
            'ask': tick[2],
            'last': tick[3],
            'volume': tick[4]
        } for tick in ticks]
        
    def _map_mt5_order_type(self, mt5_type: int) -> OrderType:
        """Map MT5 order type to internal enum"""
        mapping = {
            0: OrderType.MARKET,  # BUY
            1: OrderType.MARKET,  # SELL
            2: OrderType.LIMIT,   # BUY LIMIT
            3: OrderType.LIMIT,   # SELL LIMIT
            4: OrderType.STOP,    # BUY STOP
            5: OrderType.STOP     # SELL STOP
        }
        return mapping.get(mt5_type, OrderType.MARKET)
        
    def _map_mt5_order_state(self, state: int) -> OrderStatus:
        """Map MT5 order state to internal enum"""
        mapping = {
            0: OrderStatus.PENDING,   # ORDER_STATE_STARTED
            1: OrderStatus.PENDING,   # ORDER_STATE_PLACED
            2: OrderStatus.PENDING,   # ORDER_STATE_PARTIAL
            3: OrderStatus.FILLED,    # ORDER_STATE_FILLED
            4: OrderStatus.CANCELLED, # ORDER_STATE_CANCELLED
            5: OrderStatus.REJECTED   # ORDER_STATE_REJECTED
        }
        return mapping.get(state, OrderStatus.PENDING)
