"""Abstract broker adapter interface"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict
from datetime import datetime
from enum import Enum


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class Order:
    """Order data structure"""
    id: str
    symbol: str
    side: OrderSide
    type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_price: Optional[float] = None
    created_at: datetime = None
    filled_at: Optional[datetime] = None
    comment: str = ""
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


@dataclass
class Position:
    """Position data structure"""
    symbol: str
    side: OrderSide
    quantity: float
    open_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    open_time: datetime = None
    
    def __post_init__(self):
        if self.open_time is None:
            self.open_time = datetime.utcnow()


@dataclass
class AccountInfo:
    """Account information"""
    balance: float
    equity: float
    margin: float
    free_margin: float
    margin_level: float
    currency: str = "USD"
    unrealized_pnl: float = 0.0
    realized_pnl_today: float = 0.0


class BrokerAdapter(ABC):
    """Abstract base class for broker integrations"""
    
    @abstractmethod
    def connect(self) -> bool:
        """Connect to broker"""
        pass
        
    @abstractmethod
    def disconnect(self) -> bool:
        """Disconnect from broker"""
        pass
        
    @abstractmethod
    def is_connected(self) -> bool:
        """Check connection status"""
        pass
        
    @abstractmethod
    def place_order(self, order: Order) -> Order:
        """Place an order"""
        pass
        
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        pass
        
    @abstractmethod
    def modify_order(self, order_id: str, **kwargs) -> bool:
        """Modify an existing order"""
        pass
        
    @abstractmethod
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order details"""
        pass
        
    @abstractmethod
    def get_positions(self) -> List[Position]:
        """Get all open positions"""
        pass
        
    @abstractmethod
    def close_position(self, position_id: str) -> bool:
        """Close a position"""
        pass
        
    @abstractmethod
    def get_account_info(self) -> AccountInfo:
        """Get account information"""
        pass
        
    @abstractmethod
    def get_ticks(self, symbol: str, count: int = 100) -> List[Dict]:
        """Get recent ticks"""
        pass
