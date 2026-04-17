"""Execution module - Broker integration and order management"""

from .broker_adapter import (
    BrokerAdapter,
    Order,
    Position,
    AccountInfo,
    OrderType,
    OrderSide,
    OrderStatus
)

from .mt5_adapter import MT5Adapter
from .oanda_adapter import OANDAAdapter
from .paper_trading import PaperTradingBroker
from .order_manager import OrderManager

__all__ = [
    # Base classes
    'BrokerAdapter',
    'Order',
    'Position',
    'AccountInfo',
    'OrderType',
    'OrderSide',
    'OrderStatus',
    
    # Adapters
    'MT5Adapter',
    'OANDAAdapter',
    'PaperTradingBroker',
    
    # Managers
    'OrderManager',
]

# Factory function for creating broker adapters
def create_broker(broker_type: str, **kwargs) -> BrokerAdapter:
    """
    Factory function to create a broker adapter.
    
    Args:
        broker_type: 'mt5', 'oanda', or 'paper'
        **kwargs: Broker-specific configuration
        
    Returns:
        BrokerAdapter instance
    """
    if broker_type.lower() == 'mt5':
        from .mt5_adapter import MT5Adapter
        return MT5Adapter(
            server=kwargs.get('server'),
            login=kwargs.get('login'),
            password=kwargs.get('password')
        )
    elif broker_type.lower() == 'oanda':
        from .oanda_adapter import OANDAAdapter
        return OANDAAdapter(
            api_key=kwargs.get('api_key'),
            account_id=kwargs.get('account_id'),
            environment=kwargs.get('environment', 'practice')
        )
    elif broker_type.lower() == 'paper':
        from .paper_trading import PaperTradingBroker
        return PaperTradingBroker(
            initial_balance=kwargs.get('initial_balance', 10000.0),
            spread_pips=kwargs.get('spread_pips', 0.2)
        )
    else:
        raise ValueError(f"Unknown broker type: {broker_type}")
