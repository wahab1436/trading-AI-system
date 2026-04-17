"""Risk Engine Module - Position sizing, kill switch, and risk management"""

from .position_sizing import DynamicPositionSizer, PositionSizeResult, KellyOptimizer
from .kill_switch import KillSwitch, KillSwitchReason, KillSwitchEvent, CircuitBreaker
from .risk_limits import RiskEngine

__all__ = [
    # Position Sizing
    'DynamicPositionSizer',
    'PositionSizeResult',
    'KellyOptimizer',
    
    # Kill Switch
    'KillSwitch',
    'KillSwitchReason',
    'KillSwitchEvent',
    'CircuitBreaker',
    
    # Risk Limits
    'RiskEngine',
]

# Module version
__version__ = '1.0.0'
