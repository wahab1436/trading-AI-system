"""Realistic execution model with slippage, spread, and commissions"""

import numpy as np
from datetime import datetime, time
from typing import Dict, Tuple, Optional


class CostModel:
    """Realistic trading costs simulation"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Spread configuration (in pips)
        self.spreads = {
            'XAUUSD': {'base': 0.2, 'volatility_mult': 1.5, 'session_mult': self._session_multiplier},
            'EURUSD': {'base': 0.1, 'volatility_mult': 1.3, 'session_mult': self._session_multiplier}
        }
        
        # Slippage model
        self.slippage_mean = self.config.get('slippage_mean', 0.1)  # pips
        self.slippage_std = self.config.get('slippage_std', 0.2)   # pips
        
        # Commission
        self.commission_per_lot = self.config.get('commission_per_lot', 3.5)  # USD per round-turn lot
        
        # Swap rates (overnight financing)
        self.swap_rates = {
            'XAUUSD': {'long': -1.5, 'short': -1.5},  # USD per lot per day
            'EURUSD': {'long': -0.5, 'short': -0.5}
        }
        
    def get_spread(self, symbol: str, timestamp: datetime, volatility: float) -> float:
        """Calculate spread in pips"""
        
        spread_config = self.spreads.get(symbol, {'base': 0.2})
        base_spread = spread_config['base']
        
        # Adjust for volatility
        vol_mult = spread_config.get('volatility_mult', 1.0)
        spread = base_spread * (1 + (volatility - 1) * vol_mult * 0.5)
        
        # Adjust for trading session
        session_mult = spread_config.get('session_mult', lambda _: 1.0)(timestamp)
        spread *= session_mult
        
        return max(0.1, min(spread, 1.0))  # Cap between 0.1 and 1.0 pips
        
    def get_slippage(self, order_type: str, volatility: float, is_market_order: bool = True) -> float:
        """Calculate slippage in pips"""
        
        if not is_market_order:
            return 0.0  # Limit orders have no slippage
            
        # Higher slippage during high volatility
        slippage = np.random.normal(self.slippage_mean, self.slippage_std)
        slippage *= (1 + max(0, volatility - 1) * 0.5)
        
        # Market orders have higher slippage
        if order_type == 'market':
            slippage += 0.05
            
        return max(0, slippage)
        
    def get_commission(self, symbol: str, lot_size: float) -> float:
        """Calculate commission cost"""
        return self.commission_per_lot * lot_size
        
    def get_swap_cost(self, symbol: str, side: str, days_held: int) -> float:
        """Calculate swap/overnight financing cost"""
        swap_rate = self.swap_rates.get(symbol, {}).get(side, 0)
        return swap_rate * days_held
        
    def apply_costs(
        self,
        entry_price: float,
        exit_price: float,
        symbol: str,
        side: str,
        lot_size: float,
        timestamp: datetime,
        volatility: float
    ) -> Tuple[float, float, float]:
        """
        Apply all costs to entry and exit prices
        Returns: (adjusted_entry, adjusted_exit, total_cost)
        """
        
        spread = self.get_spread(symbol, timestamp, volatility)
        spread_value = spread * 0.0001  # Convert pips to price
        
        # Adjust entry price for spread
        if side == 'long':
            adjusted_entry = entry_price + spread_value / 2
        else:
            adjusted_entry = entry_price - spread_value / 2
            
        # Apply slippage
        slippage = self.get_slippage('market', volatility)
        slippage_value = slippage * 0.0001
        
        if side == 'long':
            adjusted_entry += slippage_value
        else:
            adjusted_entry -= slippage_value
            
        # Commission
        commission = self.get_commission(symbol, lot_size)
        
        # No adjustments to exit (costs are already accounted)
        adjusted_exit = exit_price
        
        return adjusted_entry, adjusted_exit, commission


class ExecutionModel:
    """Complete execution simulation with realistic constraints"""
    
    def __init__(self, cost_model: CostModel = None):
        self.cost_model = cost_model or CostModel()
        
        # Execution constraints
        self.max_slippage_pips = 0.5
        self.min_execution_latency_ms = 50
        self.max_execution_latency_ms = 200
        
        # Partial fill simulation
        self.partial_fill_probability = 0.05  # 5% chance of partial fill
        self.partial_fill_ratio = 0.5  # 50% fill on partial
        
    def execute_market_order(
        self,
        symbol: str,
        side: str,
        lot_size: float,
        current_price: float,
        timestamp: datetime,
        volatility: float,
        bid_ask_spread: float = None
    ) -> Dict:
        """Simulate market order execution"""
        
        # Simulate latency
        latency_ms = np.random.uniform(self.min_execution_latency_ms, self.max_execution_latency_ms)
        
        # Get spread
        if bid_ask_spread is None:
            spread = self.cost_model.get_spread(symbol, timestamp, volatility)
        else:
            spread = bid_ask_spread
            
        spread_value = spread * 0.0001
        
        # Calculate execution price
        if side == 'buy':
            execution_price = current_price + spread_value / 2
        else:
            execution_price = current_price - spread_value / 2
            
        # Apply slippage
        slippage = self.cost_model.get_slippage('market', volatility)
        slippage_value = slippage * 0.0001
        
        if side == 'buy':
            execution_price += slippage_value
        else:
            execution_price -= slippage_value
            
        # Check if partial fill
        filled_quantity = lot_size
        
        if np.random.random() < self.partial_fill_probability:
            filled_quantity = lot_size * self.partial_fill_ratio
            partial_fill = True
        else:
            partial_fill = False
            
        # Calculate commission
        commission = self.cost_model.get_commission(symbol, filled_quantity)
        
        return {
            'executed': True,
            'execution_price': execution_price,
            'filled_quantity': filled_quantity,
            'commission': commission,
            'latency_ms': latency_ms,
            'partial_fill': partial_fill,
            'slippage_pips': slippage,
            'spread_pips': spread
        }
        
    def execute_limit_order(
        self,
        symbol: str,
        side: str,
        limit_price: float,
        current_price: float,
        timestamp: datetime,
        volatility: float
    ) -> Dict:
        """Simulate limit order execution (no slippage)"""
        
        # Check if limit price is reachable
        if side == 'buy' and limit_price >= current_price:
            return {'executed': False, 'reason': 'Limit price above market'}
        if side == 'sell' and limit_price <= current_price:
            return {'executed': False, 'reason': 'Limit price below market'}
            
        # Simulate latency
        latency_ms = np.random.uniform(self.min_execution_latency_ms, self.max_execution_latency_ms)
        
        # No slippage for limit orders
        execution_price = limit_price
        
        # Calculate commission
        commission = self.cost_model.get_commission(symbol, 1.0)  # Assume 1 lot for simplicity
        
        return {
            'executed': True,
            'execution_price': execution_price,
            'filled_quantity': 1.0,
            'commission': commission,
            'latency_ms': latency_ms,
            'partial_fill': False,
            'slippage_pips': 0,
            'spread_pips': self.cost_model.get_spread(symbol, timestamp, volatility)
        }
        
    def _session_multiplier(self, timestamp: datetime) -> float:
        """Session-based multiplier for spreads"""
        hour = timestamp.hour
        
        # Asian session (low liquidity, higher spreads)
        if 0 <= hour < 8:
            return 1.3
        # London session (high liquidity)
        elif 8 <= hour < 13:
            return 0.8
        # London-NY overlap (highest liquidity)
        elif 13 <= hour < 17:
            return 0.7
        # NY session
        elif 17 <= hour < 22:
            return 0.9
        # Late session
        else:
            return 1.1
