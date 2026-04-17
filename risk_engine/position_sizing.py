"""Dynamic position sizing based on account risk and market conditions"""

import logging
from typing import Dict, Optional
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PositionSizeResult:
    """Result of position size calculation"""
    lot_size: float
    risk_amount_usd: float
    risk_percentage: float
    stop_loss_pips: float
    confidence_multiplier: float
    volatility_multiplier: float


class DynamicPositionSizer:
    """
    Dynamic position sizing engine with multiple adjustment factors:
    - Base: Fixed percentage risk per trade (default 1%)
    - Confidence scaling: Lower confidence = smaller position
    - Volatility scaling: Higher volatility = smaller position
    - Kelly criterion: Optimal bet sizing based on historical performance
    - Consecutive loss reduction: Reduce size after losses
    - Account growth scaling: Scale up slowly after profits
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Base risk parameters
        self.base_risk_pct = config.get('max_risk_per_trade_pct', 0.01)
        self.min_risk_pct = config.get('min_risk_per_trade_pct', 0.005)
        self.max_risk_pct = config.get('max_risk_per_trade_pct_absolute', 0.02)
        
        # Position limits
        self.min_lot_size = config.get('min_lot_size', 0.01)
        self.max_lot_size = config.get('max_lot_size', 1.0)
        self.lot_step = config.get('lot_step', 0.01)
        
        # Scaling factors
        self.confidence_threshold = config.get('confidence_threshold', 0.65)
        self.high_confidence_threshold = config.get('high_confidence_threshold', 0.80)
        self.max_confidence_multiplier = config.get('max_confidence_multiplier', 1.2)
        self.min_confidence_multiplier = config.get('min_confidence_multiplier', 0.3)
        
        # Volatility scaling
        self.volatility_scaling_enabled = config.get('volatility_scaling_enabled', True)
        self.volatility_lookback = config.get('volatility_lookback', 20)
        self.volatility_target = config.get('volatility_target', 0.005)  # Target ATR as % of price
        self.max_volatility_multiplier = config.get('max_volatility_multiplier', 1.5)
        self.min_volatility_multiplier = config.get('min_volatility_multiplier', 0.5)
        
        # Kelly Criterion
        self.kelly_enabled = config.get('kelly_enabled', False)
        self.kelly_fraction = config.get('kelly_fraction', 0.25)  # Fractional Kelly
        self.kelly_lookback_trades = config.get('kelly_lookback_trades', 100)
        
        # Drawdown scaling
        self.drawdown_scaling_enabled = config.get('drawdown_scaling_enabled', True)
        self.drawdown_threshold = config.get('drawdown_threshold', 0.05)  # 5% drawdown
        self.max_drawdown_reduction = config.get('max_drawdown_reduction', 0.5)  # Reduce by 50%
        
        # Consecutive loss scaling
        self.loss_scaling_enabled = config.get('loss_scaling_enabled', True)
        self.loss_reduction_start = config.get('loss_reduction_start', 2)  # Start reducing after 2 losses
        self.max_loss_reduction = config.get('max_loss_reduction', 0.75)  # Reduce by 75% max
        
        # Profit scaling (slow scaling up)
        self.profit_scaling_enabled = config.get('profit_scaling_enabled', True)
        self.profit_scale_threshold = config.get('profit_scale_threshold', 0.10)  # 10% profit
        self.max_profit_multiplier = config.get('max_profit_multiplier', 1.5)  # Scale up to 150%
        
        # Tracking state
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.peak_equity = None
        self.trade_history = []
        self.win_rate_history = []
        
    def calculate_position_size(
        self,
        account_balance: float,
        stop_loss_pips: float,
        pip_value: float = 10.0,
        confidence: float = 0.75,
        current_atr: Optional[float] = None,
        current_price: Optional[float] = None,
        volatility_regime: str = "normal"
    ) -> PositionSizeResult:
        """
        Calculate optimal position size with all scaling factors
        
        Args:
            account_balance: Current account balance
            stop_loss_pips: Stop loss distance in pips
            pip_value: Value per pip per lot (default $10 for XAUUSD)
            confidence: Model confidence (0-1)
            current_atr: Current ATR value for volatility scaling
            current_price: Current price for volatility scaling
            volatility_regime: 'low', 'normal', 'high', 'extreme'
        
        Returns:
            PositionSizeResult with calculated lot size and metadata
        """
        
        # 1. Base risk amount
        risk_amount = account_balance * self.base_risk_pct
        
        # 2. Calculate base lot size from risk
        sl_cost_per_lot = stop_loss_pips * pip_value
        if sl_cost_per_lot <= 0:
            logger.warning(f"Invalid stop loss distance: {stop_loss_pips} pips")
            sl_cost_per_lot = 50 * pip_value  # Default 50 pips
            
        base_lot_size = risk_amount / sl_cost_per_lot
        
        # 3. Apply confidence scaling
        confidence_multiplier = self._calculate_confidence_multiplier(confidence)
        
        # 4. Apply volatility scaling
        volatility_multiplier = self._calculate_volatility_multiplier(
            current_atr, current_price, volatility_regime
        )
        
        # 5. Apply drawdown scaling
        drawdown_multiplier = self._calculate_drawdown_multiplier(account_balance)
        
        # 6. Apply consecutive loss scaling
        loss_multiplier = self._calculate_loss_multiplier()
        
        # 7. Apply profit scaling
        profit_multiplier = self._calculate_profit_multiplier(account_balance)
        
        # 8. Apply Kelly Criterion (if enabled)
        kelly_multiplier = self._calculate_kelly_multiplier()
        
        # Combine all multipliers
        total_multiplier = (
            confidence_multiplier *
            volatility_multiplier *
            drawdown_multiplier *
            loss_multiplier *
            profit_multiplier *
            kelly_multiplier
        )
        
        # Clamp multiplier to reasonable range
        total_multiplier = max(0.1, min(2.0, total_multiplier))
        
        # Calculate final risk percentage
        final_risk_pct = self.base_risk_pct * total_multiplier
        final_risk_pct = max(self.min_risk_pct, min(self.max_risk_pct, final_risk_pct))
        
        # Recalculate lot size with final risk
        final_risk_amount = account_balance * final_risk_pct
        lot_size = final_risk_amount / sl_cost_per_lot
        
        # Apply hard limits
        lot_size = max(self.min_lot_size, min(self.max_lot_size, lot_size))
        
        # Round to step size
        lot_size = round(lot_size / self.lot_step) * self.lot_step
        
        logger.info(
            f"Position sizing: base={base_lot_size:.2f}, "
            f"final={lot_size:.2f}, "
            f"multipliers: conf={confidence_multiplier:.2f}, "
            f"vol={volatility_multiplier:.2f}, "
            f"dd={drawdown_multiplier:.2f}, "
            f"loss={loss_multiplier:.2f}, "
            f"total={total_multiplier:.2f}"
        )
        
        return PositionSizeResult(
            lot_size=lot_size,
            risk_amount_usd=final_risk_amount,
            risk_percentage=final_risk_pct * 100,
            stop_loss_pips=stop_loss_pips,
            confidence_multiplier=confidence_multiplier,
            volatility_multiplier=volatility_multiplier
        )
        
    def _calculate_confidence_multiplier(self, confidence: float) -> float:
        """Scale position based on model confidence"""
        
        if confidence < self.confidence_threshold:
            # Below threshold, reduce size significantly
            ratio = confidence / self.confidence_threshold
            return max(self.min_confidence_multiplier, ratio * 0.5)
        elif confidence > self.high_confidence_threshold:
            # High confidence, scale up slightly
            ratio = (confidence - self.high_confidence_threshold) / (1 - self.high_confidence_threshold)
            return 1.0 + ratio * (self.max_confidence_multiplier - 1.0)
        else:
            # Normal range
            return 1.0
            
    def _calculate_volatility_multiplier(
        self,
        current_atr: Optional[float],
        current_price: Optional[float],
        volatility_regime: str
    ) -> float:
        """Scale position based on market volatility"""
        
        if not self.volatility_scaling_enabled:
            return 1.0
            
        # Use volatility regime from market state
        regime_multipliers = {
            'low': 1.2,      # Low volatility = larger positions
            'normal': 1.0,   # Normal volatility = standard
            'high': 0.7,     # High volatility = smaller positions
            'extreme': 0.4   # Extreme volatility = very small positions
        }
        
        regime_multiplier = regime_multipliers.get(volatility_regime, 1.0)
        
        # If ATR data available, calculate additional scaling
        if current_atr and current_price and current_price > 0:
            atr_pct = current_atr / current_price
            target_pct = self.volatility_target
            
            if atr_pct > 0:
                atr_multiplier = target_pct / atr_pct
                atr_multiplier = max(self.min_volatility_multiplier, 
                                    min(self.max_volatility_multiplier, atr_multiplier))
                return regime_multiplier * atr_multiplier
                
        return regime_multiplier
        
    def _calculate_drawdown_multiplier(self, current_equity: float) -> float:
        """Scale down position during drawdown"""
        
        if not self.drawdown_scaling_enabled:
            return 1.0
            
        if self.peak_equity is None:
            self.peak_equity = current_equity
        else:
            self.peak_equity = max(self.peak_equity, current_equity)
            
        drawdown = (self.peak_equity - current_equity) / self.peak_equity
        
        if drawdown <= 0:
            return 1.0
        elif drawdown >= self.drawdown_threshold:
            # Scale down linearly based on drawdown depth
            reduction = min(self.max_drawdown_reduction, drawdown * 2)
            return 1.0 - reduction
        else:
            # Small drawdown, slight reduction
            return 1.0 - (drawdown / self.drawdown_threshold) * 0.3
            
    def _calculate_loss_multiplier(self) -> float:
        """Reduce position size after consecutive losses"""
        
        if not self.loss_scaling_enabled:
            return 1.0
            
        if self.consecutive_losses < self.loss_reduction_start:
            return 1.0
            
        # Progressive reduction
        losses_beyond_start = self.consecutive_losses - self.loss_reduction_start + 1
        reduction = min(self.max_loss_reduction, losses_beyond_start * 0.15)
        
        return 1.0 - reduction
        
    def _calculate_profit_multiplier(self, current_equity: float) -> float:
        """Slowly scale up position after profits"""
        
        if not self.profit_scaling_enabled:
            return 1.0
            
        if self.peak_equity is None:
            self.peak_equity = current_equity
            
        profit_from_peak = (current_equity - self.peak_equity) / self.peak_equity
        
        if profit_from_peak <= 0:
            return 1.0
            
        # Scale up slowly as profit increases
        scaling = 1.0 + (profit_from_peak / self.profit_scale_threshold) * (self.max_profit_multiplier - 1.0)
        
        return min(self.max_profit_multiplier, scaling)
        
    def _calculate_kelly_multiplier(self) -> float:
        """Apply Kelly Criterion based on historical performance"""
        
        if not self.kelly_enabled or len(self.trade_history) < self.kelly_lookback_trades:
            return 1.0
            
        # Get recent trades
        recent_trades = self.trade_history[-self.kelly_lookback_trades:]
        
        # Calculate win rate and average win/loss
        wins = [t for t in recent_trades if t > 0]
        losses = [t for t in recent_trades if t < 0]
        
        if not wins or not losses:
            return 1.0
            
        win_rate = len(wins) / len(recent_trades)
        avg_win = sum(wins) / len(wins)
        avg_loss = abs(sum(losses) / len(losses))
        
        if avg_loss == 0:
            return 1.0
            
        # Kelly formula: f* = (p * b - q) / b
        # where p = win rate, q = loss rate, b = avg win / avg loss
        b = avg_win / avg_loss
        kelly_fraction = (win_rate * b - (1 - win_rate)) / b
        
        # Apply fractional Kelly and clamp
        kelly_fraction = max(0, min(0.5, kelly_fraction * self.kelly_fraction))
        
        # Convert to multiplier (1 = standard, <1 = reduce, >1 = increase)
        # Kelly fraction of 0.02 means we should risk 2% of capital
        target_risk = kelly_fraction
        current_risk = self.base_risk_pct
        
        if current_risk > 0:
            return target_risk / current_risk
        else:
            return 1.0
            
    def update_trade_result(self, pnl: float, realized: bool = True):
        """Update internal state after trade closes"""
        
        if not realized:
            return
            
        self.trade_history.append(pnl)
        
        if pnl < 0:
            self.consecutive_losses += 1
            self.consecutive_wins = 0
        else:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            
        # Keep trade history limited
        if len(self.trade_history) > 1000:
            self.trade_history = self.trade_history[-500:]
            
        logger.debug(f"Trade result updated: P&L={pnl:.2f}, consecutive_losses={self.consecutive_losses}")
        
    def reset_peak_equity(self, new_equity: float):
        """Reset peak equity (e.g., after withdrawal)"""
        self.peak_equity = new_equity
        
    def get_status(self) -> Dict:
        """Get current position sizing status"""
        return {
            'base_risk_pct': self.base_risk_pct * 100,
            'consecutive_losses': self.consecutive_losses,
            'consecutive_wins': self.consecutive_wins,
            'peak_equity': self.peak_equity,
            'recent_trades': len(self.trade_history[-20:]),
            'win_rate_recent': self._calculate_recent_win_rate()
        }
        
    def _calculate_recent_win_rate(self, lookback: int = 20) -> float:
        """Calculate win rate from recent trades"""
        if len(self.trade_history) < lookback:
            return 0.5
            
        recent = self.trade_history[-lookback:]
        wins = sum(1 for t in recent if t > 0)
        return wins / len(recent)


class KellyOptimizer:
    """
    Advanced Kelly Criterion optimizer for position sizing.
    Used to find optimal risk percentage based on historical performance.
    """
    
    def __init__(self):
        self.trade_history = []
        
    def add_trade(self, r_multiple: float):
        """Add trade result (R-multiple: profit/loss relative to risk)"""
        self.trade_history.append(r_multiple)
        
    def calculate_optimal_kelly(self, lookback: int = 100) -> Dict:
        """
        Calculate optimal Kelly fraction based on trade history
        
        Returns:
            Dictionary with optimal f, Sharpe, and confidence intervals
        """
        
        if len(self.trade_history) < lookback:
            return {'optimal_f': 0.02, 'confidence': 'low', 'sharpe': 0}
            
        trades = self.trade_history[-lookback:]
        
        # Calculate statistics
        wins = [t for t in trades if t > 0]
        losses = [t for t in trades if t < 0]
        
        if not wins or not losses:
            return {'optimal_f': 0.01, 'confidence': 'low', 'sharpe': 0}
            
        win_rate = len(wins) / len(trades)
        avg_win = sum(wins) / len(wins)
        avg_loss = abs(sum(losses) / len(losses))
        
        if avg_loss == 0:
            return {'optimal_f': win_rate, 'confidence': 'low', 'sharpe': 0}
            
        # Kelly formula
        b = avg_win / avg_loss
        kelly_f = (win_rate * b - (1 - win_rate)) / b
        
        # Calculate Sharpe ratio for confidence
        returns = trades
        sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        
        # Determine confidence level based on sample size and consistency
        if len(trades) >= 200 and sharpe > 1.0:
            confidence = 'high'
            multiplier = 0.5  # Full Kelly half
        elif len(trades) >= 100 and sharpe > 0.5:
            confidence = 'medium'
            multiplier = 0.33  # One-third Kelly
        else:
            confidence = 'low'
            multiplier = 0.25  # Quarter Kelly
            
        optimal_f = max(0.01, min(0.15, kelly_f * multiplier))
        
        return {
            'optimal_f': optimal_f,
            'confidence': confidence,
            'sharpe': sharpe,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'kelly_raw': kelly_f,
            'sample_size': len(trades)
        }
