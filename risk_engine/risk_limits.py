"""Risk management and position sizing"""

import logging
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from collections import deque

logger = logging.getLogger(__name__)


class RiskEngine:
    """Core risk management system"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Risk limits
        self.max_risk_per_trade_pct = config.get('max_risk_per_trade_pct', 0.01)
        self.max_daily_loss_pct = config.get('max_daily_loss_pct', 0.03)
        self.max_concurrent_trades = config.get('max_concurrent_trades', 3)
        self.max_consecutive_losses = config.get('max_consecutive_losses', 3)
        self.max_drawdown_pct = config.get('max_drawdown_pct', 0.10)
        
        # Tracking
        self.daily_pnl = 0.0
        self.daily_reset = datetime.utcnow().date()
        self.consecutive_losses = 0
        self.peak_balance = None
        self.kill_switch_triggered = False
        self.kill_switch_time = None
        self.trade_history = deque(maxlen=100)
        
    def approve_trade(self, signal: Dict, account_balance: float, current_pnl: float) -> bool:
        """Check if trade meets all risk criteria"""
        
        # Check kill switch
        if self.kill_switch_triggered:
            if self._can_reset_kill_switch():
                self.kill_switch_triggered = False
                logger.info("Kill switch reset")
            else:
                logger.warning("Kill switch active - trade rejected")
                return False
                
        # Daily reset
        self._check_daily_reset()
        
        # Update daily P&L
        self.daily_pnl = current_pnl
        
        # Check daily loss limit
        daily_loss_limit = account_balance * self.max_daily_loss_pct
        if self.daily_pnl < -daily_loss_limit:
            logger.error(f"Daily loss limit reached: {self.daily_pnl:.2f}")
            self._trigger_kill_switch("Daily loss limit exceeded")
            return False
            
        # Check drawdown limit
        if self.peak_balance is None:
            self.peak_balance = account_balance
            
        drawdown = (self.peak_balance - account_balance) / self.peak_balance
        if drawdown > self.max_drawdown_pct:
            logger.error(f"Max drawdown reached: {drawdown:.2%}")
            self._trigger_kill_switch(f"Drawdown limit exceeded: {drawdown:.2%}")
            return False
            
        # Check consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            logger.warning(f"Max consecutive losses reached: {self.consecutive_losses}")
            return False
            
        return True
        
    def calculate_lot_size(
        self,
        account_balance: float,
        stop_loss_pips: float,
        pip_value: float = 10.0,
        confidence: float = 0.75
    ) -> float:
        """Calculate position size based on risk per trade"""
        
        # Calculate base risk amount
        risk_amount = account_balance * self.max_risk_per_trade_pct
        
        # Calculate lot size
        sl_cost_per_lot = stop_loss_pips * pip_value
        lot_size = risk_amount / sl_cost_per_lot
        
        # Apply confidence scaling
        if confidence < 0.80:
            confidence_multiplier = max(0.5, confidence / 0.80)
            lot_size *= confidence_multiplier
            
        # Apply hard limits
        min_lot = self.config.get('min_lot_size', 0.01)
        max_lot = self.config.get('max_lot_size', 1.0)
        
        lot_size = max(min_lot, min(lot_size, max_lot))
        
        # Round to step size
        step = self.config.get('lot_step', 0.01)
        lot_size = round(lot_size / step) * step
        
        logger.info(f"Calculated lot size: {lot_size} (risk: ${risk_amount:.2f})")
        return lot_size
        
    def update_trade_outcome(self, pnl: float):
        """Update tracking metrics after trade closes"""
        
        self.trade_history.append(pnl)
        
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
            
        # Update peak balance
        if self.peak_balance:
            self.peak_balance = max(self.peak_balance, self.peak_balance + pnl)
            
    def _check_daily_reset(self):
        """Reset daily counters if new day"""
        now = datetime.utcnow()
        if now.date() > self.daily_reset:
            self.daily_reset = now.date()
            self.daily_pnl = 0.0
            self.consecutive_losses = 0
            logger.info("Daily risk counters reset")
            
    def _trigger_kill_switch(self, reason: str):
        """Trigger kill switch"""
        self.kill_switch_triggered = True
        self.kill_switch_time = datetime.utcnow()
        logger.error(f"KILL SWITCH TRIGGERED: {reason}")
        
    def _can_reset_kill_switch(self) -> bool:
        """Check if kill switch can be reset"""
        if self.kill_switch_time:
            cooldown_hours = self.config.get('kill_switch_cooldown_hours', 24)
            if datetime.utcnow() - self.kill_switch_time > timedelta(hours=cooldown_hours):
                return True
        return False
        
    def get_risk_status(self) -> Dict:
        """Get current risk status"""
        return {
            'kill_switch_active': self.kill_switch_triggered,
            'consecutive_losses': self.consecutive_losses,
            'daily_pnl': self.daily_pnl,
            'peak_balance': self.peak_balance,
            'recent_trades': len(self.trade_history)
        }
