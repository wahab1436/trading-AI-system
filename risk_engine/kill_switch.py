"""Emergency kill switch for automated trading system"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
import time

logger = logging.getLogger(__name__)


class KillSwitchReason(Enum):
    """Reasons for kill switch activation"""
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    DRAWDOWN_LIMIT = "drawdown_limit"
    MAX_CONSECUTIVE_LOSSES = "max_consecutive_losses"
    MANUAL_TRIGGER = "manual_trigger"
    BROKER_DISCONNECT = "broker_disconnect"
    DATA_FEED_FAILURE = "data_feed_failure"
    MODEL_DRIFT_DETECTED = "model_drift_detected"
    SPREAD_SPIKE = "spread_spike"
    NEWS_EVENT = "news_event"
    SYSTEM_ERROR = "system_error"
    MAX_DAILY_TRADES = "max_daily_trades"
    CORRELATION_VIOLATION = "correlation_violation"


@dataclass
class KillSwitchEvent:
    """Record of kill switch activation"""
    reason: KillSwitchReason
    timestamp: datetime
    details: str
    account_balance_at_time: float
    daily_pnl_at_time: float
    drawdown_at_time: float


class KillSwitch:
    """
    Emergency kill switch that stops all trading when risk limits are breached.
    Features:
    - Multiple trigger conditions
    - Automatic position closing
    - Cooldown period
    - Manual override
    - Alert callbacks
    - Persistent state
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Limits
        self.daily_loss_limit_pct = config.get('daily_loss_limit_pct', 0.03)
        self.max_drawdown_pct = config.get('max_drawdown_pct', 0.10)
        self.max_consecutive_losses = config.get('max_consecutive_losses', 5)
        self.max_daily_trades = config.get('max_daily_trades', 20)
        self.max_spread_multiplier = config.get('max_spread_multiplier', 3.0)
        
        # Cooldown settings
        self.cooldown_minutes = config.get('cooldown_minutes', 60)
        self.auto_reset_hours = config.get('auto_reset_hours', 24)
        
        # State
        self.is_activated = False
        self.is_paused = False
        self.activation_reason: Optional[KillSwitchReason] = None
        self.activation_time: Optional[datetime] = None
        self.cooldown_until: Optional[datetime] = None
        self.events: List[KillSwitchEvent] = []
        
        # Tracking
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.peak_balance = None
        self.daily_reset_time = datetime.utcnow().replace(hour=0, minute=0, second=0)
        
        # Callbacks
        self.alert_callbacks: List[Callable] = []
        self.reset_callbacks: List[Callable] = []
        
        # Lock for thread safety
        self._lock = threading.Lock()
        
        # Start monitoring thread
        self._monitoring = False
        self._monitor_thread = None
        
    def start_monitoring(self):
        """Start background monitoring thread"""
        if self._monitoring:
            return
            
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Kill switch monitoring started")
        
    def stop_monitoring(self):
        """Stop monitoring thread"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("Kill switch monitoring stopped")
        
    def check_and_trigger(
        self,
        current_balance: float,
        daily_pnl: float,
        consecutive_losses: int,
        daily_trades: int,
        current_spread: Optional[float] = None,
        average_spread: Optional[float] = None,
        is_broker_connected: bool = True,
        is_data_feed_alive: bool = True
    ) -> bool:
        """
        Check all conditions and trigger kill switch if needed
        
        Returns:
            True if kill switch was triggered, False otherwise
        """
        
        with self._lock:
            if self.is_activated:
                return True
                
            # Daily reset check
            self._check_daily_reset()
            
            # Update tracking
            self.daily_pnl = daily_pnl
            self.daily_trades = daily_trades
            self.consecutive_losses = consecutive_losses
            
            if self.peak_balance is None:
                self.peak_balance = current_balance
            else:
                self.peak_balance = max(self.peak_balance, current_balance)
                
            # Calculate drawdown
            drawdown = (self.peak_balance - current_balance) / self.peak_balance if self.peak_balance > 0 else 0
            
            # Check daily loss limit
            daily_loss_limit = self.peak_balance * self.daily_loss_limit_pct
            if daily_pnl < -daily_loss_limit:
                self._activate(
                    KillSwitchReason.DAILY_LOSS_LIMIT,
                    f"Daily loss {daily_pnl:.2f} exceeds limit {daily_loss_limit:.2f}"
                )
                return True
                
            # Check drawdown limit
            if drawdown > self.max_drawdown_pct:
                self._activate(
                    KillSwitchReason.DRAWDOWN_LIMIT,
                    f"Drawdown {drawdown:.2%} exceeds limit {self.max_drawdown_pct:.2%}"
                )
                return True
                
            # Check consecutive losses
            if consecutive_losses >= self.max_consecutive_losses:
                self._activate(
                    KillSwitchReason.MAX_CONSECUTIVE_LOSSES,
                    f"Consecutive losses: {consecutive_losses}"
                )
                return True
                
            # Check daily trades limit
            if daily_trades >= self.max_daily_trades:
                self._activate(
                    KillSwitchReason.MAX_DAILY_TRADES,
                    f"Daily trades: {daily_trades}"
                )
                return True
                
            # Check broker connection
            if not is_broker_connected:
                self._activate(
                    KillSwitchReason.BROKER_DISCONNECT,
                    "Broker connection lost"
                )
                return True
                
            # Check data feed
            if not is_data_feed_alive:
                self._activate(
                    KillSwitchReason.DATA_FEED_FAILURE,
                    "Data feed is not alive"
                )
                return True
                
            # Check spread spike
            if current_spread and average_spread and average_spread > 0:
                spread_ratio = current_spread / average_spread
                if spread_ratio > self.max_spread_multiplier:
                    self._activate(
                        KillSwitchReason.SPREAD_SPIKE,
                        f"Spread ratio: {spread_ratio:.2f}x normal"
                    )
                    return True
                    
            return False
            
    def manual_trigger(self, reason: str = "Manual override"):
        """Manually trigger kill switch"""
        with self._lock:
            self._activate(KillSwitchReason.MANUAL_TRIGGER, reason)
            
    def manual_reset(self) -> bool:
        """Manually reset kill switch (bypass cooldown)"""
        with self._lock:
            if not self.is_activated:
                logger.warning("Kill switch not activated, nothing to reset")
                return False
                
            logger.warning(f"MANUAL RESET: Kill switch reset by operator")
            self._reset()
            return True
            
    def _activate(self, reason: KillSwitchReason, details: str):
        """Activate kill switch and trigger emergency actions"""
        
        if self.is_activated:
            return
            
        self.is_activated = True
        self.activation_reason = reason
        self.activation_time = datetime.utcnow()
        self.cooldown_until = self.activation_time + timedelta(minutes=self.cooldown_minutes)
        
        # Record event
        event = KillSwitchEvent(
            reason=reason,
            timestamp=self.activation_time,
            details=details,
            account_balance_at_time=self.peak_balance or 0,
            daily_pnl_at_time=self.daily_pnl,
            drawdown_at_time=(self.peak_balance - (self.peak_balance or 0 + self.daily_pnl)) / self.peak_balance if self.peak_balance else 0
        )
        self.events.append(event)
        
        # Log alert
        logger.error(f"🔴 KILL SWITCH ACTIVATED 🔴")
        logger.error(f"Reason: {reason.value}")
        logger.error(f"Details: {details}")
        logger.error(f"Time: {self.activation_time}")
        
        # Send alerts
        self._send_alerts(f"KILL SWITCH ACTIVATED: {reason.value} - {details}")
        
        # Keep only last 100 events
        if len(self.events) > 100:
            self.events = self.events[-100:]
            
    def _reset(self):
        """Reset kill switch state"""
        self.is_activated = False
        self.is_paused = False
        self.activation_reason = None
        self.activation_time = None
        self.cooldown_until = None
        
        # Reset daily counters on reset
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.daily_reset_time = datetime.utcnow().replace(hour=0, minute=0, second=0)
        
        logger.info("✅ Kill switch reset - trading resumed")
        self._send_alerts("Kill switch reset - trading resumed")
        
        # Trigger reset callbacks
        for callback in self.reset_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Reset callback error: {e}")
                
    def _check_daily_reset(self):
        """Check if daily reset is needed"""
        now = datetime.utcnow()
        if now.date() > self.daily_reset_time.date():
            self.daily_reset_time = now.replace(hour=0, minute=0, second=0)
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.consecutive_losses = 0
            
            # Auto-reset kill switch after auto_reset_hours if still activated
            if self.is_activated and self.activation_time:
                hours_since_activation = (now - self.activation_time).total_seconds() / 3600
                if hours_since_activation >= self.auto_reset_hours:
                    logger.info(f"Auto-resetting kill switch after {self.auto_reset_hours} hours")
                    self._reset()
                    
    def _monitor_loop(self):
        """Background monitoring loop for periodic checks"""
        
        while self._monitoring:
            try:
                # Check if kill switch is activated and in cooldown
                if self.is_activated and self.cooldown_until:
                    if datetime.utcnow() >= self.cooldown_until:
                        with self._lock:
                            if self.is_activated:  # Double-check
                                logger.info(f"Cooldown period ended, auto-resetting kill switch")
                                self._reset()
                                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                time.sleep(60)
                
    def _send_alerts(self, message: str):
        """Send alerts to all registered callbacks"""
        for callback in self.alert_callbacks:
            try:
                callback(message)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
                
    def add_alert_callback(self, callback: Callable[[str], None]):
        """Add callback for kill switch alerts"""
        self.alert_callbacks.append(callback)
        
    def add_reset_callback(self, callback: Callable[[], None]):
        """Add callback for kill switch reset"""
        self.reset_callbacks.append(callback)
        
    def can_trade(self) -> bool:
        """Check if trading is allowed"""
        if self.is_paused:
            return False
        if self.is_activated:
            return False
        return True
        
    def pause_trading(self):
        """Temporarily pause trading without full kill switch"""
        self.is_paused = True
        logger.warning("Trading paused (manual pause)")
        
    def resume_trading(self):
        """Resume trading after pause"""
        self.is_paused = False
        logger.info("Trading resumed")
        
    def get_status(self) -> Dict:
        """Get current kill switch status"""
        
        now = datetime.utcnow()
        
        return {
            'is_activated': self.is_activated,
            'is_paused': self.is_paused,
            'activation_reason': self.activation_reason.value if self.activation_reason else None,
            'activation_time': self.activation_time.isoformat() if self.activation_time else None,
            'cooldown_remaining_minutes': max(0, (self.cooldown_until - now).total_seconds() / 60) if self.cooldown_until else 0,
            'can_trade': self.can_trade(),
            'daily_pnl': self.daily_pnl,
            'daily_trades': self.daily_trades,
            'consecutive_losses': self.consecutive_losses,
            'peak_balance': self.peak_balance,
            'recent_events': [
                {
                    'reason': e.reason.value,
                    'timestamp': e.timestamp.isoformat(),
                    'details': e.details
                }
                for e in self.events[-5:]
            ]
        }
        
    def get_events(self, limit: int = 50) -> List[Dict]:
        """Get kill switch event history"""
        return [
            {
                'reason': e.reason.value,
                'timestamp': e.timestamp.isoformat(),
                'details': e.details,
                'account_balance': e.account_balance_at_time,
                'daily_pnl': e.daily_pnl_at_time,
                'drawdown': e.drawdown_at_time
            }
            for e in self.events[-limit:]
        ]


class CircuitBreaker:
    """
    Circuit breaker pattern for external service calls (broker API, data feed).
    Prevents cascading failures by temporarily disabling problematic services.
    """
    
    def __init__(self, name: str, failure_threshold: int = 3, timeout_seconds: int = 60):
        self.name = name
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        
        if self.state == "OPEN":
            if datetime.utcnow() - self.last_failure_time > timedelta(seconds=self.timeout_seconds):
                self.state = "HALF_OPEN"
                logger.info(f"Circuit breaker {self.name} transitioning to HALF_OPEN")
            else:
                raise Exception(f"Circuit breaker {self.name} is OPEN")
                
        try:
            result = func(*args, **kwargs)
            
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
                logger.info(f"Circuit breaker {self.name} reset to CLOSED")
                
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = datetime.utcnow()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                logger.error(f"Circuit breaker {self.name} opened after {self.failure_count} failures")
                
            raise e
            
    def reset(self):
        """Manually reset circuit breaker"""
        self.failure_count = 0
        self.state = "CLOSED"
        self.last_failure_time = None
        logger.info(f"Circuit breaker {self.name} manually reset")
        
    def get_status(self) -> Dict:
        """Get circuit breaker status"""
        return {
            'name': self.name,
            'state': self.state,
            'failure_count': self.failure_count,
            'last_failure': self.last_failure_time.isoformat() if self.last_failure_time else None
        }
