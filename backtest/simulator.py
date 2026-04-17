"""Core backtest simulation engine"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Individual trade record"""
    id: int
    symbol: str
    direction: int  # 1 = long, -1 = short
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    quantity: float = 0.01
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    exit_reason: str = ""  # 'tp', 'sl', 'manual', 'eod'
    
    @property
    def pnl_pips(self) -> float:
        if self.exit_price is None:
            return 0.0
        if self.direction == 1:
            return (self.exit_price - self.entry_price) * 10000  # For forex
        else:
            return (self.entry_price - self.exit_price) * 10000
            
    @property
    def pnl_dollars(self) -> float:
        # Assuming 1 pip = $10 for 1 lot on XAUUSD
        pip_value = 10.0
        return self.pnl_pips * self.quantity * pip_value
        
    @property
    def r_multiple(self) -> float:
        """Risk multiple: profit / risk"""
        risk = abs(self.entry_price - self.stop_loss) if self.stop_loss else 0
        if risk == 0:
            return 0
        profit = self.exit_price - self.entry_price if self.direction == 1 else self.entry_price - self.exit_price
        return profit / risk
        
    @property
    def duration_minutes(self) -> float:
        if self.exit_time and self.entry_time:
            return (self.exit_time - self.entry_time).total_seconds() / 60
        return 0.0


@dataclass
class BacktestResult:
    """Complete backtest results"""
    # Core metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    
    # P&L metrics
    total_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    profit_factor: float = 0.0
    net_profit: float = 0.0
    
    # Risk metrics
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # Trade metrics
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_r_multiple: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_duration_minutes: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    
    # Equity curve
    equity_curve: List[float] = field(default_factory=list)
    drawdown_curve: List[float] = field(default_factory=list)
    trades: List[Trade] = field(default_factory=list)
    
    # Metadata
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    initial_capital: float = 10000.0
    final_capital: float = 10000.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'total_pnl': self.total_pnl,
            'profit_factor': self.profit_factor,
            'net_profit': self.net_profit,
            'max_drawdown_pct': self.max_drawdown_pct,
            'sharpe_ratio': self.sharpe_ratio,
            'avg_r_multiple': self.avg_r_multiple,
            'max_consecutive_losses': self.max_consecutive_losses,
            'total_trades': self.total_trades
        }
        
    def summary(self) -> str:
        """Generate human-readable summary"""
        return f"""
╔══════════════════════════════════════════════════════════════╗
║                    BACKTEST RESULTS                          ║
╠══════════════════════════════════════════════════════════════╣
║ Period: {self.start_date.date()} → {self.end_date.date()}
║ Initial Capital: ${self.initial_capital:,.2f}
║ Final Capital: ${self.final_capital:,.2f}
║ Net Profit: ${self.net_profit:,.2f} ({self.net_profit/self.initial_capital*100:.1f}%)
╠══════════════════════════════════════════════════════════════╣
║ TRADE STATISTICS:
║   Total Trades: {self.total_trades}
║   Win Rate: {self.win_rate:.1f}% ({self.winning_trades}/{self.total_trades})
║   Profit Factor: {self.profit_factor:.2f}
║   Avg R Multiple: {self.avg_r_multiple:.2f}R
╠══════════════════════════════════════════════════════════════╣
║ RISK METRICS:
║   Max Drawdown: {self.max_drawdown_pct:.1f}%
║   Sharpe Ratio: {self.sharpe_ratio:.2f}
║   Max Consecutive Losses: {self.max_consecutive_losses}
╚══════════════════════════════════════════════════════════════╝
"""


class BacktestSimulator:
    """Main backtest simulation engine"""
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        commission_per_lot: float = 3.5,
        slippage_pips: float = 0.3,
        spread_pips: float = 0.2
    ):
        self.initial_capital = initial_capital
        self.commission_per_lot = commission_per_lot
        self.slippage_pips = slippage_pips
        self.spread_pips = spread_pips
        
        # State
        self.capital = initial_capital
        self.equity_curve = [initial_capital]
        self.trades: List[Trade] = []
        self.open_position: Optional[Trade] = None
        self.trade_counter = 0
        
        # Performance tracking
        self.peak_equity = initial_capital
        self.drawdowns = []
        
    def run(
        self,
        data: pd.DataFrame,
        signal_generator: Callable,
        risk_per_trade: float = 0.01,
        max_concurrent_trades: int = 1
    ) -> BacktestResult:
        """
        Run backtest on historical data
        
        Args:
            data: OHLCV DataFrame with 'timestamp', 'open', 'high', 'low', 'close'
            signal_generator: Function that returns signal dict {'direction': 1/-1/0, 'confidence': float}
            risk_per_trade: Percentage of capital to risk per trade
            max_concurrent_trades: Maximum number of concurrent positions
        """
        
        self.capital = self.initial_capital
        self.equity_curve = [self.initial_capital]
        self.trades = []
        self.open_position = None
        self.peak_equity = self.initial_capital
        self.drawdowns = []
        
        # Ensure data is sorted
        data = data.sort_values('timestamp').reset_index(drop=True)
        
        # Run simulation candle by candle
        for i in range(len(data)):
            current_candle = data.iloc[i]
            next_candle = data.iloc[i + 1] if i + 1 < len(data) else None
            
            # Check existing position
            if self.open_position:
                self._check_exit(self.open_position, current_candle, next_candle)
                
            # Generate signal
            if self.open_position is None or max_concurrent_trades > 1:
                signal = signal_generator(data.iloc[:i+1], current_candle)
                
                if signal and signal.get('direction') != 0:
                    self._enter_trade(signal, current_candle, risk_per_trade)
                    
            # Update equity curve
            self._update_equity(current_candle)
            
        # Close any open position at end
        if self.open_position:
            self._close_trade(self.open_position, data.iloc[-1], "eod")
            
        # Calculate metrics
        return self._calculate_results(data)
        
    def _enter_trade(self, signal: dict, candle: pd.Series, risk_per_trade: float):
        """Enter a new trade"""
        
        direction = signal['direction']
        confidence = signal.get('confidence', 0.7)
        entry_price = candle['close']
        
        # Apply spread
        if direction == 1:  # Long
            entry_price = candle['ask'] if 'ask' in candle else candle['close'] + self.spread_pips * 0.0001
        else:  # Short
            entry_price = candle['bid'] if 'bid' in candle else candle['close'] - self.spread_pips * 0.0001
            
        # Apply slippage
        slippage = np.random.normal(0, self.slippage_pips * 0.0001)
        entry_price += slippage if direction == 1 else -slippage
        
        # Calculate stop loss and take profit using ATR
        atr = self._calculate_atr(candle.get('atr', 0.005))
        sl_distance = atr * 1.5  # 1.5x ATR
        tp_distance = atr * 2.5  # 2.5x ATR (1.67 R:R)
        
        if direction == 1:
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + tp_distance
        else:
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - tp_distance
            
        # Calculate position size based on risk
        risk_amount = self.capital * risk_per_trade
        risk_pips = sl_distance * 10000
        pip_value = 10.0
        lot_size = risk_amount / (risk_pips * pip_value)
        
        # Apply confidence scaling
        if confidence < 0.8:
            lot_size *= (confidence / 0.8)
            
        # Hard limits
        lot_size = max(0.01, min(lot_size, 1.0))
        lot_size = round(lot_size / 0.01) * 0.01
        
        # Create trade
        self.trade_counter += 1
        self.open_position = Trade(
            id=self.trade_counter,
            symbol=signal.get('symbol', 'XAUUSD'),
            direction=direction,
            entry_time=candle['timestamp'],
            entry_price=entry_price,
            quantity=lot_size,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        # Deduct commission
        commission = self.commission_per_lot * lot_size
        self.capital -= commission
        
        logger.debug(f"ENTER {direction}: {lot_size} lots @ {entry_price:.2f} | SL: {stop_loss:.2f} | TP: {take_profit:.2f}")
        
    def _check_exit(self, trade: Trade, current_candle: pd.Series, next_candle: Optional[pd.Series]):
        """Check if trade should exit"""
        
        high = current_candle['high']
        low = current_candle['low']
        
        if trade.direction == 1:  # Long position
            # Check take profit
            if high >= trade.take_profit:
                exit_price = trade.take_profit
                self._close_trade(trade, current_candle, "tp")
            # Check stop loss
            elif low <= trade.stop_loss:
                exit_price = trade.stop_loss
                self._close_trade(trade, current_candle, "sl")
        else:  # Short position
            # Check take profit
            if low <= trade.take_profit:
                exit_price = trade.take_profit
                self._close_trade(trade, current_candle, "tp")
            # Check stop loss
            elif high >= trade.stop_loss:
                exit_price = trade.stop_loss
                self._close_trade(trade, current_candle, "sl")
                
    def _close_trade(self, trade: Trade, candle: pd.Series, reason: str):
        """Close an open trade"""
        
        # Apply slippage on exit
        slippage = np.random.normal(0, self.slippage_pips * 0.0001)
        exit_price = candle['close']
        
        if trade.direction == 1:
            exit_price = candle['bid'] if 'bid' in candle else candle['close'] - self.spread_pips * 0.0001
            exit_price -= slippage
        else:
            exit_price = candle['ask'] if 'ask' in candle else candle['close'] + self.spread_pips * 0.0001
            exit_price += slippage
            
        trade.exit_time = candle['timestamp']
        trade.exit_price = exit_price
        trade.exit_reason = reason
        
        # Update capital with P&L
        self.capital += trade.pnl_dollars
        
        # Deduct commission on exit (round-turn)
        commission = self.commission_per_lot * trade.quantity
        self.capital -= commission
        
        # Store trade
        self.trades.append(trade)
        self.open_position = None
        
        logger.debug(f"EXIT {reason}: P&L = ${trade.pnl_dollars:.2f} ({trade.r_multiple:.2f}R)")
        
    def _update_equity(self, current_candle: pd.Series):
        """Update equity curve with unrealized P&L"""
        
        equity = self.capital
        
        if self.open_position:
            current_price = current_candle['close']
            
            if self.open_position.direction == 1:
                unrealized = (current_price - self.open_position.entry_price) * self.open_position.quantity * 100
            else:
                unrealized = (self.open_position.entry_price - current_price) * self.open_position.quantity * 100
                
            equity += unrealized
            
        self.equity_curve.append(equity)
        
        # Track drawdown
        if equity > self.peak_equity:
            self.peak_equity = equity
            
        drawdown = (self.peak_equity - equity) / self.peak_equity * 100
        self.drawdowns.append(drawdown)
        
    def _calculate_results(self, data: pd.DataFrame) -> BacktestResult:
        """Calculate all performance metrics"""
        
        if not self.trades:
            return BacktestResult(initial_capital=self.initial_capital)
            
        # Basic counts
        winning_trades = [t for t in self.trades if t.pnl_dollars > 0]
        losing_trades = [t for t in self.trades if t.pnl_dollars < 0]
        
        total_pnl = sum(t.pnl_dollars for t in self.trades)
        gross_profit = sum(t.pnl_dollars for t in winning_trades)
        gross_loss = abs(sum(t.pnl_dollars for t in losing_trades))
        
        # Calculate drawdown
        max_drawdown = max(self.drawdowns) if self.drawdowns else 0
        
        # Calculate Sharpe ratio (assuming 252 trading days, 15m candles = 26 per day)
        returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
        sharpe = (np.mean(returns) / np.std(returns) * np.sqrt(252 * 26)) if np.std(returns) > 0 else 0
        
        # Calculate Sortino (downside deviation only)
        downside_returns = returns[returns < 0]
        sortino = (np.mean(returns) / np.std(downside_returns) * np.sqrt(252 * 26)) if len(downside_returns) > 0 and np.std(downside_returns) > 0 else 0
        
        # Consecutive wins/losses
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_wins = 0
        current_losses = 0
        
        for trade in self.trades:
            if trade.pnl_dollars > 0:
                current_wins += 1
                current_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, current_losses)
                
        return BacktestResult(
            total_trades=len(self.trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=len(winning_trades) / len(self.trades) * 100,
            total_pnl=total_pnl,
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            profit_factor=gross_profit / gross_loss if gross_loss > 0 else 0,
            net_profit=total_pnl,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=(total_pnl / self.initial_capital * 100) / (max_drawdown / 100) if max_drawdown > 0 else 0,
            avg_win=sum(t.pnl_dollars for t in winning_trades) / len(winning_trades) if winning_trades else 0,
            avg_loss=sum(t.pnl_dollars for t in losing_trades) / len(losing_trades) if losing_trades else 0,
            avg_r_multiple=sum(t.r_multiple for t in self.trades) / len(self.trades),
            largest_win=max((t.pnl_dollars for t in winning_trades), default=0),
            largest_loss=min((t.pnl_dollars for t in losing_trades), default=0),
            avg_duration_minutes=sum(t.duration_minutes for t in self.trades) / len(self.trades),
            max_consecutive_wins=max_consecutive_wins,
            max_consecutive_losses=max_consecutive_losses,
            equity_curve=self.equity_curve,
            drawdown_curve=self.drawdowns,
            trades=self.trades,
            start_date=data['timestamp'].iloc[0] if len(data) > 0 else None,
            end_date=data['timestamp'].iloc[-1] if len(data) > 0 else None,
            initial_capital=self.initial_capital,
            final_capital=self.equity_curve[-1] if self.equity_curve else self.initial_capital
        )
        
    def _calculate_atr(self, default: float = 0.005) -> float:
        """Calculate or return ATR value"""
        # In real implementation, calculate from data
        return default
