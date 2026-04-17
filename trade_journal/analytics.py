"""Trade analytics and performance metrics calculation"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from .schema import TradeRecord, DailySummary, TradeStatus


class PerformanceMetrics:
    """Container for performance metrics"""
    
    def __init__(self):
        # Core metrics
        self.total_trades: int = 0
        self.winning_trades: int = 0
        self.losing_trades: int = 0
        self.win_rate: float = 0.0
        
        # P&L metrics
        self.total_pnl: float = 0.0
        self.gross_profit: float = 0.0
        self.gross_loss: float = 0.0
        self.net_profit: float = 0.0
        self.profit_factor: float = 0.0
        
        # Trade metrics
        self.avg_win: float = 0.0
        self.avg_loss: float = 0.0
        self.largest_win: float = 0.0
        self.largest_loss: float = 0.0
        self.avg_r_multiple: float = 0.0
        self.best_r_multiple: float = 0.0
        self.worst_r_multiple: float = 0.0
        
        # Risk metrics
        self.expectancy: float = 0.0
        self.sharpe_ratio: float = 0.0
        self.sortino_ratio: float = 0.0
        self.max_drawdown: float = 0.0
        self.max_drawdown_pct: float = 0.0
        self.avg_drawdown: float = 0.0
        
        # Duration metrics
        self.avg_trade_duration_minutes: float = 0.0
        self.max_trade_duration_minutes: float = 0.0
        
        # Sequence metrics
        self.max_consecutive_wins: int = 0
        self.max_consecutive_losses: int = 0
        self.current_consecutive_wins: int = 0
        self.current_consecutive_losses: int = 0
        
        # Statistical metrics
        self.pnl_std: float = 0.0
        self.pnl_skew: float = 0.0
        self.pnl_kurtosis: float = 0.0
        
    def to_dict(self) -> dict:
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'total_pnl': self.total_pnl,
            'gross_profit': self.gross_profit,
            'gross_loss': self.gross_loss,
            'net_profit': self.net_profit,
            'profit_factor': self.profit_factor,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'largest_win': self.largest_win,
            'largest_loss': self.largest_loss,
            'avg_r_multiple': self.avg_r_multiple,
            'best_r_multiple': self.best_r_multiple,
            'worst_r_multiple': self.worst_r_multiple,
            'expectancy': self.expectancy,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_pct': self.max_drawdown_pct,
            'avg_drawdown': self.avg_drawdown,
            'avg_trade_duration_minutes': self.avg_trade_duration_minutes,
            'max_trade_duration_minutes': self.max_trade_duration_minutes,
            'max_consecutive_wins': self.max_consecutive_wins,
            'max_consecutive_losses': self.max_consecutive_losses,
            'pnl_std': self.pnl_std,
            'pnl_skew': self.pnl_skew,
            'pnl_kurtosis': self.pnl_kurtosis
        }


class TradeAnalytics:
    """Analytics engine for trade performance analysis"""
    
    def __init__(self, trade_logger=None):
        self.trade_logger = trade_logger
        
    def calculate_metrics(self, trades: List[TradeRecord]) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        
        metrics = PerformanceMetrics()
        
        if not trades:
            return metrics
            
        # Filter closed trades only
        closed_trades = [t for t in trades if t.status == TradeStatus.CLOSED]
        
        if not closed_trades:
            return metrics
            
        metrics.total_trades = len(closed_trades)
        
        # Separate winners and losers
        winners = [t for t in closed_trades if t.pnl_dollars > 0]
        losers = [t for t in closed_trades if t.pnl_dollars < 0]
        
        metrics.winning_trades = len(winners)
        metrics.losing_trades = len(losers)
        metrics.win_rate = metrics.winning_trades / metrics.total_trades if metrics.total_trades > 0 else 0
        
        # P&L calculations
        metrics.total_pnl = sum(t.pnl_dollars for t in closed_trades)
        metrics.gross_profit = sum(t.pnl_dollars for t in winners) if winners else 0
        metrics.gross_loss = abs(sum(t.pnl_dollars for t in losers)) if losers else 0
        metrics.net_profit = metrics.gross_profit - metrics.gross_loss
        metrics.profit_factor = metrics.gross_profit / metrics.gross_loss if metrics.gross_loss > 0 else float('inf')
        
        # Win/loss averages
        metrics.avg_win = metrics.gross_profit / metrics.winning_trades if metrics.winning_trades > 0 else 0
        metrics.avg_loss = metrics.gross_loss / metrics.losing_trades if metrics.losing_trades > 0 else 0
        metrics.largest_win = max((t.pnl_dollars for t in winners), default=0)
        metrics.largest_loss = min((t.pnl_dollars for t in losers), default=0)
        
        # R-multiple metrics
        r_values = [t.r_multiple for t in closed_trades if t.r_multiple != 0]
        if r_values:
            metrics.avg_r_multiple = np.mean(r_values)
            metrics.best_r_multiple = max(r_values)
            metrics.worst_r_multiple = min(r_values)
            
        # Expectancy
        metrics.expectancy = (metrics.win_rate * metrics.avg_win) - ((1 - metrics.win_rate) * metrics.avg_loss)
        
        # Sharpe ratio (assuming risk-free rate = 0)
        pnl_series = [t.pnl_dollars for t in closed_trades]
        if len(pnl_series) > 1 and np.std(pnl_series) > 0:
            metrics.sharpe_ratio = np.mean(pnl_series) / np.std(pnl_series) * np.sqrt(252)  # Annualized
        
        # Sortino ratio (downside deviation)
        downside_returns = [r for r in pnl_series if r < 0]
        if downside_returns and np.std(downside_returns) > 0:
            metrics.sortino_ratio = np.mean(pnl_series) / np.std(downside_returns) * np.sqrt(252)
            
        # Drawdown analysis
        cumulative = np.cumsum(pnl_series)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = cumulative - running_max
        metrics.max_drawdown = abs(min(drawdowns)) if len(drawdowns) > 0 else 0
        metrics.max_drawdown_pct = metrics.max_drawdown / (metrics.total_pnl + metrics.max_drawdown) if metrics.total_pnl + metrics.max_drawdown > 0 else 0
        metrics.avg_drawdown = abs(np.mean(drawdowns[drawdowns < 0])) if any(drawdowns < 0) else 0
        
        # Duration metrics
        durations = [t.duration_minutes for t in closed_trades if t.duration_minutes > 0]
        if durations:
            metrics.avg_trade_duration_minutes = np.mean(durations)
            metrics.max_trade_duration_minutes = max(durations)
            
        # Consecutive win/loss streaks
        current_win_streak = 0
        current_loss_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        
        for trade in closed_trades:
            if trade.pnl_dollars > 0:
                current_win_streak += 1
                current_loss_streak = 0
                max_win_streak = max(max_win_streak, current_win_streak)
            elif trade.pnl_dollars < 0:
                current_loss_streak += 1
                current_win_streak = 0
                max_loss_streak = max(max_loss_streak, current_loss_streak)
                
        metrics.max_consecutive_wins = max_win_streak
        metrics.max_consecutive_losses = max_loss_streak
        metrics.current_consecutive_wins = current_win_streak
        metrics.current_consecutive_losses = current_loss_streak
        
        # Statistical moments
        if len(pnl_series) > 1:
            metrics.pnl_std = np.std(pnl_series)
            metrics.pnl_skew = pd.Series(pnl_series).skew()
            metrics.pnl_kurtosis = pd.Series(pnl_series).kurtosis()
            
        return metrics
        
    def calculate_daily_summaries(self, trades: List[TradeRecord]) -> List[DailySummary]:
        """Calculate daily performance summaries"""
        
        # Group by date
        trades_by_date = defaultdict(list)
        
        for trade in trades:
            if trade.exit_time:
                date_key = trade.exit_time.date().isoformat()
                trades_by_date[date_key].append(trade)
                
        summaries = []
        
        for date, daily_trades in trades_by_date.items():
            metrics = self.calculate_metrics(daily_trades)
            
            summary = DailySummary(
                date=date,
                total_trades=metrics.total_trades,
                winning_trades=metrics.winning_trades,
                losing_trades=metrics.losing_trades,
                total_pnl=metrics.total_pnl,
                avg_win=metrics.avg_win,
                avg_loss=metrics.avg_loss,
                win_rate=metrics.win_rate,
                profit_factor=metrics.profit_factor,
                largest_win=metrics.largest_win,
                largest_loss=metrics.largest_loss,
                expectancy=metrics.expectancy,
                sharpe_ratio=metrics.sharpe_ratio,
                max_drawdown=metrics.max_drawdown
            )
            
            summaries.append(summary)
            
        return sorted(summaries, key=lambda x: x.date)
        
    def analyze_by_session(self, trades: List[TradeRecord]) -> Dict:
        """Analyze performance by trading session"""
        
        session_stats = defaultdict(lambda: {
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'total_pnl': 0,
            'avg_r': 0,
            'r_values': []
        })
        
        for trade in trades:
            if trade.status == TradeStatus.CLOSED and trade.session:
                session_stats[trade.session]['trades'] += 1
                session_stats[trade.session]['total_pnl'] += trade.pnl_dollars
                session_stats[trade.session]['r_values'].append(trade.r_multiple)
                
                if trade.pnl_dollars > 0:
                    session_stats[trade.session]['wins'] += 1
                elif trade.pnl_dollars < 0:
                    session_stats[trade.session]['losses'] += 1
                    
        # Calculate derived metrics
        result = {}
        for session, stats in session_stats.items():
            total = stats['trades']
            wins = stats['wins']
            
            result[session] = {
                'total_trades': total,
                'win_rate': wins / total if total > 0 else 0,
                'total_pnl': stats['total_pnl'],
                'avg_pnl': stats['total_pnl'] / total if total > 0 else 0,
                'avg_r': np.mean(stats['r_values']) if stats['r_values'] else 0,
                'profit_factor': self._calculate_profit_factor(stats['r_values'])
            }
            
        return result
        
    def analyze_by_model_version(self, trades: List[TradeRecord]) -> Dict:
        """Analyze performance by model version"""
        
        version_stats = defaultdict(lambda: {
            'trades': [],
            'r_values': []
        })
        
        for trade in trades:
            if trade.status == TradeStatus.CLOSED and trade.model_version:
                version_stats[trade.model_version]['trades'].append(trade)
                version_stats[trade.model_version]['r_values'].append(trade.r_multiple)
                
        result = {}
        for version, stats in version_stats.items():
            metrics = self.calculate_metrics(stats['trades'])
            
            result[version] = {
                'total_trades': metrics.total_trades,
                'win_rate': metrics.win_rate,
                'profit_factor': metrics.profit_factor,
                'avg_r': metrics.avg_r_multiple,
                'sharpe_ratio': metrics.sharpe_ratio,
                'max_drawdown': metrics.max_drawdown,
                'total_pnl': metrics.total_pnl
            }
            
        return result
        
    def analyze_by_confidence_bucket(self, trades: List[TradeRecord], buckets: List[float] = None) -> Dict:
        """Analyze performance by confidence score buckets"""
        
        if buckets is None:
            buckets = [0.6, 0.7, 0.8, 0.9]
            
        bucket_stats = defaultdict(lambda: {'trades': [], 'r_values': []})
        
        # Need to join with signals for confidence scores
        # Simplified: assuming trade has confidence_score attribute
        for trade in trades:
            if hasattr(trade, 'confidence_score') and trade.confidence_score > 0:
                for bucket in buckets:
                    if trade.confidence_score <= bucket:
                        bucket_stats[bucket]['trades'].append(trade)
                        bucket_stats[bucket]['r_values'].append(trade.r_multiple)
                        break
                else:
                    bucket_stats[1.0]['trades'].append(trade)
                    bucket_stats[1.0]['r_values'].append(trade.r_multiple)
                    
        result = {}
        for bucket, stats in bucket_stats.items():
            metrics = self.calculate_metrics(stats['trades'])
            
            result[f"{bucket*100:.0f}%"] = {
                'total_trades': metrics.total_trades,
                'win_rate': metrics.win_rate,
                'avg_r': metrics.avg_r_multiple,
                'profit_factor': metrics.profit_factor
            }
            
        return result
        
    def analyze_learning_curve(self, trades: List[TradeRecord], window_size: int = 50) -> Dict:
        """Analyze rolling performance over time"""
        
        if len(trades) < window_size:
            return {'error': f'Need at least {window_size} trades for learning curve'}
            
        # Sort by exit time
        sorted_trades = sorted([t for t in trades if t.exit_time], key=lambda x: x.exit_time)
        
        rolling_wins = []
        rolling_r = []
        rolling_pnl = []
        
        for i in range(window_size, len(sorted_trades) + 1):
            window = sorted_trades[i - window_size:i]
            metrics = self.calculate_metrics(window)
            
            rolling_wins.append(metrics.win_rate)
            rolling_r.append(metrics.avg_r_multiple)
            rolling_pnl.append(metrics.total_pnl / window_size)  # Avg per trade
            
        return {
            'window_size': window_size,
            'rolling_win_rate': rolling_wins,
            'rolling_avg_r': rolling_r,
            'rolling_avg_pnl': rolling_pnl,
            'trend_win_rate': self._calculate_trend(rolling_wins),
            'trend_avg_r': self._calculate_trend(rolling_r)
        }
        
    def _calculate_profit_factor(self, r_values: List[float]) -> float:
        """Calculate profit factor from R-multiples"""
        positive_r = [r for r in r_values if r > 0]
        negative_r = [abs(r) for r in r_values if r < 0]
        
        total_profit = sum(positive_r)
        total_loss = sum(negative_r)
        
        return total_profit / total_loss if total_loss > 0 else float('inf')
        
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction"""
        if len(values) < 2:
            return 'neutral'
            
        # Simple linear regression slope
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.001:
            return 'improving'
        elif slope < -0.001:
            return 'declining'
        else:
            return 'stable'
            
    def generate_report(self, trades: List[TradeRecord]) -> Dict:
        """Generate comprehensive analytics report"""
        
        metrics = self.calculate_metrics(trades)
        daily_summaries = self.calculate_daily_summaries(trades)
        session_analysis = self.analyze_by_session(trades)
        version_analysis = self.analyze_by_model_version(trades)
        learning_curve = self.analyze_learning_curve(trades)
        
        # Calculate monthly returns
        monthly_returns = self._calculate_monthly_returns(trades)
        
        return {
            'summary_metrics': metrics.to_dict(),
            'daily_summaries': [s.to_dict() for s in daily_summaries],
            'session_analysis': session_analysis,
            'model_version_analysis': version_analysis,
            'learning_curve': learning_curve,
            'monthly_returns': monthly_returns,
            'report_generated': datetime.utcnow().isoformat()
        }
        
    def _calculate_monthly_returns(self, trades: List[TradeRecord]) -> Dict:
        """Calculate monthly returns"""
        
        monthly_pnl = defaultdict(float)
        
        for trade in trades:
            if trade.exit_time:
                month_key = trade.exit_time.strftime("%Y-%m")
                monthly_pnl[month_key] += trade.pnl_dollars
                
        return dict(monthly_pnl)
        
    def print_summary(self, metrics: PerformanceMetrics):
        """Print formatted performance summary"""
        
        print("\n" + "="*60)
        print("TRADING PERFORMANCE SUMMARY")
        print("="*60)
        
        print(f"\n📊 TRADE STATISTICS")
        print(f"   Total Trades:     {metrics.total_trades}")
        print(f"   Winning Trades:   {metrics.winning_trades}")
        print(f"   Losing Trades:    {metrics.losing_trades}")
        print(f"   Win Rate:         {metrics.win_rate:.2%}")
        
        print(f"\n💰 P&L STATISTICS")
        print(f"   Total P&L:        ${metrics.total_pnl:,.2f}")
        print(f"   Gross Profit:     ${metrics.gross_profit:,.2f}")
        print(f"   Gross Loss:       ${metrics.gross_loss:,.2f}")
        print(f"   Profit Factor:    {metrics.profit_factor:.2f}")
        print(f"   Expectancy:       ${metrics.expectancy:.2f}")
        
        print(f"\n📈 TRADE QUALITY")
        print(f"   Avg Win:          ${metrics.avg_win:,.2f}")
        print(f"   Avg Loss:         ${metrics.avg_loss:,.2f}")
        print(f"   Largest Win:      ${metrics.largest_win:,.2f}")
        print(f"   Largest Loss:     ${metrics.largest_loss:,.2f}")
        print(f"   Avg R-Multiple:   {metrics.avg_r_multiple:.2f}R")
        
        print(f"\n⚠️ RISK METRICS")
        print(f"   Sharpe Ratio:     {metrics.sharpe_ratio:.2f}")
        print(f"   Sortino Ratio:    {metrics.sortino_ratio:.2f}")
        print(f"   Max Drawdown:     ${metrics.max_drawdown:,.2f} ({metrics.max_drawdown_pct:.2%})")
        print(f"   Avg Drawdown:     ${metrics.avg_drawdown:,.2f}")
        
        print(f"\n⏱️ DURATION")
        print(f"   Avg Trade Duration: {metrics.avg_trade_duration_minutes:.0f} min")
        print(f"   Max Trade Duration: {metrics.max_trade_duration_minutes:.0f} min")
        
        print(f"\n📉 SEQUENCE")
        print(f"   Max Consecutive Wins:   {metrics.max_consecutive_wins}")
        print(f"   Max Consecutive Losses: {metrics.max_consecutive_losses}")
        
        print(f"\n📊 STATISTICAL")
        print(f"   P&L Std Dev:      ${metrics.pnl_std:,.2f}")
        print(f"   P&L Skew:         {metrics.pnl_skew:.2f}")
        print(f"   P&L Kurtosis:     {metrics.pnl_kurtosis:.2f}")
        
        print("\n" + "="*60)
