"""Performance metrics calculation for backtest results"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from scipy import stats


class PerformanceMetrics:
    """Advanced performance metrics calculator"""
    
    @staticmethod
    def calculate_all(trades: List, equity_curve: List[float], returns: List[float]) -> Dict:
        """Calculate all performance metrics"""
        
        return {
            # Return metrics
            'total_return': PerformanceMetrics.total_return(equity_curve),
            'annualized_return': PerformanceMetrics.annualized_return(returns),
            'cumulative_return': PerformanceMetrics.cumulative_return(equity_curve),
            
            # Risk metrics
            'volatility': PerformanceMetrics.volatility(returns),
            'max_drawdown': PerformanceMetrics.max_drawdown(equity_curve),
            'max_drawdown_duration': PerformanceMetrics.max_drawdown_duration(equity_curve),
            'value_at_risk': PerformanceMetrics.value_at_risk(returns, 0.95),
            'conditional_var': PerformanceMetrics.conditional_var(returns, 0.95),
            
            # Risk-adjusted returns
            'sharpe_ratio': PerformanceMetrics.sharpe_ratio(returns),
            'sortino_ratio': PerformanceMetrics.sortino_ratio(returns),
            'calmar_ratio': PerformanceMetrics.calmar_ratio(returns, equity_curve),
            'sterling_ratio': PerformanceMetrics.sterling_ratio(returns, equity_curve),
            
            # Trade statistics
            'win_rate': PerformanceMetrics.win_rate(trades),
            'profit_factor': PerformanceMetrics.profit_factor(trades),
            'expectancy': PerformanceMetrics.expectancy(trades),
            'average_r_multiple': PerformanceMetrics.avg_r_multiple(trades),
            'recovery_factor': PerformanceMetrics.recovery_factor(trades, equity_curve),
            
            # Consistency metrics
            'z_score': PerformanceMetrics.z_score(trades),
            'kelly_criterion': PerformanceMetrics.kelly_criterion(trades),
            'system_quality_number': PerformanceMetrics.system_quality_number(trades),
        }
        
    @staticmethod
    def total_return(equity_curve: List[float]) -> float:
        """Calculate total return percentage"""
        if not equity_curve or equity_curve[0] == 0:
            return 0
        return (equity_curve[-1] - equity_curve[0]) / equity_curve[0] * 100
        
    @staticmethod
    def annualized_return(returns: List[float], periods_per_year: int = 252 * 26) -> float:
        """Calculate annualized return (15m candles = 26 per day)"""
        if not returns:
            return 0
        total_return = np.prod([1 + r for r in returns]) - 1
        years = len(returns) / periods_per_year
        if years <= 0:
            return 0
        return (1 + total_return) ** (1 / years) - 1
        
    @staticmethod
    def cumulative_return(equity_curve: List[float]) -> List[float]:
        """Calculate cumulative return series"""
        if not equity_curve or equity_curve[0] == 0:
            return []
        return [(eq - equity_curve[0]) / equity_curve[0] * 100 for eq in equity_curve]
        
    @staticmethod
    def volatility(returns: List[float], periods_per_year: int = 252 * 26) -> float:
        """Calculate annualized volatility"""
        if len(returns) < 2:
            return 0
        return np.std(returns) * np.sqrt(periods_per_year) * 100
        
    @staticmethod
    def max_drawdown(equity_curve: List[float]) -> float:
        """Calculate maximum drawdown percentage"""
        if not equity_curve:
            return 0
            
        peak = equity_curve[0]
        max_dd = 0
        
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100
            max_dd = max(max_dd, dd)
            
        return max_dd
        
    @staticmethod
    def max_drawdown_duration(equity_curve: List[float]) -> int:
        """Calculate maximum drawdown duration in periods"""
        if not equity_curve:
            return 0
            
        peak_idx = 0
        max_duration = 0
        current_duration = 0
        
        for i, eq in enumerate(equity_curve):
            if eq >= equity_curve[peak_idx]:
                peak_idx = i
                current_duration = 0
            else:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
                
        return max_duration
        
    @staticmethod
    def value_at_risk(returns: List[float], confidence: float = 0.95) -> float:
        """Calculate Value at Risk (VaR)"""
        if not returns:
            return 0
        return np.percentile(returns, (1 - confidence) * 100) * 100
        
    @staticmethod
    def conditional_var(returns: List[float], confidence: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (CVaR / Expected Shortfall)"""
        if not returns:
            return 0
        var = np.percentile(returns, (1 - confidence) * 100)
        tail_returns = [r for r in returns if r <= var]
        if not tail_returns:
            return var * 100
        return np.mean(tail_returns) * 100
        
    @staticmethod
    def sharpe_ratio(returns: List[float], risk_free_rate: float = 0.02, periods_per_year: int = 252 * 26) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2 or np.std(returns) == 0:
            return 0
            
        excess_returns = [r - risk_free_rate / periods_per_year for r in returns]
        return (np.mean(excess_returns) / np.std(returns)) * np.sqrt(periods_per_year)
        
    @staticmethod
    def sortino_ratio(returns: List[float], risk_free_rate: float = 0.02, periods_per_year: int = 252 * 26) -> float:
        """Calculate Sortino ratio (uses downside deviation only)"""
        if len(returns) < 2:
            return 0
            
        excess_returns = [r - risk_free_rate / periods_per_year for r in returns]
        downside_returns = [r for r in excess_returns if r < 0]
        
        if not downside_returns or np.std(downside_returns) == 0:
            return 0 if np.mean(excess_returns) <= 0 else float('inf')
            
        return (np.mean(excess_returns) / np.std(downside_returns)) * np.sqrt(periods_per_year)
        
    @staticmethod
    def calmar_ratio(returns: List[float], equity_curve: List[float]) -> float:
        """Calculate Calmar ratio (annualized return / max drawdown)"""
        annual_ret = PerformanceMetrics.annualized_return(returns)
        max_dd = PerformanceMetrics.max_drawdown(equity_curve)
        
        if max_dd == 0:
            return 0 if annual_ret <= 0 else float('inf')
            
        return annual_ret / max_dd
        
    @staticmethod
    def sterling_ratio(returns: List[float], equity_curve: List[float]) -> float:
        """Calculate Sterling ratio (annualized return / average drawdown)"""
        annual_ret = PerformanceMetrics.annualized_return(returns)
        
        # Calculate average of top 3 drawdowns
        equity_array = np.array(equity_curve)
        peak = equity_array[0]
        drawdowns = []
        
        for eq in equity_array:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100
            drawdowns.append(dd)
            
        top_dd = sorted(drawdowns, reverse=True)[:3]
        avg_dd = np.mean(top_dd) if top_dd else 0
        
        if avg_dd == 0:
            return 0
            
        return annual_ret / avg_dd
        
    @staticmethod
    def win_rate(trades) -> float:
        """Calculate win rate percentage"""
        if not trades:
            return 0
        winning = sum(1 for t in trades if t.pnl_dollars > 0)
        return winning / len(trades) * 100
        
    @staticmethod
    def profit_factor(trades) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        gross_profit = sum(t.pnl_dollars for t in trades if t.pnl_dollars > 0)
        gross_loss = abs(sum(t.pnl_dollars for t in trades if t.pnl_dollars < 0))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0
            
        return gross_profit / gross_loss
        
    @staticmethod
    def expectancy(trades) -> float:
        """Calculate average P&L per trade"""
        if not trades:
            return 0
        return np.mean([t.pnl_dollars for t in trades])
        
    @staticmethod
    def avg_r_multiple(trades) -> float:
        """Calculate average risk multiple"""
        if not trades:
            return 0
        return np.mean([t.r_multiple for t in trades])
        
    @staticmethod
    def recovery_factor(trades, equity_curve: List[float]) -> float:
        """Calculate recovery factor (total profit / max drawdown)"""
        total_profit = sum(t.pnl_dollars for t in trades)
        max_dd = PerformanceMetrics.max_drawdown(equity_curve)
        
        if max_dd == 0:
            return 0
            
        return total_profit / max_dd
        
    @staticmethod
    def z_score(trades) -> float:
        """Calculate Z-score for trade sequence (tests for non-randomness)"""
        if len(trades) < 2:
            return 0
            
        # Convert to win/loss sequence
        sequence = [1 if t.pnl_dollars > 0 else 0 for t in trades]
        
        # Calculate runs
        runs = 1
        for i in range(1, len(sequence)):
            if sequence[i] != sequence[i-1]:
                runs += 1
                
        n = len(sequence)
        n1 = sum(sequence)
        n2 = n - n1
        
        if n1 == 0 or n2 == 0:
            return 0
            
        expected_runs = (2 * n1 * n2) / n + 1
        std_runs = np.sqrt((2 * n1 * n2 * (2 * n1 * n2 - n)) / (n ** 2 * (n - 1)))
        
        if std_runs == 0:
            return 0
            
        return (runs - expected_runs) / std_runs
        
    @staticmethod
    def kelly_criterion(trades) -> float:
        """Calculate Kelly Criterion for optimal position sizing"""
        if not trades:
            return 0
            
        win_rate = PerformanceMetrics.win_rate(trades) / 100
        avg_win = np.mean([t.pnl_dollars for t in trades if t.pnl_dollars > 0]) if any(t.pnl_dollars > 0 for t in trades) else 0
        avg_loss = abs(np.mean([t.pnl_dollars for t in trades if t.pnl_dollars < 0])) if any(t.pnl_dollars < 0 for t in trades) else 0
        
        if avg_loss == 0:
            return win_rate
            
        b = avg_win / avg_loss  # Win/loss ratio
        kelly = win_rate - ((1 - win_rate) / b)
        
        return max(0, min(kelly, 0.25))  # Cap at 25%
        
    @staticmethod
    def system_quality_number(trades) -> float:
        """Calculate System Quality Number (SQN)"""
        if len(trades) < 30:
            return 0
            
        r_multiples = [t.r_multiple for t in trades]
        mean_r = np.mean(r_multiples)
        std_r = np.std(r_multiples)
        
        if std_r == 0:
            return 0
            
        return mean_r / std_r * np.sqrt(len(trades))
        
    @staticmethod
    def monte_carlo_confidence(pnl_series: List[float], n_simulations: int = 10000) -> Dict:
        """Calculate Monte Carlo confidence intervals"""
        if not pnl_series:
            return {}
            
        # Bootstrap resampling
        np.random.seed(42)
        simulated_returns = []
        
        for _ in range(n_simulations):
            resampled = np.random.choice(pnl_series, size=len(pnl_series), replace=True)
            simulated_returns.append(np.sum(resampled))
            
        return {
            'mean': np.mean(simulated_returns),
            'std': np.std(simulated_returns),
            'percentile_5': np.percentile(simulated_returns, 5),
            'percentile_25': np.percentile(simulated_returns, 25),
            'percentile_50': np.percentile(simulated_returns, 50),
            'percentile_75': np.percentile(simulated_returns, 75),
            'percentile_95': np.percentile(simulated_returns, 95),
            'probability_positive': np.mean(np.array(simulated_returns) > 0) * 100,
            'probability_ruin': np.mean(np.array(simulated_returns) < -10000) * 100  # $10k loss
        }


class TradeMetrics:
    """Individual trade analysis metrics"""
    
    @staticmethod
    def analyze_trade(trade) -> Dict:
        """Analyze a single trade"""
        return {
            'duration_minutes': trade.duration_minutes,
            'pnl_dollars': trade.pnl_dollars,
            'pnl_pips': trade.pnl_pips,
            'r_multiple': trade.r_multiple,
            'exit_reason': trade.exit_reason,
            'is_winner': trade.pnl_dollars > 0,
            'is_breaker': trade.r_multiple >= 2.0,  # 2R+ winner
            'is_home_run': trade.r_multiple >= 5.0  # 5R+ winner
        }
        
    @staticmethod
    def analyze_by_session(trades, session_tagger) -> Dict:
        """Analyze trades by trading session"""
        session_stats = {}
        
        for trade in trades:
            session = session_tagger.get_session(trade.entry_time)
            if session not in session_stats:
                session_stats[session] = {'trades': [], 'pnl': 0, 'wins': 0}
                
            session_stats[session]['trades'].append(trade)
            session_stats[session]['pnl'] += trade.pnl_dollars
            if trade.pnl_dollars > 0:
                session_stats[session]['wins'] += 1
                
        # Calculate metrics per session
        for session, stats in session_stats.items():
            stats['win_rate'] = stats['wins'] / len(stats['trades']) * 100 if stats['trades'] else 0
            stats['avg_pnl'] = stats['pnl'] / len(stats['trades']) if stats['trades'] else 0
            
        return session_stats
        
    @staticmethod
    def analyze_by_day_of_week(trades) -> Dict:
        """Analyze trades by day of week"""
        day_stats = {i: {'trades': [], 'pnl': 0} for i in range(5)}  # Mon-Fri
        
        for trade in trades:
            dow = trade.entry_time.weekday()
            if dow < 5:
                day_stats[dow]['trades'].append(trade)
                day_stats[dow]['pnl'] += trade.pnl_dollars
                
        return {
            'Monday': day_stats[0],
            'Tuesday': day_stats[1],
            'Wednesday': day_stats[2],
            'Thursday': day_stats[3],
            'Friday': day_stats[4]
        }
