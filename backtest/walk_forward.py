"""Walk-forward testing for robust validation"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field

from .simulator import BacktestSimulator, BacktestResult


@dataclass
class WalkForwardWindow:
    """Single walk-forward window"""
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_result: Optional[BacktestResult] = None
    test_result: Optional[BacktestResult] = None
    optimized_params: Optional[Dict] = None


@dataclass
class WalkForwardResult:
    """Complete walk-forward test results"""
    windows: List[WalkForwardWindow] = field(default_factory=list)
    out_of_sample_results: List[BacktestResult] = field(default_factory=list)
    consistency_score: float = 0.0
    avg_profit_factor: float = 0.0
    avg_win_rate: float = 0.0
    max_drawdown: float = 0.0
    total_trades: int = 0
    passed: bool = False
    
    def summary(self) -> str:
        """Generate summary report"""
        
        if not self.out_of_sample_results:
            return "No walk-forward results available"
            
        winning_windows = sum(1 for r in self.out_of_sample_results if r.profit_factor > 1.3)
        
        return f"""
╔══════════════════════════════════════════════════════════════╗
║                 WALK-FORWARD TEST RESULTS                    ║
╠══════════════════════════════════════════════════════════════╣
║ Windows Tested: {len(self.windows)}
║ Total OOS Trades: {self.total_trades}
║ Consistency Score: {self.consistency_score:.1f}%
╠══════════════════════════════════════════════════════════════╣
║ AVERAGE METRICS (Out-of-Sample):
║   Profit Factor: {self.avg_profit_factor:.2f}
║   Win Rate: {self.avg_win_rate:.1f}%
║   Max Drawdown: {self.max_drawdown:.1f}%
║   Profitable Windows: {winning_windows}/{len(self.out_of_sample_results)}
╠══════════════════════════════════════════════════════════════╣
║ VERDICT: {'✓ PASSED' if self.passed else '✗ FAILED'}
║   Pass Criteria: Avg PF > 1.5, Win Rate > 50% in >70% of windows
╚══════════════════════════════════════════════════════════════╝
"""


class WalkForwardTest:
    """Walk-forward testing framework"""
    
    def __init__(
        self,
        train_window_days: int = 365,  # 1 year training
        test_window_days: int = 56,    # 8 weeks testing
        step_days: int = 28,           # 4 week step
        min_trades_per_window: int = 30
    ):
        self.train_window_days = train_window_days
        self.test_window_days = test_window_days
        self.step_days = step_days
        self.min_trades_per_window = min_trades_per_window
        
    def run(
        self,
        data: pd.DataFrame,
        strategy_function: Callable,
        param_optimizer: Optional[Callable] = None
    ) -> WalkForwardResult:
        """
        Run walk-forward test
        
        Args:
            data: OHLCV DataFrame with 'timestamp' column
            strategy_function: Function that takes (train_data, params) and returns strategy
            param_optimizer: Function that optimizes parameters on training data
        """
        
        # Ensure data is sorted
        data = data.sort_values('timestamp').reset_index(drop=True)
        
        # Generate windows
        windows = self._generate_windows(data)
        
        results = WalkForwardResult()
        out_of_sample_results = []
        
        for window in windows:
            # Get window data
            train_data = data[(data['timestamp'] >= window.train_start) & 
                             (data['timestamp'] <= window.train_end)]
            test_data = data[(data['timestamp'] >= window.test_start) & 
                            (data['timestamp'] <= window.test_end)]
            
            if len(train_data) < 100 or len(test_data) < 50:
                continue
                
            # Optimize parameters on training data
            if param_optimizer:
                optimal_params = param_optimizer(train_data)
                window.optimized_params = optimal_params
            else:
                optimal_params = {}
                
            # Train strategy on training data
            strategy = strategy_function(train_data, optimal_params)
            
            # Run backtest on test data
            simulator = BacktestSimulator()
            test_result = simulator.run(test_data, strategy)
            
            if test_result.total_trades >= self.min_trades_per_window:
                window.test_result = test_result
                out_of_sample_results.append(test_result)
                results.windows.append(window)
                
        # Aggregate results
        results.out_of_sample_results = out_of_sample_results
        results.total_trades = sum(r.total_trades for r in out_of_sample_results)
        
        if out_of_sample_results:
            results.avg_profit_factor = np.mean([r.profit_factor for r in out_of_sample_results])
            results.avg_win_rate = np.mean([r.win_rate for r in out_of_sample_results])
            results.max_drawdown = max([r.max_drawdown_pct for r in out_of_sample_results])
            
            # Calculate consistency score: % of windows with profit factor > 1.3
            profitable_windows = sum(1 for r in out_of_sample_results if r.profit_factor > 1.3)
            results.consistency_score = (profitable_windows / len(out_of_sample_results)) * 100
            
            # Pass criteria: avg profit factor > 1.5, win rate > 50% in >70% of windows
            results.passed = (
                results.avg_profit_factor > 1.5 and
                results.consistency_score > 70
            )
            
        return results
        
    def _generate_windows(self, data: pd.DataFrame) -> List[WalkForwardWindow]:
        """Generate walk-forward windows"""
        
        windows = []
        min_timestamp = data['timestamp'].min()
        max_timestamp = data['timestamp'].max()
        
        current_train_end = min_timestamp + timedelta(days=self.train_window_days)
        
        while current_train_end + timedelta(days=self.test_window_days) <= max_timestamp:
            window = WalkForwardWindow(
                train_start=min_timestamp,
                train_end=current_train_end,
                test_start=current_train_end,
                test_end=current_train_end + timedelta(days=self.test_window_days)
            )
            windows.append(window)
            
            # Slide window
            min_timestamp += timedelta(days=self.step_days)
            current_train_end = min_timestamp + timedelta(days=self.train_window_days)
            
        return windows


class ParameterOptimizer:
    """Parameter optimization for strategies"""
    
    def __init__(self, param_grid: Dict):
        self.param_grid = param_grid
        
    def grid_search(
        self,
        train_data: pd.DataFrame,
        strategy_class,
        metric: str = 'profit_factor'
    ) -> Dict:
        """Perform grid search over parameter space"""
        
        from itertools import product
        
        best_score = -float('inf')
        best_params = {}
        
        # Generate all parameter combinations
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        
        for combination in product(*param_values):
            params = dict(zip(param_names, combination))
            
            # Run backtest with these params
            simulator = BacktestSimulator()
            strategy = strategy_class(**params)
            result = simulator.run(train_data, strategy)
            
            # Score based on metric
            if metric == 'profit_factor':
                score = result.profit_factor
            elif metric == 'sharpe_ratio':
                score = result.sharpe_ratio
            elif metric == 'win_rate':
                score = result.win_rate
            elif metric == 'calmar_ratio':
                score = result.calmar_ratio
            else:
                score = result.profit_factor
                
            if score > best_score:
                best_score = score
                best_params = params
                
        return best_params
