"""Monte Carlo simulation for risk analysis"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class MonteCarloResult:
    """Monte Carlo simulation results"""
    n_simulations: int = 10000
    final_equity_mean: float = 0.0
    final_equity_std: float = 0.0
    max_drawdown_mean: float = 0.0
    max_drawdown_std: float = 0.0
    ruin_probability: float = 0.0  # Probability of >30% drawdown
    profit_probability: float = 0.0
    
    # Percentiles
    percentile_5: float = 0.0
    percentile_25: float = 0.0
    percentile_50: float = 0.0
    percentile_75: float = 0.0
    percentile_95: float = 0.0
    
    # Drawdown distribution
    drawdown_percentiles: Dict = field(default_factory=dict)
    
    # Recovery time distribution
    recovery_times: List[int] = field(default_factory=list)
    
    # All simulation paths
    simulation_paths: List[List[float]] = field(default_factory=list)
    
    def summary(self) -> str:
        """Generate summary report"""
        
        return f"""
╔══════════════════════════════════════════════════════════════╗
║                  MONTE CARLO SIMULATION                      ║
╠══════════════════════════════════════════════════════════════╣
║ Simulations: {self.n_simulations:,}
║ Ruin Probability (>30% DD): {self.ruin_probability:.2f}%
║ Probability of Profit: {self.profit_probability:.2f}%
╠══════════════════════════════════════════════════════════════╣
║ FINAL EQUITY PERCENTILES:
║   5%: ${self.percentile_5:,.2f}
║   25%: ${self.percentile_25:,.2f}
║   50%: ${self.percentile_50:,.2f}
║   75%: ${self.percentile_75:,.2f}
║   95%: ${self.percentile_95:,.2f}
╠══════════════════════════════════════════════════════════════╣
║ MAX DRAWDOWN PERCENTILES:
║   5%: {self.drawdown_percentiles.get(5, 0):.1f}%
║   25%: {self.drawdown_percentiles.get(25, 0):.1f}%
║   50%: {self.drawdown_percentiles.get(50, 0):.1f}%
║   75%: {self.drawdown_percentiles.get(75, 0):.1f}%
║   95%: {self.drawdown_percentiles.get(95, 0):.1f}%
╚══════════════════════════════════════════════════════════════╝
"""


class MonteCarloSimulator:
    """Monte Carlo simulation for trading strategy risk analysis"""
    
    def __init__(self, n_simulations: int = 10000):
        self.n_simulations = n_simulations
        self.random_seed = 42
        
    def run(
        self,
        trades: List,
        initial_capital: float = 10000.0,
        max_drawdown_threshold: float = 0.30,  # 30% drawdown = ruin
        bootstrap: bool = True
    ) -> MonteCarloResult:
        """
        Run Monte Carlo simulation on trade sequence
        
        Args:
            trades: List of trade objects with pnl_dollars attribute
            initial_capital: Starting capital
            max_drawdown_threshold: Drawdown % considered as ruin
            bootstrap: Use bootstrap resampling (True) or shuffle (False)
        """
        
        if not trades:
            return MonteCarloResult()
            
        np.random.seed(self.random_seed)
        
        # Extract P&L sequence
        pnl_sequence = [t.pnl_dollars for t in trades]
        
        # Run simulations
        final_equities = []
        max_drawdowns = []
        all_paths = []
        
        for sim in range(self.n_simulations):
            if bootstrap:
                # Bootstrap resampling (with replacement)
                simulated_pnl = np.random.choice(pnl_sequence, size=len(pnl_sequence), replace=True)
            else:
                # Random shuffle (without replacement)
                simulated_pnl = np.random.permutation(pnl_sequence)
                
            # Calculate equity curve
            equity_curve = [initial_capital]
            for pnl in simulated_pnl:
                equity_curve.append(equity_curve[-1] + pnl)
                
            # Calculate metrics
            final_equity = equity_curve[-1]
            max_dd = self._calculate_max_drawdown(equity_curve)
            
            final_equities.append(final_equity)
            max_drawdowns.append(max_dd)
            all_paths.append(equity_curve)
            
        # Calculate statistics
        final_equities = np.array(final_equities)
        max_drawdowns = np.array(max_drawdowns)
        
        # Ruin probability (drawdown > threshold)
        ruin_probability = np.mean(max_drawdowns > max_drawdown_threshold) * 100
        
        # Profit probability
        profit_probability = np.mean(final_equities > initial_capital) * 100
        
        # Percentiles
        percentiles = [5, 25, 50, 75, 95]
        final_equity_percentiles = np.percentile(final_equities, percentiles)
        drawdown_percentiles = np.percentile(max_drawdowns, percentiles)
        
        # Calculate recovery time distribution
        recovery_times = self._calculate_recovery_times(all_paths, initial_capital)
        
        return MonteCarloResult(
            n_simulations=self.n_simulations,
            final_equity_mean=np.mean(final_equities),
            final_equity_std=np.std(final_equities),
            max_drawdown_mean=np.mean(max_drawdowns),
            max_drawdown_std=np.std(max_drawdowns),
            ruin_probability=ruin_probability,
            profit_probability=profit_probability,
            percentile_5=final_equity_percentiles[0],
            percentile_25=final_equity_percentiles[1],
            percentile_50=final_equity_percentiles[2],
            percentile_75=final_equity_percentiles[3],
            percentile_95=final_equity_percentiles[4],
            drawdown_percentiles={
                5: drawdown_percentiles[0],
                25: drawdown_percentiles[1],
                50: drawdown_percentiles[2],
                75: drawdown_percentiles[3],
                95: drawdown_percentiles[4]
            },
            recovery_times=recovery_times,
            simulation_paths=all_paths[:100]  # Store first 100 for visualization
        )
        
    def run_with_position_sizing(
        self,
        trades: List,
        initial_capital: float = 10000.0,
        risk_per_trade: float = 0.01,
        max_drawdown_threshold: float = 0.30
    ) -> MonteCarloResult:
        """
        Run Monte Carlo with dynamic position sizing based on risk percentage
        
        Args:
            trades: List of trades with r_multiple attribute
            initial_capital: Starting capital
            risk_per_trade: Percentage of capital to risk per trade
            max_drawdown_threshold: Drawdown % considered as ruin
        """
        
        np.random.seed(self.random_seed)
        
        # Extract R-multiples
        r_multiples = [t.r_multiple for t in trades]
        
        final_equities = []
        max_drawdowns = []
        
        for sim in range(self.n_simulations):
            # Bootstrap R-multiples
            simulated_r = np.random.choice(r_multiples, size=len(r_multiples), replace=True)
            
            # Simulate equity curve with compounding
            equity = initial_capital
            equity_curve = [equity]
            peak = equity
            
            for r in simulated_r:
                # Risk amount for this trade
                risk_amount = equity * risk_per_trade
                
                # P&L = risk_amount * r_multiple
                pnl = risk_amount * r
                equity += pnl
                equity_curve.append(equity)
                
                # Track peak for drawdown
                if equity > peak:
                    peak = equity
                    
            max_dd = (peak - min(equity_curve)) / peak * 100 if peak > 0 else 0
            
            final_equities.append(equity)
            max_drawdowns.append(max_dd)
            
        # Calculate statistics
        final_equities = np.array(final_equities)
        max_drawdowns = np.array(max_drawdowns)
        
        ruin_probability = np.mean(max_drawdowns > max_drawdown_threshold * 100) * 100
        profit_probability = np.mean(final_equities > initial_capital) * 100
        
        return MonteCarloResult(
            n_simulations=self.n_simulations,
            final_equity_mean=np.mean(final_equities),
            final_equity_std=np.std(final_equities),
            max_drawdown_mean=np.mean(max_drawdowns),
            max_drawdown_std=np.std(max_drawdowns),
            ruin_probability=ruin_probability,
            profit_probability=profit_probability,
            percentile_5=np.percentile(final_equities, 5),
            percentile_25=np.percentile(final_equities, 25),
            percentile_50=np.percentile(final_equities, 50),
            percentile_75=np.percentile(final_equities, 75),
            percentile_95=np.percentile(final_equities, 95),
            drawdown_percentiles={
                5: np.percentile(max_drawdowns, 5),
                25: np.percentile(max_drawdowns, 25),
                50: np.percentile(max_drawdowns, 50),
                75: np.percentile(max_drawdowns, 75),
                95: np.percentile(max_drawdowns, 95)
            }
        )
        
    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Calculate maximum drawdown percentage"""
        peak = equity_curve[0]
        max_dd = 0
        
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100 if peak > 0 else 0
            max_dd = max(max_dd, dd)
            
        return max_dd
        
    def _calculate_recovery_times(
        self,
        paths: List[List[float]],
        initial_capital: float
    ) -> List[int]:
        """Calculate time to recover from max drawdown"""
        
        recovery_times = []
        
        for path in paths:
            # Find max drawdown period
            peak = initial_capital
            peak_idx = 0
            trough_idx = 0
            trough = initial_capital
            
            for i, eq in enumerate(path):
                if eq > peak:
                    peak = eq
                    peak_idx = i
                elif eq < trough:
                    trough = eq
                    trough_idx = i
                    
            # Find recovery to peak after trough
            recovery_idx = trough_idx
            for i in range(trough_idx, len(path)):
                if path[i] >= peak:
                    recovery_idx = i
                    break
                    
            recovery_time = recovery_idx - trough_idx
            if recovery_time > 0:
                recovery_times.append(recovery_time)
                
        return recovery_times
        
    def calculate_confidence_interval(
        self,
        pnl_series: List[float],
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate confidence interval for expected P&L"""
        
        if not pnl_series:
            return (0, 0)
            
        # Bootstrap confidence interval
        np.random.seed(self.random_seed)
        bootstrap_means = []
        
        for _ in range(10000):
            sample = np.random.choice(pnl_series, size=len(pnl_series), replace=True)
            bootstrap_means.append(np.mean(sample))
            
        lower = np.percentile(bootstrap_means, (1 - confidence) / 2 * 100)
        upper = np.percentile(bootstrap_means, (1 + confidence) / 2 * 100)
        
        return (lower, upper)
