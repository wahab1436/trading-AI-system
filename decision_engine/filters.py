"""Confidence and quality filters for trade signals"""

import logging
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque

from .signal_generator import TradeSignal, SignalType, SignalStrength

logger = logging.getLogger(__name__)


@dataclass
class FilterResult:
    """Result of applying a filter"""
    passed: bool
    reason: Optional[str] = None
    adjusted_confidence: Optional[float] = None


class BaseFilter:
    """Base class for all filters"""
    
    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled
        
    def apply(self, signal: TradeSignal, context: Dict) -> FilterResult:
        """Apply filter to signal"""
        raise NotImplementedError
        
    def __repr__(self):
        return f"{self.name}(enabled={self.enabled})"


class ConfidenceFilter(BaseFilter):
    """
    Filters signals based on confidence threshold.
    Also provides dynamic threshold adjustment based on market conditions.
    """
    
    def __init__(
        self,
        min_confidence: float = 0.65,
        adaptive: bool = True,
        name: str = "ConfidenceFilter"
    ):
        super().__init__(name)
        self.min_confidence = min_confidence
        self.adaptive = adaptive
        
        # For adaptive threshold
        self.recent_performance = deque(maxlen=50)
        self.base_threshold = min_confidence
        
    def apply(self, signal: TradeSignal, context: Dict) -> FilterResult:
        if not self.enabled:
            return FilterResult(passed=True)
            
        threshold = self._get_adaptive_threshold(context) if self.adaptive else self.min_confidence
        
        if signal.confidence < threshold:
            return FilterResult(
                passed=False,
                reason=f"Confidence {signal.confidence:.2%} < threshold {threshold:.2%}"
            )
            
        return FilterResult(passed=True)
        
    def _get_adaptive_threshold(self, context: Dict) -> float:
        """Adjust threshold based on market volatility and recent performance"""
        
        threshold = self.base_threshold
        
        # Increase threshold in high volatility
        volatility = context.get('volatility_regime', 1.0)
        if volatility > 1.5:
            threshold += 0.10
        elif volatility > 1.2:
            threshold += 0.05
            
        # Adjust based on recent win rate
        if len(self.recent_performance) > 10:
            win_rate = sum(self.recent_performance) / len(self.recent_performance)
            if win_rate < 0.4:
                threshold += 0.10  # Be more selective after losses
            elif win_rate > 0.6:
                threshold -= 0.05  # Can be slightly less selective
                
        return min(max(threshold, 0.50), 0.85)
        
    def update_performance(self, won: bool):
        """Update filter with trade outcome"""
        self.recent_performance.append(1 if won else 0)


class SpreadFilter(BaseFilter):
    """Filters signals when spread is too high"""
    
    def __init__(
        self,
        max_spread_pips: float = 3.0,
        symbol_spreads: Optional[Dict[str, float]] = None,
        name: str = "SpreadFilter"
    ):
        super().__init__(name)
        self.max_spread_pips = max_spread_pips
        self.symbol_spreads = symbol_spreads or {
            "XAUUSD": 0.5,
            "EURUSD": 0.3
        }
        
    def apply(self, signal: TradeSignal, context: Dict) -> FilterResult:
        if not self.enabled:
            return FilterResult(passed=True)
            
        current_spread = context.get('current_spread', 0)
        symbol = signal.symbol
        
        # Get average spread for symbol
        avg_spread = self.symbol_spreads.get(symbol, 0.5)
        max_allowed = avg_spread * 3  # 3x average
        
        if current_spread > max_allowed:
            return FilterResult(
                passed=False,
                reason=f"Spread {current_spread:.2f} > max {max_allowed:.2f}"
            )
            
        return FilterResult(passed=True)


class VolumeFilter(BaseFilter):
    """Filters signals during low volume periods"""
    
    def __init__(
        self,
        min_volume_ratio: float = 0.5,
        name: str = "VolumeFilter"
    ):
        super().__init__(name)
        self.min_volume_ratio = min_volume_ratio
        
    def apply(self, signal: TradeSignal, context: Dict) -> FilterResult:
        if not self.enabled:
            return FilterResult(passed=True)
            
        current_volume = context.get('current_volume', 0)
        average_volume = context.get('average_volume', 1)
        
        if average_volume > 0:
            volume_ratio = current_volume / average_volume
            if volume_ratio < self.min_volume_ratio:
                return FilterResult(
                    passed=False,
                    reason=f"Low volume: {volume_ratio:.2%} of average"
                )
                
        return FilterResult(passed=True)


class TimeFilter(BaseFilter):
    """Filters signals during undesirable trading sessions"""
    
    def __init__(
        self,
        allowed_sessions: Optional[List[str]] = None,
        excluded_hours: Optional[List[int]] = None,
        name: str = "TimeFilter"
    ):
        super().__init__(name)
        self.allowed_sessions = allowed_sessions or ["london", "overlap", "new_york"]
        self.excluded_hours = excluded_hours or [0, 1, 2, 3, 4, 5, 6, 7]  # Asian session
        
    def apply(self, signal: TradeSignal, context: Dict) -> FilterResult:
        if not self.enabled:
            return FilterResult(passed=True)
            
        current_session = context.get('session', 'asian')
        current_hour = datetime.utcnow().hour
        
        # Check session
        if current_session not in self.allowed_sessions:
            return FilterResult(
                passed=False,
                reason=f"Session {current_session} not allowed"
            )
            
        # Check hour
        if current_hour in self.excluded_hours:
            return FilterResult(
                passed=False,
                reason=f"Hour {current_hour} excluded"
            )
            
        # Check weekend
        if datetime.utcnow().weekday() >= 5:  # Saturday or Sunday
            return FilterResult(
                passed=False,
                reason="Weekend trading disabled"
            )
            
        return FilterResult(passed=True)


class NewsFilter(BaseFilter):
    """Filters signals around high-impact news events"""
    
    def __init__(
        self,
        pre_news_minutes: int = 30,
        post_news_minutes: int = 30,
        name: str = "NewsFilter"
    ):
        super().__init__(name)
        self.pre_news_minutes = pre_news_minutes
        self.post_news_minutes = post_news_minutes
        
    def apply(self, signal: TradeSignal, context: Dict) -> FilterResult:
        if not self.enabled:
            return FilterResult(passed=True)
            
        upcoming_news = context.get('upcoming_news', [])
        now = datetime.utcnow()
        
        for news in upcoming_news:
            news_time = news.get('time')
            if not news_time:
                continue
                
            time_until = (news_time - now).total_seconds() / 60
            
            # Check if we're in blackout period
            if -self.post_news_minutes <= time_until <= self.pre_news_minutes:
                return FilterResult(
                    passed=False,
                    reason=f"News blackout: {news.get('event', 'unknown')} at {news_time}"
                )
                
        return FilterResult(passed=True)


class CorrelationFilter(BaseFilter):
    """Filters signals when correlated positions are already open"""
    
    def __init__(
        self,
        correlation_threshold: float = 0.7,
        name: str = "CorrelationFilter"
    ):
        super().__init__(name)
        self.correlation_threshold = correlation_threshold
        
        # Correlation matrix for symbols
        self.correlations = {
            "XAUUSD": {"EURUSD": 0.65, "GBPUSD": 0.60, "USDJPY": -0.40},
            "EURUSD": {"XAUUSD": 0.65, "GBPUSD": 0.85, "USDJPY": -0.50}
        }
        
    def apply(self, signal: TradeSignal, context: Dict) -> FilterResult:
        if not self.enabled:
            return FilterResult(passed=True)
            
        open_positions = context.get('open_positions', [])
        symbol = signal.symbol
        
        for position in open_positions:
            pos_symbol = position.get('symbol')
            pos_side = position.get('side')
            
            if pos_symbol == symbol:
                # Same symbol - check direction
                if pos_side == signal.signal_type.value:
                    return FilterResult(
                        passed=False,
                        reason=f"Already have {pos_side} position in {symbol}"
                    )
            else:
                # Different symbol - check correlation
                corr = self.correlations.get(symbol, {}).get(pos_symbol, 0)
                if abs(corr) > self.correlation_threshold:
                    # Same direction on highly correlated symbols
                    if (pos_side == "BUY" and signal.signal_type == SignalType.BUY) or \
                       (pos_side == "SELL" and signal.signal_type == SignalType.SELL):
                        return FilterResult(
                            passed=False,
                            reason=f"Correlated position {pos_symbol} ({pos_side}) already open (corr={corr:.2f})"
                        )
                        
        return FilterResult(passed=True)


class MaxDrawdownFilter(BaseFilter):
    """Filters signals when account drawdown exceeds limit"""
    
    def __init__(
        self,
        max_drawdown_pct: float = 0.10,
        name: str = "MaxDrawdownFilter"
    ):
        super().__init__(name)
        self.max_drawdown_pct = max_drawdown_pct
        self.peak_balance = None
        
    def apply(self, signal: TradeSignal, context: Dict) -> FilterResult:
        if not self.enabled:
            return FilterResult(passed=True)
            
        current_balance = context.get('account_balance', 0)
        
        if self.peak_balance is None:
            self.peak_balance = current_balance
        else:
            self.peak_balance = max(self.peak_balance, current_balance)
            
        if self.peak_balance > 0:
            drawdown = (self.peak_balance - current_balance) / self.peak_balance
            
            if drawdown > self.max_drawdown_pct:
                return FilterResult(
                    passed=False,
                    reason=f"Drawdown {drawdown:.2%} exceeds limit {self.max_drawdown_pct:.2%}"
                )
                
        return FilterResult(passed=True)


class ConsistencyFilter(BaseFilter):
    """Filters signals based on recent performance consistency"""
    
    def __init__(
        self,
        min_win_rate: float = 0.40,
        lookback_trades: int = 20,
        name: str = "ConsistencyFilter"
    ):
        super().__init__(name)
        self.min_win_rate = min_win_rate
        self.lookback_trades = lookback_trades
        self.recent_outcomes = deque(maxlen=lookback_trades)
        
    def apply(self, signal: TradeSignal, context: Dict) -> FilterResult:
        if not self.enabled:
            return FilterResult(passed=True)
            
        if len(self.recent_outcomes) >= self.lookback_trades // 2:
            win_rate = sum(self.recent_outcomes) / len(self.recent_outcomes)
            
            if win_rate < self.min_win_rate:
                return FilterResult(
                    passed=False,
                    reason=f"Recent win rate {win_rate:.2%} below {self.min_win_rate:.2%}"
                )
                
        return FilterResult(passed=True)
        
    def update_outcome(self, won: bool):
        """Update with trade outcome"""
        self.recent_outcomes.append(1 if won else 0)


class ConfluenceFilter(BaseFilter):
    """
    Ensures signal has sufficient confluence from multiple factors.
    Filters out signals that are too isolated or have conflicting indicators.
    """
    
    def __init__(
        self,
        min_confluence_score: int = 3,
        name: str = "ConfluenceFilter"
    ):
        super().__init__(name)
        self.min_confluence_score = min_confluence_score
        
    def apply(self, signal: TradeSignal, context: Dict) -> FilterResult:
        if not self.enabled:
            return FilterResult(passed=True)
            
        confluence_score = self._calculate_confluence_score(signal, context)
        
        if confluence_score < self.min_confluence_score:
            return FilterResult(
                passed=False,
                reason=f"Confluence score {confluence_score} < {self.min_confluence_score}"
            )
            
        # Optionally adjust confidence based on confluence
        adjusted_confidence = min(signal.confidence * (1 + confluence_score * 0.05), 0.95)
        
        return FilterResult(
            passed=True,
            adjusted_confidence=adjusted_confidence
        )
        
    def _calculate_confluence_score(self, signal: TradeSignal, context: Dict) -> int:
        """Calculate confluence score from multiple factors"""
        
        score = 0
        
        # HTF alignment
        if signal.htf_bias == (1 if signal.signal_type == SignalType.BUY else -1):
            score += 2
        elif signal.htf_bias == 0:
            score += 1
            
        # Order block proximity
        if signal.nearest_ob_distance and signal.nearest_ob_distance < 1.5:
            score += 2
        elif signal.nearest_ob_distance and signal.nearest_ob_distance < 3.0:
            score += 1
            
        # FVG presence
        if signal.fvg_present:
            score += 2
            
        # BOS confirmation
        if signal.bos_recent:
            score += 1
            
        # CHoCH detection (strong signal)
        if signal.choch_detected:
            score += 2
            
        # Market state alignment
        if signal.market_state == "trending_bull" and signal.signal_type == SignalType.BUY:
            score += 2
        elif signal.market_state == "trending_bear" and signal.signal_type == SignalType.SELL:
            score += 2
            
        return score


class FilterChain:
    """
    Chains multiple filters together for sequential processing.
    Stops at first failing filter.
    """
    
    def __init__(self, filters: Optional[List[BaseFilter]] = None):
        self.filters = filters or []
        
    def add_filter(self, filter_obj: BaseFilter) -> 'FilterChain':
        """Add a filter to the chain"""
        self.filters.append(filter_obj)
        return self
        
    def remove_filter(self, filter_name: str) -> bool:
        """Remove a filter by name"""
        for i, f in enumerate(self.filters):
            if f.name == filter_name:
                self.filters.pop(i)
                return True
        return False
        
    def apply_all(
        self,
        signal: TradeSignal,
        context: Dict
    ) -> Tuple[bool, List[FilterResult], Optional[float]]:
        """
        Apply all filters in sequence.
        
        Returns:
            (passed, results, adjusted_confidence)
        """
        results = []
        adjusted_confidence = signal.confidence
        
        for filter_obj in self.filters:
            result = filter_obj.apply(signal, context)
            results.append(result)
            
            if not result.passed:
                logger.debug(f"Signal rejected by {filter_obj.name}: {result.reason}")
                return False, results, None
                
            if result.adjusted_confidence:
                adjusted_confidence = result.adjusted_confidence
                
        return True, results, adjusted_confidence
        
    def get_enabled_filters(self) -> List[BaseFilter]:
        """Get list of enabled filters"""
        return [f for f in self.filters if f.enabled]
        
    def get_filter_status(self) -> Dict[str, bool]:
        """Get status of all filters"""
        return {f.name: f.enabled for f in self.filters}
        
    def enable_all(self):
        """Enable all filters"""
        for f in self.filters:
            f.enabled = True
            
    def disable_all(self):
        """Disable all filters (for testing)"""
        for f in self.filters:
            f.enabled = False


class AdaptiveFilterChain(FilterChain):
    """
    Filter chain that adapts based on market conditions.
    Can enable/disable filters dynamically.
    """
    
    def __init__(self, filters: Optional[List[BaseFilter]] = None):
        super().__init__(filters)
        self.market_regime = "normal"  # normal, high_volatility, low_volatility, trending
        
    def update_market_regime(self, context: Dict):
        """Update market regime based on context"""
        
        volatility = context.get('volatility_regime', 1.0)
        
        if volatility > 1.5:
            self.market_regime = "high_volatility"
        elif volatility < 0.7:
            self.market_regime = "low_volatility"
        else:
            self.market_regime = "normal"
            
        # Adjust filters based on regime
        self._adjust_filters()
        
    def _adjust_filters(self):
        """Enable/disable filters based on market regime"""
        
        for f in self.filters:
            if f.name == "SpreadFilter":
                # Stricter spread limits in high volatility
                if hasattr(f, 'max_spread_pips'):
                    if self.market_regime == "high_volatility":
                        f.max_spread_pips = 2.0
                    else:
                        f.max_spread_pips = 3.0
                        
            elif f.name == "ConfidenceFilter":
                # Higher confidence requirement in high volatility
                if hasattr(f, 'min_confidence'):
                    if self.market_regime == "high_volatility":
                        f.min_confidence = 0.75
                    elif self.market_regime == "low_volatility":
                        f.min_confidence = 0.60
                    else:
                        f.min_confidence = 0.65
                        
    def apply_all(
        self,
        signal: TradeSignal,
        context: Dict
    ) -> Tuple[bool, List[FilterResult], Optional[float]]:
        """Apply filters with market regime awareness"""
        
        # Update regime first
        self.update_market_regime(context)
        
        return super().apply_all(signal, context)


# Pre-configured filter chains for different trading modes

def get_paper_trading_filters() -> FilterChain:
    """Get filter chain for paper trading (more lenient)"""
    return FilterChain([
        ConfidenceFilter(min_confidence=0.60, adaptive=False),
        SpreadFilter(max_spread_pips=4.0),
        TimeFilter(allowed_sessions=["london", "overlap", "new_york"]),
        ConfluenceFilter(min_confluence_score=2)
    ])


def get_live_trading_filters() -> FilterChain:
    """Get filter chain for live trading (stricter)"""
    return FilterChain([
        ConfidenceFilter(min_confidence=0.70, adaptive=True),
        SpreadFilter(max_spread_pips=2.5),
        VolumeFilter(min_volume_ratio=0.6),
        TimeFilter(allowed_sessions=["london", "overlap"]),
        NewsFilter(pre_news_minutes=30, post_news_minutes=30),
        CorrelationFilter(correlation_threshold=0.7),
        MaxDrawdownFilter(max_drawdown_pct=0.08),
        ConfluenceFilter(min_confluence_score=3)
    ])


def get_aggressive_filters() -> FilterChain:
    """Get filter chain for aggressive trading (more signals)"""
    return FilterChain([
        ConfidenceFilter(min_confidence=0.55, adaptive=False),
        SpreadFilter(max_spread_pips=5.0),
        ConfluenceFilter(min_confluence_score=1)
    ])


def get_conservative_filters() -> FilterChain:
    """Get filter chain for conservative trading (fewer, higher quality signals)"""
    return FilterChain([
        ConfidenceFilter(min_confidence=0.80, adaptive=True),
        SpreadFilter(max_spread_pips=1.5),
        VolumeFilter(min_volume_ratio=0.8),
        TimeFilter(allowed_sessions=["overlap"]),
        NewsFilter(pre_news_minutes=45, post_news_minutes=45),
        CorrelationFilter(correlation_threshold=0.5),
        MaxDrawdownFilter(max_drawdown_pct=0.05),
        ConsistencyFilter(min_win_rate=0.50, lookback_trades=30),
        ConfluenceFilter(min_confluence_score=4)
    ])
