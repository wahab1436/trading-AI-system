"""Trade signal generation from CNN + SMC + Fusion model outputs"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, List, Tuple
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Types of trading signals"""
    BUY = "BUY"
    SELL = "SELL"
    NO_TRADE = "NO_TRADE"
    HOLD = "HOLD"  # For existing positions


class SignalStrength(Enum):
    """Signal strength levels"""
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    VERY_STRONG = 4


@dataclass
class TradeSignal:
    """Complete trade signal with all metadata"""
    
    # Core signal data
    signal_type: SignalType
    confidence: float
    strength: SignalStrength
    
    # Model outputs
    buy_probability: float
    sell_probability: float
    notrade_probability: float
    
    # CNN embedding info
    cnn_embedding_hash: Optional[str] = None
    pattern_detected: Optional[str] = None
    
    # SMC context
    htf_bias: Optional[int] = None  # -1 bear, 0 neutral, 1 bull
    nearest_ob_distance: Optional[float] = None
    fvg_present: bool = False
    bos_recent: bool = False
    choch_detected: bool = False
    market_state: Optional[str] = None  # trending_bull, trending_bear, ranging
    
    # Market context
    symbol: str = "XAUUSD"
    timeframe: str = "15m"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Price levels
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    # Risk parameters
    suggested_lot_size: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    risk_percent: float = 1.0
    
    # Metadata
    model_version: str = "v1"
    signal_id: Optional[str] = None
    
    def __post_init__(self):
        if self.signal_id is None:
            import uuid
            self.signal_id = str(uuid.uuid4())[:8]
            
    def is_valid(self) -> bool:
        """Check if signal meets minimum requirements"""
        if self.signal_type == SignalType.NO_TRADE:
            return True
            
        # Minimum confidence threshold
        if self.confidence < 0.60:
            return False
            
        # Check for conflicting signals
        if self.buy_probability > 0.6 and self.sell_probability > 0.6:
            return False
            
        return True
        
    def to_dict(self) -> dict:
        """Convert to dictionary for logging"""
        return {
            'signal_id': self.signal_id,
            'signal_type': self.signal_type.value,
            'confidence': self.confidence,
            'strength': self.strength.value,
            'buy_prob': self.buy_probability,
            'sell_prob': self.sell_probability,
            'notrade_prob': self.notrade_probability,
            'htf_bias': self.htf_bias,
            'fvg_present': self.fvg_present,
            'bos_recent': self.bos_recent,
            'choch_detected': self.choch_detected,
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'risk_reward_ratio': self.risk_reward_ratio,
            'model_version': self.model_version
        }


class SignalGenerator:
    """
    Generates trade signals from model predictions and market context.
    
    Combines:
    1. CNN pattern recognition
    2. SMC feature analysis
    3. Multi-timeframe confluence
    4. Market regime detection
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Confidence thresholds
        self.min_confidence = self.config.get('min_confidence', 0.65)
        self.high_confidence = self.config.get('high_confidence', 0.80)
        self.very_high_confidence = self.config.get('very_high_confidence', 0.90)
        
        # Minimum probability differences
        self.min_prob_diff = self.config.get('min_prob_diff', 0.15)
        
        # SMC confirmation requirements
        self.require_htf_confluence = self.config.get('require_htf_confluence', True)
        self.require_fvg_or_ob = self.config.get('require_fvg_or_ob', True)
        
        # Signal cooldown to prevent overtrading
        self.cooldown_minutes = self.config.get('cooldown_minutes', 15)
        self.last_signal_time: Dict[str, datetime] = {}
        
        # Daily limits
        self.max_daily_signals = self.config.get('max_daily_signals', 10)
        self.daily_signal_count = 0
        self.daily_reset_time = datetime.utcnow().replace(hour=0, minute=0, second=0)
        
    def generate_signal(
        self,
        buy_prob: float,
        sell_prob: float,
        notrade_prob: float,
        smc_features: Dict,
        current_price: float,
        atr: float,
        symbol: str = "XAUUSD"
    ) -> TradeSignal:
        """
        Generate trade signal from model probabilities and SMC context.
        
        Args:
            buy_prob: Probability of BUY (0-1)
            sell_prob: Probability of SELL (0-1)
            notrade_prob: Probability of NO_TRADE (0-1)
            smc_features: Dictionary of SMC features
            current_price: Current market price
            atr: Average True Range for SL/TP calculation
            symbol: Trading symbol
            
        Returns:
            TradeSignal object
        """
        
        # Check cooldown
        if not self._check_cooldown(symbol):
            return self._create_no_trade_signal(buy_prob, sell_prob, notrade_prob, "Cooldown active")
            
        # Check daily limit
        self._check_daily_reset()
        if self.daily_signal_count >= self.max_daily_signals:
            return self._create_no_trade_signal(buy_prob, sell_prob, notrade_prob, "Daily limit reached")
            
        # Extract SMC features
        htf_bias = smc_features.get('htf_bias', 0)
        nearest_bull_ob = smc_features.get('dist_nearest_bull_ob', 10)
        nearest_bear_ob = smc_features.get('dist_nearest_bear_ob', 10)
        fvg_bull_open = smc_features.get('fvg_bull_open', False)
        fvg_bear_open = smc_features.get('fvg_bear_open', False)
        bos_bull = smc_features.get('bos_count_bull', 0) > 0
        bos_bear = smc_features.get('bos_count_bear', 0) > 0
        choch = smc_features.get('choch_detected', False)
        market_state = smc_features.get('market_state', 0)  # 0=ranging, 1=bull, 2=bear
        hh_hl_ratio = smc_features.get('hh_hl_ratio', 0.5)
        lh_ll_ratio = smc_features.get('lh_ll_ratio', 0.5)
        
        # Determine raw signal direction
        buy_score = buy_prob
        sell_score = sell_prob
        
        # Apply SMC filters
        buy_score = self._apply_buy_filters(buy_score, htf_bias, fvg_bull_open, nearest_bull_ob, bos_bull, market_state)
        sell_score = self._apply_sell_filters(sell_score, htf_bias, fvg_bear_open, nearest_bear_ob, bos_bear, market_state)
        
        # Normalize
        total = buy_score + sell_score
        if total > 0:
            buy_score = buy_score / total
            sell_score = sell_score / total
            
        # Determine signal type
        signal_type = SignalType.NO_TRADE
        confidence = 0.0
        
        if buy_score > sell_score and buy_score >= self.min_confidence:
            if self._check_buy_confluence(htf_bias, fvg_bull_open, nearest_bull_ob, bos_bull, market_state):
                signal_type = SignalType.BUY
                confidence = buy_score
        elif sell_score > buy_score and sell_score >= self.min_confidence:
            if self._check_sell_confluence(htf_bias, fvg_bear_open, nearest_bear_ob, bos_bear, market_state):
                signal_type = SignalType.SELL
                confidence = sell_score
                
        # Calculate signal strength
        strength = self._calculate_strength(confidence, signal_type, smc_features)
        
        # Calculate price levels if signal is BUY/SELL
        entry_price = current_price
        stop_loss = None
        take_profit = None
        risk_reward = None
        
        if signal_type != SignalType.NO_TRADE:
            stop_loss, take_profit, risk_reward = self._calculate_sl_tp(
                signal_type, current_price, atr, confidence
            )
            
        # Create signal
        signal = TradeSignal(
            signal_type=signal_type,
            confidence=confidence,
            strength=strength,
            buy_probability=buy_prob,
            sell_probability=sell_prob,
            notrade_probability=notrade_prob,
            htf_bias=htf_bias,
            nearest_ob_distance=min(nearest_bull_ob, nearest_bear_ob) if signal_type != SignalType.NO_TRADE else None,
            fvg_present=fvg_bull_open or fvg_bear_open,
            bos_recent=bos_bull or bos_bear,
            choch_detected=choch,
            market_state={0: 'ranging', 1: 'trending_bull', 2: 'trending_bear'}.get(market_state, 'unknown'),
            symbol=symbol,
            entry_price=entry_price if signal_type != SignalType.NO_TRADE else None,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward_ratio=risk_reward,
            pattern_detected=self._detect_pattern(smc_features)
        )
        
        # Log signal
        if signal_type != SignalType.NO_TRADE:
            self.daily_signal_count += 1
            self.last_signal_time[symbol] = datetime.utcnow()
            logger.info(f"Signal generated: {signal_type.value} with {confidence:.2%} confidence")
        else:
            logger.debug(f"No trade signal: buy={buy_prob:.2f}, sell={sell_prob:.2f}")
            
        return signal
        
    def _apply_buy_filters(
        self,
        buy_score: float,
        htf_bias: int,
        fvg_bull_open: bool,
        nearest_bull_ob: float,
        bos_bull: bool,
        market_state: int
    ) -> float:
        """Apply SMC filters to BUY score"""
        
        multiplier = 1.0
        
        # HTF confluence
        if self.require_htf_confluence:
            if htf_bias == 1:  # Bullish HTF
                multiplier *= 1.3
            elif htf_bias == -1:  # Bearish HTF
                multiplier *= 0.4  # Strong penalty
                
        # Order block proximity (within 2 ATR)
        if nearest_bull_ob < 2.0:
            multiplier *= 1.2
        elif nearest_bull_ob < 5.0:
            multiplier *= 1.05
            
        # FVG presence
        if fvg_bull_open:
            multiplier *= 1.15
            
        # BOS confirmation
        if bos_bull:
            multiplier *= 1.1
            
        # Market state
        if market_state == 1:  # Trending bullish
            multiplier *= 1.2
        elif market_state == 0:  # Ranging
            multiplier *= 0.8
            
        return buy_score * min(multiplier, 2.0)
        
    def _apply_sell_filters(
        self,
        sell_score: float,
        htf_bias: int,
        fvg_bear_open: bool,
        nearest_bear_ob: float,
        bos_bear: bool,
        market_state: int
    ) -> float:
        """Apply SMC filters to SELL score"""
        
        multiplier = 1.0
        
        # HTF confluence
        if self.require_htf_confluence:
            if htf_bias == -1:  # Bearish HTF
                multiplier *= 1.3
            elif htf_bias == 1:  # Bullish HTF
                multiplier *= 0.4
                
        # Order block proximity
        if nearest_bear_ob < 2.0:
            multiplier *= 1.2
        elif nearest_bear_ob < 5.0:
            multiplier *= 1.05
            
        # FVG presence
        if fvg_bear_open:
            multiplier *= 1.15
            
        # BOS confirmation
        if bos_bear:
            multiplier *= 1.1
            
        # Market state
        if market_state == 2:  # Trending bearish
            multiplier *= 1.2
        elif market_state == 0:  # Ranging
            multiplier *= 0.8
            
        return sell_score * min(multiplier, 2.0)
        
    def _check_buy_confluence(
        self,
        htf_bias: int,
        fvg_bull_open: bool,
        nearest_bull_ob: float,
        bos_bull: bool,
        market_state: int
    ) -> bool:
        """Check if BUY signal has sufficient confluence"""
        
        confluence_score = 0
        
        if htf_bias == 1:
            confluence_score += 2
        elif htf_bias == 0:
            confluence_score += 1
            
        if fvg_bull_open:
            confluence_score += 2
            
        if nearest_bull_ob < 2.0:
            confluence_score += 2
        elif nearest_bull_ob < 5.0:
            confluence_score += 1
            
        if bos_bull:
            confluence_score += 1
            
        if market_state == 1:  # Trending bullish
            confluence_score += 2
            
        # Minimum confluence required
        return confluence_score >= 3
        
    def _check_sell_confluence(
        self,
        htf_bias: int,
        fvg_bear_open: bool,
        nearest_bear_ob: float,
        bos_bear: bool,
        market_state: int
    ) -> bool:
        """Check if SELL signal has sufficient confluence"""
        
        confluence_score = 0
        
        if htf_bias == -1:
            confluence_score += 2
        elif htf_bias == 0:
            confluence_score += 1
            
        if fvg_bear_open:
            confluence_score += 2
            
        if nearest_bear_ob < 2.0:
            confluence_score += 2
        elif nearest_bear_ob < 5.0:
            confluence_score += 1
            
        if bos_bear:
            confluence_score += 1
            
        if market_state == 2:  # Trending bearish
            confluence_score += 2
            
        return confluence_score >= 3
        
    def _calculate_strength(
        self,
        confidence: float,
        signal_type: SignalType,
        smc_features: Dict
    ) -> SignalStrength:
        """Calculate signal strength based on confidence and confluence"""
        
        if signal_type == SignalType.NO_TRADE:
            return SignalStrength.WEAK
            
        # Base strength from confidence
        if confidence >= self.very_high_confidence:
            base_strength = 4
        elif confidence >= self.high_confidence:
            base_strength = 3
        elif confidence >= self.min_confidence:
            base_strength = 2
        else:
            base_strength = 1
            
        # Add bonus for confluence
        confluence_bonus = 0
        
        if smc_features.get('htf_bias') in [1, -1]:
            confluence_bonus += 1
        if smc_features.get('fvg_bull_open') or smc_features.get('fvg_bear_open'):
            confluence_bonus += 1
        if smc_features.get('bos_count_bull', 0) > 0 or smc_features.get('bos_count_bear', 0) > 0:
            confluence_bonus += 1
        if smc_features.get('choch_detected', False):
            confluence_bonus += 1
            
        total_strength = min(base_strength + confluence_bonus // 2, 4)
        
        return SignalStrength(total_strength)
        
    def _calculate_sl_tp(
        self,
        signal_type: SignalType,
        entry_price: float,
        atr: float,
        confidence: float
    ) -> Tuple[float, float, float]:
        """Calculate stop loss and take profit levels"""
        
        # ATR multipliers
        sl_multiplier = self.config.get('sl_atr_multiplier', 1.5)
        base_tp_multiplier = self.config.get('tp_atr_multiplier', 2.5)
        
        # Adjust TP based on confidence
        if confidence >= 0.90:
            tp_multiplier = base_tp_multiplier * 1.2
        elif confidence >= 0.80:
            tp_multiplier = base_tp_multiplier * 1.0
        elif confidence >= 0.70:
            tp_multiplier = base_tp_multiplier * 0.8
        else:
            tp_multiplier = base_tp_multiplier * 0.7
            
        if signal_type == SignalType.BUY:
            stop_loss = entry_price - (atr * sl_multiplier)
            take_profit = entry_price + (atr * tp_multiplier)
        else:  # SELL
            stop_loss = entry_price + (atr * sl_multiplier)
            take_profit = entry_price - (atr * tp_multiplier)
            
        # Calculate R:R ratio
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        risk_reward = reward / risk if risk > 0 else 0
        
        return stop_loss, take_profit, risk_reward
        
    def _detect_pattern(self, smc_features: Dict) -> Optional[str]:
        """Detect specific chart patterns from SMC features"""
        
        patterns = []
        
        # Order block bounce
        if smc_features.get('dist_nearest_bull_ob', 10) < 1.5:
            patterns.append("OB_BOUNCE_BULL")
        if smc_features.get('dist_nearest_bear_ob', 10) < 1.5:
            patterns.append("OB_BOUNCE_BEAR")
            
        # FVG fill
        if smc_features.get('fvg_bull_open', False):
            patterns.append("FVG_BULL")
        if smc_features.get('fvg_bear_open', False):
            patterns.append("FVG_BEAR")
            
        # Break of structure
        if smc_features.get('bos_count_bull', 0) > 2:
            patterns.append("BOS_BULL")
        if smc_features.get('bos_count_bear', 0) > 2:
            patterns.append("BOS_BEAR")
            
        # Change of character
        if smc_features.get('choch_detected', False):
            patterns.append("CHoCH")
            
        return patterns[0] if patterns else None
        
    def _check_cooldown(self, symbol: str) -> bool:
        """Check if signal cooldown is active"""
        
        if symbol not in self.last_signal_time:
            return True
            
        time_since_last = (datetime.utcnow() - self.last_signal_time[symbol]).total_seconds() / 60
        return time_since_last >= self.cooldown_minutes
        
    def _check_daily_reset(self):
        """Reset daily counter at midnight UTC"""
        
        now = datetime.utcnow()
        if now.date() > self.daily_reset_time.date():
            self.daily_reset_time = now.replace(hour=0, minute=0, second=0)
            self.daily_signal_count = 0
            
    def _create_no_trade_signal(
        self,
        buy_prob: float,
        sell_prob: float,
        notrade_prob: float,
        reason: str = "No trade condition"
    ) -> TradeSignal:
        """Create a NO_TRADE signal with reason"""
        
        signal = TradeSignal(
            signal_type=SignalType.NO_TRADE,
            confidence=0.0,
            strength=SignalStrength.WEAK,
            buy_probability=buy_prob,
            sell_probability=sell_prob,
            notrade_probability=notrade_prob
        )
        
        logger.debug(f"No trade: {reason}")
        return signal
        
    def get_statistics(self) -> Dict:
        """Get signal generator statistics"""
        
        return {
            'daily_signals_used': self.daily_signal_count,
            'max_daily_signals': self.max_daily_signals,
            'cooldown_minutes': self.cooldown_minutes,
            'min_confidence': self.min_confidence
        }
