"""Market state detection - trending, ranging, volatile regimes"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from enum import Enum


class MarketRegime(Enum):
    """Market regime types"""
    STRONG_BULL = 2
    WEAK_BULL = 1
    NEUTRAL = 0
    WEAK_BEAR = -1
    STRONG_BEAR = -2


class VolatilityRegime(Enum):
    """Volatility regime types"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    EXTREME = 3


class MarketStateDetector:
    """
    Detects market state including:
    - Trend direction and strength
    - Volatility regime
    - Ranging vs trending
    - Market phase (accumulation, markup, distribution, markdown)
    """
    
    def __init__(
        self,
        atr_period: int = 14,
        trend_period: int = 20,
        volatility_percentiles: Tuple[float, float, float] = (25, 50, 75)
    ):
        self.atr_period = atr_period
        self.trend_period = trend_period
        self.volatility_percentiles = volatility_percentiles
        
    def compute_atr(self, df: pd.DataFrame) -> pd.Series:
        """Calculate ATR"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.ewm(span=self.atr_period, adjust=False).mean()
    
    def detect_trend(
        self,
        df: pd.DataFrame,
        lookback: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Detect trend direction and strength using multiple methods
        """
        if lookback is None:
            lookback = self.trend_period
            
        df_slice = df.tail(lookback)
        close_prices = df_slice['close'].values
        
        # Method 1: Linear regression slope
        x = np.arange(len(close_prices))
        slope, intercept = np.polyfit(x, close_prices, 1)
        slope_normalized = slope / close_prices.mean() if close_prices.mean() > 0 else 0
        
        # Method 2: Moving average alignment
        ema_9 = df_slice['close'].ewm(span=9, adjust=False).mean()
        ema_21 = df_slice['close'].ewm(span=21, adjust=False).mean()
        ema_50 = df_slice['close'].ewm(span=50, adjust=False).mean()
        
        # Check alignment (bullish: 9 > 21 > 50)
        ma_alignment = 0
        if ema_9.iloc[-1] > ema_21.iloc[-1] > ema_50.iloc[-1]:
            ma_alignment = 1  # Bullish alignment
        elif ema_9.iloc[-1] < ema_21.iloc[-1] < ema_50.iloc[-1]:
            ma_alignment = -1  # Bearish alignment
            
        # Method 3: ADX (Average Directional Index)
        adx, plus_di, minus_di = self._calculate_adx(df_slice)
        
        # Method 4: Higher highs/lower lows ratio
        highs, lows = self._find_swing_points(df_slice)
        
        hh_count = 0
        ll_count = 0
        
        for i in range(1, len(highs)):
            if df_slice['high'].iloc[highs[i]] > df_slice['high'].iloc[highs[i-1]]:
                hh_count += 1
                
        for i in range(1, len(lows)):
            if df_slice['low'].iloc[lows[i]] < df_slice['low'].iloc[lows[i-1]]:
                ll_count += 1
                
        structure_bias = (hh_count - ll_count) / max(1, len(highs) + len(lows))
        
        # Combined trend score (-1 to 1)
        trend_score = (
            slope_normalized * 0.3 +
            ma_alignment * 0.3 +
            (plus_di.iloc[-1] - minus_di.iloc[-1]) / 100 * 0.2 +
            structure_bias * 0.2
        )
        
        # Clip to [-1, 1]
        trend_score = max(-1, min(1, trend_score))
        
        # Determine regime
        if trend_score > 0.5:
            regime = MarketRegime.STRONG_BULL
        elif trend_score > 0.15:
            regime = MarketRegime.WEAK_BULL
        elif trend_score < -0.5:
            regime = MarketRegime.STRONG_BEAR
        elif trend_score < -0.15:
            regime = MarketRegime.WEAK_BEAR
        else:
            regime = MarketRegime.NEUTRAL
            
        return {
            'trend_score': trend_score,
            'trend_direction': 1 if trend_score > 0 else -1 if trend_score < 0 else 0,
            'trend_strength': abs(trend_score),
            'regime': regime.value,
            'regime_name': regime.name,
            'adx': adx.iloc[-1] if len(adx) > 0 else 0,
            'plus_di': plus_di.iloc[-1] if len(plus_di) > 0 else 0,
            'minus_di': minus_di.iloc[-1] if len(minus_di) > 0 else 0,
            'ma_alignment': ma_alignment,
            'slope_normalized': slope_normalized
        }
    
    def detect_volatility_regime(
        self,
        df: pd.DataFrame,
        lookback_days: int = 90  # 3 months
    ) -> Dict:
        """
        Detect volatility regime using ATR percentiles
        """
        atr = self.compute_atr(df)
        
        # Calculate ATR as percentage of price
        atr_pct = atr / df['close'] * 100
        
        # Get historical distribution
        historical_atr = atr_pct.tail(lookback_days)
        
        if len(historical_atr) < 50:
            return {
                'volatility_regime': VolatilityRegime.NORMAL.value,
                'atr_percentile': 50,
                'current_atr_pct': atr_pct.iloc[-1] if len(atr_pct) > 0 else 0,
                'mean_atr_pct': historical_atr.mean() if len(historical_atr) > 0 else 0,
                'volatility_multiplier': 1.0
            }
            
        # Calculate current percentile
        current_atr = atr_pct.iloc[-1]
        percentile = (historical_atr < current_atr).sum() / len(historical_atr) * 100
        
        # Determine regime
        p25, p50, p75 = self.volatility_percentiles
        
        if percentile > p75:
            if percentile > 90:
                regime = VolatilityRegime.EXTREME
            else:
                regime = VolatilityRegime.HIGH
        elif percentile < p25:
            regime = VolatilityRegime.LOW
        else:
            regime = VolatilityRegime.NORMAL
            
        # Calculate volatility multiplier (for position sizing)
        vol_multiplier = current_atr / historical_atr.median() if historical_atr.median() > 0 else 1.0
        
        return {
            'volatility_regime': regime.value,
            'regime_name': regime.name,
            'atr_percentile': percentile,
            'current_atr_pct': current_atr,
            'mean_atr_pct': historical_atr.mean(),
            'volatility_multiplier': min(2.0, max(0.5, vol_multiplier))  # Clamp between 0.5-2.0
        }
    
    def detect_market_phase(
        self,
        df: pd.DataFrame,
        lookback: int = 100
    ) -> Dict[str, str]:
        """
        Detect Wyckoff market phases:
        - Accumulation (smart money buying)
        - Markup (uptrend)
        - Distribution (smart money selling)
        - Markdown (downtrend)
        """
        df_slice = df.tail(lookback)
        
        trend_data = self.detect_trend(df_slice)
        volatility_data = self.detect_volatility_regime(df_slice)
        
        # Detect range boundaries
        range_high = df_slice['high'].rolling(window=20).max().iloc[-1]
        range_low = df_slice['low'].rolling(window=20).min().iloc[-1]
        range_size = (range_high - range_low) / range_low
        
        # Check for accumulation characteristics
        is_ranging = range_size < 0.03  # Less than 3% range
        low_volatility = volatility_data['volatility_regime'] <= 1  # LOW or NORMAL
        volume_increasing = df_slice['volume'].iloc[-10:].mean() > df_slice['volume'].iloc[-30:-10].mean() if 'volume' in df_slice.columns else False
        
        # Check for breakout
        breakout_above = df_slice['close'].iloc[-1] > range_high
        breakout_below = df_slice['close'].iloc[-1] < range_low
        
        # Determine phase
        phase = "UNKNOWN"
        phase_confidence = 0.0
        
        if is_ranging and low_volatility and volume_increasing:
            phase = "ACCUMULATION"
            phase_confidence = 0.7
        elif is_ranging and not volume_increasing:
            phase = "DISTRIBUTION"
            phase_confidence = 0.6
        elif breakout_above and trend_data['trend_score'] > 0.3:
            phase = "MARKUP"
            phase_confidence = 0.8
        elif breakout_below and trend_data['trend_score'] < -0.3:
            phase = "MARKDOWN"
            phase_confidence = 0.8
        elif trend_data['trend_score'] > 0.15:
            phase = "MARKUP"
            phase_confidence = 0.5
        elif trend_data['trend_score'] < -0.15:
            phase = "MARKDOWN"
            phase_confidence = 0.5
            
        return {
            'phase': phase,
            'phase_confidence': phase_confidence,
            'is_accumulation': phase == "ACCUMULATION",
            'is_distribution': phase == "DISTRIBUTION",
            'is_markup': phase == "MARKUP",
            'is_markdown': phase == "MARKDOWN"
        }
    
    def detect_ranging(
        self,
        df: pd.DataFrame,
        lookback: int = 50,
        max_range_pct: float = 0.03  # 3% max range to be considered ranging
    ) -> Dict:
        """
        Detect if market is ranging (consolidation)
        """
        df_slice = df.tail(lookback)
        
        # Calculate price range
        price_high = df_slice['high'].max()
        price_low = df_slice['low'].min()
        range_pct = (price_high - price_low) / price_low
        
        # Calculate volatility (lower volatility = more ranging)
        atr = self.compute_atr(df_slice)
        avg_atr_pct = (atr / df_slice['close']).mean() * 100
        
        # Calculate directional movement
        close_changes = df_slice['close'].pct_change().abs()
        avg_move = close_changes.mean()
        
        # Check for multiple tests of boundaries
        near_high = df_slice['close'] > price_high * 0.98
        near_low = df_slice['close'] < price_low * 1.02
        
        tests_of_high = near_high.sum()
        tests_of_low = near_low.sum()
        
        # Ranging score (0 to 1)
        ranging_score = 0.0
        
        if range_pct < max_range_pct:
            ranging_score += 0.4
            
        if avg_atr_pct < 0.5:  # Low volatility
            ranging_score += 0.3
            
        if avg_move < 0.005:  # Small average moves
            ranging_score += 0.2
            
        if tests_of_high > 2 and tests_of_low > 2:  # Multiple boundary tests
            ranging_score += 0.1
            
        is_ranging = ranging_score > 0.5
        
        return {
            'is_ranging': is_ranging,
            'ranging_score': ranging_score,
            'range_high': price_high,
            'range_low': price_low,
            'range_pct': range_pct,
            'tests_of_high': tests_of_high,
            'tests_of_low': tests_of_low
        }
    
    def get_market_state_features(
        self,
        df: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Extract comprehensive market state features for fusion model
        """
        trend = self.detect_trend(df)
        volatility = self.detect_volatility_regime(df)
        ranging = self.detect_ranging(df)
        phase = self.detect_market_phase(df)
        
        # Calculate additional momentum indicators
        close_prices = df['close']
        rsi = self._calculate_rsi(close_prices, 14)
        macd, macd_signal, macd_hist = self._calculate_macd(close_prices)
        
        # Calculate support/resistance proximity
        recent_high = df['high'].rolling(window=20).max().iloc[-1]
        recent_low = df['low'].rolling(window=20).min().iloc[-1]
        current_price = close_prices.iloc[-1]
        
        dist_to_high = (recent_high - current_price) / current_price
        dist_to_low = (current_price - recent_low) / current_price
        
        return {
            # Trend features
            'market_state': trend['regime'],  # -2 to 2
            'trend_score': trend['trend_score'],
            'trend_strength': trend['trend_strength'],
            'adx': trend['adx'],
            'plus_di': trend['plus_di'],
            'minus_di': trend['minus_di'],
            
            # Volatility features
            'volatility_regime': volatility['volatility_regime'],
            'atr_percentile': volatility['atr_percentile'] / 100,  # Normalize to 0-1
            'volatility_multiplier': volatility['volatility_multiplier'],
            'current_atr_pct': volatility['current_atr_pct'],
            
            # Ranging features
            'is_ranging': 1 if ranging['is_ranging'] else 0,
            'ranging_score': ranging['ranging_score'],
            'range_pct': ranging['range_pct'],
            
            # Phase features
            'is_accumulation': 1 if phase['is_accumulation'] else 0,
            'is_distribution': 1 if phase['is_distribution'] else 0,
            'phase_confidence': phase['phase_confidence'],
            
            # Momentum features
            'rsi': rsi.iloc[-1] if len(rsi) > 0 else 50,
            'rsi_trend': rsi.iloc[-1] - rsi.iloc[-5] if len(rsi) >= 5 else 0,
            'macd': macd.iloc[-1] if len(macd) > 0 else 0,
            'macd_histogram': macd_hist.iloc[-1] if len(macd_hist) > 0 else 0,
            
            # Support/Resistance
            'dist_to_recent_high': dist_to_high,
            'dist_to_recent_low': dist_to_low,
            'is_at_high': 1 if current_price >= recent_high * 0.995 else 0,
            'is_at_low': 1 if current_price <= recent_low * 1.005 else 0,
        }
    
    def _calculate_adx(
        self,
        df: pd.DataFrame,
        period: int = 14
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate ADX, +DI, -DI"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Calculate +DM and -DM
        plus_dm = high.diff()
        minus_dm = low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        minus_dm = abs(minus_dm)
        
        # Calculate TR
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Smooth using Wilder's smoothing
        atr = tr.ewm(span=period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)
        
        # Calculate DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(span=period, adjust=False).mean()
        
        return adx, plus_di, minus_di
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(
        self,
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist
    
    def _find_swing_points(
        self,
        df: pd.DataFrame,
        lookback: int = 5
    ) -> Tuple[List[int], List[int]]:
        """Find swing highs and lows"""
        highs = []
        lows = []
        
        for i in range(lookback, len(df) - lookback):
            if all(df['high'].iloc[i] >= df['high'].iloc[i - j] for j in range(1, lookback + 1)) and \
               all(df['high'].iloc[i] >= df['high'].iloc[i + j] for j in range(1, lookback + 1)):
                highs.append(i)
                
            if all(df['low'].iloc[i] <= df['low'].iloc[i - j] for j in range(1, lookback + 1)) and \
               all(df['low'].iloc[i] <= df['low'].iloc[i + j] for j in range(1, lookback + 1)):
                lows.append(i)
                
        return highs, lows
