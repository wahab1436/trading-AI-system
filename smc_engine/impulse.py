"""Impulse strength detection - identifies strong directional moves"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ImpulseCandle:
    """Represents an impulse candle"""
    index: int
    direction: str  # 'bullish' or 'bearish'
    body_size: float
    total_range: float
    body_to_range_ratio: float
    volume: float
    close_price: float
    open_price: float


class ImpulseDetector:
    """
    Detects impulse candles and calculates impulse strength
    Impulses are strong directional moves with large bodies and momentum
    """
    
    def __init__(
        self,
        atr_period: int = 14,
        min_body_ratio: float = 0.6,  # Body must be at least 60% of total range
        min_impulse_multiplier: float = 1.2,  # Must be > 1.2x ATR
        lookback_candles: int = 50
    ):
        self.atr_period = atr_period
        self.min_body_ratio = min_body_ratio
        self.min_impulse_multiplier = min_impulse_multiplier
        self.lookback_candles = lookback_candles
        
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
    
    def detect_impulse_candles(
        self,
        df: pd.DataFrame,
        lookback: Optional[int] = None
    ) -> List[ImpulseCandle]:
        """
        Detect impulse candles in the dataframe
        """
        if lookback is None:
            lookback = self.lookback_candles
            
        df_slice = df.tail(lookback).copy()
        atr = self.compute_atr(df)
        
        impulses = []
        
        for i in range(len(df_slice)):
            row = df_slice.iloc[i]
            idx = len(df) - lookback + i
            
            # Calculate candle metrics
            body_size = abs(row['close'] - row['open'])
            total_range = row['high'] - row['low']
            body_ratio = body_size / total_range if total_range > 0 else 0
            
            # Check if it's an impulse
            atr_value = atr.iloc[idx] if idx < len(atr) else atr.iloc[-1]
            is_impulse = (
                body_ratio >= self.min_body_ratio and
                body_size >= atr_value * self.min_impulse_multiplier
            )
            
            if is_impulse:
                direction = 'bullish' if row['close'] > row['open'] else 'bearish'
                
                impulse = ImpulseCandle(
                    index=idx,
                    direction=direction,
                    body_size=body_size,
                    total_range=total_range,
                    body_to_range_ratio=body_ratio,
                    volume=row.get('volume', 0),
                    close_price=row['close'],
                    open_price=row['open']
                )
                impulses.append(impulse)
                
        return impulses
    
    def calculate_impulse_strength(
        self,
        df: pd.DataFrame,
        impulse: ImpulseCandle
    ) -> float:
        """
        Calculate normalized impulse strength
        Higher values = stronger impulse
        """
        atr = self.compute_atr(df)
        atr_value = atr.iloc[impulse.index] if impulse.index < len(atr) else atr.iloc[-1]
        
        # Base strength from body size relative to ATR
        base_strength = impulse.body_size / atr_value
        
        # Adjust for body-to-range ratio
        ratio_adjustment = impulse.body_to_range_ratio / self.min_body_ratio
        
        # Adjust for volume (if available)
        volume_adjustment = 1.0
        if impulse.volume > 0 and 'volume' in df.columns:
            avg_volume = df['volume'].rolling(window=20).mean().iloc[impulse.index]
            if avg_volume > 0:
                volume_adjustment = min(2.0, impulse.volume / avg_volume)
                
        total_strength = base_strength * ratio_adjustment * volume_adjustment
        
        return total_strength
    
    def find_consecutive_impulses(
        self,
        df: pd.DataFrame,
        lookback: int = 20
    ) -> Dict[str, int]:
        """
        Find consecutive impulse candles in same direction
        """
        impulses = self.detect_impulse_candles(df, lookback)
        
        if not impulses:
            return {'max_bull_consecutive': 0, 'max_bear_consecutive': 0, 'current_streak': 0}
        
        # Count consecutive impulses
        bull_streak = 0
        bear_streak = 0
        max_bull_streak = 0
        max_bear_streak = 0
        
        for impulse in impulses:
            if impulse.direction == 'bullish':
                bull_streak += 1
                bear_streak = 0
                max_bull_streak = max(max_bull_streak, bull_streak)
            else:
                bear_streak += 1
                bull_streak = 0
                max_bear_streak = max(max_bear_streak, bear_streak)
                
        # Get current streak (from most recent impulses)
        current_streak = 0
        current_direction = None
        
        for impulse in reversed(impulses[-5:]):  # Last 5 impulses
            if current_direction is None:
                current_direction = impulse.direction
                current_streak = 1
            elif impulse.direction == current_direction:
                current_streak += 1
            else:
                break
                
        return {
            'max_bull_consecutive': max_bull_streak,
            'max_bear_consecutive': max_bear_streak,
            'current_streak': current_streak,
            'current_streak_direction': current_direction
        }
    
    def calculate_impulse_velocity(
        self,
        df: pd.DataFrame,
        lookback: int = 5
    ) -> Dict[str, float]:
        """
        Calculate price velocity during impulse phases
        Velocity = price change per candle
        """
        impulses = self.detect_impulse_candles(df, lookback * 2)
        
        if len(impulses) < 2:
            return {'bull_velocity': 0, 'bear_velocity': 0, 'net_velocity': 0}
        
        # Calculate velocities for recent impulses
        bull_velocities = []
        bear_velocities = []
        
        for i in range(1, len(impulses)):
            prev = impulses[i-1]
            curr = impulses[i]
            
            # Calculate velocity (price change per candle)
            if curr.direction == 'bullish':
                velocity = (curr.close_price - prev.close_price) / curr.body_size if curr.body_size > 0 else 0
                bull_velocities.append(velocity)
            else:
                velocity = (prev.close_price - curr.close_price) / curr.body_size if curr.body_size > 0 else 0
                bear_velocities.append(velocity)
                
        return {
            'bull_velocity': np.mean(bull_velocities) if bull_velocities else 0,
            'bear_velocity': np.mean(bear_velocities) if bear_velocities else 0,
            'net_velocity': (np.mean(bull_velocities) if bull_velocities else 0) - 
                           (np.mean(bear_velocities) if bear_velocities else 0)
        }
    
    def calculate_impulse_absorption(
        self,
        df: pd.DataFrame,
        current_price: float,
        lookback: int = 20
    ) -> Dict[str, float]:
        """
        Calculate how well price is absorbing impulse moves
        High absorption = rejection / wicks
        """
        df_slice = df.tail(lookback)
        
        # Calculate average wick size relative to body
        upper_wicks = df_slice['high'] - df_slice[['open', 'close']].max(axis=1)
        lower_wicks = df_slice[['open', 'close']].min(axis=1) - df_slice['low']
        
        avg_body = abs(df_slice['close'] - df_slice['open']).mean()
        avg_upper_wick = upper_wicks.mean()
        avg_lower_wick = lower_wicks.mean()
        
        # Absorption ratio (high absorption = large wicks relative to body)
        upper_absorption = avg_upper_wick / avg_body if avg_body > 0 else 0
        lower_absorption = avg_lower_wick / avg_body if avg_body > 0 else 0
        
        # Recent rejection candles (long wicks)
        recent_rejections = []
        for i in range(len(df_slice)):
            row = df_slice.iloc[i]
            body = abs(row['close'] - row['open'])
            upper_wick = row['high'] - max(row['open'], row['close'])
            lower_wick = min(row['open'], row['close']) - row['low']
            
            if upper_wick > body * 1.5:  # Long upper wick = rejection from above
                recent_rejections.append('bullish_rejection')
            if lower_wick > body * 1.5:  # Long lower wick = rejection from below
                recent_rejections.append('bearish_rejection')
                
        return {
            'upper_absorption_ratio': upper_absorption,
            'lower_absorption_ratio': lower_absorption,
            'recent_bullish_rejections': recent_rejections.count('bullish_rejection'),
            'recent_bearish_rejections': recent_rejections.count('bearish_rejection'),
            'absorption_score': (upper_absorption + lower_absorption) / 2
        }
    
    def get_impulse_features(
        self,
        df: pd.DataFrame,
        current_price: float
    ) -> Dict[str, float]:
        """
        Extract impulse features for fusion model
        """
        impulses = self.detect_impulse_candles(df)
        consecutive = self.find_consecutive_impulses(df)
        velocity = self.calculate_impulse_velocity(df)
        absorption = self.calculate_impulse_absorption(df, current_price)
        
        atr = self.compute_atr(df).iloc[-1]
        
        # Calculate impulse strength for most recent impulse
        recent_impulse_strength = 0
        recent_impulse_direction = 0
        
        if impulses:
            recent = impulses[-1]
            recent_impulse_strength = self.calculate_impulse_strength(df, recent)
            recent_impulse_direction = 1 if recent.direction == 'bullish' else -1
            
        # Count impulses by type
        bullish_impulses = sum(1 for i in impulses if i.direction == 'bullish')
        bearish_impulses = sum(1 for i in impulses if i.direction == 'bearish')
        total_impulses = len(impulses)
        
        return {
            # Impulse strength
            'impulse_strength': recent_impulse_strength,
            'impulse_direction': recent_impulse_direction,
            
            # Impulse counts
            'bull_impulse_count': bullish_impulses,
            'bear_impulse_count': bearish_impulses,
            'bull_impulse_ratio': bullish_impulses / total_impulses if total_impulses > 0 else 0.5,
            
            # Consecutive impulses
            'consecutive_bull_impulses': consecutive['max_bull_consecutive'],
            'consecutive_bear_impulses': consecutive['max_bear_consecutive'],
            'current_impulse_streak': consecutive['current_streak'],
            
            # Velocity
            'bull_velocity': velocity['bull_velocity'],
            'bear_velocity': velocity['bear_velocity'],
            'net_velocity': velocity['net_velocity'],
            
            # Absorption
            'absorption_score': absorption['absorption_score'],
            'rejections_count': absorption['recent_bullish_rejections'] + absorption['recent_bearish_rejections'],
            
            # Momentum (price change over last N candles)
            'momentum_5': (df['close'].iloc[-1] - df['close'].iloc[-6]) / atr if len(df) >= 6 else 0,
            'momentum_10': (df['close'].iloc[-1] - df['close'].iloc[-11]) / atr if len(df) >= 11 else 0,
            'momentum_20': (df['close'].iloc[-1] - df['close'].iloc[-21]) / atr if len(df) >= 21 else 0,
        }
    
    def get_impulse_warning(
        self,
        df: pd.DataFrame,
        current_price: float
    ) -> Dict[str, bool]:
        """
        Detect impulse warnings (exhaustion, divergence)
        """
        impulses = self.detect_impulse_candles(df, 30)
        
        if len(impulses) < 3:
            return {
                'impulse_exhaustion': False,
                'bullish_divergence': False,
                'bearish_divergence': False
            }
        
        # Check for impulse exhaustion (decreasing strength)
        recent_strengths = [self.calculate_impulse_strength(df, i) for i in impulses[-5:]]
        impulse_exhaustion = len(recent_strengths) >= 3 and recent_strengths[-1] < recent_strengths[-2] < recent_strengths[-3]
        
        # Check for divergence between price and impulse strength
        price_change = df['close'].iloc[-1] - df['close'].iloc[-len(impulses)]
        avg_strength_change = recent_strengths[-1] - recent_strengths[0] if recent_strengths else 0
        
        bullish_divergence = price_change < 0 and avg_strength_change > 0  # Price down, strength up
        bearish_divergence = price_change > 0 and avg_strength_change < 0  # Price up, strength down
        
        return {
            'impulse_exhaustion': impulse_exhaustion,
            'bullish_divergence': bullish_divergence,
            'bearish_divergence': bearish_divergence
        }s
