"""Market structure detection (BOS, CHoCH, swings)"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional


class StructureDetector:
    """Detects market structure elements like swings, BOS, CHoCH"""
    
    def __init__(self, atr_period: int = 14, swing_threshold: float = 0.5):
        self.atr_period = atr_period
        self.swing_threshold = swing_threshold  # In ATR units
        
    def find_swing_points(
        self,
        df: pd.DataFrame,
        lookback: int = 5
    ) -> Tuple[List[int], List[int]]:
        """Find swing highs and lows using zigzag algorithm"""
        
        highs = []
        lows = []
        
        for i in range(lookback, len(df) - lookback):
            # Swing high
            if all(df['high'].iloc[i] >= df['high'].iloc[i - j] for j in range(1, lookback + 1)) and \
               all(df['high'].iloc[i] >= df['high'].iloc[i + j] for j in range(1, lookback + 1)):
                highs.append(i)
                
            # Swing low
            if all(df['low'].iloc[i] <= df['low'].iloc[i - j] for j in range(1, lookback + 1)) and \
               all(df['low'].iloc[i] <= df['low'].iloc[i + j] for j in range(1, lookback + 1)):
                lows.append(i)
                
        return highs, lows
        
    def detect_bos(
        self,
        df: pd.DataFrame,
        highs: List[int],
        lows: List[int]
    ) -> Tuple[int, int]:
        """Detect Break of Structure (BOS) events"""
        
        bos_bull = 0  # Bullish BOS (break above swing high)
        bos_bear = 0  # Bearish BOS (break below swing low)
        
        if len(highs) >= 2:
            last_swing_high = df['high'].iloc[highs[-1]]
            if df['close'].iloc[-1] > last_swing_high:
                bos_bull = 1
                
        if len(lows) >= 2:
            last_swing_low = df['low'].iloc[lows[-1]]
            if df['close'].iloc[-1] < last_swing_low:
                bos_bear = 1
                
        return bos_bull, bos_bear
        
    def detect_choch(
        self,
        df: pd.DataFrame,
        highs: List[int],
        lows: List[int],
        lookback: int = 20
    ) -> bool:
        """Detect Change of Character (CHoCH) - trend reversal"""
        
        if len(highs) < 3 or len(lows) < 3:
            return False
            
        # Get recent swings within lookback
        recent_highs = [h for h in highs if h > len(df) - lookback]
        recent_lows = [l for l in lows if l > len(df) - lookback]
        
        if len(recent_highs) < 2 or len(recent_lows) < 2:
            return False
            
        # Check if we had higher highs and higher lows (uptrend)
        was_uptrend = (
            df['high'].iloc[recent_highs[-2]] < df['high'].iloc[recent_highs[-1]] and
            df['low'].iloc[recent_lows[-2]] < df['low'].iloc[recent_lows[-1]]
        )
        
        # Now check for lower low (potential reversal)
        if was_uptrend and df['low'].iloc[-1] < df['low'].iloc[recent_lows[-1]]:
            return True
            
        # Check if we had lower highs and lower lows (downtrend)
        was_downtrend = (
            df['high'].iloc[recent_highs[-2]] > df['high'].iloc[recent_highs[-1]] and
            df['low'].iloc[recent_lows[-2]] > df['low'].iloc[recent_lows[-1]]
        )
        
        # Now check for higher high (potential reversal)
        if was_downtrend and df['high'].iloc[-1] > df['high'].iloc[recent_highs[-1]]:
            return True
            
        return False
        
    def compute_structure_scores(
        self,
        df: pd.DataFrame,
        lookback: int = 20
    ) -> Dict[str, float]:
        """Compute structure scores (HH/HL ratio, LH/LL ratio)"""
        
        highs, lows = self.find_swing_points(df.tail(lookback + 10))
        
        # Filter swings within lookback
        start_idx = len(df) - lookback
        recent_highs = [h for h in highs if h >= start_idx]
        recent_lows = [l for l in lows if l >= start_idx]
        
        # Calculate HH/HL ratio (bullish structure)
        hh_count = 0
        hl_count = 0
        
        for i in range(1, len(recent_highs)):
            if df['high'].iloc[recent_highs[i]] > df['high'].iloc[recent_highs[i-1]]:
                hh_count += 1
                
        for i in range(1, len(recent_lows)):
            if df['low'].iloc[recent_lows[i]] > df['low'].iloc[recent_lows[i-1]]:
                hl_count += 1
                
        total_bull_swings = max(1, len(recent_highs) - 1 + len(recent_lows) - 1)
        bullish_ratio = (hh_count + hl_count) / total_bull_swings
        
        # Calculate LH/LL ratio (bearish structure)
        lh_count = 0
        ll_count = 0
        
        for i in range(1, len(recent_highs)):
            if df['high'].iloc[recent_highs[i]] < df['high'].iloc[recent_highs[i-1]]:
                lh_count += 1
                
        for i in range(1, len(recent_lows)):
            if df['low'].iloc[recent_lows[i]] < df['low'].iloc[recent_lows[i-1]]:
                ll_count += 1
                
        bearish_ratio = (lh_count + ll_count) / total_bull_swings
        
        return {
            'hh_hl_ratio': bullish_ratio,
            'lh_ll_ratio': bearish_ratio,
            'bos_count_bull': sum(1 for _ in range(lookback) if self.detect_bos(df.iloc[:_])[0]),
            'bos_count_bear': sum(1 for _ in range(lookback) if self.detect_bos(df.iloc[:_])[1]),
            'choch_detected': 1 if self.detect_choch(df, highs, lows) else 0
        }
