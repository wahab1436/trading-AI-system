"""Fair Value Gap (FVG) detection for inefficiencies"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional


class FVGDetector:
    """Detects Fair Value Gaps (imbalances) in price"""
    
    def __init__(self, atr_period: int = 14):
        self.atr_period = atr_period
        
    def find_bullish_fvg(
        self,
        df: pd.DataFrame,
        lookback: int = 50
    ) -> List[Dict]:
        """Find bullish FVGs (gap up that remains unfilled)"""
        
        fvgs = []
        
        for i in range(2, min(lookback, len(df))):
            # Bullish FVG: gap between candle[i-2].high and candle[i].low
            # after a bullish move
            if df['close'].iloc[i] > df['open'].iloc[i]:  # Bullish candle
                gap_top = df['high'].iloc[i-2]
                gap_bottom = df['low'].iloc[i]
                
                if gap_bottom > gap_top:  # There's a gap
                    fvg = {
                        'type': 'bullish',
                        'index': i,
                        'top': gap_top,
                        'bottom': gap_bottom,
                        'size': gap_bottom - gap_top,
                        'filled': False
                    }
                    fvgs.append(fvg)
                    
        return fvgs
        
    def find_bearish_fvg(
        self,
        df: pd.DataFrame,
        lookback: int = 50
    ) -> List[Dict]:
        """Find bearish FVGs (gap down that remains unfilled)"""
        
        fvgs = []
        
        for i in range(2, min(lookback, len(df))):
            # Bearish FVG: gap between candle[i-2].low and candle[i].high
            # after a bearish move
            if df['close'].iloc[i] < df['open'].iloc[i]:  # Bearish candle
                gap_top = df['high'].iloc[i]
                gap_bottom = df['low'].iloc[i-2]
                
                if gap_top < gap_bottom:  # There's a gap (inverted)
                    fvg = {
                        'type': 'bearish',
                        'index': i,
                        'top': gap_bottom,
                        'bottom': gap_top,
                        'size': gap_bottom - gap_top,
                        'filled': False
                    }
                    fvgs.append(fvg)
                    
        return fvgs
        
    def update_fvg_fill(
        self,
        fvgs: List[Dict],
        current_high: float,
        current_low: float
    ) -> List[Dict]:
        """Update FVG fill status"""
        
        updated_fvgs = []
        
        for fvg in fvgs:
            if not fvg['filled']:
                # Check if price has filled the gap
                if fvg['type'] == 'bullish':
                    if current_low <= fvg['top']:
                        fvg['filled'] = True
                else:  # bearish
                    if current_high >= fvg['top']:
                        fvg['filled'] = True
                        
            # Keep recent, unfilled FVGs
            if not fvg['filled']:
                updated_fvgs.append(fvg)
                
        return updated_fvgs
        
    def get_features(
        self,
        df: pd.DataFrame,
        current_price: float
    ) -> Dict[str, float]:
        """Extract FVG features for fusion model"""
        
        bullish_fvgs = self.find_bullish_fvg(df)
        bearish_fvgs = self.find_bearish_fvg(df)
        
        atr = self.compute_atr(df).iloc[-1]
        
        # Check for unfilled FVGs within 2 ATR
        bull_open = False
        bear_open = False
        
        for fvg in bullish_fvgs:
            if not fvg['filled'] and abs(current_price - fvg['top']) < 2 * atr:
                bull_open = True
                break
                
        for fvg in bearish_fvgs:
            if not fvg['filled'] and abs(current_price - fvg['top']) < 2 * atr:
                bear_open = True
                break
                
        return {
            'fvg_bull_open': 1 if bull_open else 0,
            'fvg_bear_open': 1 if bear_open else 0,
            'fvg_bull_count': len([f for f in bullish_fvgs if not f['filled']]),
            'fvg_bear_count': len([f for f in bearish_fvgs if not f['filled']])
        }
        
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
