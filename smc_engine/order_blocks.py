"""Order block detection for institutional entry zones"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple


class OrderBlockDetector:
    """Detects bullish and bearish order blocks"""
    
    def __init__(self, atr_period: int = 14, impulse_multiplier: float = 1.5):
        self.atr_period = atr_period
        self.impulse_multiplier = impulse_multiplier
        
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
        
    def find_bullish_order_blocks(
        self,
        df: pd.DataFrame,
        lookback: int = 50
    ) -> List[Dict]:
        """Find bullish order blocks (last bearish candle before bullish impulse)"""
        
        obs = []
        atr = self.compute_atr(df)
        
        for i in range(lookback, len(df)):
            # Check for bullish impulse (close > high of previous candle)
            if df['close'].iloc[i] > df['high'].iloc[i-1]:
                impulse_size = df['close'].iloc[i] - df['open'].iloc[i]
                avg_atr = atr.iloc[i]
                
                # Must be a strong impulse
                if impulse_size > avg_atr * self.impulse_multiplier:
                    # Look for bearish candle before impulse
                    ob_candle_idx = i - 1
                    
                    if df['close'].iloc[ob_candle_idx] < df['open'].iloc[ob_candle_idx]:  # Bearish
                        ob = {
                            'type': 'bullish',
                            'index': ob_candle_idx,
                            'high': df['high'].iloc[ob_candle_idx],
                            'low': df['low'].iloc[ob_candle_idx],
                            'strength': impulse_size / avg_atr,
                            'age': 0,  # Candles since formation
                            'mitigated': False,
                            'mitigation_count': 0
                        }
                        obs.append(ob)
                        
        return obs
        
    def find_bearish_order_blocks(
        self,
        df: pd.DataFrame,
        lookback: int = 50
    ) -> List[Dict]:
        """Find bearish order blocks (last bullish candle before bearish impulse)"""
        
        obs = []
        atr = self.compute_atr(df)
        
        for i in range(lookback, len(df)):
            # Check for bearish impulse (close < low of previous candle)
            if df['close'].iloc[i] < df['low'].iloc[i-1]:
                impulse_size = df['open'].iloc[i] - df['close'].iloc[i]
                avg_atr = atr.iloc[i]
                
                # Must be a strong impulse
                if impulse_size > avg_atr * self.impulse_multiplier:
                    # Look for bullish candle before impulse
                    ob_candle_idx = i - 1
                    
                    if df['close'].iloc[ob_candle_idx] > df['open'].iloc[ob_candle_idx]:  # Bullish
                        ob = {
                            'type': 'bearish',
                            'index': ob_candle_idx,
                            'high': df['high'].iloc[ob_candle_idx],
                            'low': df['low'].iloc[ob_candle_idx],
                            'strength': impulse_size / avg_atr,
                            'age': 0,
                            'mitigated': False,
                            'mitigation_count': 0
                        }
                        obs.append(ob)
                        
        return obs
        
    def update_order_blocks(
        self,
        obs: List[Dict],
        current_price: float,
        atr: float
    ) -> List[Dict]:
        """Update order block status (age, mitigation)"""
        
        updated_obs = []
        
        for ob in obs:
            ob['age'] += 1
            
            # Check if price has touched the order block
            if ob['type'] == 'bullish':
                if current_price <= ob['high']:
                    ob['mitigation_count'] += 1
                    if ob['mitigation_count'] >= 2:
                        ob['mitigated'] = True
            else:  # bearish
                if current_price >= ob['low']:
                    ob['mitigation_count'] += 1
                    if ob['mitigation_count'] >= 2:
                        ob['mitigated'] = True
                        
            # Only keep recent, unmitigated OBs
            if ob['age'] < 100 and not ob['mitigated']:
                updated_obs.append(ob)
                
        return updated_obs
        
    def get_features(
        self,
        df: pd.DataFrame,
        current_price: float
    ) -> Dict[str, float]:
        """Extract order block features for fusion model"""
        
        bullish_obs = self.find_bullish_order_blocks(df)
        bearish_obs = self.find_bearish_order_blocks(df)
        
        atr = self.compute_atr(df).iloc[-1]
        
        # Distance to nearest order block
        dist_bull = float('inf')
        dist_bear = float('inf')
        
        for ob in bullish_obs:
            dist = current_price - ob['high']  # Price above OB high
            if 0 < dist < dist_bull:
                dist_bull = dist
                
        for ob in bearish_obs:
            dist = ob['low'] - current_price  # Price below OB low
            if 0 < dist < dist_bear:
                dist_bear = dist
                
        return {
            'dist_nearest_bull_ob': dist_bull / atr if dist_bull != float('inf') else 10.0,
            'dist_nearest_bear_ob': dist_bear / atr if dist_bear != float('inf') else 10.0,
            'bull_ob_strength': max([ob['strength'] for ob in bullish_obs], default=0),
            'bear_ob_strength': max([ob['strength'] for ob in bearish_obs], default=0),
            'bull_ob_count': len(bullish_obs),
            'bear_ob_count': len(bearish_obs)
        }
