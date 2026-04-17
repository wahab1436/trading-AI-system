"""Liquidity pool detection - identifies institutional liquidity zones"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class LiquidityPool:
    """Represents a liquidity pool (equal highs/lows or trendline liquidity)"""
    type: str  # 'high', 'low', 'trendline'
    price: float
    strength: int  # Number of touches/contacts
    volume: float
    timestamp: pd.Timestamp
    is_taken: bool = False
    take_time: Optional[pd.Timestamp] = None


class LiquidityDetector:
    """
    Detects liquidity pools including:
    - Equal highs/lows (double tops/bottoms)
    - Trendline liquidity
    - Range liquidity
    - Stop clusters
    """
    
    def __init__(
        self,
        atr_period: int = 14,
        equal_price_threshold: float = 0.0005,  # 0.05% for forex, 0.5$ for gold
        min_touches: int = 2,
        lookback_candles: int = 100
    ):
        self.atr_period = atr_period
        self.equal_price_threshold = equal_price_threshold
        self.min_touches = min_touches
        self.lookback_candles = lookback_candles
        
    def compute_atr(self, df: pd.DataFrame) -> pd.Series:
        """Calculate ATR for threshold scaling"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.ewm(span=self.atr_period, adjust=False).mean()
    
    def find_equal_highs(
        self,
        df: pd.DataFrame,
        lookback: Optional[int] = None
    ) -> List[LiquidityPool]:
        """
        Find equal highs (resistance liquidity pools)
        These are price levels where multiple highs cluster
        """
        if lookback is None:
            lookback = self.lookback_candles
            
        df_slice = df.tail(lookback)
        highs = df_slice['high'].values
        timestamps = df_slice['timestamp'].values
        
        # Group similar highs
        pools = []
        used_indices = set()
        
        # Calculate dynamic threshold based on ATR
        atr = self.compute_atr(df_slice).iloc[-1] if len(df_slice) > self.atr_period else self.equal_price_threshold
        threshold = max(self.equal_price_threshold, atr * 0.1)  # 10% of ATR as threshold
        
        for i in range(len(highs)):
            if i in used_indices:
                continue
                
            # Find all highs within threshold of current high
            similar_indices = []
            for j in range(i + 1, len(highs)):
                if abs(highs[j] - highs[i]) <= threshold:
                    similar_indices.append(j)
                    
            # Check if we have enough touches
            if len(similar_indices) + 1 >= self.min_touches:
                # Calculate average price
                all_highs = [highs[i]] + [highs[idx] for idx in similar_indices]
                avg_price = np.mean(all_highs)
                
                # Calculate strength based on number of touches and volume
                volume_sum = df_slice['volume'].iloc[[i] + similar_indices].sum() if 'volume' in df_slice.columns else 0
                
                pool = LiquidityPool(
                    type='high',
                    price=avg_price,
                    strength=len(all_highs),
                    volume=volume_sum,
                    timestamp=timestamps[i]
                )
                pools.append(pool)
                
                # Mark as used
                used_indices.add(i)
                for idx in similar_indices:
                    used_indices.add(idx)
                    
        return pools
    
    def find_equal_lows(
        self,
        df: pd.DataFrame,
        lookback: Optional[int] = None
    ) -> List[LiquidityPool]:
        """
        Find equal lows (support liquidity pools)
        These are price levels where multiple lows cluster
        """
        if lookback is None:
            lookback = self.lookback_candles
            
        df_slice = df.tail(lookback)
        lows = df_slice['low'].values
        timestamps = df_slice['timestamp'].values
        
        pools = []
        used_indices = set()
        
        atr = self.compute_atr(df_slice).iloc[-1] if len(df_slice) > self.atr_period else self.equal_price_threshold
        threshold = max(self.equal_price_threshold, atr * 0.1)
        
        for i in range(len(lows)):
            if i in used_indices:
                continue
                
            similar_indices = []
            for j in range(i + 1, len(lows)):
                if abs(lows[j] - lows[i]) <= threshold:
                    similar_indices.append(j)
                    
            if len(similar_indices) + 1 >= self.min_touches:
                all_lows = [lows[i]] + [lows[idx] for idx in similar_indices]
                avg_price = np.mean(all_lows)
                
                volume_sum = df_slice['volume'].iloc[[i] + similar_indices].sum() if 'volume' in df_slice.columns else 0
                
                pool = LiquidityPool(
                    type='low',
                    price=avg_price,
                    strength=len(all_lows),
                    volume=volume_sum,
                    timestamp=timestamps[i]
                )
                pools.append(pool)
                
                used_indices.add(i)
                for idx in similar_indices:
                    used_indices.add(idx)
                    
        return pools
    
    def find_double_tops_bottoms(
        self,
        df: pd.DataFrame,
        lookback: int = 50
    ) -> Dict[str, List[Dict]]:
        """
        Detect double top/bottom formations (major liquidity grabs)
        """
        highs, lows = self._find_swing_points(df.tail(lookback))
        
        double_tops = []
        double_bottoms = []
        
        # Find double tops (two swing highs at similar level)
        for i in range(len(highs) - 1):
            for j in range(i + 1, len(highs)):
                price_diff = abs(df['high'].iloc[highs[j]] - df['high'].iloc[highs[i]])
                atr_val = self.compute_atr(df).iloc[highs[j]]
                
                if price_diff <= atr_val * 0.15:  # Within 15% of ATR
                    # Check if there's a lower low between them (valid double top)
                    between_lows = [l for l in lows if highs[i] < l < highs[j]]
                    if between_lows:
                        lowest_between = min(df['low'].iloc[l] for l in between_lows)
                        if lowest_between < df['low'].iloc[highs[i]] and lowest_between < df['low'].iloc[highs[j]]:
                            double_tops.append({
                                'left_high_index': highs[i],
                                'right_high_index': highs[j],
                                'left_price': df['high'].iloc[highs[i]],
                                'right_price': df['high'].iloc[highs[j]],
                                'neckline': lowest_between,
                                'strength': 2
                            })
                            break  # Found a pair, move to next
                            
        # Find double bottoms (two swing lows at similar level)
        for i in range(len(lows) - 1):
            for j in range(i + 1, len(lows)):
                price_diff = abs(df['low'].iloc[lows[j]] - df['low'].iloc[lows[i]])
                atr_val = self.compute_atr(df).iloc[lows[j]]
                
                if price_diff <= atr_val * 0.15:
                    between_highs = [h for h in highs if lows[i] < h < lows[j]]
                    if between_highs:
                        highest_between = max(df['high'].iloc[h] for h in between_highs)
                        if highest_between > df['high'].iloc[lows[i]] and highest_between > df['high'].iloc[lows[j]]:
                            double_bottoms.append({
                                'left_low_index': lows[i],
                                'right_low_index': lows[j],
                                'left_price': df['low'].iloc[lows[i]],
                                'right_price': df['low'].iloc[lows[j]],
                                'neckline': highest_between,
                                'strength': 2
                            })
                            break
                            
        return {
            'double_tops': double_tops,
            'double_bottoms': double_bottoms
        }
    
    def find_range_liquidity(
        self,
        df: pd.DataFrame,
        lookback: int = 50,
        min_range_candles: int = 10
    ) -> Dict[str, Optional[float]]:
        """
        Detect trading range and identify liquidity at range boundaries
        """
        df_slice = df.tail(lookback)
        
        # Find range boundaries using rolling highs/lows
        rolling_high = df_slice['high'].rolling(window=min_range_candles, center=True).max()
        rolling_low = df_slice['low'].rolling(window=min_range_candles, center=True).min()
        
        # Check if we're in a range (low volatility, bounded price)
        range_high = rolling_high.iloc[-1]
        range_low = rolling_low.iloc[-1]
        range_size = (range_high - range_low) / range_low
        
        atr = self.compute_atr(df_slice).iloc[-1]
        avg_price = (range_high + range_low) / 2
        
        is_ranging = range_size < (atr / avg_price) * 2  # Range less than 2x ATR
        
        return {
            'is_ranging': is_ranging,
            'range_high': range_high if is_ranging else None,
            'range_low': range_low if is_ranging else None,
            'range_size_pct': range_size,
            'liquidity_above_range': range_high + atr * 0.5 if is_ranging else None,
            'liquidity_below_range': range_low - atr * 0.5 if is_ranging else None
        }
    
    def find_stop_clusters(
        self,
        df: pd.DataFrame,
        current_price: float,
        lookback: int = 100
    ) -> Dict[str, List[float]]:
        """
        Identify potential stop loss clusters
        Stops typically cluster just beyond swing highs/lows
        """
        df_slice = df.tail(lookback)
        highs, lows = self._find_swing_points(df_slice)
        
        atr = self.compute_atr(df_slice).iloc[-1]
        
        # Stops above swing highs (for short positions)
        stop_above = []
        for high_idx in highs[-10:]:  # Last 10 swing highs
            high_price = df_slice['high'].iloc[high_idx]
            # Stops typically 5-15 pips above swing highs
            stop_level = high_price + atr * 0.15
            stop_above.append(stop_level)
            
        # Stops below swing lows (for long positions)
        stop_below = []
        for low_idx in lows[-10:]:
            low_price = df_slice['low'].iloc[low_idx]
            stop_level = low_price - atr * 0.15
            stop_below.append(stop_level)
            
        # Group nearby stops
        clustered_stops_above = self._cluster_prices(stop_above, atr * 0.1)
        clustered_stops_below = self._cluster_prices(stop_below, atr * 0.1)
        
        return {
            'stop_clusters_above': clustered_stops_above,
            'stop_clusters_below': clustered_stops_below,
            'nearest_stop_above': min(clustered_stops_above) if clustered_stops_above else None,
            'nearest_stop_below': max(clustered_stops_below) if clustered_stops_below else None
        }
    
    def _find_swing_points(
        self,
        df: pd.DataFrame,
        lookback: int = 5
    ) -> Tuple[List[int], List[int]]:
        """Find swing highs and lows"""
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
    
    def _cluster_prices(self, prices: List[float], threshold: float) -> List[float]:
        """Group nearby prices into clusters"""
        if not prices:
            return []
            
        prices = sorted(prices)
        clusters = []
        current_cluster = [prices[0]]
        
        for price in prices[1:]:
            if price - current_cluster[-1] <= threshold:
                current_cluster.append(price)
            else:
                clusters.append(np.mean(current_cluster))
                current_cluster = [price]
                
        clusters.append(np.mean(current_cluster))
        return clusters
    
    def get_liquidity_features(
        self,
        df: pd.DataFrame,
        current_price: float
    ) -> Dict[str, float]:
        """
        Extract liquidity features for fusion model
        """
        # Find liquidity pools
        high_pools = self.find_equal_highs(df)
        low_pools = self.find_equal_lows(df)
        range_data = self.find_range_liquidity(df)
        stop_data = self.find_stop_clusters(df, current_price)
        
        atr = self.compute_atr(df).iloc[-1]
        
        # Distance to nearest liquidity
        nearest_high_liquidity = float('inf')
        nearest_low_liquidity = float('inf')
        
        for pool in high_pools:
            if not pool.is_taken:
                dist = pool.price - current_price
                if 0 < dist < nearest_high_liquidity:
                    nearest_high_liquidity = dist
                    
        for pool in low_pools:
            if not pool.is_taken:
                dist = current_price - pool.price
                if 0 < dist < nearest_low_liquidity:
                    nearest_low_liquidity = dist
                    
        # Calculate liquidity strength
        high_liquidity_strength = sum(p.strength for p in high_pools if not p.is_taken) / len(high_pools) if high_pools else 0
        low_liquidity_strength = sum(p.strength for p in low_pools if not p.is_taken) / len(low_pools) if low_pools else 0
        
        # Double top/bottom detection
        patterns = self.find_double_tops_bottoms(df)
        has_double_top = len(patterns['double_tops']) > 0
        has_double_bottom = len(patterns['double_bottoms']) > 0
        
        return {
            # Distance to liquidity (ATR-normalized)
            'liq_high_distance': nearest_high_liquidity / atr if nearest_high_liquidity != float('inf') else 10.0,
            'liq_low_distance': nearest_low_liquidity / atr if nearest_low_liquidity != float('inf') else 10.0,
            
            # Liquidity strength
            'liq_high_strength': high_liquidity_strength,
            'liq_low_strength': low_liquidity_strength,
            'liq_high_count': len([p for p in high_pools if not p.is_taken]),
            'liq_low_count': len([p for p in low_pools if not p.is_taken]),
            
            # Range liquidity
            'is_ranging': 1 if range_data['is_ranging'] else 0,
            'range_size_pct': range_data['range_size_pct'],
            
            # Stop clusters
            'stop_cluster_above_dist': (stop_data['nearest_stop_above'] - current_price) / atr if stop_data['nearest_stop_above'] else 10.0,
            'stop_cluster_below_dist': (current_price - stop_data['nearest_stop_below']) / atr if stop_data['nearest_stop_below'] else 10.0,
            
            # Patterns
            'has_double_top': 1 if has_double_top else 0,
            'has_double_bottom': 1 if has_double_bottom else 0,
        }
    
    def update_liquidity_taken(
        self,
        pools: List[LiquidityPool],
        current_price: float,
        atr: float
    ) -> List[LiquidityPool]:
        """Update which liquidity pools have been taken (price reached them)"""
        
        for pool in pools:
            if not pool.is_taken:
                if pool.type == 'high' and current_price >= pool.price:
                    pool.is_taken = True
                    pool.take_time = pd.Timestamp.now()
                elif pool.type == 'low' and current_price <= pool.price:
                    pool.is_taken = True
                    pool.take_time = pd.Timestamp.now()
                    
        return pools
