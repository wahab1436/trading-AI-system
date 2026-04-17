"""Feature vector builder that combines SMC features into a structured format"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class SMCFeatureVector:
    """Complete SMC feature vector for a single prediction point"""
    
    # Structure features (from structure.py)
    hh_hl_ratio: float = 0.0      # Bullish structure score (0-1)
    lh_ll_ratio: float = 0.0      # Bearish structure score (0-1)
    bos_count_bull: int = 0       # Bullish BOS events last 50 candles
    bos_count_bear: int = 0       # Bearish BOS events last 50 candles
    choch_detected: int = 0       # Change of Character detected (0/1)
    
    # Order Block features (from order_blocks.py)
    dist_nearest_bull_ob: float = 10.0   # Distance to bullish OB (ATR units)
    dist_nearest_bear_ob: float = 10.0   # Distance to bearish OB (ATR units)
    bull_ob_strength: float = 0.0        # Strength of nearest bullish OB
    bear_ob_strength: float = 0.0        # Strength of nearest bearish OB
    bull_ob_count: int = 0               # Number of active bullish OBs
    bear_ob_count: int = 0               # Number of active bearish OBs
    
    # Fair Value Gap features (from fvg.py)
    fvg_bull_open: int = 0        # Unfilled bullish FVG within 2 ATR (0/1)
    fvg_bear_open: int = 0        # Unfilled bearish FVG within 2 ATR (0/1)
    fvg_bull_count: int = 0       # Number of unfilled bullish FVGs
    fvg_bear_count: int = 0       # Number of unfilled bearish FVGs
    
    # Liquidity features (from liquidity.py)
    liq_high_distance: float = 10.0    # Distance to nearest equal highs (ATR)
    liq_low_distance: float = 10.0     # Distance to nearest equal lows (ATR)
    liq_high_count: int = 0            # Number of nearby liquidity highs
    liq_low_count: int = 0             # Number of nearby liquidity lows
    
    # Impulse features (from impulse.py)
    impulse_strength: float = 0.0      # Last impulse body/ATR ratio
    impulse_direction: int = 0         # 1=bullish, -1=bearish, 0=none
    impulse_age: int = 0               # Candles since last impulse
    
    # Market state features
    market_state: int = 0              # 0=ranging, 1=trending_bull, 2=trending_bear
    volatility_regime: float = 0.5     # ATR percentile vs 3-month lookback (0-1)
    
    # Session and HTF features
    session_code: int = 0              # 0=Asian, 1=London, 2=Overlap, 3=NY
    htf_bias: int = 0                  # -1=bear, 0=neutral, 1=bull
    
    # Additional derived features
    spread_pips: float = 0.0           # Current spread in pips
    time_of_day: float = 0.0           # Hour of day (0-23) normalized
    day_of_week: int = 0               # 0=Monday, 6=Sunday
    
    def to_array(self) -> np.ndarray:
        """Convert feature vector to numpy array"""
        return np.array([
            self.hh_hl_ratio,
            self.lh_ll_ratio,
            self.bos_count_bull,
            self.bos_count_bear,
            self.choch_detected,
            self.dist_nearest_bull_ob,
            self.dist_nearest_bear_ob,
            self.bull_ob_strength,
            self.bear_ob_strength,
            self.bull_ob_count,
            self.bear_ob_count,
            self.fvg_bull_open,
            self.fvg_bear_open,
            self.fvg_bull_count,
            self.fvg_bear_count,
            self.liq_high_distance,
            self.liq_low_distance,
            self.liq_high_count,
            self.liq_low_count,
            self.impulse_strength,
            self.impulse_direction,
            self.impulse_age,
            self.market_state,
            self.volatility_regime,
            self.session_code,
            self.htf_bias,
            self.spread_pips,
            self.time_of_day,
            self.day_of_week
        ], dtype=np.float32)
    
    def to_dict(self) -> Dict:
        """Convert feature vector to dictionary"""
        return {
            'hh_hl_ratio': self.hh_hl_ratio,
            'lh_ll_ratio': self.lh_ll_ratio,
            'bos_count_bull': self.bos_count_bull,
            'bos_count_bear': self.bos_count_bear,
            'choch_detected': self.choch_detected,
            'dist_nearest_bull_ob': self.dist_nearest_bull_ob,
            'dist_nearest_bear_ob': self.dist_nearest_bear_ob,
            'bull_ob_strength': self.bull_ob_strength,
            'bear_ob_strength': self.bear_ob_strength,
            'bull_ob_count': self.bull_ob_count,
            'bear_ob_count': self.bear_ob_count,
            'fvg_bull_open': self.fvg_bull_open,
            'fvg_bear_open': self.fvg_bear_open,
            'fvg_bull_count': self.fvg_bull_count,
            'fvg_bear_count': self.fvg_bear_count,
            'liq_high_distance': self.liq_high_distance,
            'liq_low_distance': self.liq_low_distance,
            'liq_high_count': self.liq_high_count,
            'liq_low_count': self.liq_low_count,
            'impulse_strength': self.impulse_strength,
            'impulse_direction': self.impulse_direction,
            'impulse_age': self.impulse_age,
            'market_state': self.market_state,
            'volatility_regime': self.volatility_regime,
            'session_code': self.session_code,
            'htf_bias': self.htf_bias,
            'spread_pips': self.spread_pips,
            'time_of_day': self.time_of_day,
            'day_of_week': self.day_of_week
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SMCFeatureVector':
        """Create feature vector from dictionary"""
        return cls(**{k: data.get(k, v) for k, v in cls.__dataclass_fields__.items()})


class FeatureBuilder:
    """
    Builds complete feature vectors from raw data sources.
    Integrates all SMC components into a unified feature vector.
    """
    
    def __init__(self):
        self.feature_names = self._get_feature_names()
        
    def _get_feature_names(self) -> List[str]:
        """Get list of feature names in order"""
        return [
            'hh_hl_ratio', 'lh_ll_ratio', 'bos_count_bull', 'bos_count_bear',
            'choch_detected', 'dist_nearest_bull_ob', 'dist_nearest_bear_ob',
            'bull_ob_strength', 'bear_ob_strength', 'bull_ob_count', 'bear_ob_count',
            'fvg_bull_open', 'fvg_bear_open', 'fvg_bull_count', 'fvg_bear_count',
            'liq_high_distance', 'liq_low_distance', 'liq_high_count', 'liq_low_count',
            'impulse_strength', 'impulse_direction', 'impulse_age',
            'market_state', 'volatility_regime', 'session_code', 'htf_bias',
            'spread_pips', 'time_of_day', 'day_of_week'
        ]
        
    def build_from_components(
        self,
        structure_features: Dict,
        ob_features: Dict,
        fvg_features: Dict,
        liquidity_features: Dict,
        impulse_features: Dict,
        market_state: int,
        volatility_regime: float,
        session_code: int,
        htf_bias: int,
        spread_pips: float = 0.0,
        timestamp: Optional[datetime] = None
    ) -> SMCFeatureVector:
        """
        Build feature vector from individual component outputs
        
        Args:
            structure_features: Output from StructureDetector.compute_structure_scores()
            ob_features: Output from OrderBlockDetector.get_features()
            fvg_features: Output from FVGDetector.get_features()
            liquidity_features: Output from LiquidityDetector.get_features()
            impulse_features: Output from ImpulseDetector.get_features()
            market_state: 0=ranging, 1=trending_bull, 2=trending_bear
            volatility_regime: ATR percentile (0-1)
            session_code: 0=Asian, 1=London, 2=Overlap, 3=NY
            htf_bias: -1=bear, 0=neutral, 1=bull
            spread_pips: Current spread in pips
            timestamp: Current timestamp for time features
        """
        
        # Extract time features if timestamp provided
        time_of_day = 0.0
        day_of_week = 0
        if timestamp:
            time_of_day = timestamp.hour + timestamp.minute / 60.0
            day_of_week = timestamp.weekday()
        
        return SMCFeatureVector(
            # Structure
            hh_hl_ratio=structure_features.get('hh_hl_ratio', 0.0),
            lh_ll_ratio=structure_features.get('lh_ll_ratio', 0.0),
            bos_count_bull=structure_features.get('bos_count_bull', 0),
            bos_count_bear=structure_features.get('bos_count_bear', 0),
            choch_detected=structure_features.get('choch_detected', 0),
            
            # Order Blocks
            dist_nearest_bull_ob=ob_features.get('dist_nearest_bull_ob', 10.0),
            dist_nearest_bear_ob=ob_features.get('dist_nearest_bear_ob', 10.0),
            bull_ob_strength=ob_features.get('bull_ob_strength', 0.0),
            bear_ob_strength=ob_features.get('bear_ob_strength', 0.0),
            bull_ob_count=ob_features.get('bull_ob_count', 0),
            bear_ob_count=ob_features.get('bear_ob_count', 0),
            
            # FVG
            fvg_bull_open=fvg_features.get('fvg_bull_open', 0),
            fvg_bear_open=fvg_features.get('fvg_bear_open', 0),
            fvg_bull_count=fvg_features.get('fvg_bull_count', 0),
            fvg_bear_count=fvg_features.get('fvg_bear_count', 0),
            
            # Liquidity
            liq_high_distance=liquidity_features.get('liq_high_distance', 10.0),
            liq_low_distance=liquidity_features.get('liq_low_distance', 10.0),
            liq_high_count=liquidity_features.get('liq_high_count', 0),
            liq_low_count=liquidity_features.get('liq_low_count', 0),
            
            # Impulse
            impulse_strength=impulse_features.get('impulse_strength', 0.0),
            impulse_direction=impulse_features.get('impulse_direction', 0),
            impulse_age=impulse_features.get('impulse_age', 0),
            
            # Market state
            market_state=market_state,
            volatility_regime=volatility_regime,
            session_code=session_code,
            htf_bias=htf_bias,
            
            # Additional
            spread_pips=spread_pips,
            time_of_day=time_of_day / 24.0,  # Normalize to 0-1
            day_of_week=day_of_week
        )
        
    def build_from_dataframe(
        self,
        df: pd.DataFrame,
        idx: int,
        precomputed_features: Optional[Dict] = None
    ) -> SMCFeatureVector:
        """
        Build feature vector from dataframe row
        
        Args:
            df: DataFrame containing precomputed feature columns
            idx: Row index
            precomputed_features: Optional dict of precomputed feature dicts
        """
        
        row = df.iloc[idx]
        
        # Extract from dataframe columns (assuming they exist)
        return SMCFeatureVector(
            hh_hl_ratio=row.get('hh_hl_ratio', 0.0),
            lh_ll_ratio=row.get('lh_ll_ratio', 0.0),
            bos_count_bull=int(row.get('bos_count_bull', 0)),
            bos_count_bear=int(row.get('bos_count_bear', 0)),
            choch_detected=int(row.get('choch_detected', 0)),
            dist_nearest_bull_ob=row.get('dist_nearest_bull_ob', 10.0),
            dist_nearest_bear_ob=row.get('dist_nearest_bear_ob', 10.0),
            bull_ob_strength=row.get('bull_ob_strength', 0.0),
            bear_ob_strength=row.get('bear_ob_strength', 0.0),
            bull_ob_count=int(row.get('bull_ob_count', 0)),
            bear_ob_count=int(row.get('bear_ob_count', 0)),
            fvg_bull_open=int(row.get('fvg_bull_open', 0)),
            fvg_bear_open=int(row.get('fvg_bear_open', 0)),
            fvg_bull_count=int(row.get('fvg_bull_count', 0)),
            fvg_bear_count=int(row.get('fvg_bear_count', 0)),
            liq_high_distance=row.get('liq_high_distance', 10.0),
            liq_low_distance=row.get('liq_low_distance', 10.0),
            liq_high_count=int(row.get('liq_high_count', 0)),
            liq_low_count=int(row.get('liq_low_count', 0)),
            impulse_strength=row.get('impulse_strength', 0.0),
            impulse_direction=int(row.get('impulse_direction', 0)),
            impulse_age=int(row.get('impulse_age', 0)),
            market_state=int(row.get('market_state', 0)),
            volatility_regime=row.get('volatility_regime', 0.5),
            session_code=int(row.get('session_code', 0)),
            htf_bias=int(row.get('htf_bias', 0)),
            spread_pips=row.get('spread_pips', 0.0),
            time_of_day=row.get('time_of_day', 0.0),
            day_of_week=int(row.get('day_of_week', 0))
        )
        
    def build_batch(
        self,
        structure_features_list: List[Dict],
        ob_features_list: List[Dict],
        fvg_features_list: List[Dict],
        liquidity_features_list: List[Dict],
        impulse_features_list: List[Dict],
        market_states: List[int],
        volatility_regimes: List[float],
        session_codes: List[int],
        htf_biases: List[int],
        timestamps: Optional[List[datetime]] = None
    ) -> np.ndarray:
        """Build batch of feature vectors as numpy array"""
        
        vectors = []
        
        for i in range(len(structure_features_list)):
            timestamp = timestamps[i] if timestamps else None
            
            vector = self.build_from_components(
                structure_features=structure_features_list[i],
                ob_features=ob_features_list[i],
                fvg_features=fvg_features_list[i],
                liquidity_features=liquidity_features_list[i],
                impulse_features=impulse_features_list[i],
                market_state=market_states[i],
                volatility_regime=volatility_regimes[i],
                session_code=session_codes[i],
                htf_bias=htf_biases[i],
                timestamp=timestamp
            )
            vectors.append(vector.to_array())
            
        return np.array(vectors, dtype=np.float32)
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names"""
        return self.feature_names
    
    def get_feature_count(self) -> int:
        """Return number of features"""
        return len(self.feature_names)


def build_smc_feature_vector(
    df: pd.DataFrame,
    current_idx: int,
    structure_detector=None,
    ob_detector=None,
    fvg_detector=None,
    atr_period: int = 14
) -> SMCFeatureVector:
    """
    Convenience function to build complete SMC feature vector from OHLCV data
    
    Args:
        df: OHLCV DataFrame
        current_idx: Current index position
        structure_detector: StructureDetector instance
        ob_detector: OrderBlockDetector instance
        fvg_detector: FVGDetector instance
        atr_period: ATR period for normalization
        
    Returns:
        Complete SMCFeatureVector
    """
    
    from ..smc_engine.structure import StructureDetector
    from ..smc_engine.order_blocks import OrderBlockDetector
    from ..smc_engine.fvg import FVGDetector
    
    if structure_detector is None:
        structure_detector = StructureDetector()
    if ob_detector is None:
        ob_detector = OrderBlockDetector()
    if fvg_detector is None:
        fvg_detector = FVGDetector()
    
    # Get window of data
    window_size = 100
    start_idx = max(0, current_idx - window_size)
    df_window = df.iloc[start_idx:current_idx + 1].copy()
    
    if len(df_window) < 50:
        logger.warning(f"Insufficient data for feature building: {len(df_window)} candles")
        return SMCFeatureVector()
    
    # Compute features
    structure_features = structure_detector.compute_structure_scores(df_window)
    ob_features = ob_detector.get_features(df_window, df_window['close'].iloc[-1])
    fvg_features = fvg_detector.get_features(df_window, df_window['close'].iloc[-1])
    
    # Calculate ATR for volatility regime
    atr_values = []
    for i in range(atr_period, len(df_window)):
        tr = max(
            df_window['high'].iloc[i] - df_window['low'].iloc[i],
            abs(df_window['high'].iloc[i] - df_window['close'].iloc[i-1]),
            abs(df_window['low'].iloc[i] - df_window['close'].iloc[i-1])
        )
        atr_values.append(tr)
    
    current_atr = np.mean(atr_values[-atr_period:]) if atr_values else 0.01
    historical_atr = np.mean(atr_values) if atr_values else 0.01
    volatility_regime = min(1.0, current_atr / historical_atr) if historical_atr > 0 else 0.5
    
    # Determine market state
    price_change = (df_window['close'].iloc[-1] - df_window['close'].iloc[-20]) / df_window['close'].iloc[-20]
    if abs(price_change) < 0.005:  # Less than 0.5% change
        market_state = 0  # Ranging
    elif price_change > 0:
        market_state = 1  # Trending bull
    else:
        market_state = 2  # Trending bear
        
    # Session code (simplified - would use SessionTagger)
    hour = df_window['timestamp'].iloc[-1].hour if 'timestamp' in df_window.columns else 12
    if 0 <= hour < 8:
        session_code = 0  # Asian
    elif 8 <= hour < 13:
        session_code = 1  # London
    elif 13 <= hour < 17:
        session_code = 2  # Overlap
    else:
        session_code = 3  # New York
        
    # HTF bias (simplified - would use higher timeframe data)
    if len(df_window) >= 100:
        htf_trend = (df_window['close'].iloc[-1] - df_window['close'].iloc[-20]) / df_window['close'].iloc[-20]
        if htf_trend > 0.002:
            htf_bias = 1
        elif htf_trend < -0.002:
            htf_bias = -1
        else:
            htf_bias = 0
    else:
        htf_bias = 0
        
    builder = FeatureBuilder()
    
    return builder.build_from_components(
        structure_features=structure_features,
        ob_features=ob_features,
        fvg_features=fvg_features,
        liquidity_features={'liq_high_distance': 10.0, 'liq_low_distance': 10.0, 
                           'liq_high_count': 0, 'liq_low_count': 0},
        impulse_features={'impulse_strength': 0.0, 'impulse_direction': 0, 'impulse_age': 0},
        market_state=market_state,
        volatility_regime=volatility_regime,
        session_code=session_code,
        htf_bias=htf_bias,
        timestamp=df_window['timestamp'].iloc[-1] if 'timestamp' in df_window.columns else None
    )
