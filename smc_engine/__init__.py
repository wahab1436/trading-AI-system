"""
SMC (Smart Money Concepts) Engine
Complete implementation of institutional trading concepts

Modules:
- structure: Market structure (BOS, CHoCH, swings, HH/HL)
- order_blocks: Order block detection (bullish/bearish OBs)
- fvg: Fair Value Gap detection
- liquidity: Liquidity pool mapping (equal highs/lows, stop clusters)
- impulse: Impulse strength and momentum detection
- market_state: Market regime detection (trend, ranging, volatility)

Usage:
    from smc_engine import (
        StructureDetector,
        OrderBlockDetector,
        FVGDetector,
        LiquidityDetector,
        ImpulseDetector,
        MarketStateDetector,
        get_complete_smc_features
    )
"""

from .structure import StructureDetector
from .order_blocks import OrderBlockDetector
from .fvg import FVGDetector
from .liquidity import LiquidityDetector, LiquidityPool
from .impulse import ImpulseDetector, ImpulseCandle
from .market_state import MarketStateDetector, MarketRegime, VolatilityRegime

__version__ = "1.0.0"

__all__ = [
    # Detectors
    'StructureDetector',
    'OrderBlockDetector', 
    'FVGDetector',
    'LiquidityDetector',
    'ImpulseDetector',
    'MarketStateDetector',
    
    # Data classes
    'LiquidityPool',
    'ImpulseCandle',
    
    # Enums
    'MarketRegime',
    'VolatilityRegime',
]


def get_complete_smc_features(df, current_price):
    """
    Convenience function to get all SMC features at once
    
    Args:
        df: DataFrame with OHLCV data (at least 50-100 candles)
        current_price: Current market price
        
    Returns:
        Dictionary with all SMC features combined
    """
    # Initialize detectors
    structure_detector = StructureDetector()
    order_block_detector = OrderBlockDetector()
    fvg_detector = FVGDetector()
    liquidity_detector = LiquidityDetector()
    impulse_detector = ImpulseDetector()
    market_state_detector = MarketStateDetector()
    
    # Get all features
    features = {}
    
    # Structure features
    structure_features = structure_detector.compute_structure_scores(df)
    features.update(structure_features)
    
    # Order block features
    ob_features = order_block_detector.get_features(df, current_price)
    features.update(ob_features)
    
    # FVG features
    fvg_features = fvg_detector.get_features(df, current_price)
    features.update(fvg_features)
    
    # Liquidity features
    liquidity_features = liquidity_detector.get_liquidity_features(df, current_price)
    features.update(liquidity_features)
    
    # Impulse features
    impulse_features = impulse_detector.get_impulse_features(df, current_price)
    features.update(impulse_features)
    
    # Market state features
    market_features = market_state_detector.get_market_state_features(df)
    features.update(market_features)
    
    return features


# Default feature names (for fusion model)
SMC_FEATURE_NAMES = [
    # Structure (4 features)
    'hh_hl_ratio',
    'lh_ll_ratio', 
    'bos_count_bull',
    'bos_count_bear',
    'choch_detected',
    
    # Order Blocks (6 features)
    'dist_nearest_bull_ob',
    'dist_nearest_bear_ob',
    'bull_ob_strength',
    'bear_ob_strength',
    'bull_ob_count',
    'bear_ob_count',
    
    # FVG (4 features)
    'fvg_bull_open',
    'fvg_bear_open',
    'fvg_bull_count',
    'fvg_bear_count',
    
    # Liquidity (10 features)
    'liq_high_distance',
    'liq_low_distance',
    'liq_high_strength',
    'liq_low_strength',
    'liq_high_count',
    'liq_low_count',
    'is_ranging',
    'range_size_pct',
    'stop_cluster_above_dist',
    'stop_cluster_below_dist',
    'has_double_top',
    'has_double_bottom',
    
    # Impulse (12 features)
    'impulse_strength',
    'impulse_direction',
    'bull_impulse_count',
    'bear_impulse_count',
    'bull_impulse_ratio',
    'consecutive_bull_impulses',
    'consecutive_bear_impulses',
    'current_impulse_streak',
    'bull_velocity',
    'bear_velocity',
    'net_velocity',
    'absorption_score',
    'rejections_count',
    'momentum_5',
    'momentum_10',
    'momentum_20',
    
    # Market State (18 features)
    'market_state',
    'trend_score',
    'trend_strength',
    'adx',
    'plus_di',
    'minus_di',
    'volatility_regime',
    'atr_percentile',
    'volatility_multiplier',
    'current_atr_pct',
    'ranging_score',
    'range_pct',
    'is_accumulation',
    'is_distribution',
    'phase_confidence',
    'rsi',
    'rsi_trend',
    'macd',
    'macd_histogram',
    'dist_to_recent_high',
    'dist_to_recent_low',
    'is_at_high',
    'is_at_low',
]

# Total: ~70 SMC features (reduced from 16 in blueprint - now comprehensive)
