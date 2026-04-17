"""Prefect flow for labeling OHLCV data"""

from prefect import flow, task
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from data_engine.validator import DataValidator

logger = logging.getLogger(__name__)


@task
def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range"""
    
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    return tr.ewm(span=period, adjust=False).mean()


@task
def label_candles(
    df: pd.DataFrame,
    lookahead: int = 8,
    atr_multiplier: float = 1.2
) -> pd.DataFrame:
    """Label candles as BUY/SELL/NO_TRADE based on future movement"""
    
    df = df.copy()
    
    # Calculate ATR
    df['ATR'] = compute_atr(df)
    
    # Future close
    df['future_close'] = df['close'].shift(-lookahead)
    
    # Future movement
    df['future_move'] = df['future_close'] - df['close']
    
    # Threshold
    threshold = df['ATR'] * atr_multiplier
    
    # Assign labels
    df['label'] = 0  # NO_TRADE default
    df.loc[df['future_move'] > threshold, 'label'] = 1  # BUY
    df.loc[df['future_move'] < -threshold, 'label'] = -1  # SELL
    
    # Remove rows with NaN
    df = df.dropna()
    
    return df


@task
def compute_htf_bias(df_htf: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """Compute higher timeframe trend bias"""
    
    if len(df_htf) < lookback:
        return pd.Series([0] * len(df_htf))
        
    # Find swing highs and lows
    highs = df_htf['high'].rolling(lookback // 2, center=True).max()
    lows = df_htf['low'].rolling(lookback // 2, center=True).min()
    
    # Determine trend
    higher_highs = df_htf['high'] > highs.shift(1)
    higher_lows = df_htf['low'] > lows.shift(1)
    
    bias = pd.Series(0, index=df_htf.index)
    bias[higher_highs & higher_lows] = 1  # Bullish
    bias[~(higher_highs | higher_lows)] = -1  # Bearish
    
    return bias


@task
def apply_htf_filter(df: pd.DataFrame, df_h4: pd.DataFrame) -> pd.DataFrame:
    """Filter labels based on H4 trend bias"""
    
    if len(df_h4) == 0:
        return df
        
    # Align H4 bias to 15m timeframe
    df_h4 = df_h4.copy()
    df_h4['timestamp'] = pd.to_datetime(df_h4['timestamp'])
    df_h4.set_index('timestamp', inplace=True)
    
    # Resample to 15m
    h4_bias_15m = df_h4['bias'].resample('15T').ffill()
    
    # Merge
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    df['htf_bias'] = h4_bias_15m.reindex(df.index, method='ffill').fillna(0)
    
    # Filter: Only BUY when H4 bias is bullish, only SELL when bearish
    df['label_filtered'] = df['label']
    mask_bullish = (df['label'] == 1) & (df['htf_bias'] != 1)
    mask_bearish = (df['label'] == -1) & (df['htf_bias'] != -1)
    
    df.loc[mask_bullish | mask_bearish, 'label_filtered'] = 0
    
    df.reset_index(inplace=True)
    
    return df


@task
def balance_labels(df: pd.DataFrame, target_ratios: dict = None) -> pd.DataFrame:
    """Balance label distribution to prevent skew"""
    
    if target_ratios is None:
        target_ratios = {1: 0.4, -1: 0.4, 0: 0.2}  # 40% BUY, 40% SELL, 20% NO_TRADE
        
    df = df.copy()
    
    # Get current distribution
    label_counts = df['label_filtered'].value_counts()
    
    # Determine target counts
    total = len(df)
    target_counts = {label: int(total * ratio) for label, ratio in target_ratios.items()}
    
    # Undersample majority classes
    balanced_dfs = []
    
    for label, target_count in target_counts.items():
        label_df = df[df['label_filtered'] == label]
        
        if len(label_df) > target_count:
            # Undersample
            label_df = label_df.sample(n=target_count, random_state=42)
            
        balanced_dfs.append(label_df)
        
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=42)  # Shuffle
    
    logger.info(f"Balanced labels: {balanced_df['label_filtered'].value_counts().to_dict()}")
    
    return balanced_df


@flow(name="label_data")
def label_data_flow(
    input_dir: str = "data/raw/",
    output_dir: str = "data/labels/",
    h4_data_dir: str = "data/raw/"
):
    """Main flow for labeling OHLCV data with HTF filter"""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    dfs = {}
    for file in input_path.glob("*.parquet"):
        symbol = file.stem.split('_')[0]
        timeframe = file.stem.split('_')[1]
        
        df = pd.read_parquet(file)
        dfs[(symbol, timeframe)] = df
        logger.info(f"Loaded {len(df)} candles from {file.name}")
        
    # Process each symbol
    for (symbol, timeframe), df in dfs.items():
        if timeframe != "15m":
            continue
            
        logger.info(f"Processing {symbol} 15m data...")
        
        # Label candles
        df_labeled = label_candles(df)
        
        # Apply H4 filter if available
        if (symbol, "4h") in dfs:
            df_h4 = dfs[(symbol, "4h")]
            df_h4['bias'] = compute_htf_bias(df_h4)
            df_labeled = apply_htf_filter(df_labeled, df_h4)
            
        # Balance labels
        df_balanced = balance_labels(df_labeled)
        
        # Save
        output_file = output_path / f"{symbol}_15m_labels.parquet"
        df_balanced.to_parquet(output_file)
        logger.info(f"Saved labels to {output_file}")
        
    return output_path


if __name__ == "__main__":
    label_data_flow()
