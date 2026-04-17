"""Data validation layer - ensures data quality before pipeline ingestion"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple, Any
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of data validation"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    fixed_count: int = 0
    
    def add_error(self, error: str):
        self.errors.append(error)
        self.is_valid = False
        
    def add_warning(self, warning: str):
        self.warnings.append(warning)
        
    def __bool__(self):
        return self.is_valid


class DataValidator:
    """Validates and cleans OHLCV data"""
    
    def __init__(self, atr_period: int = 14, max_spike_multiplier: float = 5.0):
        self.atr_period = atr_period
        self.max_spike_multiplier = max_spike_multiplier
        
    def validate_and_clean(self, df: pd.DataFrame) -> ValidationResult:
        """Main validation and cleaning pipeline"""
        
        result = ValidationResult(is_valid=True)
        
        if df is None or len(df) == 0:
            result.add_error("DataFrame is empty or None")
            return result
            
        # Run all validations
        result = self._check_schema(df, result)
        result = self._remove_duplicates(df, result)
        result = self._fill_gaps(df, result)
        result = self._remove_outliers(df, result)
        result = self._validate_timestamps(df, result)
        result = self._validate_ohlc_integrity(df, result)
        
        return result
        
    def _check_schema(self, df: pd.DataFrame, result: ValidationResult) -> ValidationResult:
        """Validate column schema and data types"""
        
        required_columns = ['open', 'high', 'low', 'close', 'timestamp']
        
        for col in required_columns:
            if col not in df.columns:
                result.add_error(f"Missing required column: {col}")
                
        # Check for NaN values
        nan_counts = df[required_columns].isna().sum()
        for col, count in nan_counts.items():
            if count > 0:
                result.add_warning(f"Column {col} has {count} NaN values")
                
        # Validate data types
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    result.add_error(f"Column {col} is not numeric")
                    
        return result
        
    def _remove_duplicates(self, df: pd.DataFrame, result: ValidationResult) -> ValidationResult:
        """Remove duplicate rows based on timestamp and symbol"""
        
        original_len = len(df)
        
        # Check for duplicate timestamps
        if 'timestamp' in df.columns:
            df.drop_duplicates(subset=['timestamp'], keep='first', inplace=True)
            
        duplicates_removed = original_len - len(df)
        if duplicates_removed > 0:
            result.fixed_count += duplicates_removed
            result.add_warning(f"Removed {duplicates_removed} duplicate rows")
            
        return result
        
    def _fill_gaps(self, df: pd.DataFrame, result: ValidationResult) -> ValidationResult:
        """Detect and fill missing candles"""
        
        if 'timestamp' not in df.columns:
            return result
            
        # Sort by timestamp
        df.sort_values('timestamp', inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        # Detect expected frequency
        if len(df) >= 2:
            time_diffs = df['timestamp'].diff().dropna()
            if len(time_diffs) > 0:
                expected_freq = time_diffs.mode().iloc[0]
                
                # Check for gaps
                gaps = time_diffs[time_diffs > expected_freq * 1.5]
                
                if len(gaps) > 0:
                    result.add_warning(f"Found {len(gaps)} gaps in data")
                    
                    # Forward fill for small gaps
                    df.set_index('timestamp', inplace=True)
                    df = df.asfreq(expected_freq)
                    
                    # Forward fill missing values
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        if col in df.columns:
                            df[col] = df[col].fillna(method='ffill')
                            
                    df.reset_index(inplace=True)
                    result.fixed_count += len(gaps)
                    
        return result
        
    def _remove_outliers(self, df: pd.DataFrame, result: ValidationResult) -> ValidationResult:
        """Remove price spikes that are > max_spike_multiplier * ATR"""
        
        if len(df) < self.atr_period + 1:
            return result
            
        # Calculate ATR
        atr = self._compute_atr(df)
        
        # Calculate candle ranges
        df['candle_range'] = df['high'] - df['low']
        
        # Detect spikes
        spike_mask = df['candle_range'] > atr * self.max_spike_multiplier
        
        if spike_mask.any():
            spike_count = spike_mask.sum()
            result.add_warning(f"Found {spike_count} outlier candles")
            
            # Replace outliers with interpolated values
            for col in ['open', 'high', 'low', 'close']:
                df.loc[spike_mask, col] = np.nan
                
            # Interpolate
            df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].interpolate()
            
            result.fixed_count += spike_count
            
        # Clean up temporary columns
        df.drop(columns=['candle_range'], inplace=True)
        
        return result
        
    def _compute_atr(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range"""
        
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        return tr.ewm(span=self.atr_period, adjust=False).mean()
        
    def _validate_timestamps(self, df: pd.DataFrame, result: ValidationResult) -> ValidationResult:
        """Validate and normalize timestamps"""
        
        if 'timestamp' not in df.columns:
            return result
            
        # Convert to UTC if not already
        if df['timestamp'].dt.tz is None:
            df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
        else:
            df['timestamp'] = df['timestamp'].dt.tz_convert('UTC')
            
        # Check for future timestamps
        now = datetime.now().astimezone()
        future_mask = df['timestamp'] > now
        
        if future_mask.any():
            result.add_error(f"Found {future_mask.sum()} future timestamps")
            
        return result
        
    def _validate_ohlc_integrity(self, df: pd.DataFrame, result: ValidationResult) -> ValidationResult:
        """Validate that high >= low, high >= open/close, low <= open/close"""
        
        # Check high >= low
        invalid_hl = df['high'] < df['low']
        if invalid_hl.any():
            result.add_error(f"Found {invalid_hl.sum()} candles with high < low")
            # Fix by swapping
            mask = invalid_hl
            df.loc[mask, ['high', 'low']] = df.loc[mask, ['low', 'high']].values
            
        # Check high >= open and high >= close
        invalid_high = (df['high'] < df['open']) | (df['high'] < df['close'])
        if invalid_high.any():
            result.add_warning(f"Found {invalid_high.sum()} candles with invalid high")
            df.loc[invalid_high, 'high'] = df.loc[invalid_high, ['open', 'close']].max(axis=1)
            
        # Check low <= open and low <= close
        invalid_low = (df['low'] > df['open']) | (df['low'] > df['close'])
        if invalid_low.any():
            result.add_warning(f"Found {invalid_low.sum()} candles with invalid low")
            df.loc[invalid_low, 'low'] = df.loc[invalid_low, ['open', 'close']].min(axis=1)
            
        return result
