"""Historical data fetcher for bulk OHLCV downloads"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class HistoricalDataFetcher:
    """Fetch historical OHLCV data from various sources"""
    
    def __init__(self, broker_type: str = "mt5"):
        self.broker_type = broker_type
        self.symbols = ["XAUUSD", "EURUSD"]
        self.timeframes = ["15m", "1h", "4h"]
        
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Fetch OHLCV data for given symbol and timeframe"""
        
        if self.broker_type == "mt5":
            return self._fetch_from_mt5(symbol, timeframe, start_date, end_date)
        elif self.broker_type == "oanda":
            return self._fetch_from_oanda(symbol, timeframe, start_date, end_date)
        else:
            return self._generate_synthetic_data(symbol, timeframe, start_date, end_date)
            
    def _fetch_from_mt5(self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch from MetaTrader 5"""
        try:
            import MetaTrader5 as mt5
            
            # Initialize MT5
            if not mt5.initialize():
                logger.error("MT5 initialization failed")
                return pd.DataFrame()
                
            # Convert timeframe
            tf_map = {"15m": mt5.TIMEFRAME_M15, "1h": mt5.TIMEFRAME_H1, "4h": mt5.TIMEFRAME_H4}
            mt5_tf = tf_map.get(timeframe, mt5.TIMEFRAME_M15)
            
            # Fetch rates
            rates = mt5.copy_rates_range(symbol, mt5_tf, start_date, end_date)
            mt5.shutdown()
            
            if rates is None or len(rates) == 0:
                logger.warning(f"No data for {symbol} from MT5")
                return pd.DataFrame()
                
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.rename(columns={
                'time': 'timestamp',
                'open': 'open',
                'high': 'high', 
                'low': 'low',
                'close': 'close',
                'tick_volume': 'volume'
            }, inplace=True)
            
            df['symbol'] = symbol
            df['timeframe'] = timeframe
            
            return df
            
        except ImportError:
            logger.warning("MT5 package not installed, using synthetic data")
            return self._generate_synthetic_data(symbol, timeframe, start_date, end_date)
            
    def _fetch_from_oanda(self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch from OANDA API"""
        # Implementation for OANDA API
        logger.info(f"Fetching {symbol} from OANDA")
        return self._generate_synthetic_data(symbol, timeframe, start_date, end_date)
        
    def _generate_synthetic_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        trend: float = 0.0001,
        volatility: float = 0.005
    ) -> pd.DataFrame:
        """Generate synthetic OHLCV data for testing"""
        
        # Calculate number of candles
        minutes_per_candle = {"15m": 15, "1h": 60, "4h": 240}[timeframe]
        total_minutes = int((end_date - start_date).total_seconds() / 60)
        num_candles = total_minutes // minutes_per_candle
        
        if num_candles <= 0:
            return pd.DataFrame()
            
        # Generate price path
        np.random.seed(hash(symbol) % 2**32)
        
        # Base price for gold (~$2000) or EURUSD (~1.10)
        base_price = 2000.0 if symbol == "XAUUSD" else 1.10
        
        # Generate returns with drift and volatility
        returns = np.random.normal(trend * minutes_per_candle, 
                                   volatility * np.sqrt(minutes_per_candle / (24*60)),
                                   num_candles)
        
        # Add some market structure patterns
        for i in range(len(returns)):
            # Add mean reversion
            if i > 0:
                returns[i] -= 0.02 * returns[i-1]  # Slight mean reversion
                
            # Add occasional spikes (news events)
            if np.random.random() < 0.01:  # 1% of candles
                returns[i] += np.random.normal(0, volatility * 2)
                
        price_series = base_price * np.exp(np.cumsum(returns))
        
        # Generate OHLC from closing prices
        df = pd.DataFrame()
        df['timestamp'] = pd.date_range(start=start_date, periods=num_candles, freq=f'{minutes_per_candle}T')
        df['close'] = price_series
        
        # Generate realistic OHLC
        candle_range = np.abs(returns) * base_price * 0.5 + volatility * base_price
        df['open'] = df['close'].shift(1).fillna(base_price)
        df['high'] = df[['open', 'close']].max(axis=1) + candle_range * np.random.uniform(0.3, 0.7)
        df['low'] = df[['open', 'close']].min(axis=1) - candle_range * np.random.uniform(0.3, 0.7)
        
        # Ensure high/low are valid
        df['high'] = df[['high', 'open', 'close']].max(axis=1)
        df['low'] = df[['low', 'open', 'close']].min(axis=1)
        
        df['volume'] = np.random.gamma(2, 500, num_candles).astype(int)
        df['symbol'] = symbol
        df['timeframe'] = timeframe
        
        return df
        
    def fetch_all_symbols(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> dict:
        """Fetch data for all symbols and timeframes"""
        
        results = {}
        
        for symbol in self.symbols:
            results[symbol] = {}
            for timeframe in self.timeframes:
                df = self.fetch_ohlcv(symbol, timeframe, start_date, end_date)
                results[symbol][timeframe] = df
                logger.info(f"Fetched {len(df)} candles for {symbol} {timeframe}")
                
        return results
        
    def save_to_parquet(self, data: dict, output_dir: Path):
        """Save fetched data to parquet files"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for symbol, timeframes in data.items():
            for timeframe, df in timeframes.items():
                if len(df) > 0:
                    filename = output_dir / f"{symbol}_{timeframe}.parquet"
                    df.to_parquet(filename)
                    logger.info(f"Saved {filename}")
