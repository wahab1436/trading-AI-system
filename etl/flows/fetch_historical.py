"""Prefect flow for historical data fetching"""

from prefect import flow, task
from prefect.task_runners import ConcurrentTaskRunner
from datetime import datetime, timedelta
from pathlib import Path
import logging
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from data_engine.historical_fetch import HistoricalDataFetcher
from data_engine.validator import DataValidator

logger = logging.getLogger(__name__)


@task
def fetch_data_task(
    symbol: str,
    timeframe: str,
    start_date: datetime,
    end_date: datetime
):
    """Task to fetch data for single symbol/timeframe"""
    
    fetcher = HistoricalDataFetcher(broker_type="mt5")
    df = fetcher.fetch_ohlcv(symbol, timeframe, start_date, end_date)
    
    logger.info(f"Fetched {len(df)} candles for {symbol} {timeframe}")
    return df


@task
def validate_data_task(df, symbol: str, timeframe: str):
    """Task to validate fetched data"""
    
    validator = DataValidator()
    result = validator.validate_and_clean(df)
    
    if not result.is_valid:
        logger.error(f"Validation failed for {symbol} {timeframe}: {result.errors}")
        raise ValueError(f"Data validation failed: {result.errors}")
        
    logger.info(f"Validation passed for {symbol} {timeframe}. Fixed: {result.fixed_count} issues")
    return df


@task
def save_data_task(df, symbol: str, timeframe: str, output_dir: Path):
    """Task to save validated data"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = output_dir / f"{symbol}_{timeframe}.parquet"
    df.to_parquet(filename)
    
    logger.info(f"Saved {filename}")
    return str(filename)


@flow(name="fetch_historical_data", task_runner=ConcurrentTaskRunner())
def fetch_historical_flow(
    symbols: list = None,
    timeframes: list = None,
    years_back: int = 5,
    output_dir: str = "data/raw/"
):
    """Main flow for fetching historical OHLCV data"""
    
    if symbols is None:
        symbols = ["XAUUSD", "EURUSD"]
        
    if timeframes is None:
        timeframes = ["15m", "1h", "4h"]
        
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years_back * 365)
    
    output_path = Path(output_dir)
    
    # Fetch all combinations
    futures = []
    for symbol in symbols:
        for timeframe in timeframes:
            future = fetch_data_task.submit(
                symbol, timeframe, start_date, end_date
            )
            futures.append((symbol, timeframe, future))
            
    # Validate and save
    for symbol, timeframe, future in futures:
        df = future.result()
        
        if len(df) > 0:
            df = validate_data_task.submit(df, symbol, timeframe).result()
            save_data_task.submit(df, symbol, timeframe, output_path)
            
    logger.info("Historical data fetch complete!")


if __name__ == "__main__":
    fetch_historical_flow()
