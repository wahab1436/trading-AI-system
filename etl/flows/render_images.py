"""Prefect flow for rendering chart images from OHLCV data"""

from prefect import flow, task
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import logging
from concurrent.futures import ProcessPoolExecutor
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)


@task
def load_and_prepare_data(input_dir: str, label_dir: str) -> pd.DataFrame:
    """Load OHLCV data and corresponding labels"""
    
    input_path = Path(input_dir)
    label_path = Path(label_dir)
    
    all_data = []
    
    for file in input_path.glob("*_15m.parquet"):
        symbol = file.stem.split('_')[0]
        
        # Load OHLCV
        df = pd.read_parquet(file)
        
        # Load labels
        label_file = label_path / f"{symbol}_15m_labels.parquet"
        if label_file.exists():
            df_labels = pd.read_parquet(label_file)
            
            # Merge
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df_labels['timestamp'] = pd.to_datetime(df_labels['timestamp'])
            
            df = df.merge(df_labels[['timestamp', 'label_filtered']], on='timestamp', how='inner')
            df['symbol'] = symbol
            
            all_data.append(df)
            
    result = pd.concat(all_data, ignore_index=True)
    logger.info(f"Prepared {len(result)} rows with labels")
    
    return result


class ChartRenderer:
    """Renders candlestick charts from OHLCV data"""
    
    def __init__(self, width: int = 380, height: int = 380, candles: int = 50):
        self.width = width
        self.height = height
        self.candles = candles
        
        # Chart area dimensions
        self.chart_top = 40
        self.chart_bottom = height - 60
        self.chart_height = self.chart_bottom - self.chart_top
        
        # Volume area
        self.volume_top = self.chart_bottom + 10
        self.volume_height = 40
        
        # Colors
        self.bg_color = (0, 0, 0)  # Black
        self.bull_color = (0, 255, 0)  # Green
        self.bear_color = (255, 0, 0)  # Red
        self.ema_20_color = (0, 191, 255)  # DeepSkyBlue
        self.ema_50_color = (255, 140, 0)  # DarkOrange
        self.grid_color = (40, 40, 40)  # Dark gray
        
    def render(self, df_window: pd.DataFrame) -> Image.Image:
        """Render a single chart from a window of data"""
        
        img = Image.new('RGB', (self.width, self.height), self.bg_color)
        draw = ImageDraw.Draw(img)
        
        if len(df_window) < self.candles:
            return img
            
        # Calculate scaling
        price_min = df_window['low'].min()
        price_max = df_window['high'].max()
        price_range = price_max - price_min
        price_scale = self.chart_height / price_range if price_range > 0 else 1
        
        # Volume scaling
        volume_max = df_window['volume'].max() if 'volume' in df_window else 1
        volume_scale = self.volume_height / volume_max if volume_max > 0 else 1
        
        # Candle width
        candle_width = (self.width - 40) / self.candles
        
        # Draw grid
        self._draw_grid(draw, price_min, price_max, price_scale)
        
        # Draw candles
        for i, (idx, row) in enumerate(df_window.iterrows()):
            x = 20 + i * candle_width
            
            # Price coordinates
            open_y = self._price_to_y(row['open'], price_min, price_scale)
            close_y = self._price_to_y(row['close'], price_min, price_scale)
            high_y = self._price_to_y(row['high'], price_min, price_scale)
            low_y = self._price_to_y(row['low'], price_min, price_scale)
            
            # Determine color
            is_bull = row['close'] >= row['open']
            color = self.bull_color if is_bull else self.bear_color
            
            # Draw wick
            draw.line([(x + candle_width/2, high_y), (x + candle_width/2, low_y)], fill=color, width=1)
            
            # Draw body
            body_top = min(open_y, close_y)
            body_bottom = max(open_y, close_y)
            draw.rectangle(
                [(x, body_top), (x + candle_width - 1, body_bottom)],
                fill=color
            )
            
            # Draw volume
            if 'volume' in row:
                volume_height = row['volume'] * volume_scale
                vol_y = self.volume_top + self.volume_height - volume_height
                draw.rectangle(
                    [(x, vol_y), (x + candle_width - 1, self.volume_top + self.volume_height)],
                    fill=(100, 100, 100)
                )
                
        # Draw EMAs
        if len(df_window) >= 20:
            self._draw_ema(draw, df_window, 'ema_20', self.ema_20_color, candle_width, price_min, price_scale)
            
        if len(df_window) >= 50:
            self._draw_ema(draw, df_window, 'ema_50', self.ema_50_color, candle_width, price_min, price_scale)
            
        return img
        
    def _draw_grid(self, draw: ImageDraw, price_min: float, price_max: float, price_scale: float):
        """Draw price grid lines"""
        
        # Horizontal grid lines
        num_lines = 5
        for i in range(num_lines + 1):
            y = self.chart_top + (i / num_lines) * self.chart_height
            draw.line([(10, y), (self.width - 10, y)], fill=self.grid_color, width=1)
            
        # Vertical grid lines
        num_vlines = 10
        for i in range(num_vlines + 1):
            x = 20 + (i / num_vlines) * (self.width - 40)
            draw.line([(x, self.chart_top), (x, self.chart_bottom)], fill=self.grid_color, width=1)
            
    def _draw_ema(self, draw: ImageDraw, df: pd.DataFrame, col: str, color: tuple,
                  candle_width: float, price_min: float, price_scale: float):
        """Draw EMA line"""
        
        points = []
        for i, (idx, row) in enumerate(df.iterrows()):
            if col in row and not pd.isna(row[col]):
                x = 20 + i * candle_width
                y = self._price_to_y(row[col], price_min, price_scale)
                points.append((x, y))
                
        if len(points) > 1:
            draw.line(points, fill=color, width=2)
            
    def _price_to_y(self, price: float, price_min: float, price_scale: float) -> float:
        """Convert price to y coordinate"""
        return self.chart_bottom - (price - price_min) * price_scale


@task
def render_images_task(
    df: pd.DataFrame,
    output_dir: str,
    candles_per_image: int = 50
):
    """Render images for each window of data"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    renderer = ChartRenderer(width=380, height=380, candles=candles_per_image)
    
    images_rendered = 0
    
    # Group by symbol
    for symbol, symbol_df in df.groupby('symbol'):
        symbol_df = symbol_df.sort_values('timestamp').reset_index(drop=True)
        
        # Create sliding windows
        for start_idx in range(len(symbol_df) - candles_per_image):
            end_idx = start_idx + candles_per_image
            window = symbol_df.iloc[start_idx:end_idx].copy()
            
            # Calculate EMAs
            window['ema_20'] = window['close'].ewm(span=20, adjust=False).mean()
            window['ema_50'] = window['close'].ewm(span=50, adjust=False).mean()
            
            # Get label for the prediction point (last candle's label or future)
            label = window.iloc[-1].get('label_filtered', 0)
            
            # Render image
            img = renderer.render(window)
            
            # Save with timestamp
            timestamp = window['timestamp'].iloc[-1].strftime("%Y%m%d_%H%M%S")
            filename = output_path / f"{symbol}_{timestamp}_label_{label}.png"
            img.save(filename)
            
            images_rendered += 1
            
            if images_rendered % 1000 == 0:
                logger.info(f"Rendered {images_rendered} images")
                
    logger.info(f"Total images rendered: {images_rendered}")
    return images_rendered


@flow(name="render_images")
def render_images_flow(
    input_dir: str = "data/raw/",
    label_dir: str = "data/labels/",
    output_dir: str = "data/images/"
):
    """Main flow for rendering chart images"""
    
    # Load data with labels
    df = load_and_prepare_data(input_dir, label_dir)
    
    # Split into train/val/test
    timestamps = df['timestamp'].sort_values()
    train_cutoff = timestamps.quantile(0.7)
    val_cutoff = timestamps.quantile(0.85)
    
    train_df = df[df['timestamp'] <= train_cutoff]
    val_df = df[(df['timestamp'] > train_cutoff) & (df['timestamp'] <= val_cutoff)]
    test_df = df[df['timestamp'] > val_cutoff]
    
    # Render images for each split
    splits = {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }
    
    for split_name, split_df in splits.items():
        split_dir = Path(output_dir) / split_name
        logger.info(f"Rendering {split_name} set with {len(split_df)} rows")
        count = render_images_task(split_df, str(split_dir))
        logger.info(f"Rendered {count} images for {split_name}")
        
    return output_dir


if __name__ == "__main__":
    render_images_flow()
