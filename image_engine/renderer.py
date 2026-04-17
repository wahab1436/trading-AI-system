"""Chart rendering engine for converting OHLCV to images"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class CandlestickRenderer:
    """Professional candlestick chart renderer with consistent styling"""
    
    def __init__(
        self,
        width: int = 380,
        height: int = 380,
        candles_per_chart: int = 50,
        background_color: str = "#000000",
        bull_color: str = "#00FF00",
        bear_color: str = "#FF0000",
        ema_20_color: str = "#00BFFF",
        ema_50_color: str = "#FF8C00"
    ):
        self.width = width
        self.height = height
        self.candles = candles_per_chart
        
        # Parse colors
        self.bg_color = self._hex_to_rgb(background_color)
        self.bull_color = self._hex_to_rgb(bull_color)
        self.bear_color = self._hex_to_rgb(bear_color)
        self.ema_20_color = self._hex_to_rgb(ema_20_color)
        self.ema_50_color = self._hex_to_rgb(ema_50_color)
        
        # Chart area
        self.chart_top = 40
        self.chart_bottom = height - 60
        self.chart_height = self.chart_bottom - self.chart_top
        self.chart_left = 20
        self.chart_right = width - 20
        self.chart_width = self.chart_right - self.chart_left
        
        # Volume area
        self.volume_top = self.chart_bottom + 10
        self.volume_height = 40
        
        # Grid
        self.grid_color = (40, 40, 40)
        
    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB tuple"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
    def render(self, df: pd.DataFrame) -> Image.Image:
        """Render a candlestick chart from OHLCV data"""
        
        img = Image.new('RGB', (self.width, self.height), self.bg_color)
        draw = ImageDraw.Draw(img)
        
        if len(df) < self.candles:
            logger.warning(f"Data has only {len(df)} candles, need {self.candles}")
            return img
            
        # Use last N candles
        df = df.tail(self.candles).copy()
        
        # Calculate scaling
        price_min = df['low'].min()
        price_max = df['high'].max()
        price_padding = (price_max - price_min) * 0.05
        price_min -= price_padding
        price_max += price_padding
        price_range = price_max - price_min
        
        if price_range > 0:
            price_scale = self.chart_height / price_range
        else:
            price_scale = 1
            
        # Volume scaling
        if 'volume' in df.columns:
            volume_max = df['volume'].max()
            volume_scale = self.volume_height / volume_max if volume_max > 0 else 1
        else:
            volume_scale = 0
            
        # Candle width
        candle_width = self.chart_width / self.candles
        candle_spacing = candle_width * 0.2
        candle_body_width = candle_width - candle_spacing
        
        # Draw grid
        self._draw_grid(draw, price_min, price_max)
        
        # Draw candles
        for i, (idx, row) in enumerate(df.iterrows()):
            x = self.chart_left + i * candle_width + candle_spacing / 2
            
            # Price coordinates
            open_y = self._price_to_y(row['open'], price_min, price_scale)
            close_y = self._price_to_y(row['close'], price_min, price_scale)
            high_y = self._price_to_y(row['high'], price_min, price_scale)
            low_y = self._price_to_y(row['low'], price_min, price_scale)
            
            # Determine color
            is_bull = row['close'] >= row['open']
            color = self.bull_color if is_bull else self.bear_color
            
            # Draw wick
            draw.line(
                [(x + candle_body_width/2, high_y), (x + candle_body_width/2, low_y)],
                fill=color, width=1
            )
            
            # Draw body
            body_top = min(open_y, close_y)
            body_bottom = max(open_y, close_y)
            
            if body_bottom - body_top < 1:
                body_bottom = body_top + 1
                
            draw.rectangle(
                [(x, body_top), (x + candle_body_width - 1, body_bottom)],
                fill=color
            )
            
            # Draw volume
            if 'volume' in df.columns and volume_scale > 0:
                volume_height = row['volume'] * volume_scale
                vol_y = self.volume_top + self.volume_height - volume_height
                draw.rectangle(
                    [(x, vol_y), (x + candle_body_width - 1, self.volume_top + self.volume_height)],
                    fill=(80, 80, 80)
                )
                
        # Draw EMAs
        if 'close' in df.columns:
            df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
            df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
            self._draw_line(draw, df, 'ema_20', self.ema_20_color, candle_width, price_min, price_scale)
            self._draw_line(draw, df, 'ema_50', self.ema_50_color, candle_width, price_min, price_scale)
            
        return img
        
    def _draw_grid(self, draw: ImageDraw, price_min: float, price_max: float):
        """Draw grid lines"""
        
        # Horizontal grid lines
        num_lines = 6
        for i in range(num_lines + 1):
            y = self.chart_top + (i / num_lines) * self.chart_height
            draw.line([(self.chart_left - 5, y), (self.chart_right + 5, y)], fill=self.grid_color, width=1)
            
            # Price labels
            price = price_min + (i / num_lines) * (price_max - price_min)
            price_label = f"{price:.2f}"
            draw.text((self.chart_left - 35, y - 5), price_label, fill=(100, 100, 100))
            
        # Vertical grid lines
        num_vlines = 10
        for i in range(num_vlines + 1):
            x = self.chart_left + (i / num_vlines) * self.chart_width
            draw.line([(x, self.chart_top), (x, self.chart_bottom)], fill=self.grid_color, width=1)
            
    def _draw_line(self, draw: ImageDraw, df: pd.DataFrame, col: str, color: Tuple[int, int, int],
                   candle_width: float, price_min: float, price_scale: float):
        """Draw a line (EMA) on the chart"""
        
        points = []
        for i, (idx, row) in enumerate(df.iterrows()):
            if col in row and not pd.isna(row[col]):
                x = self.chart_left + i * candle_width + candle_width / 2
                y = self._price_to_y(row[col], price_min, price_scale)
                points.append((x, y))
                
        if len(points) > 1:
            draw.line(points, fill=color, width=2)
            
    def _price_to_y(self, price: float, price_min: float, price_scale: float) -> float:
        """Convert price to y coordinate"""
        return self.chart_bottom - (price - price_min) * price_scale
        
    def render_batch(self, dfs: list) -> list:
        """Render multiple charts in batch"""
        images = []
        for df in dfs:
            images.append(self.render(df))
        return images


class ImageValidator:
    """Validates that rendered images meet specifications"""
    
    def __init__(self, expected_width: int = 380, expected_height: int = 380):
        self.expected_width = expected_width
        self.expected_height = expected_height
        
    def validate(self, img: Image.Image) -> Tuple[bool, str]:
        """Validate image dimensions and format"""
        
        if img.width != self.expected_width:
            return False, f"Wrong width: {img.width} != {self.expected_width}"
            
        if img.height != self.expected_height:
            return False, f"Wrong height: {img.height} != {self.expected_height}"
            
        if img.mode != 'RGB':
            return False, f"Wrong mode: {img.mode} != RGB"
            
        return True, "Valid"
        
    def compute_hash(self, img: Image.Image) -> str:
        """Compute perceptual hash for image"""
        import hashlib
        return hashlib.md5(img.tobytes()).hexdigest()
