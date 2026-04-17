"""Live WebSocket feed handler for real-time market data"""

import asyncio
import json
import logging
import threading
from datetime import datetime
from typing import Callable, Dict, List, Optional, Any
from dataclasses import dataclass, field
from queue import Queue
import websocket
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Tick:
    """Market tick data structure"""
    symbol: str
    bid: float
    ask: float
    timestamp: datetime
    volume: float = 0.0
    
    @property
    def spread(self) -> float:
        return self.ask - self.bid
    
    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2


@dataclass
class OHLCV:
    """OHLCV candle data structure"""
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: datetime
    timeframe: str = "15m"
    
    def to_dict(self) -> dict:
        return {
            'symbol': self.symbol,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'timestamp': self.timestamp,
            'timeframe': self.timeframe
        }


class WebSocketHandler:
    """Handles WebSocket connection to broker data feed"""
    
    def __init__(self, broker_type: str = "mt5"):
        self.broker_type = broker_type
        self.ws = None
        self.is_connected = False
        self.callbacks: List[Callable] = []
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.reconnect_delay = 5  # seconds
        
    def connect(self, url: str, api_key: Optional[str] = None):
        """Establish WebSocket connection"""
        try:
            self.ws = websocket.WebSocketApp(
                url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
                header={"Authorization": f"Bearer {api_key}"} if api_key else {}
            )
            
            # Run in separate thread
            wst = threading.Thread(target=self.ws.run_forever)
            wst.daemon = True
            wst.start()
            
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            self._schedule_reconnect()
            
    def _on_open(self, ws):
        """Called on successful connection"""
        self.is_connected = True
        self.reconnect_attempts = 0
        logger.info("WebSocket connected")
        
        # Subscribe to symbols
        self.subscribe(["XAUUSD", "EURUSD"])
        
    def _on_message(self, ws, message):
        """Handle incoming messages"""
        try:
            data = json.loads(message)
            tick = self._parse_tick(data)
            for callback in self.callbacks:
                callback(tick)
        except Exception as e:
            logger.error(f"Error parsing message: {e}")
            
    def _on_error(self, ws, error):
        """Handle WebSocket errors"""
        logger.error(f"WebSocket error: {error}")
        self.is_connected = False
        
    def _on_close(self, ws, close_status_code, close_msg):
        """Handle connection close"""
        logger.warning("WebSocket closed")
        self.is_connected = False
        self._schedule_reconnect()
        
    def _schedule_reconnect(self):
        """Schedule reconnection attempt"""
        if self.reconnect_attempts < self.max_reconnect_attempts:
            self.reconnect_attempts += 1
            delay = self.reconnect_delay * self.reconnect_attempts
            logger.info(f"Scheduling reconnect in {delay}s (attempt {self.reconnect_attempts})")
            threading.Timer(delay, self._reconnect).start()
            
    def _reconnect(self):
        """Attempt to reconnect"""
        logger.info("Attempting to reconnect...")
        self.connect(self.current_url)
        
    def subscribe(self, symbols: List[str]):
        """Subscribe to price feeds"""
        subscribe_msg = {
            "type": "subscribe",
            "symbols": symbols
        }
        if self.ws and self.is_connected:
            self.ws.send(json.dumps(subscribe_msg))
            
    def _parse_tick(self, data: dict) -> Tick:
        """Parse broker-specific tick data"""
        # Implementation depends on broker
        return Tick(
            symbol=data.get('symbol', 'XAUUSD'),
            bid=data.get('bid', 0),
            ask=data.get('ask', 0),
            timestamp=datetime.fromtimestamp(data.get('time', datetime.now().timestamp())),
            volume=data.get('volume', 0)
        )
        
    def add_callback(self, callback: Callable):
        """Add callback for tick updates"""
        self.callbacks.append(callback)
        
    def close(self):
        """Close WebSocket connection"""
        if self.ws:
            self.ws.close()
            self.is_connected = False


class LiveFeed:
    """Main live feed manager that aggregates ticks into OHLCV candles"""
    
    def __init__(self, symbol: str = "XAUUSD", timeframe: str = "15m"):
        self.symbol = symbol
        self.timeframe = timeframe
        self.timeframe_minutes = self._parse_timeframe(timeframe)
        self.current_candle: Optional[OHLCV] = None
        self.candle_callbacks: List[Callable] = []
        self.tick_buffer: List[Tick] = []
        self.last_candle_close = None
        
        # OHLCV accumulators
        self.candle_open = None
        self.candle_high = None
        self.candle_low = None
        self.candle_close = None
        self.candle_volume = 0.0
        self.candle_start_time = None
        
    def _parse_timeframe(self, timeframe: str) -> int:
        """Convert timeframe string to minutes"""
        mapping = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '4h': 240, '1d': 1440
        }
        return mapping.get(timeframe, 15)
        
    def process_tick(self, tick: Tick):
        """Process incoming tick and update current candle"""
        if tick.symbol != self.symbol:
            return
            
        self.tick_buffer.append(tick)
        
        # Calculate current candle time boundary
        current_minute = tick.timestamp.minute
        candle_minute = (current_minute // self.timeframe_minutes) * self.timeframe_minutes
        candle_time = tick.timestamp.replace(minute=candle_minute, second=0, microsecond=0)
        
        # Check if we need to close current candle
        if self.current_candle and candle_time > self.current_candle.timestamp:
            self._close_current_candle()
            self._start_new_candle(candle_time, tick)
        elif not self.current_candle:
            self._start_new_candle(candle_time, tick)
        else:
            # Update current candle
            self.candle_high = max(self.candle_high, tick.ask)
            self.candle_low = min(self.candle_low, tick.bid)
            self.candle_close = tick.mid
            self.candle_volume += tick.volume
            
    def _start_new_candle(self, timestamp: datetime, tick: Tick):
        """Initialize a new candle"""
        self.candle_start_time = timestamp
        self.candle_open = tick.mid
        self.candle_high = tick.ask
        self.candle_low = tick.bid
        self.candle_close = tick.mid
        self.candle_volume = tick.volume
        
    def _close_current_candle(self):
        """Finalize and emit current candle"""
        if self.candle_open is None:
            return
            
        self.current_candle = OHLCV(
            symbol=self.symbol,
            open=self.candle_open,
            high=self.candle_high,
            low=self.candle_low,
            close=self.candle_close,
            volume=self.candle_volume,
            timestamp=self.candle_start_time,
            timeframe=self.timeframe
        )
        
        # Notify callbacks
        for callback in self.candle_callbacks:
            callback(self.current_candle)
            
    def add_candle_callback(self, callback: Callable[[OHLCV], None]):
        """Register callback for completed candles"""
        self.candle_callbacks.append(callback)
        
    def get_current_candle(self) -> Optional[OHLCV]:
        """Get the currently forming candle"""
        if self.candle_open is None:
            return None
        return OHLCV(
            symbol=self.symbol,
            open=self.candle_open,
            high=self.candle_high,
            low=self.candle_low,
            close=self.candle_close,
            volume=self.candle_volume,
            timestamp=self.candle_start_time,
            timeframe=self.timeframe
        )
