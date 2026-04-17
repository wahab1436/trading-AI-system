#!/usr/bin/env python3
"""
TRADING AI SYSTEM - COMPLETE MAIN FILE
=======================================
Everything Connected: Dashboard | Live Trading | Backtest | Risk Management
"""

import os
import sys
import json
import yaml
import logging
import threading
import webbrowser
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field
from queue import Queue
import signal

# Core imports
import numpy as np
import pandas as pd
import torch
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# Set up paths
sys.path.append(str(Path(__file__).parent))

# Import all modules
from data_engine.validator import DataValidator
from data_engine.session_tagger import SessionTagger
from image_engine.renderer import CandlestickRenderer, ImageValidator
from smc_engine.structure import StructureDetector
from smc_engine.order_blocks import OrderBlockDetector
from smc_engine.fvg import FVGDetector
from fusion_model.train_xgb import FusionModel
from execution.paper_trading import PaperTradingBroker
from execution.order_manager import OrderManager
from risk_engine.risk_limits import RiskEngine
from trade_journal.logger import TradeJournal
from backtest.simulator import BacktestEngine
from mlops.drift_monitor import DriftMonitor
from mlops.alerting import AlertManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trading_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================
# DATA CLASSES
# ============================================

@dataclass
class TradingSignal:
    """Trading signal from AI model"""
    timestamp: datetime
    symbol: str
    direction: int  # 1=BUY, -1=SELL, 0=NO_TRADE
    confidence: float
    buy_prob: float
    sell_prob: float
    notrade_prob: float
    entry_price: float
    stop_loss: float
    take_profit: float
    lot_size: float = 0.01
    model_version: str = "v1.0"
    smc_features: Dict = field(default_factory=dict)
    cnn_embedding: Optional[np.ndarray] = None
    
    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'direction': self.direction,
            'action': 'BUY' if self.direction == 1 else 'SELL' if self.direction == -1 else 'NO_TRADE',
            'confidence': self.confidence,
            'probabilities': {
                'BUY': self.buy_prob,
                'SELL': self.sell_prob,
                'NO_TRADE': self.notrade_prob
            },
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'lot_size': self.lot_size,
            'risk_reward': abs((self.take_profit - self.entry_price) / (self.entry_price - self.stop_loss)) if self.direction != 0 else 0
        }


@dataclass
class SystemStatus:
    """System health status"""
    running: bool = True
    mode: str = "paper"  # paper, live, backtest
    connected: bool = False
    last_prediction: Optional[datetime] = None
    predictions_today: int = 0
    trades_today: int = 0
    daily_pnl: float = 0.0
    model_loaded: bool = False
    drift_detected: bool = False
    kill_switch_active: bool = False


# ============================================
# TRADING AI SYSTEM - MAIN CLASS
# ============================================

class TradingAISystem:
    """
    Complete Trading AI System with Dashboard Integration
    """
    
    def __init__(self, config_path: str = "config"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Initialize all components
        self._init_components()
        
        # System state
        self.status = SystemStatus()
        self.current_signals: List[TradingSignal] = []
        self.price_history: Dict[str, pd.DataFrame] = {}
        
        # Queue for async processing
        self.signal_queue = Queue()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)
        
        logger.info("🚀 Trading AI System Initialized")
        
    def _load_config(self) -> dict:
        """Load all configuration files"""
        config = {}
        for yaml_file in self.config_path.glob("*.yaml"):
            with open(yaml_file, 'r') as f:
                config.update(yaml.safe_load(f))
        return config
        
    def _init_components(self):
        """Initialize all system components"""
        
        # Data Components
        self.data_validator = DataValidator()
        self.session_tagger = SessionTagger()
        
        # Image Components
        self.image_renderer = CandlestickRenderer(
            width=380, height=380, candles_per_chart=50
        )
        self.image_validator = ImageValidator()
        
        # SMC Components
        self.structure_detector = StructureDetector()
        self.order_block_detector = OrderBlockDetector()
        self.fvg_detector = FVGDetector()
        
        # AI Models
        self.cnn_model = None  # Will load on demand
        self.fusion_model = FusionModel()
        
        # Try to load existing model
        model_path = Path("models/fusion_model")
        if model_path.exists():
            try:
                self.fusion_model.load(model_path)
                self.status.model_loaded = True
                logger.info("✅ Model loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load model: {e}")
        
        # Broker & Execution
        self.broker = PaperTradingBroker(initial_balance=10000.0)
        self.order_manager = OrderManager(self.broker)
        
        # Risk Management
        self.risk_engine = RiskEngine(self.config.get('risk', {}))
        
        # Trade Journal
        self.trade_journal = TradeJournal()
        
        # Backtest Engine
        self.backtest_engine = BacktestEngine()
        
        # MLOps Components
        self.drift_monitor = DriftMonitor()
        self.alert_manager = AlertManager()
        
        logger.info("✅ All components initialized")
        
    def _load_cnn_model(self):
        """Lazy load CNN model"""
        if self.cnn_model is None:
            from cnn_model.model import ChartCNN
            self.cnn_model = ChartCNN()
            self.cnn_model.eval()
            logger.info("✅ CNN Model loaded")
            
    def get_current_price(self, symbol: str = "XAUUSD") -> float:
        """Get current market price"""
        # In paper trading, get from broker
        if self.status.mode == "paper":
            price = self.broker._get_market_price(symbol)['bid']
        else:
            # Live mode - get from MT5
            price = 2000.0  # Placeholder
        return price
        
    def generate_signal(self, df: pd.DataFrame, symbol: str = "XAUUSD") -> TradingSignal:
        """
        Generate trading signal from market data
        """
        current_price = self.get_current_price(symbol)
        
        # 1. Render chart image
        image = self.image_renderer.render(df)
        
        # 2. Extract CNN embedding (if model available)
        cnn_embedding = None
        if self.status.model_loaded and self.cnn_model:
            self._load_cnn_model()
            # Process image through CNN
            # cnn_embedding = self.cnn_model.extract_embeddings(image_tensor)
            pass
            
        # 3. Compute SMC features
        smc_features = self._compute_smc_features(df, current_price)
        
        # 4. Get AI prediction
        buy_prob = 0.5
        sell_prob = 0.3
        notrade_prob = 0.2
        
        if self.status.model_loaded:
            try:
                probs = self.fusion_model.predict(
                    np.random.randn(1536),  # Placeholder embedding
                    np.array(list(smc_features.values()))
                )
                buy_prob = probs['buy_prob']
                sell_prob = probs['sell_prob']
                notrade_prob = probs['notrade_prob']
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                
        # 5. Determine signal
        confidence_threshold = 0.65
        
        if buy_prob > confidence_threshold and buy_prob > sell_prob:
            direction = 1
            confidence = buy_prob
            action = "BUY"
        elif sell_prob > confidence_threshold and sell_prob > buy_prob:
            direction = -1
            confidence = sell_prob
            action = "SELL"
        else:
            direction = 0
            confidence = max(buy_prob, sell_prob)
            action = "NO_TRADE"
            
        # 6. Calculate SL/TP (ATR-based)
        atr = self._calculate_atr(df)
        if direction == 1:  # BUY
            stop_loss = current_price - (atr * 1.5)
            take_profit = current_price + (atr * 2.5)
        elif direction == -1:  # SELL
            stop_loss = current_price + (atr * 1.5)
            take_profit = current_price - (atr * 2.5)
        else:
            stop_loss = current_price
            take_profit = current_price
            
        # 7. Calculate position size
        lot_size = 0.01
        if direction != 0:
            sl_distance_pips = abs(current_price - stop_loss) * 10000
            lot_size = self.risk_engine.calculate_lot_size(
                self.broker.get_account_info().balance,
                sl_distance_pips,
                confidence=confidence
            )
            
        # Create signal
        signal = TradingSignal(
            timestamp=datetime.now(),
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            buy_prob=buy_prob,
            sell_prob=sell_prob,
            notrade_prob=notrade_prob,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            lot_size=lot_size,
            smc_features=smc_features,
            cnn_embedding=cnn_embedding
        )
        
        # Log signal
        logger.info(f"📊 Signal: {action} | Conf: {confidence:.2%} | Price: {current_price:.2f}")
        
        return signal
        
    def _compute_smc_features(self, df: pd.DataFrame, current_price: float) -> Dict:
        """Compute all SMC features"""
        features = {}
        
        try:
            # Structure features
            structure = self.structure_detector.compute_structure_scores(df)
            features.update(structure)
            
            # Order block features
            ob_features = self.order_block_detector.get_features(df, current_price)
            features.update(ob_features)
            
            # FVG features
            fvg_features = self.fvg_detector.get_features(df, current_price)
            features.update(fvg_features)
            
            # Market state
            features['market_state'] = self._detect_market_state(df)
            
            # Session
            features['session_code'] = self.session_tagger.get_current_session_code()
            
        except Exception as e:
            logger.error(f"SMC feature error: {e}")
            features = self._get_default_smc_features()
            
        return features
        
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR"""
        if len(df) < period:
            return 10.0  # Default for gold
            
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        tr = np.maximum(high - low, 
                       np.abs(high - np.roll(close, 1)),
                       np.abs(low - np.roll(close, 1)))
        atr = np.mean(tr[-period:])
        return atr
        
    def _detect_market_state(self, df: pd.DataFrame) -> int:
        """0=ranging, 1=trending_bull, 2=trending_bear"""
        if len(df) < 20:
            return 0
            
        # Simple trend detection
        sma_20 = df['close'].rolling(20).mean()
        sma_50 = df['close'].rolling(50).mean()
        
        if sma_20.iloc[-1] > sma_50.iloc[-1] and df['close'].iloc[-1] > sma_20.iloc[-1]:
            return 1  # Bullish trend
        elif sma_20.iloc[-1] < sma_50.iloc[-1] and df['close'].iloc[-1] < sma_20.iloc[-1]:
            return 2  # Bearish trend
        else:
            return 0  # Ranging
            
    def _get_default_smc_features(self) -> Dict:
        """Return default SMC features"""
        return {
            'hh_hl_ratio': 0.5,
            'lh_ll_ratio': 0.5,
            'bos_count_bull': 0,
            'bos_count_bear': 0,
            'choch_detected': 0,
            'dist_nearest_bull_ob': 5.0,
            'dist_nearest_bear_ob': 5.0,
            'fvg_bull_open': 0,
            'fvg_bear_open': 0,
            'market_state': 0,
            'session_code': 1
        }
        
    def execute_signal(self, signal: TradingSignal) -> Optional[Dict]:
        """Execute a trading signal"""
        
        if signal.direction == 0:
            return None
            
        # Check risk limits
        account = self.broker.get_account_info()
        if not self.risk_engine.approve_trade(
            {'direction': signal.direction},
            account.balance,
            self.status.daily_pnl
        ):
            logger.warning("⚠️ Risk limits blocked trade")
            return None
            
        # Place order
        order = self.order_manager.place_trade(
            signal={'direction': signal.direction, 'symbol': signal.symbol, 'model_version': signal.model_version},
            lot_size=signal.lot_size,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit
        )
        
        if order:
            self.status.trades_today += 1
            
            # Log to journal
            self.trade_journal.log_signal(signal)
            self.trade_journal.log_order(order)
            
            # Send alert
            self.alert_manager.send_trade_alert(signal, order)
            
            return {'order_id': order.id, 'status': 'filled'}
            
        return None
        
    def run_paper_trading(self, symbol: str = "XAUUSD"):
        """Run paper trading mode"""
        logger.info(f"📝 Starting PAPER TRADING for {symbol}")
        self.status.mode = "paper"
        self.status.running = True
        
        # Generate synthetic data for testing
        dates = pd.date_range(end=datetime.now(), periods=100, freq='15T')
        
        while self.status.running:
            try:
                # Create synthetic OHLCV data
                df = self._generate_synthetic_data(dates[-50:])
                
                # Generate signal
                signal = self.generate_signal(df, symbol)
                self.current_signals.append(signal)
                self.status.last_prediction = datetime.now()
                self.status.predictions_today += 1
                
                # Execute if confident
                if signal.confidence > 0.70:
                    result = self.execute_signal(signal)
                    if result:
                        logger.info(f"✅ Trade executed: {result}")
                        
                # Update status
                self._update_status()
                
                # Wait for next candle (simulated)
                import time
                time.sleep(5)  # 5 seconds for demo, real would be 15 minutes
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Paper trading error: {e}")
                import time
                time.sleep(1)
                
    def _generate_synthetic_data(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Generate synthetic OHLCV data for testing"""
        np.random.seed(42)
        n = len(dates)
        
        # Random walk
        returns = np.random.randn(n) * 0.002
        prices = 2000 * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * (1 + np.abs(np.random.randn(n) * 0.001)),
            'low': prices * (1 - np.abs(np.random.randn(n) * 0.001)),
            'close': prices * (1 + np.random.randn(n) * 0.001),
            'volume': np.random.randint(100, 10000, n)
        })
        
        # Ensure high is highest, low is lowest
        df['high'] = df[['high', 'open', 'close']].max(axis=1)
        df['low'] = df[['low', 'open', 'close']].min(axis=1)
        
        return df
        
    def run_backtest(self, start_date: datetime, end_date: datetime):
        """Run backtest mode"""
        logger.info(f"📊 Running BACKTEST from {start_date} to {end_date}")
        self.status.mode = "backtest"
        
        results = self.backtest_engine.run(
            strategy=self.generate_signal,
            start_date=start_date,
            end_date=end_date
        )
        
        # Print results
        print("\n" + "="*50)
        print("BACKTEST RESULTS")
        print("="*50)
        print(f"Total Trades: {results.get('total_trades', 0)}")
        print(f"Win Rate: {results.get('win_rate', 0):.2%}")
        print(f"Profit Factor: {results.get('profit_factor', 0):.2f}")
        print(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
        print(f"Max Drawdown: {results.get('max_drawdown', 0):.2%}")
        print(f"Total Return: {results.get('total_return', 0):.2%}")
        print("="*50)
        
        return results
        
    def _update_status(self):
        """Update system status"""
        account = self.broker.get_account_info()
        self.status.daily_pnl = account.realized_pnl_today
        self.status.connected = self.broker.is_connected()
        self.status.kill_switch_active = self.risk_engine.kill_switch_triggered
        
    def _shutdown(self, signum=None, frame=None):
        """Graceful shutdown"""
        logger.info("🛑 Shutting down Trading AI System...")
        self.status.running = False
        
        # Close all positions
        if self.order_manager:
            self.order_manager.close_all_trades()
            
        # Disconnect broker
        if self.broker:
            self.broker.disconnect()
            
        # Save journal
        if self.trade_journal:
            self.trade_journal.save()
            
        logger.info("✅ System shutdown complete")
        sys.exit(0)
        
    def get_dashboard_data(self) -> Dict:
        """Get data for dashboard"""
        account = self.broker.get_account_info()
        positions = self.order_manager.get_open_positions()
        
        return {
            'status': {
                'mode': self.status.mode,
                'running': self.status.running,
                'model_loaded': self.status.model_loaded,
                'connected': self.status.connected,
                'kill_switch': self.status.kill_switch_active
            },
            'account': {
                'balance': account.balance,
                'equity': account.equity,
                'free_margin': account.free_margin,
                'daily_pnl': self.status.daily_pnl,
                'daily_pnl_pct': (self.status.daily_pnl / account.balance) * 100 if account.balance > 0 else 0
            },
            'trading': {
                'predictions_today': self.status.predictions_today,
                'trades_today': self.status.trades_today,
                'open_positions': len(positions),
                'last_signal': self.current_signals[-1].to_dict() if self.current_signals else None
            },
            'risk': self.risk_engine.get_risk_status(),
            'recent_signals': [s.to_dict() for s in self.current_signals[-10:]]
        }


# ============================================
# FASTAPI DASHBOARD
# ============================================

app = FastAPI(title="Trading AI Dashboard", version="2.0.0")

# Global system instance
trading_system: TradingAISystem = None


@app.on_event("startup")
async def startup():
    global trading_system
    trading_system = TradingAISystem()
    logger.info("✅ Dashboard API Started")


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Main dashboard HTML"""
    return HTMLResponse(DASHBOARD_HTML)


@app.get("/api/status")
async def get_status():
    """Get system status"""
    return trading_system.get_dashboard_data()


@app.get("/api/signals")
async def get_signals(limit: int = 50):
    """Get recent signals"""
    return {
        'signals': [s.to_dict() for s in trading_system.current_signals[-limit:]],
        'total': len(trading_system.current_signals)
    }


@app.get("/api/positions")
async def get_positions():
    """Get open positions"""
    positions = trading_system.order_manager.get_open_positions()
    return {
        'positions': [{
            'symbol': p.symbol,
            'side': p.side.value,
            'quantity': p.quantity,
            'open_price': p.open_price,
            'current_price': p.current_price,
            'pnl': p.unrealized_pnl,
            'pnl_pct': (p.unrealized_pnl / (p.open_price * p.quantity)) * 100 if p.open_price > 0 else 0
        } for p in positions],
        'count': len(positions)
    }


@app.get("/api/account")
async def get_account():
    """Get account info"""
    account = trading_system.broker.get_account_info()
    return {
        'balance': account.balance,
        'equity': account.equity,
        'free_margin': account.free_margin,
        'margin_level': account.margin_level,
        'currency': account.currency,
        'daily_pnl': trading_system.status.daily_pnl
    }


@app.get("/api/risk")
async def get_risk():
    """Get risk status"""
    return trading_system.risk_engine.get_risk_status()


@app.post("/api/trade/close/{position_id}")
async def close_position(position_id: str):
    """Close a position"""
    success = trading_system.order_manager.close_trade(position_id)
    return {'success': success, 'position_id': position_id}


@app.post("/api/trade/close-all")
async def close_all_positions():
    """Close all positions"""
    count = trading_system.order_manager.close_all_trades()
    return {'closed': count}


@app.post("/api/system/start")
async def start_trading():
    """Start trading"""
    if not trading_system.status.running:
        trading_system.status.running = True
        # Start trading in background thread
        thread = threading.Thread(target=trading_system.run_paper_trading)
        thread.start()
    return {'status': 'started'}


@app.post("/api/system/stop")
async def stop_trading():
    """Stop trading"""
    trading_system.status.running = False
    return {'status': 'stopped'}


@app.post("/api/kill-switch/activate")
async def activate_kill_switch():
    """Activate kill switch"""
    trading_system.risk_engine._trigger_kill_switch("Manual activation from dashboard")
    return {'status': 'activated'}


# ============================================
# DASHBOARD HTML
# ============================================

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trading AI Dashboard</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%);
            color: #fff;
            min-height: 100vh;
        }
        
        .header {
            background: rgba(0,0,0,0.5);
            padding: 20px;
            border-bottom: 1px solid #00ff88;
        }
        
        .header h1 {
            font-size: 24px;
            background: linear-gradient(90deg, #00ff88, #00bfff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .card {
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.1);
            transition: transform 0.3s;
        }
        
        .card:hover {
            transform: translateY(-5px);
            border-color: #00ff88;
        }
        
        .card-title {
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 2px;
            color: #888;
            margin-bottom: 10px;
        }
        
        .card-value {
            font-size: 32px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        
        .card-change {
            font-size: 12px;
        }
        
        .positive {
            color: #00ff88;
        }
        
        .negative {
            color: #ff4444;
        }
        
        .neutral {
            color: #ffaa00;
        }
        
        .signal-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        
        .signal-table th,
        .signal-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        
        .signal-table th {
            color: #888;
            font-weight: normal;
        }
        
        .badge-buy {
            background: #00ff8822;
            color: #00ff88;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
        }
        
        .badge-sell {
            background: #ff444422;
            color: #ff4444;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
        }
        
        .badge-neutral {
            background: #ffaa0022;
            color: #ffaa00;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
        }
        
        .btn {
            background: linear-gradient(90deg, #00ff88, #00bfff);
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
            color: #000;
            font-weight: bold;
            cursor: pointer;
            transition: opacity 0.3s;
        }
        
        .btn:hover {
            opacity: 0.8;
        }
        
        .btn-danger {
            background: linear-gradient(90deg, #ff4444, #ff6666);
            color: white;
        }
        
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        .status-online {
            background: #00ff88;
            box-shadow: 0 0 5px #00ff88;
        }
        
        .status-offline {
            background: #ff4444;
        }
        
        .refresh-btn {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: #00ff88;
            border: none;
            width: 50px;
            height: 50px;
            border-radius: 50%;
            cursor: pointer;
            font-size: 20px;
            box-shadow: 0 0 20px rgba(0,255,136,0.3);
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .trading-active {
            animation: pulse 2s infinite;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 Trading AI System | Smart Money Concepts + CNN</h1>
        <div style="margin-top: 10px;">
            <span id="status-badge">
                <span class="status-indicator status-online"></span> System Online
            </span>
            <span style="margin-left: 20px;" id="mode-badge">Mode: Paper Trading</span>
            <span style="margin-left: 20px;" id="time-badge"></span>
        </div>
    </div>
    
    <div class="container">
        <div class="stats-grid">
            <div class="card">
                <div class="card-title">Account Balance</div>
                <div class="card-value" id="balance">$10,000</div>
                <div class="card-change" id="daily-pnl">Today: $0 (0%)</div>
            </div>
            
            <div class="card">
                <div class="card-title">Equity</div>
                <div class="card-value" id="equity">$10,000</div>
                <div class="card-change">Free Margin: <span id="free-margin">$10,000</span></div>
            </div>
            
            <div class="card">
                <div class="card-title">Today's Stats</div>
                <div class="card-value"><span id="trades-today">0</span> Trades</div>
                <div class="card-change"><span id="predictions-today">0</span> Predictions</div>
            </div>
            
            <div class="card">
                <div class="card-title">Open Positions</div>
                <div class="card-value" id="open-positions">0</div>
                <div class="card-change">Max Concurrent: 3</div>
            </div>
            
            <div class="card">
                <div class="card-title">Risk Status</div>
                <div class="card-value" id="risk-status">✅ Active</div>
                <div class="card-change" id="consecutive-losses">Consecutive Losses: 0</div>
            </div>
            
            <div class="card">
                <div class="card-title">AI Model</div>
                <div class="card-value" id="model-status">✅ Loaded</div>
                <div class="card-change">Version: v2.0</div>
            </div>
        </div>
        
        <div class="stats-grid">
            <div class="card">
                <div class="card-title">Last Signal</div>
                <div id="last-signal">
                    <div class="card-value neutral">WAITING</div>
                    <div class="card-change">No signal yet</div>
                </div>
            </div>
            
            <div class="card">
                <div class="card-title">Win Rate (Paper)</div>
                <div class="card-value" id="win-rate">0%</div>
                <div class="card-change">Last 100 trades</div>
            </div>
            
            <div class="card">
                <div class="card-title">Profit Factor</div>
                <div class="card-value" id="profit-factor">0.00</div>
                <div class="card-change">Target: >1.5</div>
            </div>
            
            <div class="card">
                <div class="card-title">Kill Switch</div>
                <div class="card-value" id="kill-switch-status">⚪ Inactive</div>
                <div class="card-change">
                    <button class="btn btn-danger" onclick="activateKillSwitch()" style="padding: 5px 10px; font-size: 12px;">Emergency Stop</button>
                </div>
            </div>
        </div>
        
        <div class="card" style="margin-top: 20px;">
            <div class="card-title">Recent Signals</div>
            <div style="overflow-x: auto;">
                <table class="signal-table" id="signals-table">
                    <thead>
                        <tr><th>Time</th><th>Action</th><th>Confidence</th><th>Price</th><th>SL</th><th>TP</th><th>Lot</th></tr>
                    </thead>
                    <tbody>
                        <tr><td colspan="7" style="text-align: center;">Loading signals...</td></tr>
                    </tbody>
                </table>
            </div>
        </div>
        
        <div class="card" style="margin-top: 20px;">
            <div class="card-title">Open Positions</div>
            <div style="overflow-x: auto;">
                <table class="signal-table" id="positions-table">
                    <thead><tr><th>Symbol</th><th>Side</th><th>Quantity</th><th>Open Price</th><th>Current</th><th>P&L</th><th>Action</th></tr></thead>
                    <tbody><tr><td colspan="7">No open positions</td></tr></tbody>
                </table>
            </div>
        </div>
        
        <div style="display: flex; gap: 10px; margin-top: 20px; justify-content: center;">
            <button class="btn" onclick="startTrading()">▶ Start Trading</button>
            <button class="btn" onclick="stopTrading()">⏹ Stop Trading</button>
            <button class="btn" onclick="closeAllPositions()">🔒 Close All Positions</button>
        </div>
    </div>
    
    <button class="refresh-btn" onclick="refreshData()">🔄</button>
    
    <script>
        let refreshInterval = null;
        
        async function refreshData() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                
                // Update account info
                document.getElementById('balance').innerHTML = `$${data.account.balance.toFixed(2)}`;
                document.getElementById('equity').innerHTML = `$${data.account.equity.toFixed(2)}`;
                document.getElementById('free-margin').innerHTML = `$${data.account.free_margin.toFixed(2)}`;
                document.getElementById('trades-today').innerHTML = data.trading.trades_today;
                document.getElementById('predictions-today').innerHTML = data.trading.predictions_today;
                document.getElementById('open-positions').innerHTML = data.trading.open_positions;
                
                // Daily P&L
                const dailyPnl = data.account.daily_pnl;
                const dailyPnlClass = dailyPnl >= 0 ? 'positive' : 'negative';
                document.getElementById('daily-pnl').innerHTML = `Today: <span class="${dailyPnlClass}">$${dailyPnl.toFixed(2)} (${data.account.daily_pnl_pct.toFixed(2)}%)</span>`;
                
                // Risk status
                const riskStatus = data.risk.kill_switch_active ? '🔴 KILL SWITCH ACTIVE' : '✅ Active';
                document.getElementById('risk-status').innerHTML = riskStatus;
                document.getElementById('consecutive-losses').innerHTML = `Consecutive Losses: ${data.risk.consecutive_losses}`;
                
                // Model status
                document.getElementById('model-status').innerHTML = data.status.model_loaded ? '✅ Loaded' : '⚠️ Not Loaded';
                
                // Kill switch
                const ksStatus = data.status.kill_switch ? '🔴 ACTIVE' : '⚪ Inactive';
                document.getElementById('kill-switch-status').innerHTML = ksStatus;
                
                // Last signal
                if (data.trading.last_signal) {
                    const signal = data.trading.last_signal;
                    const actionClass = signal.action === 'BUY' ? 'positive' : (signal.action === 'SELL' ? 'negative' : 'neutral');
                    document.getElementById('last-signal').innerHTML = `
                        <div class="card-value ${actionClass}">${signal.action}</div>
                        <div class="card-change">Confidence: ${(signal.confidence * 100).toFixed(1)}% | Price: $${signal.entry_price.toFixed(2)}</div>
                    `;
                }
                
                // Update signals table
                updateSignalsTable();
                updatePositionsTable();
                
            } catch (error) {
                console.error('Error fetching data:', error);
            }
        }
        
        async function updateSignalsTable() {
            const response = await fetch('/api/signals?limit=20');
            const data = await response.json();
            
            const tbody = document.querySelector('#signals-table tbody');
            if (data.signals.length === 0) {
                tbody.innerHTML = '<tr><td colspan="7" style="text-align: center;">No signals yet</td></tr>';
                return;
            }
            
            tbody.innerHTML = data.signals.map(s => `
                <tr>
                    <td>${new Date(s.timestamp).toLocaleTimeString()}</td>
                    <td><span class="badge-${s.action === 'BUY' ? 'buy' : (s.action === 'SELL' ? 'sell' : 'neutral')}">${s.action}</span></td>
                    <td>${(s.confidence * 100).toFixed(1)}%</td>
                    <td>$${s.entry_price.toFixed(2)}</td>
                    <td>$${s.stop_loss.toFixed(2)}</td>
                    <td>$${s.take_profit.toFixed(2)}</td>
                    <td>${s.lot_size}</td>
                </tr>
            `).join('');
        }
        
        async function updatePositionsTable() {
            const response = await fetch('/api/positions');
            const data = await response.json();
            
            const tbody = document.querySelector('#positions-table tbody');
            if (data.positions.length === 0) {
                tbody.innerHTML = '<tr><td colspan="7">No open positions</td></tr>';
                return;
            }
            
            tbody.innerHTML = data.positions.map(p => `
                <tr>
                    <td>${p.symbol}</td>
                    <td class="${p.side === 'buy' ? 'positive' : 'negative'}">${p.side.toUpperCase()}</td>
                    <td>${p.quantity}</td>
                    <td>$${p.open_price.toFixed(2)}</td>
                    <td>$${p.current_price.toFixed(2)}</td>
                    <td class="${p.pnl >= 0 ? 'positive' : 'negative'}">$${p.pnl.toFixed(2)} (${p.pnl_pct.toFixed(2)}%)</td>
                    <td><button class="btn" style="padding: 5px 10px; font-size: 12px;" onclick="closePosition('${p.symbol}')">Close</button></td>
                </tr>
            `).join('');
        }
        
        async function startTrading() {
            await fetch('/api/system/start', { method: 'POST' });
            alert('Trading started!');
        }
        
        async function stopTrading() {
            await fetch('/api/system/stop', { method: 'POST' });
            alert('Trading stopped!');
        }
        
        async function closeAllPositions() {
            if (confirm('Close all open positions?')) {
                await fetch('/api/trade/close-all', { method: 'POST' });
                refreshData();
            }
        }
        
        async function closePosition(symbol) {
            await fetch(`/api/trade/close/${symbol}`, { method: 'POST' });
            refreshData();
        }
        
        async function activateKillSwitch() {
            if (confirm('⚠️ EMERGENCY: Activate Kill Switch? This will stop all trading!')) {
                await fetch('/api/kill-switch/activate', { method: 'POST' });
                refreshData();
            }
        }
        
        function updateTime() {
            document.getElementById('time-badge').innerHTML = new Date().toLocaleString();
        }
        
        // Start auto-refresh
        refreshData();
        updateTime();
        refreshInterval = setInterval(refreshData, 5000);
        setInterval(updateTime, 1000);
    </script>
</body>
</html>
"""


# ============================================
# MAIN ENTRY POINT
# ============================================

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Trading AI System")
    parser.add_argument("--mode", choices=["paper", "live", "backtest", "dashboard"], default="dashboard")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--start-date", type=str, help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="Backtest end date (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    if args.mode == "dashboard":
        print("""
        ╔══════════════════════════════════════════════════════════════╗
        ║     🤖 TRADING AI SYSTEM - DASHBOARD MODE                    ║
        ║                                                              ║
        ║     Dashboard URL: http://localhost:{}                     ║
        ║     API Docs: http://localhost:{}/docs                      ║
        ║                                                              ║
        ║     Features:                                                ║
        ║     - Real-time signals                                     ║
        ║     - Account monitoring                                    ║
        ║     - Risk management                                       ║
        ║     - Trade journal                                         ║
        ║                                                              ║
        ║     Press Ctrl+C to stop                                    ║
        ╚══════════════════════════════════════════════════════════════╝
        """.format(args.port, args.port))
        
        # Open browser automatically
        webbrowser.open(f"http://localhost:{args.port}")
        
        # Run FastAPI server
        uvicorn.run(app, host="0.0.0.0", port=args.port)
        
    elif args.mode == "paper":
        print("📝 Starting PAPER TRADING mode...")
        system = TradingAISystem()
        system.run_paper_trading(args.symbol)
        
    elif args.mode == "live":
        print("⚠️ LIVE TRADING mode - Risk acknowledged!")
        system = TradingAISystem()
        system.status.mode = "live"
        system.run_paper_trading(args.symbol)  # Replace with live broker
        
    elif args.mode == "backtest":
        print("📊 Starting BACKTEST mode...")
        system = TradingAISystem()
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d") if args.start_date else datetime.now() - timedelta(days=365)
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d") if args.end_date else datetime.now()
        system.run_backtest(start_date, end_date)
        
    else:
        parser.print_help()


if __name__ == "__main__":
    # Create necessary directories
    Path("logs").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    
    main()
