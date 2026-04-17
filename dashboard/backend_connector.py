"""Dashboard backend connector - WebSocket and REST API integration"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import pandas as pd
import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from collections import deque

import sys
sys.path.append(str(Path(__file__).parent.parent))

from execution.broker_adapter import OrderSide
from execution.mt5_adapter import MT5Adapter
from execution.paper_trading import PaperTradingBroker
from execution.order_manager import OrderManager
from risk_engine.risk_limits import RiskEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="Trading AI Dashboard API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Global state
class DashboardState:
    def __init__(self):
        self.broker = PaperTradingBroker()
        self.order_manager = OrderManager(self.broker)
        self.risk_engine = RiskEngine({})
        self.connected_clients: List[WebSocket] = []
        self.signals = deque(maxlen=1000)
        self.trades = deque(maxlen=1000)
        self.market_data = {
            'bid': 2000.0,
            'ask': 2000.2,
            'spread': 0.2,
            'timestamp': datetime.utcnow()
        }
        self.mode = 'paper'
        self.kill_switch_active = False
        
    def add_signal(self, signal: Dict):
        self.signals.appendleft(signal)
        
    def add_trade(self, trade: Dict):
        self.trades.appendleft(trade)
        
    def get_trades(self, limit: int = 100, symbol: str = None, start: datetime = None, end: datetime = None) -> List:
        trades = list(self.trades)
        if symbol:
            trades = [t for t in trades if t.get('symbol') == symbol]
        if start:
            trades = [t for t in trades if t.get('close_time', datetime.min) >= start]
        if end:
            trades = [t for t in trades if t.get('close_time', datetime.max) <= end]
        return trades[:limit]

state = DashboardState()

# Models
class OrderRequest(BaseModel):
    symbol: str
    direction: str  # BUY or SELL
    lot_size: float
    stop_loss_pips: float
    take_profit_pips: float
    confidence: float = 1.0

class SettingsUpdate(BaseModel):
    max_risk_per_trade: float
    max_concurrent_trades: int
    min_confidence: float
    sl_atr_multiplier: float
    tp_atr_multiplier: float
    alerts: Dict
    slack_webhook: str

# WebSocket Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            
    async def broadcast(self, message: Dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Broadcast error: {e}")

manager = ConnectionManager()

# Helper Functions
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token"""
    token = credentials.credentials
    # Simple token validation for demo
    if token != "demo_token_123":
        raise HTTPException(status_code=401, detail="Invalid token")
    return token

# API Endpoints
@app.get("/health")
async def health_check():
    return {"status": "healthy", "mode": state.mode, "timestamp": datetime.utcnow()}

@app.get("/account")
async def get_account_info(token: str = Depends(verify_token)):
    """Get account information"""
    if state.mode == 'paper' and isinstance(state.broker, PaperTradingBroker):
        return state.broker.get_account_info().__dict__
    else:
        account = state.broker.get_account_info()
        return {
            'balance': account.balance,
            'equity': account.equity,
            'margin': account.margin,
            'free_margin': account.free_margin,
            'margin_level': account.margin_level,
            'currency': account.currency,
            'unrealized_pnl': account.unrealized_pnl,
            'realized_pnl_today': account.realized_pnl_today
        }

@app.get("/positions")
async def get_positions(token: str = Depends(verify_token)):
    """Get open positions"""
    positions = state.order_manager.get_open_positions()
    return [{
        'id': getattr(p, 'id', str(i)),
        'symbol': p.symbol,
        'side': p.side.value,
        'quantity': p.quantity,
        'open_price': p.open_price,
        'current_price': p.current_price,
        'unrealized_pnl': p.unrealized_pnl,
        'stop_loss': p.stop_loss,
        'take_profit': p.take_profit
    } for i, p in enumerate(positions)]

@app.delete("/positions/{position_id}")
async def close_position(position_id: str, token: str = Depends(verify_token)):
    """Close a specific position"""
    success = state.order_manager.close_trade(position_id)
    if not success:
        raise HTTPException(status_code=404, detail="Position not found")
    return {"status": "closed", "position_id": position_id}

@app.post("/positions/close_all")
async def close_all_positions(token: str = Depends(verify_token)):
    """Close all open positions"""
    count = state.order_manager.close_all_trades()
    return {"closed": count}

@app.post("/trade")
async def place_trade(order: OrderRequest, token: str = Depends(verify_token)):
    """Place a trade"""
    
    # Check kill switch
    if state.kill_switch_active:
        raise HTTPException(status_code=403, detail="Kill switch active - trading disabled")
        
    # Check mode
    if state.mode == 'paper' and not isinstance(state.broker, PaperTradingBroker):
        state.broker = PaperTradingBroker()
        state.order_manager = OrderManager(state.broker)
        
    # Get market price
    market = state.broker._get_market_price(order.symbol) if isinstance(state.broker, PaperTradingBroker) else {'bid': 2000, 'ask': 2000.2}
    
    # Calculate SL/TP prices
    pip_size = 0.01 if order.symbol == 'XAUUSD' else 0.0001
    if order.direction == 'BUY':
        entry = market['ask']
        stop_loss = entry - (order.stop_loss_pips * pip_size)
        take_profit = entry + (order.take_profit_pips * pip_size)
    else:
        entry = market['bid']
        stop_loss = entry + (order.stop_loss_pips * pip_size)
        take_profit = entry - (order.take_profit_pips * pip_size)
        
    # Create signal dict
    signal = {
        'direction': 1 if order.direction == 'BUY' else -1,
        'symbol': order.symbol,
        'model_version': 'manual',
        'confidence': order.confidence
    }
    
    # Place trade
    trade_order = state.order_manager.place_trade(
        signal=signal,
        lot_size=order.lot_size,
        stop_loss=stop_loss,
        take_profit=take_profit
    )
    
    if not trade_order:
        raise HTTPException(status_code=500, detail="Trade placement failed")
        
    # Log trade
    state.add_trade({
        'id': trade_order.id,
        'symbol': order.symbol,
        'direction': order.direction,
        'lot_size': order.lot_size,
        'entry_price': trade_order.average_price,
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'open_time': datetime.utcnow(),
        'status': 'open'
    })
    
    # Broadcast to WebSocket
    await manager.broadcast({
        'type': 'trade_update',
        'data': {'action': 'opened', 'order_id': trade_order.id}
    })
    
    return {
        'order_id': trade_order.id,
        'status': 'filled',
        'entry_price': trade_order.average_price
    }

@app.get("/signals")
async def get_signals(limit: int = 50, token: str = Depends(verify_token)):
    """Get recent signals"""
    return list(state.signals)[:limit]

@app.get("/trades/history")
async def get_trade_history(
    limit: int = 100,
    symbol: str = None,
    start: str = None,
    end: str = None,
    token: str = Depends(verify_token)
):
    """Get trade history"""
    start_date = datetime.fromisoformat(start) if start else None
    end_date = datetime.fromisoformat(end) if end else None
    
    trades = state.get_trades(limit, symbol, start_date, end_date)
    return trades

@app.get("/models")
async def get_models(token: str = Depends(verify_token)):
    """Get model information and metrics"""
    return {
        'champion': {
            'name': 'Fusion Model v4.2',
            'metrics': {
                'profit_factor': 1.87,
                'win_rate': 58.3,
                'sharpe': 1.62,
                'max_drawdown': 8.3
            },
            'trained': '2024-01-15',
            'dataset': 'v4_htf_filtered'
        },
        'challenger': {
            'name': 'Fusion Model v4.3',
            'metrics': {
                'profit_factor': 1.92,
                'win_rate': 59.7,
                'sharpe': 1.71,
                'max_drawdown': 7.9
            },
            'trained': '2024-01-22',
            'dataset': 'v4_htf_filtered'
        },
        'feature_importance': {
            'cnn_embedding_pca_0': 0.142,
            'dist_to_nearest_ob': 0.098,
            'hh_hl_ratio': 0.087,
            'fvg_bull_open': 0.076,
            'htf_bias': 0.065,
            'volatility_regime': 0.054,
            'bos_count_bull': 0.043,
            'impulse_strength': 0.032,
            'session_code': 0.021,
            'liq_high_distance': 0.018
        },
        'drift': {
            'psi_score': 0.08,
            'avg_confidence': 0.72,
            'last_retrain': '2024-01-20',
            'drifted_features': []
        }
    }

@app.get("/risk/status")
async def get_risk_status(token: str = Depends(verify_token)):
    """Get risk engine status"""
    status = state.risk_engine.get_risk_status()
    account = state.broker.get_account_info() if state.broker else None
    balance = account.balance if account else 10000
    
    return {
        'kill_switch_active': state.kill_switch_active,
        'daily_loss': status.get('daily_pnl', 0),
        'daily_loss_limit': balance * 0.03,
        'drawdown': status.get('drawdown', 0),
        'drawdown_limit': 0.10,
        'consecutive_losses': status.get('consecutive_losses', 0),
        'max_consecutive_losses': 3,
        'open_positions': len(state.order_manager.get_open_positions()),
        'max_concurrent': state.order_manager.max_concurrent_trades
    }

@app.post("/risk/kill_switch")
async def trigger_kill_switch_endpoint(token: str = Depends(verify_token)):
    """Trigger kill switch"""
    state.kill_switch_active = True
    state.order_manager.close_all_trades()
    
    await manager.broadcast({
        'type': 'alert',
        'data': {
            'type': 'danger',
            'message': 'KILL SWITCH ACTIVATED - All positions closed'
        }
    })
    
    return {"status": "kill_switch_activated"}

@app.put("/mode")
async def set_mode(mode: str, token: str = Depends(verify_token)):
    """Switch between paper and live mode"""
    if mode not in ['paper', 'live']:
        raise HTTPException(status_code=400, detail="Invalid mode")
        
    state.mode = mode
    
    if mode == 'paper':
        state.broker = PaperTradingBroker()
    else:
        state.broker = MT5Adapter()
        
    state.order_manager = OrderManager(state.broker)
    
    await manager.broadcast({
        'type': 'mode_change',
        'data': {'mode': mode}
    })
    
    return {"mode": mode}

@app.put("/settings")
async def update_settings(settings: SettingsUpdate, token: str = Depends(verify_token)):
    """Update system settings"""
    # Update risk engine config
    state.risk_engine.config.update({
        'max_risk_per_trade_pct': settings.max_risk_per_trade / 100,
        'max_concurrent_trades': settings.max_concurrent_trades,
        'min_confidence': settings.min_confidence
    })
    
    # Update order manager
    state.order_manager.max_concurrent_trades = settings.max_concurrent_trades
    
    return {"status": "settings_updated"}

# WebSocket Endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    
    try:
        # Send initial data
        await websocket.send_json({
            'type': 'connected',
            'data': {'mode': state.mode}
        })
        
        while True:
            # Receive messages from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle different message types
            if message.get('type') == 'ping':
                await websocket.send_json({'type': 'pong'})
                
            elif message.get('type') == 'subscribe':
                # Subscribe to specific data streams
                pass
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("WebSocket disconnected")

# Simulated Market Data Generator
async def market_data_simulator():
    """Simulate real-time market data"""
    import random
    import math
    
    base_price = 2000.0
    trend = 0
    
    while True:
        # Random walk with mean reversion
        trend += random.gauss(0, 0.5)
        trend *= 0.99  # Mean reversion
        
        price_change = random.gauss(trend * 0.1, 1.5)
        base_price += price_change
        
        # Keep within reasonable range
        base_price = max(1900, min(2100, base_price))
        
        spread = 0.2 + random.gauss(0, 0.05)
        spread = max(0.1, min(0.5, spread))
        
        state.market_data = {
            'bid': base_price,
            'ask': base_price + spread,
            'spread': spread,
            'timestamp': datetime.utcnow()
        }
        
        # Broadcast to all connected clients
        await manager.broadcast({
            'type': 'market_data',
            'data': state.market_data
        })
        
        await asyncio.sleep(1)  # Update every second

# Simulated AI Signal Generator
async def ai_signal_simulator():
    """Simulate AI predictions"""
    import random
    
    while True:
        # Generate random signals for demo
        if random.random() < 0.05:  # 5% chance per 15 seconds
            direction = random.choice(['BUY', 'SELL', 'NO_TRADE'])
            confidence = 0.6 + random.random() * 0.35
            
            if direction != 'NO_TRADE':
                signal = {
                    'type': 'signal',
                    'data': {
                        'direction': direction,
                        'confidence': confidence,
                        'symbol': 'XAUUSD',
                        'timestamp': datetime.utcnow().isoformat(),
                        'has_ob': random.random() > 0.5,
                        'has_fvg': random.random() > 0.7,
                        'smc_features': {
                            'dist_to_ob': random.uniform(0, 5),
                            'trend_strength': random.uniform(-1, 1),
                            'volatility': random.uniform(0.5, 2)
                        }
                    }
                }
                
                state.add_signal(signal['data'])
                await manager.broadcast(signal)
                
        await asyncio.sleep(15)  # Check every 15 seconds

# Startup Event
@app.on_event("startup")
async def startup_event():
    """Start background tasks"""
    asyncio.create_task(market_data_simulator())
    asyncio.create_task(ai_signal_simulator())
    logger.info("Dashboard backend started")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
