"""
Vercel Serverless Function - Lightweight Version
No NumPy/pandas compilation required
"""

import json
import logging
from datetime import datetime
from typing import Optional, Dict, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="Trading AI System", version="1.0.0")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# API Models (Pure Python)
# ============================================

class PredictionRequest(BaseModel):
    symbol: str = "XAUUSD"
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float = 0


class PredictionResponse(BaseModel):
    signal: str
    confidence: float
    buy_prob: float
    sell_prob: float
    notrade_prob: float
    timestamp: datetime
    pattern_detected: str


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: datetime


# ============================================
# Pure Python Pattern Recognition (No NumPy)
# ============================================

class PatternRecognizer:
    """Pure Python pattern recognition - no heavy dependencies"""
    
    def __init__(self):
        self.patterns = {
            'bullish_engulfing': 0,
            'bearish_engulfing': 0,
            'hammer': 0,
            'shooting_star': 0,
            'doji': 0,
            'morning_star': 0,
            'evening_star': 0
        }
        
    def detect_patterns(self, open_price: float, high: float, low: float, close: float) -> Dict:
        """Detect candlestick patterns"""
        
        body = abs(close - open_price)
        candle_range = high - low
        upper_wick = high - max(open_price, close)
        lower_wick = min(open_price, close) - low
        
        patterns = []
        
        # Doji (indecision)
        if candle_range > 0 and body / candle_range < 0.1:
            patterns.append("doji")
            
        # Hammer (bullish reversal)
        if lower_wick > body * 2 and upper_wick < body * 0.3:
            patterns.append("hammer")
            
        # Shooting Star (bearish reversal)
        if upper_wick > body * 2 and lower_wick < body * 0.3:
            patterns.append("shooting_star")
            
        # Bullish Engulfing
        if close > open_price and body > candle_range * 0.6:
            patterns.append("bullish_engulfing")
            
        # Bearish Engulfing
        if close < open_price and body > candle_range * 0.6:
            patterns.append("bearish_engulfing")
            
        # Morning Star (bullish)
        if close > open_price and lower_wick > body:
            patterns.append("morning_star")
            
        # Evening Star (bearish)
        if close < open_price and upper_wick > body:
            patterns.append("evening_star")
            
        return {
            'detected': patterns,
            'primary': patterns[0] if patterns else 'none',
            'count': len(patterns)
        }
        
    def calculate_support_resistance(self, prices: List[float]) -> Dict:
        """Simple support/resistance levels"""
        if not prices:
            return {'support': 0, 'resistance': 0}
            
        recent = prices[-20:] if len(prices) > 20 else prices
        avg = sum(recent) / len(recent)
        
        # Find local minima and maxima
        support = min(recent)
        resistance = max(recent)
        
        return {
            'support': support,
            'resistance': resistance,
            'pivot': (support + resistance + avg) / 3
        }


class TradingModel:
    """Lightweight trading model"""
    
    def __init__(self):
        self.pattern_recognizer = PatternRecognizer()
        self.price_history: List[float] = []
        
    def predict(self, open_price: float, high: float, low: float, close: float, volume: float) -> Dict:
        """Make prediction based on patterns"""
        
        # Add to history
        self.price_history.append(close)
        if len(self.price_history) > 50:
            self.price_history.pop(0)
            
        # Detect patterns
        patterns = self.pattern_recognizer.detect_patterns(open_price, high, low, close)
        
        # Calculate technical levels
        levels = self.pattern_recognizer.calculate_support_resistance(self.price_history)
        
        # Price action
        is_bullish = close > open_price
        is_bearish = close < open_price
        body_percent = abs(close - open_price) / (high - low) if (high - low) > 0 else 0
        
        # Base probabilities
        buy_prob = 0.35
        sell_prob = 0.35
        notrade_prob = 0.30
        
        pattern_name = "none"
        
        # Adjust based on patterns
        for pattern in patterns['detected']:
            if pattern in ['bullish_engulfing', 'hammer', 'morning_star']:
                buy_prob += 0.15
                pattern_name = pattern
            elif pattern in ['bearish_engulfing', 'shooting_star', 'evening_star']:
                sell_prob += 0.15
                pattern_name = pattern
            elif pattern == 'doji':
                notrade_prob += 0.10
                pattern_name = pattern
                
        # Adjust based on position relative to levels
        if close < levels['support'] * 1.01:
            buy_prob += 0.10  # Near support - bullish
        elif close > levels['resistance'] * 0.99:
            sell_prob += 0.10  # Near resistance - bearish
            
        # Strong move
        if body_percent > 0.6:
            if is_bullish:
                buy_prob += 0.10
            else:
                sell_prob += 0.10
                
        # Normalize
        total = buy_prob + sell_prob + notrade_prob
        buy_prob /= total
        sell_prob /= total
        notrade_prob /= total
        
        # Determine signal
        confidence_threshold = 0.55
        
        if buy_prob > confidence_threshold and buy_prob > sell_prob:
            signal = "BUY"
            confidence = buy_prob
        elif sell_prob > confidence_threshold and sell_prob > buy_prob:
            signal = "SELL"
            confidence = sell_prob
        else:
            signal = "NO_TRADE"
            confidence = max(buy_prob, sell_prob, notrade_prob)
            
        return {
            'signal': signal,
            'confidence': confidence,
            'buy_prob': round(buy_prob, 4),
            'sell_prob': round(sell_prob, 4),
            'notrade_prob': round(notrade_prob, 4),
            'patterns': patterns['detected'],
            'primary_pattern': pattern_name,
            'support': round(levels['support'], 2),
            'resistance': round(levels['resistance'], 2),
            'pivot': round(levels['pivot'], 2)
        }


# Initialize model
model = TradingModel()


# ============================================
# API Endpoints
# ============================================

@app.get("/", response_model=HealthResponse)
async def root():
    return HealthResponse(
        status="online",
        version="1.0.0",
        timestamp=datetime.now()
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now()
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Get trading prediction"""
    
    try:
        prediction = model.predict(
            request.open_price,
            request.high_price,
            request.low_price,
            request.close_price,
            request.volume
        )
        
        return PredictionResponse(
            signal=prediction['signal'],
            confidence=prediction['confidence'],
            buy_prob=prediction['buy_prob'],
            sell_prob=prediction['sell_prob'],
            notrade_prob=prediction['notrade_prob'],
            timestamp=datetime.now(),
            pattern_detected=prediction['primary_pattern']
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-batch")
async def predict_batch(requests: List[PredictionRequest]):
    """Batch prediction"""
    
    results = []
    for req in requests:
        pred = model.predict(
            req.open_price, req.high_price, 
            req.low_price, req.close_price, req.volume
        )
        results.append({
            'symbol': req.symbol,
            **pred
        })
        
    return {'predictions': results, 'count': len(results)}


@app.get("/patterns")
async def get_patterns():
    """Get available patterns"""
    return {
        'patterns': [
            {'name': 'Bullish Engulfing', 'type': 'bullish', 'confidence': 'high'},
            {'name': 'Bearish Engulfing', 'type': 'bearish', 'confidence': 'high'},
            {'name': 'Hammer', 'type': 'bullish', 'confidence': 'medium'},
            {'name': 'Shooting Star', 'type': 'bearish', 'confidence': 'medium'},
            {'name': 'Doji', 'type': 'neutral', 'confidence': 'low'},
            {'name': 'Morning Star', 'type': 'bullish', 'confidence': 'high'},
            {'name': 'Evening Star', 'type': 'bearish', 'confidence': 'high'}
        ]
    }


@app.get("/dashboard")
async def dashboard():
    """HTML Dashboard"""
    
    html = '''<!DOCTYPE html>
<html>
<head>
    <title>Trading AI System</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #0a0a0a, #1a1a2e);
            color: white;
            margin: 0;
            padding: 20px;
            min-height: 100vh;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 40px; }
        .header h1 {
            font-size: 2.5em;
            background: linear-gradient(90deg, #00ff88, #00bfff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .card {
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(10px);
        }
        .card-title {
            font-size: 1.2em;
            margin-bottom: 15px;
            color: #00ff88;
        }
        input, select {
            width: 100%;
            padding: 10px;
            margin: 8px 0;
            border-radius: 8px;
            border: 1px solid #333;
            background: rgba(0,0,0,0.5);
            color: white;
        }
        button {
            background: linear-gradient(90deg, #00ff88, #00bfff);
            color: #000;
            border: none;
            padding: 12px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            width: 100%;
            margin-top: 10px;
        }
        .result {
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        .signal-buy { background: rgba(0,255,136,0.2); border: 1px solid #00ff88; }
        .signal-sell { background: rgba(255,68,68,0.2); border: 1px solid #ff4444; }
        .signal-neutral { background: rgba(255,170,0,0.2); border: 1px solid #ffaa00; }
        .signal-text { font-size: 2em; font-weight: bold; }
        .pattern-badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            margin: 2px;
        }
        .pattern-bullish { background: #00ff8822; color: #00ff88; }
        .pattern-bearish { background: #ff444422; color: #ff4444; }
        .pattern-neutral { background: #ffaa0022; color: #ffaa00; }
        @media (max-width: 768px) { .grid { grid-template-columns: 1fr; } }
        footer { text-align: center; margin-top: 40px; color: #666; }
        .status-online { color: #00ff88; }
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>🤖 Trading AI System</h1>
        <p>Pattern Recognition + Smart Money Concepts</p>
        <p><span class="status-online">●</span> System Online | Vercel Deployment</p>
    </div>
    <div class="grid">
        <div class="card">
            <div class="card-title">📊 Market Data</div>
            <select id="symbol"><option>XAUUSD</option><option>EURUSD</option><option>GBPUSD</option></select>
            <input type="number" id="open" placeholder="Open Price" value="2000">
            <input type="number" id="high" placeholder="High Price" value="2010">
            <input type="number" id="low" placeholder="Low Price" value="1995">
            <input type="number" id="close" placeholder="Close Price" value="2005">
            <button onclick="predict()">🔮 Predict</button>
        </div>
        <div class="card">
            <div class="card-title">🎯 Prediction</div>
            <div id="result" class="result signal-neutral">
                <div class="signal-text">Enter Data</div>
                <div>Click Predict to analyze</div>
            </div>
        </div>
    </div>
    <div class="card" style="margin-top:20px">
        <div class="card-title">📈 Analysis</div>
        <div id="analysis">Waiting for prediction...</div>
    </div>
    <footer>Powered by Pattern Recognition | Real-time Analysis</footer>
</div>
<script>
async function predict() {
    const data = {
        symbol: document.getElementById('symbol').value,
        open_price: parseFloat(document.getElementById('open').value),
        high_price: parseFloat(document.getElementById('high').value),
        low_price: parseFloat(document.getElementById('low').value),
        close_price: parseFloat(document.getElementById('close').value),
        volume: 1000
    };
    const res = await fetch('/predict', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(data)
    });
    const r = await res.json();
    const resultDiv = document.getElementById('result');
    resultDiv.className = `result signal-${r.signal.toLowerCase()}`;
    resultDiv.innerHTML = `<div class="signal-text">${r.signal}</div>
        <div>Confidence: ${(r.confidence*100).toFixed(1)}%</div>
        <div>Pattern: ${r.pattern_detected}</div>`;
    document.getElementById('analysis').innerHTML = `
        <div>BUY: ${(r.buy_prob*100).toFixed(1)}% | SELL: ${(r.sell_prob*100).toFixed(1)}% | NO TRADE: ${(r.notrade_prob*100).toFixed(1)}%</div>
        <div style="margin-top:10px">Pattern-based AI | Technical Analysis Active</div>`;
}
</script>
</body>
</html>'''
    
    return HTMLResponse(html)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
