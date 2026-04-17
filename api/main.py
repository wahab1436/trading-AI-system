"""FastAPI server for trading system"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List
from datetime import datetime
import logging

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from cnn_model.model import ChartCNN
from fusion_model.train_xgb import FusionModel
from execution.broker_adapter import OrderSide

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Trading AI System API", version="1.0.0")

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

# Global model instances
cnn_model = None
fusion_model = None
order_manager = None
risk_engine = None


class PredictionRequest(BaseModel):
    """Prediction request schema"""
    symbol: str = "XAUUSD"
    image_base64: str  # Base64 encoded chart image
    smc_features: Dict  # Precomputed SMC features


class PredictionResponse(BaseModel):
    """Prediction response schema"""
    signal: str  # BUY, SELL, NO_TRADE
    confidence: float
    probabilities: Dict[str, float]
    explanation: Optional[Dict] = None
    timestamp: datetime


class TradeRequest(BaseModel):
    """Trade request schema"""
    signal: str
    confidence: float
    lot_size: float
    stop_loss: float
    take_profit: float
    symbol: str = "XAUUSD"


class TradeResponse(BaseModel):
    """Trade response schema"""
    order_id: str
    status: str
    message: str


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token"""
    # Implement JWT verification
    return credentials.credentials


@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    global cnn_model, fusion_model
    
    logger.info("Loading models...")
    
    # Load CNN model
    cnn_model = ChartCNN()
    # cnn_model.load_state_dict(torch.load('models/cnn_best.pt'))
    cnn_model.eval()
    
    # Load fusion model
    fusion_model = FusionModel()
    # fusion_model.load(Path('models/fusion_model'))
    
    logger.info("Models loaded successfully")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "models_loaded": cnn_model is not None and fusion_model is not None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    token: str = Depends(verify_token)
):
    """Get trading prediction from chart image"""
    
    try:
        # Process image (decode base64)
        import base64
        from PIL import Image
        import io
        import torch
        from torchvision import transforms
        
        image_bytes = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Transform image
        transform = transforms.Compose([
            transforms.Resize((380, 380)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        img_tensor = transform(image).unsqueeze(0)
        
        # Get CNN embedding
        with torch.no_grad():
            embedding, _ = cnn_model(img_tensor)
        embedding_np = embedding.numpy()
        
        # Prepare SMC features
        import numpy as np
        smc_array = np.array([list(request.smc_features.values())])
        
        # Get fusion model prediction
        probs = fusion_model.predict(embedding_np, smc_array)
        
        # Determine signal
        if probs['buy_prob'] > 0.65 and probs['buy_prob'] > probs['sell_prob']:
            signal = "BUY"
            confidence = probs['buy_prob']
        elif probs['sell_prob'] > 0.65 and probs['sell_prob'] > probs['buy_prob']:
            signal = "SELL"
            confidence = probs['sell_prob']
        else:
            signal = "NO_TRADE"
            confidence = max(probs['buy_prob'], probs['sell_prob'])
            
        return PredictionResponse(
            signal=signal,
            confidence=confidence,
            probabilities=probs,
            explanation=None,  # Add SHAP explanation
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/trade", response_model=TradeResponse)
async def place_trade(
    request: TradeRequest,
    token: str = Depends(verify_token)
):
    """Execute a trade"""
    
    if order_manager is None:
        raise HTTPException(status_code=503, detail="Order manager not initialized")
        
    # Check risk limits
    account = order_manager.broker.get_account_info()
    
    if not risk_engine.approve_trade(
        {'direction': 1 if request.signal == "BUY" else -1},
        account.balance,
        account.unrealized_pnl + account.realized_pnl_today
    ):
        raise HTTPException(status_code=400, detail="Risk limits exceeded")
        
    # Place trade
    signal = {
        'direction': 1 if request.signal == "BUY" else -1,
        'symbol': request.symbol,
        'model_version': 'api_v1'
    }
    
    order = order_manager.place_trade(
        signal=signal,
        lot_size=request.lot_size,
        stop_loss=request.stop_loss,
        take_profit=request.take_profit
    )
    
    if order is None:
        raise HTTPException(status_code=500, detail="Trade placement failed")
        
    return TradeResponse(
        order_id=order.id,
        status=order.status.value,
        message=f"Trade placed successfully"
    )


@app.get("/account")
async def get_account_info(token: str = Depends(verify_token)):
    """Get account information"""
    
    if order_manager is None:
        raise HTTPException(status_code=503, detail="Order manager not initialized")
        
    return order_manager.broker.get_account_info().__dict__


@app.get("/positions")
async def get_positions(token: str = Depends(verify_token)):
    """Get open positions"""
    
    if order_manager is None:
        raise HTTPException(status_code=503, detail="Order manager not initialized")
        
    positions = order_manager.get_open_positions()
    return [p.__dict__ for p in positions]


@app.delete("/positions/{position_id}")
async def close_position(position_id: str, token: str = Depends(verify_token)):
    """Close a position"""
    
    if order_manager is None:
        raise HTTPException(status_code=503, detail="Order manager not initialized")
        
    success = order_manager.close_trade(position_id)
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to close position")
        
    return {"status": "closed", "position_id": position_id}


@app.get("/risk/status")
async def get_risk_status(token: str = Depends(verify_token)):
    """Get risk engine status"""
    
    if risk_engine is None:
        raise HTTPException(status_code=503, detail="Risk engine not initialized")
        
    return risk_engine.get_risk_status()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
