"""Main entry point for Trading AI System"""

import argparse
import logging
import yaml
from pathlib import Path
from datetime import datetime

from data_engine.live_feed import LiveFeed
from data_engine.validator import DataValidator
from smc_engine.structure import StructureDetector
from smc_engine.order_blocks import OrderBlockDetector
from smc_engine.fvg import FVGDetector
from fusion_model.train_xgb import FusionModel
from execution.mt5_adapter import MT5Adapter
from execution.paper_trading import PaperTradingBroker
from execution.order_manager import OrderManager
from risk_engine.risk_limits import RiskEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TradingAISystem:
    """Main trading system orchestrator"""
    
    def __init__(self, config_path: str = "config"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Initialize components
        self.data_validator = DataValidator()
        self.structure_detector = StructureDetector()
        self.order_block_detector = OrderBlockDetector()
        self.fvg_detector = FVGDetector()
        
        # Load models
        self.fusion_model = FusionModel()
        model_path = Path("models/fusion_model")
        if model_path.exists():
            self.fusion_model.load(model_path)
            
        # Initialize broker (paper by default)
        self.broker = PaperTradingBroker()
        self.order_manager = OrderManager(self.broker)
        
        # Initialize risk engine
        self.risk_engine = RiskEngine(self.config.get('risk', {}))
        
        # Live feed
        self.live_feed = None
        
    def _load_config(self) -> dict:
        """Load configuration files"""
        config = {}
        
        for config_file in self.config_path.glob("*.yaml"):
            with open(config_file, 'r') as f:
                config.update(yaml.safe_load(f))
                
        return config
        
    def start_live_trading(self, symbol: str = "XAUUSD"):
        """Start live trading with real-time feed"""
        
        logger.info(f"Starting live trading for {symbol}")
        
        # Initialize live feed
        self.live_feed = LiveFeed(symbol=symbol, timeframe="15m")
        self.live_feed.add_candle_callback(self._on_candle_close)
        
        # Connect broker
        if not self.broker.connect():
            logger.error("Failed to connect to broker")
            return
            
        logger.info("Trading system started")
        
    def _on_candle_close(self, candle):
        """Handle completed candle"""
        
        logger.info(f"New candle closed: {candle.timestamp}")
        
        # Get recent data (would need to maintain history)
        # For now, placeholder for prediction logic
        
        # Get SMC features
        # smc_features = self._compute_smc_features(df)
        
        # Get CNN embedding from image
        # embedding = self._get_cnn_embedding(df)
        
        # Get prediction
        # prediction = self.fusion_model.predict(embedding, smc_features)
        
        # Execute trade if confidence high
        # self._execute_if_signal(prediction)
        
    def stop(self):
        """Stop the trading system"""
        if self.live_feed:
            self.live_feed.close()
        self.broker.disconnect()
        logger.info("Trading system stopped")
        
    def run_backtest(self, start_date: datetime, end_date: datetime):
        """Run backtest on historical data"""
        logger.info(f"Running backtest from {start_date} to {end_date}")
        # Implement backtest logic
        

def main():
    parser = argparse.ArgumentParser(description="Trading AI System")
    parser.add_argument("--mode", choices=["live", "paper", "backtest"], default="paper")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--config", default="config")
    
    args = parser.parse_args()
    
    system = TradingAISystem(config_path=args.config)
    
    try:
        if args.mode == "paper":
            logger.info("Starting paper trading mode")
            system.start_live_trading(args.symbol)
            
            # Keep running
            import time
            while True:
                time.sleep(1)
                
        elif args.mode == "live":
            logger.warning("LIVE TRADING MODE - Risk acknowledged")
            system.start_live_trading(args.symbol)
            
            while True:
                time.sleep(1)
                
        else:
            logger.info("Backtest mode - implement with historical data")
            
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        system.stop()
        

if __name__ == "__main__":
    main()
