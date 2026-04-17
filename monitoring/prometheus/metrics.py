"""Prometheus metrics collection for trading system monitoring"""

import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import defaultdict
import threading
import logging

logger = logging.getLogger(__name__)

# Try to import prometheus_client, fall back to mock if not installed
try:
    from prometheus_client import (
        Counter, Gauge, Histogram, Summary, Info,
        generate_latest, CollectorRegistry, REGISTRY
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Create mock classes
    class Counter:
        def __init__(self, *args, **kwargs): pass
        def inc(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
    
    class Gauge:
        def __init__(self, *args, **kwargs): pass
        def set(self, *args, **kwargs): pass
        def inc(self, *args, **kwargs): pass
        def dec(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
    
    class Histogram:
        def __init__(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
    
    class Summary:
        def __init__(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
    
    class Info:
        def __init__(self, *args, **kwargs): pass
        def info(self, *args, **kwargs): pass


class TradingMetrics:
    """Centralized metrics collection for trading system"""
    
    def __init__(self, namespace: str = "trading_ai"):
        self.namespace = namespace
        self._initialized = False
        self._lock = threading.Lock()
        
        # Metrics dictionaries
        self.counters = {}
        self.gauges = {}
        self.histograms = {}
        self.summaries = {}
        
        self._init_metrics()
        
    def _init_metrics(self):
        """Initialize all Prometheus metrics"""
        if not PROMETHEUS_AVAILABLE:
            logger.warning("prometheus_client not installed. Metrics will be mocked.")
            
        with self._lock:
            # ========== TRADING METRICS ==========
            
            # Trade counters
            self.counters['trades_total'] = Counter(
                f'{self.namespace}_trades_total',
                'Total number of trades executed',
                ['symbol', 'direction', 'outcome']
            )
            
            self.counters['trades_pending'] = Counter(
                f'{self.namespace}_trades_pending_total',
                'Total pending orders',
                ['symbol']
            )
            
            self.counters['signals_generated'] = Counter(
                f'{self.namespace}_signals_generated_total',
                'Total signals generated',
                ['symbol', 'signal_type']
            )
            
            # P&L Gauges
            self.gauges['pnl_daily'] = Gauge(
                f'{self.namespace}_pnl_daily',
                'Daily P&L in USD',
                ['symbol']
            )
            
            self.gauges['pnl_total'] = Gauge(
                f'{self.namespace}_pnl_total',
                'Total P&L in USD',
                ['symbol']
            )
            
            self.gauges['balance'] = Gauge(
                f'{self.namespace}_balance',
                'Account balance in USD',
                ['account_type']
            )
            
            self.gauges['equity'] = Gauge(
                f'{self.namespace}_equity',
                'Account equity in USD',
                ['account_type']
            )
            
            self.gauges['drawdown'] = Gauge(
                f'{self.namespace}_drawdown_percent',
                'Current drawdown percentage',
                ['symbol']
            )
            
            self.gauges['max_drawdown'] = Gauge(
                f'{self.namespace}_max_drawdown_percent',
                'Maximum drawdown percentage',
                ['symbol']
            )
            
            # Risk metrics
            self.gauges['risk_exposure'] = Gauge(
                f'{self.namespace}_risk_exposure_percent',
                'Current risk exposure percentage',
                ['symbol']
            )
            
            self.gauges['daily_loss'] = Gauge(
                f'{self.namespace}_daily_loss_percent',
                'Daily loss percentage',
                ['symbol']
            )
            
            self.gauges['consecutive_losses'] = Gauge(
                f'{self.namespace}_consecutive_losses',
                'Number of consecutive losses',
                ['symbol']
            )
            
            self.gauges['win_rate'] = Gauge(
                f'{self.namespace}_win_rate',
                'Win rate percentage',
                ['symbol', 'period']  # period: day, week, month, all
            )
            
            self.gauges['profit_factor'] = Gauge(
                f'{self.namespace}_profit_factor',
                'Profit factor (gross profit / gross loss)',
                ['symbol', 'period']
            )
            
            self.gauges['sharpe_ratio'] = Gauge(
                f'{self.namespace}_sharpe_ratio',
                'Sharpe ratio',
                ['symbol', 'period']
            )
            
            # Position metrics
            self.gauges['open_positions'] = Gauge(
                f'{self.namespace}_open_positions',
                'Number of open positions',
                ['symbol']
            )
            
            self.gauges['position_size'] = Gauge(
                f'{self.namespace}_position_size_lots',
                'Position size in lots',
                ['symbol', 'direction']
            )
            
            # ========== MODEL PERFORMANCE METRICS ==========
            
            self.counters['predictions_total'] = Counter(
                f'{self.namespace}_predictions_total',
                'Total model predictions',
                ['model_version', 'signal_type']
            )
            
            self.gauges['prediction_confidence'] = Gauge(
                f'{self.namespace}_prediction_confidence',
                'Model prediction confidence',
                ['model_version', 'signal_type']
            )
            
            self.histograms['prediction_confidence_dist'] = Histogram(
                f'{self.namespace}_prediction_confidence_distribution',
                'Distribution of prediction confidence',
                ['model_version'],
                buckets=[0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]
            )
            
            self.gauges['model_accuracy'] = Gauge(
                f'{self.namespace}_model_accuracy',
                'Model accuracy percentage',
                ['model_version', 'data_split']
            )
            
            self.gauges['model_f1_score'] = Gauge(
                f'{self.namespace}_model_f1_score',
                'Model F1 score',
                ['model_version', 'class']
            )
            
            # ========== SYSTEM HEALTH METRICS ==========
            
            self.gauges['system_status'] = Gauge(
                f'{self.namespace}_system_status',
                'System status (1=healthy, 0=unhealthy)',
                ['component']
            )
            
            self.gauges['kill_switch_active'] = Gauge(
                f'{self.namespace}_kill_switch_active',
                'Kill switch status (1=active, 0=inactive)'
            )
            
            self.histograms['api_latency'] = Histogram(
                f'{self.namespace}_api_latency_seconds',
                'API endpoint latency in seconds',
                ['endpoint', 'method'],
                buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1]
            )
            
            self.histograms['inference_latency'] = Histogram(
                f'{self.namespace}_inference_latency_seconds',
                'Model inference latency in seconds',
                ['model_type'],
                buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1]
            )
            
            self.counters['api_requests'] = Counter(
                f'{self.namespace}_api_requests_total',
                'Total API requests',
                ['endpoint', 'method', 'status_code']
            )
            
            self.counters['errors_total'] = Counter(
                f'{self.namespace}_errors_total',
                'Total errors by type',
                ['error_type', 'component']
            )
            
            # ========== DATA PIPELINE METRICS ==========
            
            self.gauges['data_lag_seconds'] = Gauge(
                f'{self.namespace}_data_lag_seconds',
                'Data lag from real-time in seconds',
                ['symbol', 'timeframe']
            )
            
            self.gauges['data_gap_count'] = Gauge(
                f'{self.namespace}_data_gap_count',
                'Number of data gaps detected',
                ['symbol', 'timeframe']
            )
            
            self.counters['candles_processed'] = Counter(
                f'{self.namespace}_candles_processed_total',
                'Total candles processed',
                ['symbol', 'timeframe']
            )
            
            # ========== DRIFT DETECTION METRICS ==========
            
            self.gauges['psi_score'] = Gauge(
                f'{self.namespace}_psi_score',
                'Population Stability Index for features',
                ['feature']
            )
            
            self.gauges['drift_detected'] = Gauge(
                f'{self.namespace}_drift_detected',
                'Drift detection status (1=drift, 0=stable)',
                ['feature']
            )
            
            # ========== BROKER CONNECTION METRICS ==========
            
            self.gauges['broker_connection_status'] = Gauge(
                f'{self.namespace}_broker_connection_status',
                'Broker connection status (1=connected, 0=disconnected)',
                ['broker']
            )
            
            self.histograms['broker_latency'] = Histogram(
                f'{self.namespace}_broker_latency_seconds',
                'Broker API latency in seconds',
                ['broker', 'operation'],
                buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2]
            )
            
            self.counters['order_placement_failures'] = Counter(
                f'{self.namespace}_order_placement_failures_total',
                'Order placement failures',
                ['broker', 'reason']
            )
            
            # ========== RETRAINING METRICS ==========
            
            self.gauges['last_retrain_timestamp'] = Gauge(
                f'{self.namespace}_last_retrain_timestamp',
                'Unix timestamp of last model retrain',
                ['model_type']
            )
            
            self.gauges['retrain_duration_seconds'] = Gauge(
                f'{self.namespace}_retrain_duration_seconds',
                'Duration of last retrain in seconds',
                ['model_type']
            )
            
            self.gauges['retrain_success'] = Gauge(
                f'{self.namespace}_retrain_success',
                'Retrain success status (1=success, 0=failure)',
                ['model_type']
            )
            
            # System info
            self.info = Info(f'{self.namespace}_info', 'System information')
            self.info.info({
                'version': '1.0.0',
                'environment': 'production'
            })
            
            self._initialized = True
            logger.info("Prometheus metrics initialized")
            
    def record_trade(self, symbol: str, direction: str, outcome: str, pnl: float):
        """Record a completed trade"""
        self.counters['trades_total'].labels(
            symbol=symbol, 
            direction=direction, 
            outcome=outcome
        ).inc()
        
        # Update P&L gauges
        current_pnl = self.gauges['pnl_total'].labels(symbol=symbol)._value.get() if hasattr(self.gauges['pnl_total'].labels(symbol=symbol), '_value') else 0
        self.gauges['pnl_total'].labels(symbol=symbol).set(current_pnl + pnl)
        
    def record_signal(self, symbol: str, signal_type: str):
        """Record a generated signal"""
        self.counters['signals_generated'].labels(
            symbol=symbol,
            signal_type=signal_type
        ).inc()
        
    def update_account_metrics(self, balance: float, equity: float, account_type: str = "live"):
        """Update account balance and equity metrics"""
        self.gauges['balance'].labels(account_type=account_type).set(balance)
        self.gauges['equity'].labels(account_type=account_type).set(equity)
        
    def update_drawdown(self, symbol: str, drawdown_pct: float, max_drawdown_pct: float):
        """Update drawdown metrics"""
        self.gauges['drawdown'].labels(symbol=symbol).set(drawdown_pct)
        self.gauges['max_drawdown'].labels(symbol=symbol).set(max_drawdown_pct)
        
    def update_win_rate(self, symbol: str, period: str, rate: float):
        """Update win rate metric"""
        self.gauges['win_rate'].labels(symbol=symbol, period=period).set(rate)
        
    def update_profit_factor(self, symbol: str, period: str, pf: float):
        """Update profit factor metric"""
        self.gauges['profit_factor'].labels(symbol=symbol, period=period).set(pf)
        
    def update_sharpe_ratio(self, symbol: str, period: str, sharpe: float):
        """Update Sharpe ratio metric"""
        self.gauges['sharpe_ratio'].labels(symbol=symbol, period=period).set(sharpe)
        
    def update_positions(self, symbol: str, count: int):
        """Update open positions count"""
        self.gauges['open_positions'].labels(symbol=symbol).set(count)
        
    def update_prediction(self, model_version: str, signal_type: str, confidence: float):
        """Record model prediction"""
        self.counters['predictions_total'].labels(
            model_version=model_version,
            signal_type=signal_type
        ).inc()
        self.gauges['prediction_confidence'].labels(
            model_version=model_version,
            signal_type=signal_type
        ).set(confidence)
        self.histograms['prediction_confidence_dist'].labels(
            model_version=model_version
        ).observe(confidence)
        
    def update_model_metrics(self, model_version: str, accuracy: float, f1_score: float):
        """Update model performance metrics"""
        self.gauges['model_accuracy'].labels(
            model_version=model_version,
            data_split='validation'
        ).set(accuracy)
        self.gauges['model_f1_score'].labels(
            model_version=model_version,
            class='macro'
        ).set(f1_score)
        
    def record_api_request(self, endpoint: str, method: str, status_code: int, duration: float):
        """Record API request metrics"""
        self.counters['api_requests'].labels(
            endpoint=endpoint,
            method=method,
            status_code=str(status_code)
        ).inc()
        self.histograms['api_latency'].labels(
            endpoint=endpoint,
            method=method
        ).observe(duration)
        
    def record_inference(self, model_type: str, duration: float):
        """Record inference latency"""
        self.histograms['inference_latency'].labels(model_type=model_type).observe(duration)
        
    def record_error(self, error_type: str, component: str):
        """Record error occurrence"""
        self.counters['errors_total'].labels(
            error_type=error_type,
            component=component
        ).inc()
        
    def update_system_status(self, component: str, healthy: bool):
        """Update system component health status"""
        self.gauges['system_status'].labels(component=component).set(1 if healthy else 0)
        
    def update_kill_switch(self, active: bool):
        """Update kill switch status"""
        self.gauges['kill_switch_active'].set(1 if active else 0)
        
    def update_broker_connection(self, broker: str, connected: bool, latency: float = None):
        """Update broker connection metrics"""
        self.gauges['broker_connection_status'].labels(broker=broker).set(1 if connected else 0)
        if latency is not None:
            self.histograms['broker_latency'].labels(
                broker=broker,
                operation='connection'
            ).observe(latency)
            
    def record_order_failure(self, broker: str, reason: str):
        """Record order placement failure"""
        self.counters['order_placement_failures'].labels(
            broker=broker,
            reason=reason
        ).inc()
        
    def update_data_quality(self, symbol: str, timeframe: str, lag_seconds: float, gaps: int):
        """Update data quality metrics"""
        self.gauges['data_lag_seconds'].labels(symbol=symbol, timeframe=timeframe).set(lag_seconds)
        self.gauges['data_gap_count'].labels(symbol=symbol, timeframe=timeframe).set(gaps)
        
    def record_candle_processed(self, symbol: str, timeframe: str):
        """Record processed candle"""
        self.counters['candles_processed'].labels(
            symbol=symbol,
            timeframe=timeframe
        ).inc()
        
    def update_drift_metrics(self, feature: str, psi_score: float, drift_detected: bool):
        """Update concept drift metrics"""
        self.gauges['psi_score'].labels(feature=feature).set(psi_score)
        self.gauges['drift_detected'].labels(feature=feature).set(1 if drift_detected else 0)
        
    def update_retrain_metrics(self, model_type: str, duration_seconds: float, success: bool):
        """Update model retraining metrics"""
        self.gauges['last_retrain_timestamp'].labels(model_type=model_type).set(time.time())
        self.gauges['retrain_duration_seconds'].labels(model_type=model_type).set(duration_seconds)
        self.gauges['retrain_success'].labels(model_type=model_type).set(1 if success else 0)
        
    def get_metrics(self) -> bytes:
        """Get all metrics in Prometheus format"""
        if PROMETHEUS_AVAILABLE:
            return generate_latest()
        return b"# Prometheus client not available\n"
        
    def get_metric_summary(self) -> Dict[str, Any]:
        """Get summary of key metrics as dictionary"""
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'metrics': {
                'trading': {
                    'total_trades': self._get_counter_value('trades_total'),
                    'open_positions': self._get_gauge_value('open_positions'),
                    'win_rate': self._get_gauge_value('win_rate'),
                    'profit_factor': self._get_gauge_value('profit_factor'),
                    'drawdown': self._get_gauge_value('drawdown')
                },
                'model': {
                    'accuracy': self._get_gauge_value('model_accuracy'),
                    'f1_score': self._get_gauge_value('model_f1_score'),
                    'avg_confidence': self._get_gauge_value('prediction_confidence')
                },
                'system': {
                    'kill_switch': self._get_gauge_value('kill_switch_active'),
                    'broker_connected': self._get_gauge_value('broker_connection_status'),
                    'errors_total': self._get_counter_value('errors_total')
                }
            }
        }
        
    def _get_counter_value(self, name: str) -> float:
        """Get counter value (mock implementation)"""
        # In real implementation, would query prometheus
        return 0.0
        
    def _get_gauge_value(self, name: str) -> float:
        """Get gauge value (mock implementation)"""
        return 0.0


class MetricsCollector:
    """Singleton metrics collector for easy access across the system"""
    
    _instance = None
    _metrics = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._metrics = TradingMetrics()
            self._initialized = True
            
    @property
    def metrics(self) -> TradingMetrics:
        return self._metrics
        
    def start_http_server(self, port: int = 8001):
        """Start Prometheus HTTP server for metrics scraping"""
        if PROMETHEUS_AVAILABLE:
            from prometheus_client import start_http_server
            start_http_server(port)
            logger.info(f"Prometheus metrics server started on port {port}")
        else:
            logger.warning("Cannot start Prometheus server - client not installed")
            
    def get_all_metrics(self) -> bytes:
        """Get all metrics for scraping"""
        return self._metrics.get_metrics()


# Global instance
metrics_collector = MetricsCollector()
