"""Grafana dashboard management"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import requests

logger = logging.getLogger(__name__)


class GrafanaDashboard:
    """Manage Grafana dashboards via API"""
    
    def __init__(
        self,
        grafana_url: str = "http://localhost:3000",
        api_key: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None
    ):
        self.grafana_url = grafana_url.rstrip('/')
        self.session = requests.Session()
        
        if api_key:
            self.session.headers.update({'Authorization': f'Bearer {api_key}'})
        elif username and password:
            self.session.auth = (username, password)
            
    def load_dashboard(self, dashboard_path: Path) -> Dict:
        """Load dashboard from JSON file"""
        with open(dashboard_path, 'r') as f:
            return json.load(f)
            
    def upload_dashboard(self, dashboard: Dict, overwrite: bool = True) -> bool:
        """Upload dashboard to Grafana"""
        
        url = f"{self.grafana_url}/api/dashboards/db"
        
        payload = {
            "dashboard": dashboard.get('dashboard', dashboard),
            "overwrite": overwrite,
            "message": dashboard.get('message', 'Updated via API')
        }
        
        try:
            response = self.session.post(url, json=payload)
            response.raise_for_status()
            logger.info(f"Dashboard uploaded successfully: {response.json().get('uid')}")
            return True
        except Exception as e:
            logger.error(f"Failed to upload dashboard: {e}")
            return False
            
    def get_dashboard(self, uid: str) -> Optional[Dict]:
        """Get dashboard by UID"""
        
        url = f"{self.grafana_url}/api/dashboards/uid/{uid}"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get dashboard: {e}")
            return None
            
    def delete_dashboard(self, uid: str) -> bool:
        """Delete dashboard by UID"""
        
        url = f"{self.grafana_url}/api/dashboards/uid/{uid}"
        
        try:
            response = self.session.delete(url)
            response.raise_for_status()
            logger.info(f"Dashboard {uid} deleted")
            return True
        except Exception as e:
            logger.error(f"Failed to delete dashboard: {e}")
            return False
            
    def create_alert_rule(
        self,
        name: str,
        query: str,
        condition: str,
        threshold: float,
        duration: str = "5m"
    ) -> bool:
        """Create a Prometheus alert rule via Grafana"""
        
        url = f"{self.grafana_url}/api/ruler/grafana/api/v1/rules"
        
        rule = {
            "name": name,
            "condition": condition,
            "data": [{
                "refId": "A",
                "relativeTimeRange": {"from": 300, "to": 0},
                "datasourceUid": "prometheus",
                "model": {
                    "expr": query,
                    "intervalMs": 1000,
                    "maxDataPoints": 43200,
                    "refId": "A"
                }
            }],
            "noDataState": "NoData",
            "execErrState": "Alerting",
            "for": duration,
            "annotations": {
                "summary": f"Alert: {name}",
                "description": f"Alert triggered when {condition} > {threshold}"
            },
            "labels": {
                "severity": "warning",
                "component": "trading_ai"
            }
        }
        
        try:
            response = self.session.post(url, json=rule)
            response.raise_for_status()
            logger.info(f"Alert rule {name} created")
            return True
        except Exception as e:
            logger.error(f"Failed to create alert rule: {e}")
            return False
            
    @staticmethod
    def create_default_alerts():
        """Create default alert rules for trading system"""
        
        alerts = [
            {
                "name": "drawdown_exceeded",
                "query": "trading_ai_drawdown_percent > 10",
                "condition": "drawdown",
                "threshold": 10,
                "duration": "1m"
            },
            {
                "name": "daily_loss_exceeded",
                "query": "trading_ai_daily_loss_percent > 3",
                "condition": "daily loss",
                "threshold": 3,
                "duration": "1m"
            },
            {
                "name": "kill_switch_active",
                "query": "trading_ai_kill_switch_active == 1",
                "condition": "kill switch",
                "threshold": 1,
                "duration": "0s"
            },
            {
                "name": "broker_disconnected",
                "query": "trading_ai_broker_connection_status == 0",
                "condition": "broker connection",
                "threshold": 0,
                "duration": "1m"
            },
            {
                "name": "high_error_rate",
                "query": "rate(trading_ai_errors_total[5m]) > 10",
                "condition": "error rate",
                "threshold": 10,
                "duration": "2m"
            },
            {
                "name": "drift_detected",
                "query": "trading_ai_drift_detected == 1",
                "condition": "drift",
                "threshold": 1,
                "duration": "5m"
            },
            {
                "name": "low_prediction_confidence",
                "query": "trading_ai_prediction_confidence < 0.6",
                "condition": "confidence",
                "threshold": 0.6,
                "duration": "15m"
            },
            {
                "name": "model_performance_degraded",
                "query": "trading_ai_model_accuracy < 0.55",
                "condition": "accuracy",
                "threshold": 0.55,
                "duration": "1h"
            }
        ]
        
        return alerts
