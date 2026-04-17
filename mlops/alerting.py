"""Alerting system for Slack, Email, and webhooks"""

import logging
import requests
import smtplib
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import threading
from queue import Queue

logger = logging.getLogger(__name__)


class AlertManager:
    """Central alert manager with multiple channels"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config = self._load_config(config_path)
        self.alert_queue = Queue()
        self.channels = []
        
        # Initialize alert channels
        if self.config.get('slack', {}).get('enabled'):
            self.channels.append(SlackAlert(self.config['slack']))
            
        if self.config.get('email', {}).get('enabled'):
            self.channels.append(EmailAlert(self.config['email']))
            
        if self.config.get('webhook', {}).get('enabled'):
            self.channels.append(WebhookAlert(self.config['webhook']))
            
        # Start background worker
        self.running = True
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()
        
        logger.info(f"Alert manager initialized with {len(self.channels)} channels")
        
    def _load_config(self, config_path: Optional[Path]) -> Dict:
        """Load alert configuration"""
        
        default_config = {
            'slack': {
                'enabled': False,
                'webhook_url': None,
                'channel': '#trading-alerts'
            },
            'email': {
                'enabled': False,
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': 587,
                'username': None,
                'password': None,
                'from_email': None,
                'to_emails': []
            },
            'webhook': {
                'enabled': False,
                'url': None,
                'headers': {}
            }
        }
        
        if config_path and config_path.exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
                
        return default_config
        
    def send_alert(
        self,
        title: str,
        message: str,
        severity: str = "info",
        metadata: Dict = None
    ):
        """Send alert through all configured channels"""
        
        alert = {
            'title': title,
            'message': message,
            'severity': severity,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        self.alert_queue.put(alert)
        logger.info(f"Alert queued: {title} [{severity}]")
        
    def _process_queue(self):
        """Process alerts from queue"""
        
        while self.running:
            try:
                alert = self.alert_queue.get(timeout=1)
                
                for channel in self.channels:
                    try:
                        channel.send(alert)
                    except Exception as e:
                        logger.error(f"Failed to send alert via {channel.__class__.__name__}: {e}")
                        
            except:
                pass
                
    def stop(self):
        """Stop alert manager"""
        self.running = False


class SlackAlert:
    """Slack webhook alert channel"""
    
    def __init__(self, config: Dict):
        self.webhook_url = config.get('webhook_url')
        self.channel = config.get('channel', '#trading-alerts')
        self.username = config.get('username', 'Trading AI System')
        
        if not self.webhook_url:
            logger.warning("Slack webhook URL not configured")
            
    def send(self, alert: Dict):
        """Send alert to Slack"""
        
        if not self.webhook_url:
            return
            
        # Color based on severity
        colors = {
            'info': '#36a64f',
            'warning': '#ffcc00',
            'error': '#ff0000',
            'critical': '#ff0000',
            'success': '#36a64f'
        }
        
        payload = {
            'channel': self.channel,
            'username': self.username,
            'attachments': [{
                'color': colors.get(alert['severity'], '#36a64f'),
                'title': alert['title'],
                'text': alert['message'],
                'fields': [
                    {
                        'title': 'Severity',
                        'value': alert['severity'].upper(),
                        'short': True
                    },
                    {
                        'title': 'Time',
                        'value': alert['timestamp'],
                        'short': True
                    }
                ],
                'footer': 'Trading AI System',
                'ts': int(datetime.now().timestamp())
            }]
        }
        
        # Add metadata fields
        if alert.get('metadata'):
            for key, value in list(alert['metadata'].items())[:5]:
                payload['attachments'][0]['fields'].append({
                    'title': key.replace('_', ' ').title(),
                    'value': str(value),
                    'short': True
                })
                
        response = requests.post(self.webhook_url, json=payload)
        response.raise_for_status()
        
        logger.debug(f"Slack alert sent: {alert['title']}")
        
    def send_trade_alert(self, trade: Dict):
        """Send specific trade alert"""
        
        direction_emoji = "🟢" if trade.get('direction') == 'BUY' else "🔴"
        
        alert = {
            'title': f"{direction_emoji} Trade Executed: {trade.get('direction', 'UNKNOWN')}",
            'message': (
                f"**Symbol:** {trade.get('symbol', 'XAUUSD')}\n"
                f"**Lot Size:** {trade.get('lot_size', 0)}\n"
                f"**Entry:** {trade.get('entry_price', 0):.2f}\n"
                f"**SL:** {trade.get('stop_loss', 0):.2f}\n"
                f"**TP:** {trade.get('take_profit', 0):.2f}\n"
                f"**Confidence:** {trade.get('confidence', 0):.1%}"
            ),
            'severity': 'info',
            'metadata': trade
        }
        
        self.send(alert)


class EmailAlert:
    """Email alert channel"""
    
    def __init__(self, config: Dict):
        self.smtp_server = config.get('smtp_server', 'smtp.gmail.com')
        self.smtp_port = config.get('smtp_port', 587)
        self.username = config.get('username')
        self.password = config.get('password')
        self.from_email = config.get('from_email')
        self.to_emails = config.get('to_emails', [])
        
    def send(self, alert: Dict):
        """Send email alert"""
        
        if not self.username or not self.password:
            logger.warning("Email credentials not configured")
            return
            
        if not self.to_emails:
            logger.warning("No recipient emails configured")
            return
            
        # Create message
        msg = MIMEMultipart()
        msg['From'] = self.from_email or self.username
        msg['To'] = ', '.join(self.to_emails)
        msg['Subject'] = f"[{alert['severity'].upper()}] Trading AI: {alert['title']}"
        
        # Create HTML body
        html_body = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                .alert {{ padding: 10px; margin: 10px 0; }}
                .info {{ background-color: #d4edda; border-left: 4px solid #28a745; }}
                .warning {{ background-color: #fff3cd; border-left: 4px solid #ffc107; }}
                .error {{ background-color: #f8d7da; border-left: 4px solid #dc3545; }}
                .critical {{ background-color: #f8d7da; border-left: 4px solid #dc3545; }}
                .success {{ background-color: #d4edda; border-left: 4px solid #28a745; }}
                .metadata {{ background-color: #f8f9fa; padding: 10px; margin-top: 10px; }}
            </style>
        </head>
        <body>
            <div class="alert {alert['severity']}">
                <h2>{alert['title']}</h2>
                <p>{alert['message']}</p>
                <p><strong>Time:</strong> {alert['timestamp']}</p>
            </div>
        """
        
        # Add metadata
        if alert.get('metadata'):
            html_body += '<div class="metadata"><h3>Additional Information:</h3><ul>'
            for key, value in alert['metadata'].items():
                html_body += f"<li><strong>{key}:</strong> {value}</li>"
            html_body += '</ul></div>'
            
        html_body += """
        </body>
        </html>
        """
        
        msg.attach(MIMEText(html_body, 'html'))
        
        # Send email
        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
                
            logger.debug(f"Email alert sent to {len(self.to_emails)} recipients")
            
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            raise
            
    def send_daily_report(self, report: Dict):
        """Send daily performance report"""
        
        alert = {
            'title': 'Daily Trading Report',
            'message': (
                f"**Date:** {report.get('date', datetime.now().date())}\n"
                f"**Trades:** {report.get('total_trades', 0)}\n"
                f"**Win Rate:** {report.get('win_rate', 0):.1%}\n"
                f"**PnL:** ${report.get('pnl', 0):.2f}\n"
                f"**Drawdown:** {report.get('max_drawdown', 0):.1%}"
            ),
            'severity': 'info',
            'metadata': report
        }
        
        self.send(alert)


class WebhookAlert:
    """Generic webhook alert channel"""
    
    def __init__(self, config: Dict):
        self.url = config.get('url')
        self.headers = config.get('headers', {})
        self.method = config.get('method', 'POST')
        
    def send(self, alert: Dict):
        """Send alert via webhook"""
        
        if not self.url:
            return
            
        response = requests.request(
            method=self.method,
            url=self.url,
            headers=self.headers,
            json=alert,
            timeout=5
        )
        response.raise_for_status()
        
        logger.debug(f"Webhook alert sent to {self.url}")


class TradingAlerts:
    """Pre-defined trading-specific alerts"""
    
    def __init__(self, alert_manager: AlertManager):
        self.alerts = alert_manager
        
    def kill_switch_triggered(self, reason: str, daily_pnl: float):
        """Alert when kill switch is triggered"""
        self.alerts.send_alert(
            title="⚠️ KILL SWITCH TRIGGERED",
            message=f"Trading has been halted.\nReason: {reason}\nDaily P&L: ${daily_pnl:.2f}",
            severity="critical",
            metadata={"reason": reason, "daily_pnl": daily_pnl}
        )
        
    def trade_executed(self, trade: Dict):
        """Alert when trade is executed"""
        direction = trade.get('direction', 'UNKNOWN')
        emoji = "🟢" if direction == "BUY" else "🔴"
        
        self.alerts.send_alert(
            title=f"{emoji} Trade Executed: {direction}",
            message=(
                f"Symbol: {trade.get('symbol', 'XAUUSD')}\n"
                f"Lot Size: {trade.get('lot_size', 0)}\n"
                f"Entry: {trade.get('entry_price', 0):.2f}\n"
                f"Confidence: {trade.get('confidence', 0):.1%}"
            ),
            severity="info",
            metadata=trade
        )
        
    def trade_closed(self, trade: Dict):
        """Alert when trade is closed"""
        pnl = trade.get('pnl', 0)
        emoji = "✅" if pnl > 0 else "❌"
        
        self.alerts.send_alert(
            title=f"{emoji} Trade Closed",
            message=(
                f"Symbol: {trade.get('symbol', 'XAUUSD')}\n"
                f"PnL: ${pnl:.2f}\n"
                f"Return: {trade.get('return_pct', 0):.1%}\n"
                f"Duration: {trade.get('duration_minutes', 0)} min"
            ),
            severity="success" if pnl > 0 else "warning",
            metadata=trade
        )
        
    def model_retrained(self, model_version: str, metrics: Dict):
        """Alert when model is retrained"""
        self.alerts.send_alert(
            title="🔄 Model Retrained",
            message=(
                f"New model version: {model_version}\n"
                f"Profit Factor: {metrics.get('profit_factor', 0):.2f}\n"
                f"Win Rate: {metrics.get('win_rate', 0):.1%}\n"
                f"Sharpe: {metrics.get('sharpe_ratio', 0):.2f}"
            ),
            severity="success",
            metadata={"model_version": model_version, **metrics}
        )
        
    def champion_promoted(self, model_version: str, metrics: Dict):
        """Alert when new champion is promoted"""
        self.alerts.send_alert(
            title="🏆 New Champion Model",
            message=(
                f"Model {model_version} has been promoted to champion!\n"
                f"Profit Factor: {metrics.get('profit_factor', 0):.2f}\n"
                f"Win Rate: {metrics.get('win_rate', 0):.1%}"
            ),
            severity="success",
            metadata={"model_version": model_version, **metrics}
        )
        
    def drift_detected(self, drift_report: Dict):
        """Alert when drift is detected"""
        self.alerts.send_alert(
            title="📊 Concept Drift Detected",
            message=(
                f"Drift Score: {drift_report.get('drift_score', 0):.2f}\n"
                f"Affected Features: {len(drift_report.get('drifted_features', []))}\n"
                f"Recommendation: Retraining may be needed"
            ),
            severity="warning",
            metadata=drift_report
        )
        
    def broker_disconnected(self, broker_name: str):
        """Alert when broker disconnects"""
        self.alerts.send_alert(
            title="🔌 Broker Disconnected",
            message=f"Connection to {broker_name} has been lost. Attempting to reconnect...",
            severity="critical",
            metadata={"broker": broker_name}
        )
        
    def daily_summary(self, stats: Dict):
        """Send daily performance summary"""
        self.alerts.send_alert(
            title="📈 Daily Trading Summary",
            message=(
                f"Date: {stats.get('date', 'Unknown')}\n"
                f"Trades: {stats.get('total_trades', 0)}\n"
                f"Win Rate: {stats.get('win_rate', 0):.1%}\n"
                f"PnL: ${stats.get('pnl', 0):.2f}\n"
                f"Max Drawdown: {stats.get('max_drawdown', 0):.1%}\n"
                f"Sharpe: {stats.get('sharpe_ratio', 0):.2f}"
            ),
            severity="info",
            metadata=stats
        )
