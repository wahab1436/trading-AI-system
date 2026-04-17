"""Alerting system for Trading AI - Slack, Email, Webhook"""

import os
import json
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import threading
import queue

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

logger = logging.getLogger(__name__)


class AlertSeverity:
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertChannel:
    """Alert channel types"""
    SLACK = "slack"
    EMAIL = "email"
    WEBHOOK = "webhook"
    TELEGRAM = "telegram"
    PUSHOVER = "pushover"


class AlertManager:
    """Centralized alert management system"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.channels = {}
        self.alert_queue = queue.Queue()
        self.running = False
        self.worker_thread = None
        
        # Initialize channels
        self._init_channels()
        
    def _init_channels(self):
        """Initialize alert channels from config"""
        
        # Slack
        slack_webhook = self.config.get('slack_webhook') or os.getenv('SLACK_WEBHOOK_URL')
        if slack_webhook:
            self.channels[AlertChannel.SLACK] = SlackAlertChannel(slack_webhook)
            
        # Email
        email_config = self.config.get('email', {})
        if email_config.get('enabled'):
            self.channels[AlertChannel.EMAIL] = EmailAlertChannel(email_config)
            
        # Webhook
        webhook_url = self.config.get('webhook_url')
        if webhook_url:
            self.channels[AlertChannel.WEBHOOK] = WebhookAlertChannel(webhook_url)
            
        # Telegram
        telegram_bot = self.config.get('telegram_bot_token') or os.getenv('TELEGRAM_BOT_TOKEN')
        telegram_chat = self.config.get('telegram_chat_id') or os.getenv('TELEGRAM_CHAT_ID')
        if telegram_bot and telegram_chat:
            self.channels[AlertChannel.TELEGRAM] = TelegramAlertChannel(telegram_bot, telegram_chat)
            
        # Pushover
        pushover_token = self.config.get('pushover_token') or os.getenv('PUSHOVER_TOKEN')
        pushover_user = self.config.get('pushover_user_key') or os.getenv('PUSHOVER_USER_KEY')
        if pushover_token and pushover_user:
            self.channels[AlertChannel.PUSHOVER] = PushoverAlertChannel(pushover_token, pushover_user)
            
    def start(self):
        """Start alert processing worker"""
        if not self.running:
            self.running = True
            self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
            self.worker_thread.start()
            logger.info("Alert manager started")
            
    def stop(self):
        """Stop alert processing"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
            logger.info("Alert manager stopped")
            
    def send_alert(
        self,
        title: str,
        message: str,
        severity: str = AlertSeverity.INFO,
        channels: Optional[List[str]] = None,
        data: Optional[Dict] = None
    ):
        """Send alert through specified channels"""
        
        alert = {
            'title': title,
            'message': message,
            'severity': severity,
            'timestamp': datetime.utcnow().isoformat(),
            'data': data or {}
        }
        
        # Add to queue for async processing
        self.alert_queue.put((alert, channels))
        
        # Log critical alerts immediately
        if severity == AlertSeverity.CRITICAL:
            logger.critical(f"ALERT: {title} - {message}")
            
    def _process_queue(self):
        """Process alerts from queue"""
        while self.running:
            try:
                alert, channels = self.alert_queue.get(timeout=1)
                self._dispatch_alert(alert, channels)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Alert processing error: {e}")
                
    def _dispatch_alert(self, alert: Dict, channels: Optional[List[str]]):
        """Dispatch alert to channels"""
        
        target_channels = channels or list(self.channels.keys())
        
        for channel_name in target_channels:
            if channel_name in self.channels:
                try:
                    self.channels[channel_name].send(alert)
                    logger.debug(f"Alert sent to {channel_name}")
                except Exception as e:
                    logger.error(f"Failed to send alert to {channel_name}: {e}")
                    
    def alert_trade_executed(self, trade: Dict):
        """Alert when trade is executed"""
        self.send_alert(
            title="Trade Executed",
            message=f"{trade.get('direction', 'UNKNOWN')} {trade.get('quantity', 0)} {trade.get('symbol', '')} @ {trade.get('price', 0)}",
            severity=AlertSeverity.INFO,
            channels=[AlertChannel.SLACK]
        )
        
    def alert_trade_closed(self, trade: Dict):
        """Alert when trade is closed"""
        pnl = trade.get('pnl', 0)
        emoji = "✅" if pnl > 0 else "❌"
        
        self.send_alert(
            title=f"{emoji} Trade Closed",
            message=f"{trade.get('symbol', '')}: P&L = ${pnl:.2f} ({trade.get('pnl_pips', 0):.1f} pips)",
            severity=AlertSeverity.INFO if pnl > 0 else AlertSeverity.WARNING,
            channels=[AlertChannel.SLACK]
        )
        
    def alert_kill_switch(self, reason: str):
        """Alert when kill switch activates"""
        self.send_alert(
            title="🚨 KILL SWITCH ACTIVATED 🚨",
            message=f"Trading stopped. Reason: {reason}",
            severity=AlertSeverity.CRITICAL,
            channels=[AlertChannel.SLACK, AlertChannel.EMAIL]
        )
        
    def alert_daily_loss(self, loss_pct: float, loss_amount: float):
        """Alert when daily loss limit reached"""
        self.send_alert(
            title="Daily Loss Limit Reached",
            message=f"Loss: {loss_pct:.2f}% (${loss_amount:.2f})",
            severity=AlertSeverity.ERROR,
            channels=[AlertChannel.SLACK, AlertChannel.EMAIL]
        )
        
    def alert_drawdown(self, drawdown_pct: float):
        """Alert when drawdown exceeds threshold"""
        self.send_alert(
            title="High Drawdown Warning",
            message=f"Current drawdown: {drawdown_pct:.2f}%",
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.SLACK]
        )
        
    def alert_drift_detected(self, feature: str, psi_score: float):
        """Alert when concept drift detected"""
        self.send_alert(
            title="Concept Drift Detected",
            message=f"Feature '{feature}' PSI = {psi_score:.3f}",
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.SLACK]
        )
        
    def alert_retrain_complete(self, model_type: str, success: bool, metrics: Dict = None):
        """Alert when model retraining completes"""
        if success:
            self.send_alert(
                title="Model Retrain Complete",
                message=f"{model_type} retrained successfully. New F1: {metrics.get('f1', 'N/A')}",
                severity=AlertSeverity.INFO,
                channels=[AlertChannel.SLACK]
            )
        else:
            self.send_alert(
                title="Model Retrain Failed",
                message=f"{model_type} retraining failed",
                severity=AlertSeverity.ERROR,
                channels=[AlertChannel.SLACK, AlertChannel.EMAIL]
            )
            
    def alert_broker_disconnect(self, broker: str):
        """Alert when broker disconnects"""
        self.send_alert(
            title="Broker Disconnected",
            message=f"Connection to {broker} lost",
            severity=AlertSeverity.ERROR,
            channels=[AlertChannel.SLACK, AlertChannel.EMAIL]
        )
        
    def alert_system_error(self, error: str, component: str):
        """Alert on system error"""
        self.send_alert(
            title=f"System Error: {component}",
            message=error[:500],
            severity=AlertSeverity.ERROR,
            channels=[AlertChannel.SLACK, AlertChannel.EMAIL]
        )


class BaseAlertChannel:
    """Base class for alert channels"""
    
    def send(self, alert: Dict):
        raise NotImplementedError


class SlackAlertChannel(BaseAlertChannel):
    """Send alerts to Slack"""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
        
    def send(self, alert: Dict):
        if not REQUESTS_AVAILABLE:
            logger.warning("requests not available for Slack alerts")
            return
            
        # Color based on severity
        colors = {
            AlertSeverity.INFO: "#36a64f",
            AlertSeverity.WARNING: "#ffcc00",
            AlertSeverity.ERROR: "#ff4444",
            AlertSeverity.CRITICAL: "#ff0000"
        }
        
        emojis = {
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.ERROR: "❌",
            AlertSeverity.CRITICAL: "🚨"
        }
        
        payload = {
            "attachments": [
                {
                    "color": colors.get(alert['severity'], "#cccccc"),
                    "title": f"{emojis.get(alert['severity'], '')} {alert['title']}",
                    "text": alert['message'],
                    "fields": [
                        {
                            "title": "Severity",
                            "value": alert['severity'].upper(),
                            "short": True
                        },
                        {
                            "title": "Time",
                            "value": alert['timestamp'],
                            "short": True
                        }
                    ],
                    "footer": "Trading AI System",
                    "ts": int(datetime.utcnow().timestamp())
                }
            ]
        }
        
        # Add data fields if present
        if alert.get('data'):
            for key, value in list(alert['data'].items())[:5]:
                payload['attachments'][0]['fields'].append({
                    "title": key.replace('_', ' ').title(),
                    "value": str(value),
                    "short": True
                })
                
        response = requests.post(self.webhook_url, json=payload)
        response.raise_for_status()


class EmailAlertChannel(BaseAlertChannel):
    """Send alerts via email"""
    
    def __init__(self, config: Dict):
        self.smtp_host = config.get('smtp_host', 'smtp.gmail.com')
        self.smtp_port = config.get('smtp_port', 587)
        self.sender = config.get('sender')
        self.password = config.get('password')
        self.recipients = config.get('recipients', [])
        
    def send(self, alert: Dict):
        # Only send critical and error alerts via email
        if alert['severity'] not in [AlertSeverity.ERROR, AlertSeverity.CRITICAL]:
            return
            
        msg = MIMEMultipart()
        msg['From'] = self.sender
        msg['To'] = ', '.join(self.recipients)
        msg['Subject'] = f"[{alert['severity'].upper()}] Trading AI: {alert['title']}"
        
        body = f"""
        Trading AI System Alert
        
        Severity: {alert['severity'].upper()}
        Time: {alert['timestamp']}
        Title: {alert['title']}
        
        Message:
        {alert['message']}
        
        Additional Data:
        {json.dumps(alert.get('data', {}), indent=2)}
        
        ---
        Trading AI System
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender, self.password)
                server.send_message(msg)
            logger.info(f"Email alert sent to {self.recipients}")
        except Exception as e:
            logger.error(f"Failed to send email: {e}")


class WebhookAlertChannel(BaseAlertChannel):
    """Send alerts to generic webhook"""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
        
    def send(self, alert: Dict):
        if not REQUESTS_AVAILABLE:
            return
            
        response = requests.post(
            self.webhook_url,
            json=alert,
            timeout=5
        )
        response.raise_for_status()


class TelegramAlertChannel(BaseAlertChannel):
    """Send alerts to Telegram"""
    
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.api_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        
    def send(self, alert: Dict):
        if not REQUESTS_AVAILABLE:
            return
            
        emojis = {
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.ERROR: "❌",
            AlertSeverity.CRITICAL: "🚨"
        }
        
        message = f"""
{emojis.get(alert['severity'], '')} *{alert['title']}*
{alert['message']}

Severity: {alert['severity'].upper()}
Time: {alert['timestamp']}
        """
        
        payload = {
            'chat_id': self.chat_id,
            'text': message,
            'parse_mode': 'Markdown'
        }
        
        response = requests.post(self.api_url, json=payload)
        response.raise_for_status()


class PushoverAlertChannel(BaseAlertChannel):
    """Send alerts to Pushover"""
    
    def __init__(self, api_token: str, user_key: str):
        self.api_token = api_token
        self.user_key = user_key
        self.api_url = "https://api.pushover.net/1/messages.json"
        
    def send(self, alert: Dict):
        if not REQUESTS_AVAILABLE:
            return
            
        priorities = {
            AlertSeverity.INFO: 0,
            AlertSeverity.WARNING: 0,
            AlertSeverity.ERROR: 1,
            AlertSeverity.CRITICAL: 2
        }
        
        payload = {
            'token': self.api_token,
            'user': self.user_key,
            'title': alert['title'],
            'message': alert['message'],
            'priority': priorities.get(alert['severity'], 0),
            'timestamp': int(datetime.utcnow().timestamp())
        }
        
        response = requests.post(self.api_url, data=payload)
        response.raise_for_status()


# Global alert manager instance
alert_manager = AlertManager()
