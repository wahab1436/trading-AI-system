"""Trade logging and storage module"""

import logging
import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd

from .schema import (
    SignalLog, TradeRecord, TradeStatus, 
    ExitReason, DailySummary, TradeJournalSchema
)

logger = logging.getLogger(__name__)


class TradeLogger:
    """Main logger for recording trades and signals"""
    
    def __init__(self, db_path: str = "data/trade_journal.db", csv_backup: bool = True):
        self.db_path = db_path
        self.csv_backup = csv_backup
        self.schema = TradeJournalSchema(db_path)
        self.current_trades: Dict[str, TradeRecord] = {}
        
        # Create backup directory
        if csv_backup:
            self.backup_dir = Path("data/trade_backups")
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            
    def log_signal(self, signal: SignalLog) -> str:
        """Log an AI signal to the journal"""
        
        conn = sqlite3.connect(self.db_path)
        
        try:
            conn.execute('''
                INSERT INTO signals (
                    signal_id, timestamp, symbol, timeframe, direction,
                    buy_prob, sell_prob, notrade_prob, cnn_embedding_hash,
                    model_version, shap_top5, htf_bias, nearest_ob_dist,
                    fvg_present, bos_recent, confidence_score, signal_strength
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal.signal_id, signal.timestamp.isoformat(), signal.symbol,
                signal.timeframe, signal.direction, signal.buy_prob,
                signal.sell_prob, signal.notrade_prob, signal.cnn_embedding_hash,
                signal.model_version, json.dumps(signal.shap_top5),
                signal.htf_bias, signal.nearest_ob_dist,
                1 if signal.fvg_present else 0,
                1 if signal.bos_recent else 0,
                signal.confidence_score, signal.signal_strength
            ))
            
            conn.commit()
            logger.info(f"Logged signal {signal.signal_id}: direction={signal.direction}")
            
        except Exception as e:
            logger.error(f"Failed to log signal: {e}")
            raise
        finally:
            conn.close()
            
        return signal.signal_id
        
    def create_trade_from_signal(self, signal: SignalLog) -> TradeRecord:
        """Create a trade record from a signal"""
        
        trade = TradeRecord(
            signal_id=signal.signal_id,
            symbol=signal.symbol,
            direction=signal.direction,
            status=TradeStatus.PENDING,
            signal_time=signal.timestamp,
            model_version=signal.model_version
        )
        
        self.current_trades[trade.trade_id] = trade
        return trade
        
    def open_trade(
        self,
        trade_id: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        lot_size: float,
        order_id: str,
        spread_at_entry: float = 0,
        execution_ms: int = 0,
        position_size_usd: float = 0,
        risk_percent: float = 0.01
    ) -> bool:
        """Mark a trade as opened/executed"""
        
        if trade_id not in self.current_trades:
            logger.error(f"Trade {trade_id} not found")
            return False
            
        trade = self.current_trades[trade_id]
        trade.status = TradeStatus.OPEN
        trade.entry_time = datetime.utcnow()
        trade.entry_price = entry_price
        trade.stop_loss = stop_loss
        trade.take_profit = take_profit
        trade.lot_size = lot_size
        trade.order_id = order_id
        trade.spread_at_entry = spread_at_entry
        trade.execution_ms = execution_ms
        trade.position_size_usd = position_size_usd
        trade.risk_percent = risk_percent
        
        # Calculate risk-reward
        pip_size = 0.01 if trade.symbol == "XAUUSD" else 0.0001
        sl_distance = abs(entry_price - stop_loss) / pip_size
        tp_distance = abs(take_profit - entry_price) / pip_size
        if sl_distance > 0:
            trade.risk_reward_ratio = tp_distance / sl_distance
            
        # Save to database
        self._save_trade(trade)
        
        logger.info(f"Opened trade {trade_id}: {lot_size} {trade.symbol} @ {entry_price}")
        return True
        
    def close_trade(
        self,
        trade_id: str,
        exit_price: float,
        exit_reason: ExitReason,
        notes: str = ""
    ) -> bool:
        """Close an open trade"""
        
        if trade_id not in self.current_trades:
            logger.error(f"Trade {trade_id} not found")
            return False
            
        trade = self.current_trades[trade_id]
        trade.exit_time = datetime.utcnow()
        trade.exit_price = exit_price
        trade.exit_reason = exit_reason
        trade.notes = notes
        trade.status = TradeStatus.CLOSED
        
        # Calculate metrics
        trade.calculate_metrics()
        
        # Update database
        self._update_closed_trade(trade)
        
        # Log to CSV backup
        if self.csv_backup:
            self._backup_to_csv(trade)
            
        logger.info(f"Closed trade {trade_id}: P&L=${trade.pnl_dollars:.2f}, R={trade.r_multiple:.2f}")
        return True
        
    def _save_trade(self, trade: TradeRecord):
        """Save trade to database"""
        
        conn = sqlite3.connect(self.db_path)
        
        try:
            conn.execute('''
                INSERT OR REPLACE INTO trades (
                    trade_id, signal_id, symbol, direction, status,
                    signal_time, entry_time, entry_price, stop_loss,
                    take_profit, order_id, lot_size, spread_at_entry,
                    execution_ms, risk_percent, risk_reward_ratio,
                    position_size_usd, model_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade.trade_id, trade.signal_id, trade.symbol,
                trade.direction.value, trade.status.value,
                trade.signal_time.isoformat() if trade.signal_time else None,
                trade.entry_time.isoformat() if trade.entry_time else None,
                trade.entry_price, trade.stop_loss, trade.take_profit,
                trade.order_id, trade.lot_size, trade.spread_at_entry,
                trade.execution_ms, trade.risk_percent,
                trade.risk_reward_ratio, trade.position_size_usd,
                trade.model_version
            ))
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"Failed to save trade: {e}")
        finally:
            conn.close()
            
    def _update_closed_trade(self, trade: TradeRecord):
        """Update trade with closing information"""
        
        conn = sqlite3.connect(self.db_path)
        
        try:
            conn.execute('''
                UPDATE trades SET
                    status = ?, exit_time = ?, exit_price = ?,
                    exit_reason = ?, pnl_pips = ?, pnl_dollars = ?,
                    r_multiple = ?, duration_minutes = ?, commission = ?,
                    swap = ?, notes = ?
                WHERE trade_id = ?
            ''', (
                trade.status.value,
                trade.exit_time.isoformat() if trade.exit_time else None,
                trade.exit_price,
                trade.exit_reason.value,
                trade.pnl_pips, trade.pnl_dollars, trade.r_multiple,
                trade.duration_minutes, trade.commission, trade.swap,
                trade.notes, trade.trade_id
            ))
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"Failed to update trade: {e}")
        finally:
            conn.close()
            
    def _backup_to_csv(self, trade: TradeRecord):
        """Backup trade to CSV file"""
        
        backup_file = self.backup_dir / f"trades_{datetime.now().strftime('%Y%m')}.csv"
        
        df = pd.DataFrame([trade.to_dict()])
        
        if backup_file.exists():
            existing = pd.read_csv(backup_file)
            df = pd.concat([existing, df], ignore_index=True)
            
        df.to_csv(backup_file, index=False)
        
    def get_trade(self, trade_id: str) -> Optional[TradeRecord]:
        """Retrieve a trade by ID"""
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        try:
            cursor = conn.execute('SELECT * FROM trades WHERE trade_id = ?', (trade_id,))
            row = cursor.fetchone()
            
            if row:
                return TradeRecord.from_dict(dict(row))
                
        except Exception as e:
            logger.error(f"Failed to get trade: {e}")
        finally:
            conn.close()
            
        return None
        
    def get_trades(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        symbol: Optional[str] = None,
        status: Optional[TradeStatus] = None
    ) -> List[TradeRecord]:
        """Get filtered list of trades"""
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        query = "SELECT * FROM trades WHERE 1=1"
        params = []
        
        if start_date:
            query += " AND entry_time >= ?"
            params.append(start_date.isoformat())
        if end_date:
            query += " AND entry_time <= ?"
            params.append(end_date.isoformat())
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if status:
            query += " AND status = ?"
            params.append(status.value)
            
        query += " ORDER BY entry_time DESC"
        
        try:
            cursor = conn.execute(query, params)
            trades = [TradeRecord.from_dict(dict(row)) for row in cursor.fetchall()]
            return trades
            
        except Exception as e:
            logger.error(f"Failed to get trades: {e}")
            return []
        finally:
            conn.close()
            
    def get_signals(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        min_confidence: float = 0.0
    ) -> List[SignalLog]:
        """Get filtered list of signals"""
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        query = "SELECT * FROM signals WHERE 1=1"
        params = []
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.isoformat())
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date.isoformat())
        if min_confidence > 0:
            query += " AND confidence_score >= ?"
            params.append(min_confidence)
            
        query += " ORDER BY timestamp DESC"
        
        try:
            cursor = conn.execute(query, params)
            signals = [SignalLog.from_dict(dict(row)) for row in cursor.fetchall()]
            return signals
            
        except Exception as e:
            logger.error(f"Failed to get signals: {e}")
            return []
        finally:
            conn.close()
            
    def get_open_trades(self) -> List[TradeRecord]:
        """Get all currently open trades"""
        return self.get_trades(status=TradeStatus.OPEN)
        
    def get_today_trades(self) -> List[TradeRecord]:
        """Get today's trades"""
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        return self.get_trades(start_date=today_start)
        
    def get_weekly_trades(self) -> List[TradeRecord]:
        """Get this week's trades"""
        today = datetime.utcnow()
        week_start = today - timedelta(days=today.weekday())
        week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
        return self.get_trades(start_date=week_start)
        
    def get_monthly_trades(self) -> List[TradeRecord]:
        """Get this month's trades"""
        month_start = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        return self.get_trades(start_date=month_start)
        
    def export_to_json(self, output_path: str) -> bool:
        """Export all trades to JSON file"""
        
        trades = self.get_trades()
        
        export_data = {
            'export_date': datetime.utcnow().isoformat(),
            'total_trades': len(trades),
            'trades': [t.to_dict() for t in trades]
        }
        
        try:
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            logger.info(f"Exported {len(trades)} trades to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export: {e}")
            return False
            
    def export_to_csv(self, output_path: str) -> bool:
        """Export all trades to CSV file"""
        
        trades = self.get_trades()
        
        if not trades:
            logger.warning("No trades to export")
            return False
            
        df = pd.DataFrame([t.to_dict() for t in trades])
        df.to_csv(output_path, index=False)
        logger.info(f"Exported {len(trades)} trades to {output_path}")
        return True


class JournalEntry:
    """Single journal entry for manual trade notes"""
    
    def __init__(self, journal_path: str = "data/trade_journal.md"):
        self.journal_path = Path(journal_path)
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        
    def add_entry(self, trade_id: str, notes: str, tags: List[str] = None):
        """Add a manual journal entry"""
        
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        tags_str = f" [{', '.join(tags)}]" if tags else ""
        
        entry = f"""
## {timestamp} - Trade {trade_id}{tags_str}

{notes}

---
"""
        with open(self.journal_path, 'a') as f:
            f.write(entry)
            
        logger.info(f"Added journal entry for trade {trade_id}")
        
    def get_entries(self, limit: int = 50) -> List[Dict]:
        """Get recent journal entries"""
        
        if not self.journal_path.exists():
            return []
            
        with open(self.journal_path, 'r') as f:
            content = f.read()
            
        # Parse markdown entries (simplified)
        entries = []
        current_entry = {}
        
        for line in content.split('\n'):
            if line.startswith('## '):
                if current_entry:
                    entries.append(current_entry)
                # Parse header
                header = line[3:]
                parts = header.split(' - ')
                current_entry = {
                    'timestamp': parts[0] if len(parts) > 0 else '',
                    'trade_id': parts[1].split(' ')[1] if len(parts) > 1 else '',
                    'notes': ''
                }
            elif current_entry and line.strip() and not line.startswith('---'):
                current_entry['notes'] += line + '\n'
                
        if current_entry:
            entries.append(current_entry)
            
        return entries[-limit:]
