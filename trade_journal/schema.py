"""Data schemas for trade journal entries"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Dict, List, Any
from enum import Enum
import uuid
import json


class OrderDirection(Enum):
    """Trade direction"""
    BUY = 1
    SELL = -1


class ExitReason(Enum):
    """Reason for trade exit"""
    TAKE_PROFIT = "tp_hit"
    STOP_LOSS = "sl_hit"
    MANUAL = "manual"
    END_OF_DAY = "eod"
    KILL_SWITCH = "kill_switch"
    BROKER_CANCELLED = "broker_cancelled"


class TradeStatus(Enum):
    """Current status of a trade"""
    PENDING = "pending"
    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class SignalLog:
    """Raw signal data from AI model"""
    signal_id: str
    timestamp: datetime
    symbol: str
    timeframe: str
    direction: int  # 1=BUY, -1=SELL, 0=NO_TRADE
    
    # Model outputs
    buy_prob: float
    sell_prob: float
    notrade_prob: float
    cnn_embedding_hash: str
    model_version: str
    
    # SHAP explanations (top features)
    shap_top5: Dict[str, float]
    
    # SMC context at signal time
    htf_bias: int
    nearest_ob_dist: float
    fvg_present: bool
    bos_recent: bool
    
    # Confidence metrics
    confidence_score: float
    signal_strength: float
    
    def to_dict(self) -> dict:
        return {
            'signal_id': self.signal_id,
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'direction': self.direction,
            'buy_prob': self.buy_prob,
            'sell_prob': self.sell_prob,
            'notrade_prob': self.notrade_prob,
            'cnn_embedding_hash': self.cnn_embedding_hash,
            'model_version': self.model_version,
            'shap_top5': json.dumps(self.shap_top5),
            'htf_bias': self.htf_bias,
            'nearest_ob_dist': self.nearest_ob_dist,
            'fvg_present': self.fvg_present,
            'bos_recent': self.bos_recent,
            'confidence_score': self.confidence_score,
            'signal_strength': self.signal_strength
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'SignalLog':
        return cls(
            signal_id=data['signal_id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            symbol=data['symbol'],
            timeframe=data['timeframe'],
            direction=data['direction'],
            buy_prob=data['buy_prob'],
            sell_prob=data['sell_prob'],
            notrade_prob=data['notrade_prob'],
            cnn_embedding_hash=data['cnn_embedding_hash'],
            model_version=data['model_version'],
            shap_top5=json.loads(data['shap_top5']) if isinstance(data['shap_top5'], str) else data['shap_top5'],
            htf_bias=data['htf_bias'],
            nearest_ob_dist=data['nearest_ob_dist'],
            fvg_present=data['fvg_present'],
            bos_recent=data['bos_recent'],
            confidence_score=data['confidence_score'],
            signal_strength=data['signal_strength']
        )


@dataclass
class TradeRecord:
    """Complete trade record with execution and outcome"""
    
    # Core identifiers
    trade_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    signal_id: str = ""
    
    # Trade metadata
    symbol: str = "XAUUSD"
    direction: OrderDirection = OrderDirection.BUY
    status: TradeStatus = TradeStatus.PENDING
    
    # Timestamps
    signal_time: Optional[datetime] = None
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    
    # Price levels
    entry_price: float = 0.0
    exit_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    
    # Order details
    order_id: str = ""
    lot_size: float = 0.0
    spread_at_entry: float = 0.0
    execution_ms: int = 0
    
    # Risk metrics
    risk_percent: float = 0.01  # 1% risk
    risk_reward_ratio: float = 0.0
    position_size_usd: float = 0.0
    
    # Outcome
    exit_reason: ExitReason = ExitReason.MANUAL
    pnl_pips: float = 0.0
    pnl_dollars: float = 0.0
    r_multiple: float = 0.0  # e.g., +1.8R, -1.0R
    duration_minutes: int = 0
    
    # Additional data
    model_version: str = ""
    commission: float = 0.0
    swap: float = 0.0
    notes: str = ""
    
    # Market context
    entry_atr: float = 0.0
    volatility_regime: float = 0.0
    session: str = ""
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage"""
        return {
            'trade_id': self.trade_id,
            'signal_id': self.signal_id,
            'symbol': self.symbol,
            'direction': self.direction.value,
            'status': self.status.value,
            'signal_time': self.signal_time.isoformat() if self.signal_time else None,
            'entry_time': self.entry_time.isoformat() if self.entry_time else None,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'order_id': self.order_id,
            'lot_size': self.lot_size,
            'spread_at_entry': self.spread_at_entry,
            'execution_ms': self.execution_ms,
            'risk_percent': self.risk_percent,
            'risk_reward_ratio': self.risk_reward_ratio,
            'position_size_usd': self.position_size_usd,
            'exit_reason': self.exit_reason.value,
            'pnl_pips': self.pnl_pips,
            'pnl_dollars': self.pnl_dollars,
            'r_multiple': self.r_multiple,
            'duration_minutes': self.duration_minutes,
            'model_version': self.model_version,
            'commission': self.commission,
            'swap': self.swap,
            'notes': self.notes,
            'entry_atr': self.entry_atr,
            'volatility_regime': self.volatility_regime,
            'session': self.session
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'TradeRecord':
        """Create from dictionary"""
        return cls(
            trade_id=data.get('trade_id', str(uuid.uuid4())),
            signal_id=data.get('signal_id', ''),
            symbol=data.get('symbol', 'XAUUSD'),
            direction=OrderDirection(data.get('direction', 1)),
            status=TradeStatus(data.get('status', 'pending')),
            signal_time=datetime.fromisoformat(data['signal_time']) if data.get('signal_time') else None,
            entry_time=datetime.fromisoformat(data['entry_time']) if data.get('entry_time') else None,
            exit_time=datetime.fromisoformat(data['exit_time']) if data.get('exit_time') else None,
            entry_price=data.get('entry_price', 0),
            exit_price=data.get('exit_price', 0),
            stop_loss=data.get('stop_loss', 0),
            take_profit=data.get('take_profit', 0),
            order_id=data.get('order_id', ''),
            lot_size=data.get('lot_size', 0),
            spread_at_entry=data.get('spread_at_entry', 0),
            execution_ms=data.get('execution_ms', 0),
            risk_percent=data.get('risk_percent', 0.01),
            risk_reward_ratio=data.get('risk_reward_ratio', 0),
            position_size_usd=data.get('position_size_usd', 0),
            exit_reason=ExitReason(data.get('exit_reason', 'manual')),
            pnl_pips=data.get('pnl_pips', 0),
            pnl_dollars=data.get('pnl_dollars', 0),
            r_multiple=data.get('r_multiple', 0),
            duration_minutes=data.get('duration_minutes', 0),
            model_version=data.get('model_version', ''),
            commission=data.get('commission', 0),
            swap=data.get('swap', 0),
            notes=data.get('notes', ''),
            entry_atr=data.get('entry_atr', 0),
            volatility_regime=data.get('volatility_regime', 0),
            session=data.get('session', '')
        )
    
    def calculate_metrics(self):
        """Calculate derived metrics after trade closes"""
        if self.entry_price > 0 and self.exit_price > 0:
            # Pip calculation for gold (0.01 = 1 pip) and forex (0.0001 = 1 pip)
            pip_size = 0.01 if self.symbol == "XAUUSD" else 0.0001
            self.pnl_pips = (self.exit_price - self.entry_price) / pip_size
            if self.direction == OrderDirection.SELL:
                self.pnl_pips = -self.pnl_pips
            
            # Dollar P&L
            pip_value = 10.0 if self.symbol == "XAUUSD" else 10.0  # Standard for 1 lot
            self.pnl_dollars = self.pnl_pips * pip_value * self.lot_size
            
            # R-multiple
            sl_distance_pips = abs(self.entry_price - self.stop_loss) / pip_size
            if sl_distance_pips > 0:
                self.r_multiple = self.pnl_pips / sl_distance_pips
            else:
                self.r_multiple = 0
                
            # Duration
            if self.entry_time and self.exit_time:
                self.duration_minutes = int((self.exit_time - self.entry_time).total_seconds() / 60)
                
            # Risk-reward ratio
            tp_distance_pips = abs(self.take_profit - self.entry_price) / pip_size
            if sl_distance_pips > 0:
                self.risk_reward_ratio = tp_distance_pips / sl_distance_pips


@dataclass
class DailySummary:
    """Daily trading summary"""
    date: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    expectancy: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            'date': self.date,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'total_pnl': self.total_pnl,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'largest_win': self.largest_win,
            'largest_loss': self.largest_loss,
            'expectancy': self.expectancy,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown
        }


class TradeJournalSchema:
    """Schema manager for trade journal database"""
    
    def __init__(self, db_path: str = "data/trade_journal.db"):
        self.db_path = db_path
        self._init_database()
        
    def _init_database(self):
        """Initialize SQLite database with proper schema"""
        import sqlite3
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Signals table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                signal_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                direction INTEGER NOT NULL,
                buy_prob REAL NOT NULL,
                sell_prob REAL NOT NULL,
                notrade_prob REAL NOT NULL,
                cnn_embedding_hash TEXT,
                model_version TEXT,
                shap_top5 TEXT,
                htf_bias INTEGER,
                nearest_ob_dist REAL,
                fvg_present INTEGER,
                bos_recent INTEGER,
                confidence_score REAL,
                signal_strength REAL
            )
        ''')
        
        # Trades table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                trade_id TEXT PRIMARY KEY,
                signal_id TEXT,
                symbol TEXT NOT NULL,
                direction INTEGER NOT NULL,
                status TEXT NOT NULL,
                signal_time TEXT,
                entry_time TEXT,
                exit_time TEXT,
                entry_price REAL,
                exit_price REAL,
                stop_loss REAL,
                take_profit REAL,
                order_id TEXT,
                lot_size REAL,
                spread_at_entry REAL,
                execution_ms INTEGER,
                risk_percent REAL,
                risk_reward_ratio REAL,
                position_size_usd REAL,
                exit_reason TEXT,
                pnl_pips REAL,
                pnl_dollars REAL,
                r_multiple REAL,
                duration_minutes INTEGER,
                model_version TEXT,
                commission REAL,
                swap REAL,
                notes TEXT,
                entry_atr REAL,
                volatility_regime REAL,
                session TEXT,
                FOREIGN KEY (signal_id) REFERENCES signals(signal_id)
            )
        ''')
        
        # Daily summaries table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_summaries (
                date TEXT PRIMARY KEY,
                total_trades INTEGER,
                winning_trades INTEGER,
                losing_trades INTEGER,
                total_pnl REAL,
                avg_win REAL,
                avg_loss REAL,
                win_rate REAL,
                profit_factor REAL,
                largest_win REAL,
                largest_loss REAL,
                expectancy REAL,
                sharpe_ratio REAL,
                max_drawdown REAL
            )
        ''')
        
        # Performance by model version
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                model_version TEXT,
                date TEXT,
                total_trades INTEGER,
                win_rate REAL,
                profit_factor REAL,
                avg_r_multiple REAL,
                PRIMARY KEY (model_version, date)
            )
        ''')
        
        # Performance by session
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS session_performance (
                session TEXT,
                date TEXT,
                total_trades INTEGER,
                win_rate REAL,
                avg_pnl REAL,
                PRIMARY KEY (session, date)
            )
        ''')
        
        conn.commit()
        conn.close()
