"""Tag candles by trading session (London, NY, Asian, Overlap)"""

import pandas as pd
from datetime import datetime, time
from typing import Dict, List, Optional


class SessionTagger:
    """Tag each candle with the dominant trading session"""
    
    # Session definitions in UTC
    SESSIONS = {
        'asian': {
            'hours': range(0, 8),
            'name': 'Asian',
            'code': 0,
            'volatility_mult': 0.7
        },
        'london': {
            'hours': range(8, 13),
            'name': 'London',
            'code': 1,
            'volatility_mult': 1.2
        },
        'overlap': {
            'hours': range(13, 17),
            'name': 'London-NY Overlap',
            'code': 2,
            'volatility_mult': 1.5
        },
        'new_york': {
            'hours': range(17, 22),
            'name': 'New York',
            'code': 3,
            'volatility_mult': 1.1
        }
    }
    
    def __init__(self):
        self.session_map = {}
        for session, config in self.SESSIONS.items():
            for hour in config['hours']:
                self.session_map[hour] = session
                
    def tag_dataframe(self, df: pd.DataFrame, timestamp_col: str = 'timestamp') -> pd.DataFrame:
        """Add session tags to dataframe"""
        
        if timestamp_col not in df.columns:
            raise ValueError(f"Column {timestamp_col} not found in dataframe")
            
        # Extract hour from timestamp
        df['hour_utc'] = df[timestamp_col].dt.hour
        
        # Map hour to session
        df['session'] = df['hour_utc'].map(self.session_map)
        
        # Add session code
        df['session_code'] = df['session'].map(
            lambda s: self.SESSIONS[s]['code'] if pd.notna(s) else -1
        )
        
        # Add session name
        df['session_name'] = df['session'].map(
            lambda s: self.SESSIONS[s]['name'] if pd.notna(s) else 'Unknown'
        )
        
        # Add volatility multiplier
        df['session_volatility'] = df['session'].map(
            lambda s: self.SESSIONS[s]['volatility_mult'] if pd.notna(s) else 1.0
        )
        
        return df
        
    def is_excluded_session(self, session: str, symbol: str = "XAUUSD") -> bool:
        """Check if session should be excluded for given symbol"""
        
        # Exclude Asian session for XAUUSD (low volatility)
        if symbol == "XAUUSD" and session == 'asian':
            return True
            
        return False
        
    def get_current_session(self) -> str:
        """Get current trading session based on UTC time"""
        
        current_hour = datetime.utcnow().hour
        return self.session_map.get(current_hour, 'asian')  # Default to Asian
        
    def get_session_stats(self, df: pd.DataFrame) -> Dict:
        """Calculate statistics per session"""
        
        stats = {}
        
        for session, config in self.SESSIONS.items():
            session_df = df[df['session'] == session]
            
            if len(session_df) > 0:
                stats[session] = {
                    'count': len(session_df),
                    'avg_range': (session_df['high'] - session_df['low']).mean(),
                    'total_volume': session_df['volume'].sum() if 'volume' in session_df else 0,
                    'volatility_mult': config['volatility_mult']
                }
                
        return stats
