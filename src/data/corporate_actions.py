"""Corporate action flagger.

Flags days with corporate actions to avoid contaminated features.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

import pandas as pd

logger = logging.getLogger(__name__)

# Corporate action calendar (manually curated from NSE announcements)
# Format: {symbol: {action_type: [dates]}}
# Action types: SPLIT, BONUS, DIVIDEND, RIGHTS, MERGER, DEMERGER
CORPORATE_ACTIONS = {
    'TMCV': {
        'DEMERGER': ['2025-11-14'],  # Tata Motors demerged: TMCV (PV) + TMC (CV)
        'SPLIT': [],
        'BONUS': [],
        'DIVIDEND': ['2025-06-13', '2025-12-16', '2026-06-13']
    },
    'TATAMOTORS': {  # Legacy ticker (delisted post-demerger)
        'DEMERGER': ['2025-11-14'],
        'SPLIT': [],
        'BONUS': [],
        'DIVIDEND': []
    },
    'VEDL': {
        'DEMERGER': ['2024-07-31'],  # Vedanta demerger
        'SPLIT': [],
        'BONUS': [],
        'DIVIDEND': ['2024-05-24', '2024-08-23', '2024-11-22', '2025-05-23']
    },
    'ITC': {
        'SPLIT': [],
        'BONUS': [],
        'DIVIDEND': ['2024-05-24', '2024-08-23', '2024-11-22', '2025-05-23']
    },
    'INFY': {
        'SPLIT': [],
        'BONUS': [],
        'DIVIDEND': ['2024-06-14', '2024-12-16', '2025-06-13']
    },
    'TCS': {
        'SPLIT': [],
        'BONUS': [],
        'DIVIDEND': ['2024-06-14', '2024-12-16', '2025-06-13']
    },
    'WIPRO': {
        'SPLIT': [],
        'BONUS': [],
        'DIVIDEND': ['2024-06-14', '2024-12-16', '2025-06-13']
    },
    'HDFCBANK': {
        'SPLIT': [],
        'BONUS': [],
        'DIVIDEND': ['2024-05-24', '2024-08-23', '2024-11-22', '2025-05-23']
    },
    'RELIANCE': {
        'SPLIT': [],
        'BONUS': [],
        'DIVIDEND': ['2024-05-24', '2024-08-23', '2024-11-22', '2025-05-23']
    },
}


def is_corporate_action_day(symbol: str, date: str | datetime) -> tuple[bool, str]:
    """Check if a date has a corporate action for a symbol.
    
    Args:
        symbol: NSE ticker
        date: Date to check
        
    Returns:
        Tuple of (is_action_day, reason)
    """
    if isinstance(date, str):
        date = pd.Timestamp(date)
    
    if symbol not in CORPORATE_ACTIONS:
        return False, "No corporate actions on record"
    
    actions = CORPORATE_ACTIONS[symbol]
    
    for action_type, dates in actions.items():
        for action_date_str in dates:
            action_date = pd.Timestamp(action_date_str)
            # Flag the day before (cum-date) and the action date itself
            if abs((date - action_date).days) <= 1:
                return True, f"{action_type} on {action_date_str}"
    
    return False, ""


def filter_corporate_action_days(
    df: pd.DataFrame, 
    symbol: str, 
    date_col: str = 'Date'
) -> pd.DataFrame:
    """Filter out corporate action days from DataFrame.
    
    Args:
        df: DataFrame with date index or column
        symbol: NSE ticker
        date_col: Name of date column (if not index)
        
    Returns:
        Filtered DataFrame with corporate action days removed
    """
    if date_col in df.columns:
        dates = pd.to_datetime(df[date_col])
    else:
        dates = df.index
    
    mask = dates.map(lambda d: not is_corporate_action_day(symbol, d)[0])
    
    return df[mask]
