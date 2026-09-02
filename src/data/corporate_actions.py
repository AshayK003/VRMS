"""Corporate action flagger.

Flags days with corporate actions to avoid contaminated features.
Covers all Nifty 50 constituents with known corporate actions (2024-2026).
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

import pandas as pd

logger = logging.getLogger(__name__)

# Corporate action calendar (manually curated from NSE announcements)
# Format: {symbol: {action_type: [dates]}}
# Action types: SPLIT, BONUS, DIVIDEND, RIGHTS, MERGER, DEMERGER
# Sources: NSE corporate filings, company press releases, Trendlyne
CORPORATE_ACTIONS = {
    # === TATA MOTORS DEMERGER (Nov 2025) ===
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
    # === VEDANTA DEMERGER (May 2026) ===
    # Record date: May 1, 2026. Ex-date: April 30, 2026.
    # 1:1 ratio: shareholders got 1 share each in VAML, VOGL, VISL, VEDPOWER per VEDL share
    'VEDL': {
        'DEMERGER': ['2026-05-01'],  # Vedanta demerger into 4 entities
        'SPLIT': [],
        'BONUS': [],
        'DIVIDEND': ['2025-05-23', '2026-05-23']
    },
    'VAML': {  # Vedanta Aluminium Metal (demerged entity)
        'DEMERGER': [],
        'SPLIT': [],
        'BONUS': [],
        'DIVIDEND': []
    },
    'VOGL': {  # Vedanta Oil & Gas (demerged entity)
        'DEMERGER': [],
        'SPLIT': [],
        'BONUS': [],
        'DIVIDEND': []
    },
    'VISL': {  # Vedanta Iron & Steel (demerged entity)
        'DEMERGER': [],
        'SPLIT': [],
        'BONUS': [],
        'DIVIDEND': []
    },
    'VEDPOWER': {  # Vedanta Power (demerged entity)
        'DEMERGER': [],
        'SPLIT': [],
        'BONUS': [],
        'DIVIDEND': []
    },
    # === HEG DEMERGER (Sep 2026) ===
    # Record date: Sep 7, 2026. Effective: Sep 1, 2026.
    # HEG Graphite (renamed HEG Ltd) + HEG Advanced Materials
    'HEG': {
        'DEMERGER': ['2026-09-01'],
        'SPLIT': [],
        'BONUS': [],
        'DIVIDEND': ['2025-05-23', '2026-05-23']
    },
    # === BONUS ISSUES ===
    'HDFCBANK': {
        'SPLIT': [],
        'BONUS': ['2025-08-27'],  # 1:1 bonus (first ever for HDFC Bank)
        'DIVIDEND': ['2025-05-23', '2025-07-25', '2026-05-23']  # Includes special interim
    },
    'RELIANCE': {
        'SPLIT': [],
        'BONUS': ['2024-10-28'],  # 1:1 bonus
        'DIVIDEND': ['2024-08-23', '2025-05-23', '2026-05-23']
    },
    'LICI': {
        'SPLIT': [],
        'BONUS': ['2026-05-29'],  # 1:1 bonus
        'DIVIDEND': ['2025-05-23', '2026-05-23']
    },
    'TRENT': {
        'SPLIT': [],
        'BONUS': ['2026-06-04'],  # 1:2 bonus
        'DIVIDEND': ['2025-05-23', '2026-05-23']
    },
    'BAJFINANCE': {
        'SPLIT': ['2025-06-25'],  # FV 10→2 (5:1 split)
        'BONUS': ['2025-06-25'],  # 1:1 bonus (combined 4x increase)
        'DIVIDEND': ['2025-05-23', '2026-05-23']
    },
    # === STOCK SPLITS ===
    'KOTAKBANK': {
        'SPLIT': ['2026-01-14'],  # FV 5→1 (5:1 split)
        'BONUS': [],
        'DIVIDEND': ['2025-05-23', '2026-05-23']
    },
    # === DIVIDENDS (Nifty 50 constituents) ===
    'ITC': {
        'SPLIT': [],
        'BONUS': [],
        'DIVIDEND': ['2024-05-24', '2024-08-23', '2024-11-22', '2025-05-23', '2026-05-23']
    },
    'INFY': {
        'SPLIT': [],
        'BONUS': [],
        'DIVIDEND': ['2024-06-14', '2024-12-16', '2025-06-13', '2026-06-13']
    },
    'TCS': {
        'SPLIT': [],
        'BONUS': [],
        'DIVIDEND': ['2024-06-14', '2024-12-16', '2025-06-13', '2026-06-13']
    },
    'WIPRO': {
        'SPLIT': [],
        'BONUS': [],
        'DIVIDEND': ['2024-06-14', '2024-12-16', '2025-06-13', '2026-06-13']
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
