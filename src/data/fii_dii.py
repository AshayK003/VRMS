"""FII/DII data fetching from NSE."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_fii_dii() -> pd.DataFrame:
    """Fetch daily FII/DII flow data from NSE.
    
    Returns:
        DataFrame with columns: Date, FII, DII, Net
    """
    try:
        from nsepython import nse_fiidii
        
        df = nse_fiidii()
        
        if df is None or df.empty:
            logger.warning("No FII/DII data returned")
            return pd.DataFrame()
        
        # Standardize column names
        df = df.rename(columns={
            'date': 'Date',
            'fii': 'FII',
            'dii': 'DII',
            'net': 'Net'
        })
        
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').sort_index()
        
        return df[['FII', 'DII', 'Net']]
        
    except Exception as e:
        logger.error(f"Failed to fetch FII/DII: {e}")
        return pd.DataFrame()


def get_fii_dii_lagged(days: int = 2) -> pd.DataFrame | None:
    """Get FII/DII data lagged by specified days.
    
    Uses T-2 final data (not T-1 provisional) to avoid revision issues.
    
    Args:
        days: Number of days to lag (default 2 for T-2 final)
        
    Returns:
        Lagged DataFrame or None
    """
    df = fetch_fii_dii()
    if df.empty:
        return None
    
    # Shift data back by specified days to use only final (revised) data
    lagged = df.shift(days)
    lagged = lagged.dropna()
    
    return lagged
