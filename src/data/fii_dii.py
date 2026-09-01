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
        
        # nse_fiidii returns: buyValue, category, date, netValue, sellValue
        # category is 'FII/FPI' or 'DII'
        # We need to pivot this into FII, DII, Net columns
        
        df['date'] = pd.to_datetime(df['date'], format='%d-%b-%Y')
        
        # Pivot: separate FII and DII rows
        fii_rows = df[df['category'] == 'FII/FPI'].copy()
        dii_rows = df[df['category'] == 'DII'].copy()
        
        result = pd.DataFrame(index=df['date'].unique())
        result.index.name = 'Date'
        
        if not fii_rows.empty:
            fii_rows = fii_rows.set_index('date')
            result['FII'] = fii_rows['netValue'].astype(float)
        
        if not dii_rows.empty:
            dii_rows = dii_rows.set_index('date')
            result['DII'] = dii_rows['netValue'].astype(float)
        
        result['Net'] = result['FII'].fillna(0).astype(float) + result['DII'].fillna(0).astype(float)
        
        result = result.sort_index()
        
        return result[['FII', 'DII', 'Net']]
        
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
