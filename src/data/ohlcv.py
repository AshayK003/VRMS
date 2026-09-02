"""OHLCV data fetching from Yahoo Finance (yfinance) for Indian equities."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


def _clean_yf_df(df: pd.DataFrame) -> pd.DataFrame:
    """Clean yfinance DataFrame: reset index, rename, tz-naive, numeric."""
    if df is None or df.empty:
        return pd.DataFrame()
    
    df = df.reset_index()
    
    # Handle both 'Date' and 'Datetime' column names
    date_col = 'Date' if 'Date' in df.columns else 'Datetime'
    
    df = df.rename(columns={
        date_col: 'Date',
        'Open': 'Open',
        'High': 'High',
        'Low': 'Low',
        'Close': 'Close',
        'Volume': 'Volume'
    })
    
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    df = df.set_index('Date').sort_index()
    
    # Ensure numeric columns
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_ohlcv(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily OHLCV data for a symbol from Yahoo Finance.
    
    Uses auto_adjust=True to handle corporate actions (splits, dividends,
    demergers) correctly. For demerged tickers (e.g., TATAMOTORS→TMCV.NS),
    the old ticker is delisted — use the new ticker symbol.
    
    Args:
        symbol: NSE ticker (e.g., 'RELIANCE', 'TCS')
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        
    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume (tz-naive)
    """
    try:
        import yfinance as yf
        
        ticker = yf.Ticker(f"{symbol}.NS")
        # auto_adjust=True ensures corporate action adjustments are applied
        df = ticker.history(start=start_date, end=end_date, auto_adjust=True)
        
        if df is None or df.empty:
            logger.warning(f"No data returned for {symbol}")
            return pd.DataFrame()
        
        return _clean_yf_df(df)
        
    except Exception as e:
        logger.error(f"Failed to fetch OHLCV for {symbol}: {e}")
        return pd.DataFrame()


def get_benchmark(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch Nifty 50 benchmark data from Yahoo Finance.
    
    Args:
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        
    Returns:
        Nifty 50 OHLCV DataFrame (tz-naive)
    """
    try:
        import yfinance as yf
        
        ticker = yf.Ticker("^NSEI")
        df = ticker.history(start=start_date, end=end_date)
        
        if df is None or df.empty:
            logger.warning("No benchmark data returned")
            return pd.DataFrame()
        
        return _clean_yf_df(df)
        
    except Exception as e:
        logger.error(f"Failed to fetch benchmark: {e}")
        return pd.DataFrame()


def fetch_vix() -> float | None:
    """Fetch current India VIX value from Yahoo Finance.
    
    Returns:
        VIX value or None if fetch fails
    """
    try:
        import yfinance as yf
        
        ticker = yf.Ticker("^INDIAVIX")
        df = ticker.history(period="1d")
        
        if df is not None and not df.empty:
            return float(df['Close'].iloc[-1])
        
        return None
        
    except Exception as e:
        logger.error(f"Failed to fetch VIX: {e}")
        return None
