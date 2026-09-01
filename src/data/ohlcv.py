"""OHLCV data fetching from NSE via nsepython."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_ohlcv(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily OHLCV data for a symbol from NSE.
    
    Args:
        symbol: NSE ticker (e.g., 'RELIANCE', 'TCS')
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        
    Returns:
        DataFrame with columns: Date, Open, High, Low, Close, Volume
    """
    try:
        from nsepython import history
        
        # nsepython.history expects symbol and date range
        df = history.get_history(symbol, start_date, end_date)
        
        if df is None or df.empty:
            logger.warning(f"No data returned for {symbol}")
            return pd.DataFrame()
        
        # Standardize column names
        df = df.rename(columns={
            'CH_TIMESTAMP': 'Date',
            'CH_OPENING_PRICE': 'Open',
            'CH_TRADE_HIGH_PRICE': 'High',
            'CH_TRADE_LOW_PRICE': 'Low',
            'CH_CLOSING_PRICE': 'Close',
            'CH_TOT_TRADED_QTY': 'Volume'
        })
        
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').sort_index()
        
        # Keep only required columns
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
    except Exception as e:
        logger.error(f"Failed to fetch OHLCV for {symbol}: {e}")
        return pd.DataFrame()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_intraday(symbol: str, interval: str = "15m", period: str = "5d") -> pd.DataFrame:
    """Fetch intraday OHLCV data from Yahoo Finance via yfinance.
    
    Args:
        symbol: Yahoo ticker (e.g., 'RELIANCE.NS', '^NSEI')
        interval: Candle interval ('1m', '5m', '15m', '1h')
        period: Data period ('1d', '5d', '1mo')
        
    Returns:
        DataFrame with OHLCV data
    """
    try:
        import yfinance as yf
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        
        if df is None or df.empty:
            return pd.DataFrame()
        
        # Standardize column names
        df = df.rename(columns={
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low',
            'Close': 'Close',
            'Volume': 'Volume'
        })
        
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
    except Exception as e:
        logger.error(f"Failed to fetch intraday for {symbol}: {e}")
        return pd.DataFrame()


def fetch_vix() -> float | None:
    """Fetch current India VIX value from NSE.
    
    Returns:
        VIX value or None if fetch fails
    """
    try:
        from nsepython import nse_get_index_quote
        
        quote = nse_get_index_quote("INDIAVIX")
        if quote and 'last' in quote:
            return float(quote['last'])
        return None
        
    except Exception as e:
        logger.error(f"Failed to fetch VIX: {e}")
        return None
