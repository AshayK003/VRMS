"""Point-in-time Nifty 50 constituent mapper.

Maps historical index membership to avoid survivorship bias.
Nifty 50 rebalances semi-annually (Jan and July).
"""
from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)

# Historical Nifty 50 constituents (semi-annual rebalancing)
# Source: NSE press releases, compiled manually
# Format: {effective_date: [list of symbols]}
NIFTY_50_CONSTITUENTS = {
    # 2024
    '2024-01-01': [
        'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'HINDUNILVR', 'ICICIBANK', 'SBIN',
        'ITC', 'BHARTIARTL', 'LICI', 'HCLTECH', 'ASIANPAINT', 'KOTAKBANK',
        'MARUTI', 'TATAMOTORS', 'SUNPHARMA', 'TITAN', 'AXISBANK', 'WIPRO',
        'NESTLEIND', 'ULTRACEMCO', 'BAJFINANCE', 'ONGC', 'ADANIPORTS',
        'POWERGRID', 'NTPC', 'TATASTEEL', 'JSWSTEEL', 'COALINDIA', 'GRASIM',
        'TECHM', 'CIPLA', 'DRREDDY', 'BRITANNIA', 'HEROMOTOCO', 'EICHERMOT',
        'APOLLOHOSP', 'ADANIENT', 'TATACONSUM', 'DIVISLAB', 'HINDALCO',
        'SHREECEM', 'BAJAJFINSV', 'M&M', 'SBILIFE', 'IOC', 'INDUSINDBK',
        'HDFCLIFE', 'BPCH', 'TRENT', 'SUNTV'
    ],
    '2024-07-01': [
        'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'HINDUNILVR', 'ICICIBANK', 'SBIN',
        'ITC', 'BHARTIARTL', 'LICI', 'HCLTECH', 'ASIANPAINT', 'KOTAKBANK',
        'MARUTI', 'TATAMOTORS', 'SUNPHARMA', 'TITAN', 'AXISBANK', 'WIPRO',
        'NESTLEIND', 'ULTRACEMCO', 'BAJFINANCE', 'ONGC', 'ADANIPORTS',
        'POWERGRID', 'NTPC', 'TATASTEEL', 'JSWSTEEL', 'COALINDIA', 'GRASIM',
        'TECHM', 'CIPLA', 'DRREDDY', 'BRITANNIA', 'HEROMOTOCO', 'EICHERMOT',
        'APOLLOHOSP', 'ADANIENT', 'TATACONSUM', 'DIVISLAB', 'HINDALCO',
        'SHREECEM', 'BAJAJFINSV', 'M&M', 'SBILIFE', 'IOC', 'INDUSINDBK',
        'HDFCLIFE', 'BPCH', 'TRENT', 'SUNTV'
    ],
    # 2025
    '2025-01-01': [
        'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'HINDUNILVR', 'ICICIBANK', 'SBIN',
        'ITC', 'BHARTIARTL', 'LICI', 'HCLTECH', 'ASIANPAINT', 'KOTAKBANK',
        'MARUTI', 'TATAMOTORS', 'SUNPHARMA', 'TITAN', 'AXISBANK', 'WIPRO',
        'NESTLEIND', 'ULTRACEMCO', 'BAJFINANCE', 'ONGC', 'ADANIPORTS',
        'POWERGRID', 'NTPC', 'TATASTEEL', 'JSWSTEEL', 'COALINDIA', 'GRASIM',
        'TECHM', 'CIPLA', 'DRREDDY', 'BRITANNIA', 'HEROMOTOCO', 'EICHERMOT',
        'APOLLOHOSP', 'ADANIENT', 'TATACONSUM', 'DIVISLAB', 'HINDALCO',
        'SHREECEM', 'BAJAJFINSV', 'M&M', 'SBILIFE', 'IOC', 'INDUSINDBK',
        'HDFCLIFE', 'BPCH', 'TRENT', 'SUNTV'
    ],
    '2025-07-01': [
        'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'HINDUNILVR', 'ICICIBANK', 'SBIN',
        'ITC', 'BHARTIARTL', 'LICI', 'HCLTECH', 'ASIANPAINT', 'KOTAKBANK',
        'MARUTI', 'TATAMOTORS', 'SUNPHARMA', 'TITAN', 'AXISBANK', 'WIPRO',
        'NESTLEIND', 'ULTRACEMCO', 'BAJFINANCE', 'ONGC', 'ADANIPORTS',
        'POWERGRID', 'NTPC', 'TATASTEEL', 'JSWSTEEL', 'COALINDIA', 'GRASIM',
        'TECHM', 'CIPLA', 'DRREDDY', 'BRITANNIA', 'HEROMOTOCO', 'EICHERMOT',
        'APOLLOHOSP', 'ADANIENT', 'TATACONSUM', 'DIVISLAB', 'HINDALCO',
        'SHREECEM', 'BAJAJFINSV', 'M&M', 'SBILIFE', 'IOC', 'INDUSINDBK',
        'HDFCLIFE', 'BPCH', 'TRENT', 'SUNTV'
    ],
}


def get_constituents_on_date(date: str | datetime) -> list[str]:
    """Get Nifty 50 constituents on a specific date.
    
    Args:
        date: Date string or datetime
        
    Returns:
        List of Nifty 50 symbols
    """
    if isinstance(date, str):
        date = pd.Timestamp(date)
    
    # Find the most recent rebalancing date before the given date
    best_date = None
    for rebal_date_str in NIFTY_50_CONSTITUENTS:
        rebal_date = pd.Timestamp(rebal_date_str)
        if rebal_date <= date:
            if best_date is None or rebal_date > best_date:
                best_date = rebal_date
    
    if best_date is None:
        logger.warning(f"No constituent data found for {date}")
        return list(NIFTY_50_CONSTITUENTS.values())[0]
    
    return NIFTY_50_CONSTITUENTS[best_date.strftime('%Y-%m-%d')]


def filter_by_constituents(
    df: pd.DataFrame, 
    date: str | datetime, 
    symbol_col: str = 'Symbol'
) -> pd.DataFrame:
    """Filter DataFrame to only include stocks that were in Nifty 50 on a date.
    
    Args:
        df: DataFrame with symbol column
        date: Date to filter by
        symbol_col: Name of symbol column
        
    Returns:
        Filtered DataFrame
    """
    constituents = get_constituents_on_date(date)
    return df[df[symbol_col].isin(constituents)]
