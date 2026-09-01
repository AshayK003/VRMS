"""Fundamental feature extraction for VRMS.

Extracts P/E, ROE, Debt/Equity, Market Cap from yfinance.
"""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def fetch_fundamentals(symbol: str) -> dict:
    """Fetch fundamental data for a symbol.
    
    Args:
        symbol: NSE ticker
        
    Returns:
        Dict with fundamental metrics
    """
    try:
        import yfinance as yf
        
        ticker = yf.Ticker(f"{symbol}.NS")
        info = ticker.info or {}
        
        return {
            'pe_ratio': info.get('trailingPE') or info.get('forwardPE'),
            'roe': info.get('returnOnEquity'),
            'debt_to_equity': info.get('debtToEquity'),
            'market_cap': info.get('marketCap'),
            'price_to_book': info.get('priceToBook'),
            'profit_margin': info.get('profitMargins'),
            'revenue_growth': info.get('revenueGrowth'),
            'earnings_growth': info.get('earningsGrowth'),
            'dividend_yield': info.get('dividendYield'),
            'free_cash_flow': info.get('freeCashflow'),
        }
        
    except Exception as e:
        logger.error(f"Failed to fetch fundamentals for {symbol}: {e}")
        return {}


def compute_fundamental_features(symbol: str) -> dict:
    """Compute fundamental features for ML.
    
    Args:
        symbol: NSE ticker
        
    Returns:
        Dict with processed fundamental features
    """
    raw = fetch_fundamentals(symbol)
    
    if not raw:
        return {
            'pe_ratio': None,
            'roe': None,
            'debt_to_equity': None,
            'market_cap_log': None,
            'price_to_book': None,
            'profit_margin': None,
            'revenue_growth': None,
            'earnings_growth': None,
            'dividend_yield': None,
            'fcf_yield': None,
        }
    
    import math
    
    # Log transform market cap
    market_cap_log = math.log10(raw['market_cap']) if raw.get('market_cap') else None
    
    # FCF yield
    fcf_yield = None
    if raw.get('free_cash_flow') and raw.get('market_cap'):
        fcf_yield = raw['free_cash_flow'] / raw['market_cap']
    
    return {
        'pe_ratio': raw.get('pe_ratio'),
        'roe': raw.get('roe'),
        'debt_to_equity': raw.get('debt_to_equity'),
        'market_cap_log': market_cap_log,
        'price_to_book': raw.get('price_to_book'),
        'profit_margin': raw.get('profit_margin'),
        'revenue_growth': raw.get('revenue_growth'),
        'earnings_growth': raw.get('earnings_growth'),
        'dividend_yield': raw.get('dividend_yield'),
        'fcf_yield': fcf_yield,
    }
