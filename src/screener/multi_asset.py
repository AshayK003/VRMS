"""Multi-Asset VIX + Momentum Screener.

Scans Nifty 50 stocks and ranks them by conviction score
based on VIX regime + momentum alignment.

Output: Top 5 picks per signal day with entry/exit levels.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


@dataclass
class StockPick:
    """A ranked stock pick."""
    symbol: str
    rank: int
    conviction: float  # 0-100
    vix_regime: str  # "fear", "neutral", "complacency"
    momentum_score: float
    price: float
    ma20: float
    ma50: float
    adx: float
    reason: str
    entry: float
    target: float
    stop_loss: float


# Nifty 50 tickers (Yahoo Finance format)
NIFTY_50 = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS',
    'ICICIBANK.NS', 'SBIN.NS', 'ITC.NS', 'BHARTIARTL.NS', 'LICI.NS',
    'HCLTECH.NS', 'ASIANPAINT.NS', 'KOTAKBANK.NS', 'MARUTI.NS', 'TATAMOTORS',
    'SUNPHARMA.NS', 'TITAN.NS', 'AXISBANK.NS', 'WIPRO.NS', 'NESTLEIND.NS',
    'ULTRACEMCO.NS', 'BAJFINANCE.NS', 'ONGC.NS', 'ADANIPORTS.NS', 'POWERGRID.NS',
    'NTPC.NS', 'TATASTEEL.NS', 'JSWSTEEL.NS', 'COALINDIA.NS', 'GRASIM.NS',
    'TECHM.NS', 'CIPLA.NS', 'DRREDDY.NS', 'BRITANNIA.NS', 'HEROMOTOCO.NS',
    'EICHERMOT.NS', 'APOLLOHOSP.NS', 'ADANIENT.NS', 'TATACONSUM.NS', 'DIVISLAB.NS',
    'HINDALCO.NS', 'SHREECEM.NS', 'BAJAJFINSV.NS', 'M&M.NS', 'SBILIFE.NS',
    'IOC.NS', 'INDUSINDBK.NS', 'HDFCLIFE.NS', 'BPCL.NS', 'TRENT.NS',
]


def fetch_stock_data(symbol: str, period: str = "6mo") -> pd.DataFrame:
    """Fetch OHLCV data for a stock with candidate suffix fallback."""
    candidates = [symbol, f"{symbol}.NS", f"{symbol}.BO"]
    for sym in candidates:
        try:
            ticker = yf.Ticker(sym)
            df = ticker.history(period=period)
            if df is not None and not df.empty and len(df) >= 50:
                df = df.reset_index()
                date_col = 'Date' if 'Date' in df.columns else 'Datetime'
                df = df.rename(columns={
                    date_col: 'Date',
                    'Open': 'Open', 'High': 'High', 'Low': 'Low',
                    'Close': 'Close', 'Volume': 'Volume'
                })
                df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
                df = df.set_index('Date').sort_index()
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                return df[['Open', 'High', 'Low', 'Close', 'Volume']]
        except Exception:
            continue
    logger.debug(f"Failed to fetch {symbol}")
    return pd.DataFrame()


def fetch_vix(period: str = "6mo") -> pd.DataFrame:
    """Fetch VIX history."""
    try:
        ticker = yf.Ticker("^INDIAVIX")
        df = ticker.history(period=period)
        
        if df is None or df.empty:
            return pd.DataFrame()
        
        df = df.reset_index()
        df = df.rename(columns={'Date': 'Date', 'Close': 'VIX'})
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        df = df.set_index('Date').sort_index()
        
        return df[['VIX']]
        
    except Exception as e:
        logger.error(f"Failed to fetch VIX: {e}")
        return pd.DataFrame()


def compute_momentum(df: pd.DataFrame) -> dict:
    """Compute momentum features for a stock."""
    if len(df) < 50:
        return {}
    
    close = df['Close']
    high = df['High']
    low = df['Low']
    
    # Drop NaN rows
    valid = close.notna() & high.notna() & low.notna()
    close = close[valid]
    high = high[valid]
    low = low[valid]
    
    if len(close) < 50:
        return {}
    
    # Moving averages
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    
    # Latest values
    price = close.iloc[-1]
    ma20_val = ma20.iloc[-1]
    ma50_val = ma50.iloc[-1]
    
    if pd.isna(price) or pd.isna(ma20_val) or pd.isna(ma50_val):
        return {}
    
    # Price vs MAs
    above_ma20 = price > ma20_val
    above_ma50 = price > ma50_val
    ma_cross = ma20_val > ma50_val
    
    # ADX calculation
    tr = np.maximum(
        high.values[1:] - low.values[1:],
        np.maximum(
            np.abs(high.values[1:] - close.values[:-1]),
            np.abs(low.values[1:] - close.values[:-1])
        )
    )
    
    up_move = high.values[1:] - high.values[:-1]
    down_move = low.values[:-1] - low.values[1:]
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean() / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean() / (atr + 1e-10)
    
    dx = 100 * np.abs(plus_di.values - minus_di.values) / (plus_di.values + minus_di.values + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().iloc[-1]
    
    if pd.isna(adx):
        adx = 0.0
    
    # Rate of change
    roc_20 = close.pct_change(20).iloc[-1] if len(close) > 20 else 0
    if pd.isna(roc_20):
        roc_20 = 0.0
    
    return {
        'price': price,
        'ma20': ma20_val,
        'ma50': ma50_val,
        'above_ma20': above_ma20,
        'above_ma50': above_ma50,
        'ma_cross': ma_cross,
        'adx': adx,
        'roc_20': roc_20,
    }


def compute_conviction(mom: dict, vix_regime: str, params: dict) -> tuple[float, str]:
    """Compute conviction score for a stock.
    
    Args:
        mom: Momentum features
        vix_regime: Current VIX regime
        params: Strategy parameters
        
    Returns:
        Tuple of (conviction_score, reason)
    """
    if not mom:
        return 0.0, "Insufficient data"
    
    score = 0.0
    reasons = []
    
    adx_threshold = params.get('adx_threshold', 25)
    
    if vix_regime == "fear":
        # In fear zone: buy stocks with strong momentum
        if mom['above_ma20'] and mom['above_ma50'] and mom['ma_cross']:
            score += 40
            reasons.append("MA20>MA50")
        
        if mom['adx'] > adx_threshold:
            score += 30
            reasons.append(f"ADX={mom['adx']:.0f}")
        
        if mom['roc_20'] > 0:
            score += 20
            reasons.append(f"ROC20={mom['roc_20']:.1%}")
        
        if mom['price'] > mom['ma20'] > mom['ma50']:
            score += 10
            reasons.append("Price>MA20>MA50")
    
    elif vix_regime == "neutral":
        # In neutral zone: trade momentum only
        if mom['above_ma20'] and mom['above_ma50'] and mom['ma_cross'] and mom['adx'] > adx_threshold:
            score += 50
            reasons.append("Strong momentum")
        
        if mom['roc_20'] > 0.05:
            score += 30
            reasons.append(f"ROC20={mom['roc_20']:.1%}")
        
        if mom['adx'] > adx_threshold:
            score += 20
            reasons.append(f"ADX={mom['adx']:.0f}")
    
    elif vix_regime == "complacency":
        # In complacency zone: sell/avoid
        if not mom['above_ma20'] or not mom['above_ma50']:
            score += 40
            reasons.append("Below MAs")
        
        if mom['adx'] > adx_threshold and mom['roc_20'] < 0:
            score += 30
            reasons.append(f"Strong downtrend")
        
        if mom['price'] < mom['ma20'] < mom['ma50']:
            score += 20
            reasons.append("Price<MA20<MA50")
    
    reason_str = ", ".join(reasons) if reasons else "No edge"
    return score, reason_str


def get_vix_regime(vix_df: pd.DataFrame, params: dict) -> tuple[str, float]:
    """Determine current VIX regime.
    
    Returns:
        Tuple of (regime, current_vix)
    """
    if vix_df.empty:
        return "neutral", 14.0
    
    vix = vix_df.iloc[-1]['VIX']
    vix_high = params.get('vix_high', 18)
    vix_low = params.get('vix_low', 14)
    
    if vix > vix_high:
        return "fear", vix
    elif vix < vix_low:
        return "complacency", vix
    else:
        return "neutral", vix


def screen_stocks(
    vix_df: pd.DataFrame,
    params: dict | None = None,
    top_n: int = 5,
) -> list[StockPick]:
    """Screen Nifty 50 stocks and return top picks.
    
    Args:
        vix_df: VIX history
        params: Strategy parameters
        top_n: Number of top picks to return
        
    Returns:
        List of StockPick objects sorted by conviction
    """
    if params is None:
        params = {
            'vix_high': 18,
            'vix_low': 14,
            'adx_threshold': 25,
            'target_pct': 0.05,
            'stop_loss_pct': 0.03,
        }
    
    # Get VIX regime
    vix_regime, current_vix = get_vix_regime(vix_df, params)
    
    if vix_regime == "complacency":
        # In complacency zone, return empty (no buys)
        return []
    
    # Scan all stocks
    picks = []
    
    for symbol in NIFTY_50:
        df = fetch_stock_data(symbol, period="6mo")
        
        if df.empty or len(df) < 50:
            continue
        
        mom = compute_momentum(df)
        
        if not mom:
            continue
        
        conviction, reason = compute_conviction(mom, vix_regime, params)
        
        if conviction < 30:
            continue
        
        # Calculate entry/target/stop
        price = mom['price']
        target = price * (1 + params['target_pct'])
        stop_loss = price * (1 - params['stop_loss_pct'])
        
        picks.append(StockPick(
            symbol=symbol.replace('.NS', ''),
            rank=0,
            conviction=conviction,
            vix_regime=vix_regime,
            momentum_score=conviction,
            price=price,
            ma20=mom['ma20'],
            ma50=mom['ma50'],
            adx=mom['adx'],
            reason=reason,
            entry=price,
            target=target,
            stop_loss=stop_loss,
        ))
    
    # Sort by conviction (descending)
    picks.sort(key=lambda x: x.conviction, reverse=True)
    
    # Assign ranks and return top N
    for i, pick in enumerate(picks[:top_n], 1):
        pick.rank = i
    
    return picks[:top_n]
