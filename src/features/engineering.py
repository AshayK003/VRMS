"""Feature engineering for VRMS — all features computed on expanding window only."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_realized_vol(df: pd.DataFrame, windows: list[int] = [5, 10, 20]) -> pd.DataFrame:
    """Compute realized volatility for multiple windows."""
    result = pd.DataFrame(index=df.index)
    returns = np.log(df['Close'] / df['Close'].shift(1))
    
    for w in windows:
        result[f'vol_{w}d'] = returns.rolling(w).std() * np.sqrt(252)
    
    return result


def compute_momentum(df: pd.DataFrame, windows: list[int] = [21, 63, 126]) -> pd.DataFrame:
    """Compute momentum (total return) for multiple windows."""
    result = pd.DataFrame(index=df.index)
    
    for w in windows:
        result[f'mom_{w}d'] = df['Close'].pct_change(w)
    
    return result


def compute_relative_strength(
    df: pd.DataFrame, 
    benchmark: pd.DataFrame, 
    windows: list[int] = [21]
) -> pd.DataFrame:
    """Compute relative strength vs benchmark."""
    result = pd.DataFrame(index=df.index)
    
    # Use merge to align dates
    stock_close = df['Close'].rename('stock_close')
    bench_close = benchmark['Close'].rename('bench_close')
    
    merged = pd.concat([stock_close, bench_close], axis=1, join='inner')
    
    if len(merged) < 2:
        for w in windows:
            result[f'rs_{w}d'] = np.nan
        return result
    
    stock_returns = merged['stock_close'].pct_change()
    bench_returns = merged['bench_close'].pct_change()
    
    for w in windows:
        stock_cum = (1 + stock_returns).rolling(w).apply(np.prod, raw=True)
        bench_cum = (1 + bench_returns).rolling(w).apply(np.prod, raw=True)
        rs = stock_cum / bench_cum - 1
        result[f'rs_{w}d'] = rs.reindex(df.index)
    
    return result


def compute_volume_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Compute volume features."""
    result = pd.DataFrame(index=df.index)
    
    avg_vol = df['Volume'].rolling(window).mean()
    result['volume_ratio'] = df['Volume'] / avg_vol
    result['circuit_flag'] = (df['Volume'] < avg_vol * 0.1).astype(int)
    
    return result


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute ADX (Average Directional Index)."""
    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values
    
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
    )
    
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean()
    
    # Return with original index (pad first value with NaN)
    result = pd.Series(np.nan, index=df.index)
    result.iloc[1:] = adx.values
    return result


def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute RSI (Relative Strength Index)."""
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute ATR (Average True Range)."""
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    
    return atr


def compute_garch_vol(df: pd.DataFrame, window: int = 252) -> pd.Series:
    """Compute GARCH(1,1) volatility (EWMA approximation)."""
    returns = np.log(df['Close'] / df['Close'].shift(1))
    alpha = 0.06
    variance = returns.ewm(alpha=alpha, adjust=False).var()
    garch_vol = np.sqrt(variance) * np.sqrt(252)
    
    return garch_vol
