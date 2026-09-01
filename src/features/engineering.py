"""Feature engineering for VRMS — all features computed on expanding window only."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_realized_vol(df: pd.DataFrame, windows: list[int] = [5, 10, 20]) -> pd.DataFrame:
    """Compute realized volatility for multiple windows.
    
    Args:
        df: OHLCV DataFrame
        windows: List of lookback windows
        
    Returns:
        DataFrame with realized vol columns
    """
    result = pd.DataFrame(index=df.index)
    returns = np.log(df['Close'] / df['Close'].shift(1))
    
    for w in windows:
        result[f'vol_{w}d'] = returns.rolling(w).std() * np.sqrt(252)
    
    return result


def compute_momentum(df: pd.DataFrame, windows: list[int] = [21, 63, 126]) -> pd.DataFrame:
    """Compute momentum (total return) for multiple windows.
    
    Uses unadjusted close to avoid corporate action contamination.
    
    Args:
        df: OHLCV DataFrame
        windows: List of lookback windows (21=1M, 63=3M, 126=6M)
        
    Returns:
        DataFrame with momentum columns
    """
    result = pd.DataFrame(index=df.index)
    
    for w in windows:
        result[f'mom_{w}d'] = df['Close'].pct_change(w)
    
    return result


def compute_relative_strength(
    df: pd.DataFrame, 
    benchmark: pd.DataFrame, 
    windows: list[int] = [21, 63]
) -> pd.DataFrame:
    """Compute relative strength vs benchmark.
    
    Args:
        df: Stock OHLCV DataFrame
        benchmark: Benchmark OHLCV DataFrame (e.g., Nifty 50)
        windows: List of lookback windows
        
    Returns:
        DataFrame with relative strength columns
    """
    result = pd.DataFrame(index=df.index)
    
    stock_returns = df['Close'].pct_change()
    bench_returns = benchmark['Close'].pct_change()
    
    for w in windows:
        stock_cum = (1 + stock_returns).rolling(w).apply(np.prod, raw=True)
        bench_cum = (1 + bench_returns).rolling(w).apply(np.prod, raw=True)
        result[f'rs_{w}d'] = stock_cum / bench_cum - 1
    
    return result


def compute_volume_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Compute volume features.
    
    Args:
        df: OHLCV DataFrame
        window: Lookback window
        
    Returns:
        DataFrame with volume features
    """
    result = pd.DataFrame(index=df.index)
    
    # Volume ratio
    avg_vol = df['Volume'].rolling(window).mean()
    result['volume_ratio'] = df['Volume'] / avg_vol
    
    # Circuit flag (volume = 0 or extremely low)
    result['circuit_flag'] = (df['Volume'] < avg_vol * 0.1).astype(int)
    
    return result


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute ADX (Average Directional Index).
    
    Args:
        df: OHLCV DataFrame
        period: ADX period
        
    Returns:
        ADX series
    """
    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values
    
    # True Range
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
    )
    
    # Directional Movement
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Wilder's smoothing
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean()
    
    return adx


def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute RSI (Relative Strength Index).
    
    Args:
        df: OHLCV DataFrame
        period: RSI period
        
    Returns:
        RSI series
    """
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute ATR (Average True Range).
    
    Args:
        df: OHLCV DataFrame
        period: ATR period
        
    Returns:
        ATR series
    """
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
    """Compute GARCH(1,1) volatility.
    
    Simplified implementation — uses EWMA as GARCH approximation.
    For full GARCH, use arch library (not included due to dependency conflicts).
    
    Args:
        df: OHLCV DataFrame
        window: Estimation window
        
    Returns:
        GARCH volatility series
    """
    returns = np.log(df['Close'] / df['Close'].shift(1))
    
    # EWMA variance (GARCH(1,1) approximation with alpha+beta=0.94)
    alpha = 0.06
    beta = 0.94
    
    variance = returns.ewm(alpha=alpha, adjust=False).var()
    garch_vol = np.sqrt(variance) * np.sqrt(252)
    
    return garch_vol
