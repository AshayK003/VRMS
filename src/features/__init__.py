"""Main feature engineering orchestrator.

Combines all features into a single DataFrame per stock.
"""
from __future__ import annotations

import logging

import pandas as pd

from .engineering import (
    compute_realized_vol,
    compute_momentum,
    compute_relative_strength,
    compute_volume_features,
    compute_adx,
    compute_rsi,
    compute_atr,
    compute_garch_vol,
)
from .fii_dii import compute_fii_dii_features
from .sentiment import compute_sentiment_features
from .fundamentals import compute_fundamental_features

logger = logging.getLogger(__name__)


def compute_all_features(
    df: pd.DataFrame,
    benchmark: pd.DataFrame | None = None,
    fii_dii_df: pd.DataFrame | None = None,
    symbol: str | None = None,
) -> pd.DataFrame:
    """Compute all features for a stock.
    
    Args:
        df: Stock OHLCV DataFrame
        benchmark: Benchmark OHLCV DataFrame (e.g., Nifty 50)
        fii_dii_df: FII/DII flow DataFrame
        symbol: NSE ticker (for sentiment/fundamentals)
        
    Returns:
        DataFrame with all features
    """
    all_features = []
    
    # Volatility features
    vol_features = compute_realized_vol(df)
    all_features.append(vol_features)
    
    # GARCH volatility
    garch = compute_garch_vol(df)
    garch_df = pd.DataFrame({'garch_vol': garch}, index=df.index)
    all_features.append(garch_df)
    
    # Momentum features
    mom_features = compute_momentum(df)
    all_features.append(mom_features)
    
    # Relative strength
    if benchmark is not None:
        rs_features = compute_relative_strength(df, benchmark)
        all_features.append(rs_features)
    
    # Volume features
    vol_ratio_features = compute_volume_features(df)
    all_features.append(vol_ratio_features)
    
    # ADX
    adx = compute_adx(df)
    adx_df = pd.DataFrame({'adx': adx}, index=df.index[1:])
    all_features.append(adx_df)
    
    # RSI
    rsi = compute_rsi(df)
    rsi_df = pd.DataFrame({'rsi': rsi}, index=df.index)
    all_features.append(rsi_df)
    
    # ATR
    atr = compute_atr(df)
    atr_df = pd.DataFrame({'atr': atr}, index=df.index)
    all_features.append(atr_df)
    
    # FII/DII features
    if fii_dii_df is not None:
        fii_dii_features = compute_fii_dii_features(fii_dii_df)
        all_features.append(fii_dii_features)
    
    # Combine all features
    combined = pd.concat(all_features, axis=1)
    
    # Drop rows with all NaN
    combined = combined.dropna(how='all')
    
    return combined
