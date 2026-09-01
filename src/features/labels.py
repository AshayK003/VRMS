"""Label generator for VRMS.

Defines precise WIN/LOSS labels with corporate action exclusion.
"""
from __future__ import annotations

import logging
from datetime import timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_labels(
    df: pd.DataFrame,
    target_pct: float = 0.05,
    horizon: int = 5,
    stop_loss_pct: float = 0.05,
) -> pd.DataFrame:
    """Generate WIN/LOSS labels for each date.
    
    Label definition:
    - WIN (1): Close at T+horizon >= (1 + target_pct) * Close at T
                AND no close <= (1 - stop_loss_pct) * Close at T in between
    - LOSS (0): Close at T+horizon < (1 + target_pct) * Close at T
                OR any close <= (1 - stop_loss_pct) * Close at T
    
    Args:
        df: OHLCV DataFrame
        target_pct: Target gain percentage (default 5%)
        horizon: Holding period in days (default 5)
        stop_loss_pct: Stop loss percentage (default 5%)
        
    Returns:
        DataFrame with 'label' column (1=WIN, 0=LOSS)
    """
    result = pd.DataFrame(index=df.index)
    result['label'] = np.nan
    
    for i in range(len(df) - horizon):
        entry_date = df.index[i]
        entry_price = df['Close'].iloc[i]
        
        # Check if we have enough data for the horizon
        if i + horizon >= len(df):
            continue
        
        # Get prices during the holding period
        period_prices = df['Close'].iloc[i+1:i+horizon+1]
        
        # Check stop loss
        stop_loss_hit = (period_prices <= entry_price * (1 - stop_loss_pct)).any()
        
        if stop_loss_hit:
            result.loc[entry_date, 'label'] = 0
            continue
        
        # Check target
        exit_price = df['Close'].iloc[i + horizon]
        target_hit = exit_price >= entry_price * (1 + target_pct)
        
        if target_hit:
            result.loc[entry_date, 'label'] = 1
        else:
            result.loc[entry_date, 'label'] = 0
    
    return result


def filter_labels_by_corporate_actions(
    labels: pd.DataFrame,
    symbol: str,
    corporate_actions: dict,
) -> pd.DataFrame:
    """Filter out labels on corporate action days.
    
    Args:
        labels: Label DataFrame
        symbol: NSE ticker
        corporate_actions: Corporate action calendar
        
    Returns:
        Filtered labels
    """
    from src.data.corporate_actions import is_corporate_action_day
    
    mask = labels.index.map(
        lambda d: not is_corporate_action_day(symbol, d)[0]
    )
    
    return labels[mask]
