"""FII/DII flow features with temporal alignment."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_fii_dii_features(fii_dii_df: pd.DataFrame, windows: list[int] = [5, 20]) -> pd.DataFrame:
    """Compute FII/DII flow features.
    
    Args:
        fii_dii_df: DataFrame with FII, DII, Net columns
        windows: Rolling windows for flow trends
        
    Returns:
        DataFrame with FII/DII features
    """
    result = pd.DataFrame(index=fii_dii_df.index)
    
    for w in windows:
        # Rolling flow trends
        result[f'fii_flow_{w}d'] = fii_dii_df['FII'].rolling(w).sum()
        result[f'dii_flow_{w}d'] = fii_dii_df['DII'].rolling(w).sum()
        result[f'net_flow_{w}d'] = fii_dii_df['Net'].rolling(w).sum()
        
        # Flow momentum (acceleration)
        result[f'fii_momentum_{w}d'] = fii_dii_df['FII'].diff(w)
        result[f'dii_momentum_{w}d'] = fii_dii_df['DII'].diff(w)
    
    # Flow direction (1 = net inflow, -1 = net outflow)
    result['flow_direction'] = np.sign(fii_dii_df['Net'])
    
    # Cumulative flow
    result['cumulative_net_flow'] = fii_dii_df['Net'].cumsum()
    
    return result


def compute_flow_correlation(
    stock_returns: pd.Series, 
    fii_dii_df: pd.DataFrame, 
    window: int = 21
) -> pd.Series:
    """Compute correlation between stock returns and FII/DII flows.
    
    Args:
        stock_returns: Stock daily returns
        fii_dii_df: FII/DII DataFrame
        window: Rolling window
        
    Returns:
        Correlation series
    """
    # Align indices
    common_idx = stock_returns.index.intersection(fii_dii_df.index)
    aligned_returns = stock_returns.loc[common_idx]
    aligned_flows = fii_dii_df['Net'].loc[common_idx]
    
    # Rolling correlation
    correlation = aligned_returns.rolling(window).corr(aligned_flows)
    
    return correlation
