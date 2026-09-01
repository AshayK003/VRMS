"""Signal generator for VRMS.

Generates Top 5 BUY signals with stop loss and target.
"""
from __future__ import annotations

import logging

import pandas as pd

from src.signals.governance import apply_governance

logger = logging.getLogger(__name__)


def generate_signals(
    predictions: pd.DataFrame,
    top_n: int = 5,
    vix: float | None = None,
    adx: float | None = None,
    ad_ratio: float | None = None,
    fii_dii_bias: str = "NEUTRAL",
    is_expiry: bool = False,
) -> list[dict]:
    """Generate Top N signals from predictions.
    
    Args:
        predictions: DataFrame with 'symbol', 'probability', 'momentum', 'rs'
        top_n: Number of signals to generate
        vix: Current VIX value
        adx: Current ADX value
        ad_ratio: Advance-decline ratio
        fii_dii_bias: FII/DII bias
        is_expiry: Whether today is expiry day
        
    Returns:
        List of signal dicts
    """
    if predictions.empty:
        return []
    
    predictions = predictions.copy()
    
    # Rank by composite score
    momentum = predictions.get('momentum', pd.Series([0] * len(predictions)))
    rs = predictions.get('rs', pd.Series([0] * len(predictions)))
    predictions['score'] = predictions['probability'] * 0.5 + momentum * 0.3 + rs * 0.2
    predictions = predictions.sort_values('score', ascending=False)
    
    signals = []
    for _, row in predictions.head(top_n).iterrows():
        signal = {
            'symbol': row['symbol'],
            'direction': 'LONG',
            'probability': row['probability'],
            'score': row['score'],
            'momentum': row.get('momentum'),
            'rs': row.get('rs'),
        }
        
        signal = apply_governance(
            signal,
            vix=vix,
            adx=adx,
            ad_ratio=ad_ratio,
            fii_dii_bias=fii_dii_bias,
            is_expiry=is_expiry,
        )
        
        if signal['allowed']:
            signal['stop_loss_pct'] = 0.05
            signal['target_pct'] = 0.05
            signals.append(signal)
    
    return signals
