"""Governance filters for VRMS signals.

Filters signals based on market context (VIX, ADX, breadth, FII/DII, expiry).
"""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Thresholds
VIX_PANIC_THRESHOLD = 22
ADX_DOLDRUMS_THRESHOLD = 15
AD_RATIO_WEAK = 0.7
AD_RATIO_STRONG = 1.5


def filter_vix_spike(vix: float | None) -> tuple[bool, float, str | None]:
    """Check if VIX is too high for reliable signals.
    
    Returns:
        Tuple of (pass, risk_multiplier, reason)
    """
    if vix is not None and vix > VIX_PANIC_THRESHOLD:
        return True, 0.5, f"VIX at {vix} (>{VIX_PANIC_THRESHOLD}) — reduce size"
    return True, 1.0, None


def filter_adx_doldrums(adx: float | None) -> tuple[bool, float, str | None]:
    """Check if market is range-bound (low ADX).
    
    Returns:
        Tuple of (pass, risk_multiplier, reason)
    """
    if adx is not None and adx < ADX_DOLDRUMS_THRESHOLD:
        return True, 0.5, f"ADX at {adx} (<{ADX_DOLDRUMS_THRESHOLD}) — reduce size"
    return True, 1.0, None


def filter_breadth(ad_ratio: float | None, direction: str) -> tuple[bool, float, str | None]:
    """Check if market breadth confirms signal direction.
    
    Returns:
        Tuple of (pass, risk_multiplier, reason)
    """
    if ad_ratio is not None:
        if direction == "LONG" and ad_ratio < AD_RATIO_WEAK:
            return True, 0.5, f"A/D ratio {ad_ratio:.2f} (<{AD_RATIO_WEAK}) — reduce size"
        if direction == "SHORT" and ad_ratio > AD_RATIO_STRONG:
            return True, 0.5, f"A/D ratio {ad_ratio:.2f} (>{AD_RATIO_STRONG}) — reduce size"
    return True, 1.0, None


def filter_fii_dii_bias(fii_dii_bias: str, direction: str) -> tuple[bool, float, str | None]:
    """Check if FII/DII flow contradicts signal direction.
    
    Returns:
        Tuple of (pass, risk_multiplier, reason)
    """
    if fii_dii_bias == "BEARISH" and direction == "LONG":
        return True, 0.5, "FII/DII bearish — reduce long size"
    if fii_dii_bias == "BULLISH" and direction == "SHORT":
        return True, 0.5, "FII/DII bullish — reduce short size"
    return True, 1.0, None


def filter_expiry_day(is_expiry: bool) -> tuple[bool, float, str | None]:
    """Block signals on expiry day.
    
    Returns:
        Tuple of (pass, risk_multiplier, reason)
    """
    if is_expiry:
        return False, 0.0, "Expiry day — signal blocked"
    return True, 1.0, None


def apply_governance(
    signal: dict,
    vix: float | None = None,
    adx: float | None = None,
    ad_ratio: float | None = None,
    fii_dii_bias: str = "NEUTRAL",
    is_expiry: bool = False,
) -> dict:
    """Apply all governance filters to a signal.
    
    Args:
        signal: Signal dict with 'direction', 'probability', etc.
        vix: Current VIX value
        adx: Current ADX value
        ad_ratio: Advance-decline ratio
        fii_dii_bias: FII/DII bias ('BULLISH', 'BEARISH', 'NEUTRAL')
        is_expiry: Whether today is expiry day
        
    Returns:
        Signal dict with 'allowed', 'risk_multiplier', 'reasons' added
    """
    filters = [
        filter_vix_spike(vix),
        filter_adx_doldrums(adx),
        filter_breadth(ad_ratio, signal.get('direction', 'LONG')),
        filter_fii_dii_bias(fii_dii_bias, signal.get('direction', 'LONG')),
        filter_expiry_day(is_expiry),
    ]
    
    allowed = True
    risk_mult = 1.0
    reasons = []
    
    for pass_filter, mult, reason in filters:
        if not pass_filter:
            allowed = False
            risk_mult = 0.0
            reasons.append(reason)
            break
        if mult < risk_mult:
            risk_mult = mult
        if reason:
            reasons.append(reason)
    
    signal['allowed'] = allowed
    signal['risk_multiplier'] = risk_mult
    signal['governance_reasons'] = reasons
    
    return signal
