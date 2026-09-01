"""Position sizing for VRMS.

Volatility-adjusted position sizing: risk a fixed budget per stop-out.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def calc_position_size(
    risk_budget_rupees: float,
    stop_distance_price: float,
    point_value: float = 1.0,
) -> float:
    """Volatility-based position size.
    
    units = risk_budget / (stop_distance_price * point_value)
    
    Args:
        risk_budget_rupees: Maximum rupees to risk per trade
        stop_distance_price: Stop loss distance in price units
        point_value: Rupees per price-point per share (1.0 for equity)
        
    Returns:
        Number of shares to buy
    """
    if stop_distance_price <= 0 or point_value <= 0:
        return 0.0
    
    risk_per_unit = stop_distance_price * point_value
    return max(0.0, risk_budget_rupees / risk_per_unit)


def calc_position_size_with_governance(
    risk_budget_rupees: float,
    stop_distance_price: float,
    risk_multiplier: float = 1.0,
    point_value: float = 1.0,
) -> float:
    """Position size with governance risk multiplier.
    
    Args:
        risk_budget_rupees: Maximum rupees to risk per trade
        stop_distance_price: Stop loss distance in price units
        risk_multiplier: Governance risk multiplier (0.5 = half size)
        point_value: Rupees per price-point per share
        
    Returns:
        Number of shares to buy
    """
    adjusted_budget = risk_budget_rupees * risk_multiplier
    return calc_position_size(adjusted_budget, stop_distance_price, point_value)
