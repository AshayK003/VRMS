"""Backtest metrics: Deflated Sharpe, bootstrap confidence intervals."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_deflated_sharpe(returns: list[float], n_trials: int = 1) -> float:
    """Compute Deflated Sharpe Ratio.
    
    Controls for multiple testing. If you tried 10 models and picked the best,
    the Deflated Sharpe accounts for that.
    
    Args:
        returns: List of trade returns
        n_trials: Number of models tested (default 1)
        
    Returns:
        Deflated Sharpe ratio
    """
    if not returns or np.std(returns) == 0:
        return 0.0
    
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
    
    # Deflation factor (Bailey & Lopez de Prado 2014)
    # DS = SR * sqrt(T) / sqrt(n_trials)
    # Simplified: DS = SR / sqrt(n_trials)
    deflated = sharpe / np.sqrt(n_trials)
    
    return deflated


def bootstrap_confidence(
    returns: list[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence intervals.
    
    Args:
        returns: List of trade returns
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level
        
    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    if not returns:
        return 0.0, 0.0, 0.0
    
    returns = np.array(returns)
    n = len(returns)
    
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(returns, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))
    
    bootstrap_means = sorted(bootstrap_means)
    
    lower_idx = int((1 - confidence) / 2 * n_bootstrap)
    upper_idx = int((1 + confidence) / 2 * n_bootstrap)
    
    mean = np.mean(returns)
    lower = bootstrap_means[lower_idx]
    upper = bootstrap_means[upper_idx]
    
    return mean, lower, upper


def compute_win_rate_ci(
    n_wins: int,
    n_trades: int,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Compute Wilson score interval for win rate.
    
    Args:
        n_wins: Number of winning trades
        n_trades: Total number of trades
        confidence: Confidence level
        
    Returns:
        Tuple of (win_rate, lower_bound, upper_bound)
    """
    if n_trades == 0:
        return 0.0, 0.0, 0.0
    
    from scipy import stats
    
    z = stats.norm.ppf((1 + confidence) / 2)
    p = n_wins / n_trades
    
    denominator = 1 + z**2 / n_trades
    center = (p + z**2 / (2 * n_trades)) / denominator
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n_trades)) / n_trades) / denominator
    
    win_rate = p
    lower = max(0, center - margin)
    upper = min(1, center + margin)
    
    return win_rate, lower, upper


def compute_profit_factor(returns: list[float]) -> float:
    """Compute profit factor (gross profits / gross losses).
    
    Args:
        returns: List of trade returns
        
    Returns:
        Profit factor (>1 is profitable)
    """
    gross_profits = sum(r for r in returns if r > 0)
    gross_losses = abs(sum(r for r in returns if r < 0))
    
    if gross_losses == 0:
        return float('inf') if gross_profits > 0 else 0.0
    
    return gross_profits / gross_losses


def compute_expectancy(returns: list[float]) -> float:
    """Compute expectancy (average return per trade).
    
    Args:
        returns: List of trade returns
        
    Returns:
        Expectancy value
    """
    if not returns:
        return 0.0
    
    return np.mean(returns)
