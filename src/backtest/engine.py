"""Walk-forward backtest engine.

Trains on expanding window, tests on next day. Never leaks future info.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Results from a walk-forward backtest."""
    dates: list[pd.Timestamp]
    predictions: list[int]
    probabilities: list[float]
    actuals: list[int]
    equity_curve: list[float]
    returns: list[float]
    win_rate: float
    sharpe: float
    max_drawdown: float
    total_return: float
    n_trades: int
    n_wins: int
    n_losses: float


def walk_forward_backtest(
    features: pd.DataFrame,
    labels: pd.Series,
    model_cls,
    train_window: int = 252,
    step: int = 1,
    transaction_cost: float = 0.003,
    **model_kwargs,
) -> BacktestResult:
    """Run walk-forward backtest.
    
    Args:
        features: Feature DataFrame
        labels: Label Series (1=WIN, 0=LOSS)
        model_cls: Model class (e.g., XGBoostClassifier)
        train_window: Minimum training window
        step: Step size (days)
        transaction_cost: Round-trip transaction cost (default 0.3%)
        **model_kwargs: Model hyperparameters
        
    Returns:
        BacktestResult with metrics
    """
    # Align features and labels
    common_idx = features.index.intersection(labels.index)
    features = features.loc[common_idx]
    labels = labels.loc[common_idx]
    
    # Drop NaN rows
    mask = ~(features.isnull().any(axis=1) | labels.isnull())
    features = features[mask]
    labels = labels[mask]
    
    dates = []
    predictions = []
    probabilities = []
    actuals = []
    equity_curve = [1.0]
    returns = []
    
    # Generate walk-forward splits
    for i in range(train_window, len(features) - 1, step):
        # Training data: 0 to i-1
        X_train = features.iloc[:i]
        y_train = labels.iloc[:i]
        
        # Test data: i
        X_test = features.iloc[i:i+1]
        y_test = labels.iloc[i]
        
        if y_test == 0 or np.isnan(y_test):
            continue
        
        # Train model
        model = model_cls(**model_kwargs)
        model.fit(X_train, y_train)
        
        # Predict
        prob = model.predict_proba(X_test).iloc[0]
        pred = 1 if prob > 0.5 else 0
        
        # Calculate return
        ret = 0.0
        if pred == 1 and y_test == 1:
            ret = 0.05 - transaction_cost  # Win: 5% gain minus costs
        elif pred == 1 and y_test == 0:
            ret = -0.05 - transaction_cost  # Loss: 5% stop-loss minus costs
        
        # Update equity curve
        new_equity = equity_curve[-1] * (1 + ret)
        equity_curve.append(new_equity)
        
        dates.append(features.index[i])
        predictions.append(pred)
        probabilities.append(prob)
        actuals.append(y_test)
        returns.append(ret)
    
    # Calculate metrics
    n_trades = len(predictions)
    n_wins = sum(1 for p, a in zip(predictions, actuals) if p == 1 and a == 1)
    n_losses = sum(1 for p, a in zip(predictions, actuals) if p == 1 and a == 0)
    
    win_rate = n_wins / (n_wins + n_losses) if (n_wins + n_losses) > 0 else 0.0
    
    # Sharpe ratio (annualized)
    if returns and np.std(returns) > 0:
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
    else:
        sharpe = 0.0
    
    # Max drawdown
    if equity_curve:
        peak = equity_curve[0]
        max_dd = 0.0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak
            if dd > max_dd:
                max_dd = dd
        max_drawdown = max_dd
    else:
        max_drawdown = 0.0
    
    total_return = equity_curve[-1] - 1.0 if equity_curve else 0.0
    
    return BacktestResult(
        dates=dates,
        predictions=predictions,
        probabilities=probabilities,
        actuals=actuals,
        equity_curve=equity_curve,
        returns=returns,
        win_rate=win_rate,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        total_return=total_return,
        n_trades=n_trades,
        n_wins=n_wins,
        n_losses=n_losses,
    )
