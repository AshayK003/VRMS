"""Walk-forward backtest for VRMS.

Trains on expanding window, tests on next N days. No look-ahead bias.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.data.ohlcv import fetch_ohlcv, get_benchmark
from src.data.fii_dii import fetch_fii_dii
from src.features.engineering import (
    compute_realized_vol, compute_momentum, compute_relative_strength,
    compute_volume_features, compute_adx, compute_rsi, compute_atr, compute_garch_vol,
)
from src.features.labels import generate_labels
from src.models.xgboost import XGBoostClassifier

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Backtest results."""
    dates: list
    predictions: list
    actuals: list
    equity_curve: list
    returns: list
    win_rate: float
    sharpe: float
    max_drawdown: float
    total_return: float
    n_trades: int
    n_wins: int
    n_losses: int
    avg_win: float
    avg_loss: float
    profit_factor: float


def run_walk_forward_backtest(
    symbols: list[str],
    start_date: str,
    end_date: str,
    train_window: int = 252,
    test_window: int = 21,
    transaction_cost: float = 0.003,
    target_pct: float = 0.05,
    stop_loss_pct: float = 0.05,
    threshold: float = 0.5,
) -> BacktestResult:
    """Run walk-forward backtest.
    
    Args:
        symbols: List of NSE tickers
        start_date: Start date
        end_date: End date
        train_window: Training window in days
        test_window: Testing window in days
        transaction_cost: Round-trip cost
        target_pct: Target gain
        stop_loss_pct: Stop loss
        threshold: Probability threshold for signals
        
    Returns:
        BacktestResult with metrics
    """
    # Fetch all data
    logger.info("Fetching data...")
    stock_data = {}
    for s in symbols[:10]:  # Limit to 10 for speed
        df = fetch_ohlcv(s, start_date, end_date)
        if not df.empty:
            stock_data[s] = df
    
    benchmark = get_benchmark(start_date, end_date)
    
    logger.info(f"Fetched {len(stock_data)} stocks")
    
    # Compute features for all stocks
    all_features = {}
    all_labels = {}
    
    for symbol, df in stock_data.items():
        try:
            feats = _compute_features(df, benchmark)
            labels = generate_labels(df, target_pct=target_pct, horizon=5, stop_loss_pct=stop_loss_pct)
            
            if not feats.empty and not labels.empty:
                all_features[symbol] = feats
                all_labels[symbol] = labels
        except Exception as e:
            logger.warning(f"Failed {symbol}: {e}")
    
    logger.info(f"Computed features for {len(all_features)} stocks")
    
    # Get all unique dates
    all_dates = set()
    for df in all_features.values():
        all_dates.update(df.index)
    all_dates = sorted(all_dates)
    
    if len(all_dates) < train_window + test_window:
        raise ValueError("Not enough data for backtest")
    
    # Walk-forward
    dates = []
    predictions = []
    actuals = []
    equity_curve = [1.0]
    returns = []
    
    logger.info(f"Running walk-forward from {all_dates[0]} to {all_dates[-1]}")
    
    i = train_window
    while i < len(all_dates) - test_window:
        train_end = all_dates[i]
        test_end = all_dates[min(i + test_window, len(all_dates) - 1)]
        
        # Build training set
        train_X = []
        train_y = []
        
        for symbol in all_features:
            X = all_features[symbol]
            y = all_labels[symbol]['label']
            
            # Get training data up to train_end
            mask = X.index <= train_end
            X_train = X[mask]
            y_train = y[X.index[mask]]
            
            # Align and drop NaN
            common = X_train.index.intersection(y_train.index)
            X_train = X_train.loc[common]
            y_train = y_train.loc[common]
            
            mask_nan = ~(X_train.isnull().any(axis=1) | y_train.isnull())
            X_train = X_train[mask_nan]
            y_train = y_train[mask_nan]
            
            if len(X_train) > 0:
                train_X.append(X_train.values.astype(np.float32))
                train_y.append(y_train.values.astype(np.int32))
        
        if not train_X:
            i += test_window
            continue
        
        X_train_all = np.vstack(train_X)
        y_train_all = np.concatenate(train_y)
        
        # Drop NaN
        mask = ~np.isnan(X_train_all).any(axis=1)
        X_train_all = X_train_all[mask]
        y_train_all = y_train_all[mask]
        
        if len(X_train_all) < 100:
            i += test_window
            continue
        
        # PCA
        n_components = min(10, X_train_all.shape[1])
        pca = PCA(n_components=n_components)
        X_train_pca = pca.fit_transform(X_train_all)
        
        # Train model
        model = XGBoostClassifier(n_estimators=50, max_depth=3, learning_rate=0.05)
        model.fit(X_train_pca, y_train_all)
        
        # Test on next window
        test_mask = [(all_dates.index(train_end) < all_dates.index(d) <= all_dates.index(test_end)) 
                      if d in all_dates else False for d in all_dates]
        test_dates = [d for d, m in zip(all_dates, test_mask) if m]
        
        if not test_dates:
            i += test_window
            continue
        
        # Get predictions for test period
        for test_date in test_dates:
            day_predictions = []
            day_actuals = []
            
            for symbol in all_features:
                X = all_features[symbol]
                y = all_labels[symbol]['label']
                
                if test_date not in X.index or test_date not in y.index:
                    continue
                
                X_test = X.loc[test_date:test_date]
                y_test = y.loc[test_date:test_date]
                
                if X_test.isnull().any(axis=1).iloc[0] or y_test.isnull().iloc[0]:
                    continue
                
                # PCA transform
                X_test_arr = X_test.values.astype(np.float32)
                X_test_pca = pca.transform(X_test_arr)
                
                # Predict
                prob = model.predict_proba(X_test_pca)[0]
                
                if prob >= threshold:
                    day_predictions.append(prob)
                    day_actuals.append(y_test.iloc[0])
            
            if day_predictions:
                # Take top prediction
                best_idx = np.argmax(day_predictions)
                pred_prob = day_predictions[best_idx]
                actual = day_actuals[best_idx]
                
                # Calculate return
                ret = 0.0
                if actual == 1:
                    ret = target_pct - transaction_cost
                else:
                    ret = -stop_loss_pct - transaction_cost
                
                dates.append(test_date)
                predictions.append(pred_prob)
                actuals.append(actual)
                equity_curve.append(equity_curve[-1] * (1 + ret))
                returns.append(ret)
        
        i += test_window
    
    # Calculate metrics
    n_trades = len(predictions)
    n_wins = sum(1 for a in actuals if a == 1)
    n_losses = sum(1 for a in actuals if a == 0)
    
    win_rate = n_wins / n_trades if n_trades > 0 else 0.0
    
    if returns and np.std(returns) > 0:
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
    else:
        sharpe = 0.0
    
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
    
    avg_win = np.mean([r for r in returns if r > 0]) if any(r > 0 for r in returns) else 0.0
    avg_loss = np.mean([r for r in returns if r < 0]) if any(r < 0 for r in returns) else 0.0
    
    gross_profits = sum(r for r in returns if r > 0)
    gross_losses = abs(sum(r for r in returns if r < 0))
    profit_factor = gross_profits / gross_losses if gross_losses > 0 else float('inf')
    
    return BacktestResult(
        dates=dates,
        predictions=predictions,
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
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=profit_factor,
    )


def _compute_features(df: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    """Compute features for a stock."""
    all_features = []
    
    all_features.append(compute_realized_vol(df, windows=[5, 10, 20]))
    
    garch = compute_garch_vol(df)
    all_features.append(pd.DataFrame({'garch_vol': garch}, index=df.index))
    
    all_features.append(compute_momentum(df, windows=[21, 63]))
    
    all_features.append(compute_relative_strength(df, benchmark, windows=[21]))
    
    all_features.append(compute_volume_features(df))
    
    adx = compute_adx(df)
    all_features.append(pd.DataFrame({'adx': adx}, index=df.index))
    
    rsi = compute_rsi(df)
    all_features.append(pd.DataFrame({'rsi': rsi}, index=df.index))
    
    atr = compute_atr(df)
    all_features.append(pd.DataFrame({'atr': atr}, index=df.index))
    
    combined = pd.concat(all_features, axis=1)
    return combined.dropna(how='all')
