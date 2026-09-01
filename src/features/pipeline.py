"""Expanding-window feature pipeline.

All features computed on expanding window only to prevent look-ahead bias.
No future information leaks into training.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardSplit:
    """A single walk-forward split."""
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_date: pd.Timestamp
    X_train: pd.DataFrame
    X_test: pd.DataFrame


class ExpandingWindowPipeline:
    """Feature pipeline with expanding-window validation.
    
    Fits all transformers on training data only, then transforms test data.
    Never leaks future information into training.
    """
    
    def __init__(
        self,
        feature_fns: list[Callable[[pd.DataFrame], pd.DataFrame]],
        scaler: StandardScaler | None = None,
    ):
        self.feature_fns = feature_fns
        self.scaler = scaler or StandardScaler()
        self._is_fitted = False
    
    def fit_transform_train(
        self, 
        df: pd.DataFrame, 
        train_end: pd.Timestamp
    ) -> pd.DataFrame:
        """Fit on training data and transform training data.
        
        Args:
            df: Full OHLCV DataFrame
            train_end: Last date for training
            
        Returns:
            Transformed training features
        """
        train_df = df.loc[:train_end]
        
        # Compute features on train data only
        features = self._compute_features(train_df)
        
        # Fit scaler on train data only
        self.scaler.fit(features)
        self._is_fitted = True
        
        # Transform train data
        transformed = self.scaler.transform(features)
        return pd.DataFrame(
            transformed, 
            index=features.index, 
            columns=features.columns
        )
    
    def transform_test(self, df: pd.DataFrame, test_date: pd.Timestamp) -> pd.DataFrame:
        """Transform test data using fitted scaler.
        
        Args:
            df: Full OHLCV DataFrame
            test_date: Date to predict for
            
        Returns:
            Transformed test features
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline not fitted. Call fit_transform_train first.")
        
        test_df = df.loc[test_date:test_date]
        features = self._compute_features(test_df)
        
        # Transform using train's scaler params (no refit)
        transformed = self.scaler.transform(features)
        return pd.DataFrame(
            transformed, 
            index=features.index, 
            columns=features.columns
        )
    
    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all features on a DataFrame."""
        all_features = []
        
        for fn in self.feature_fns:
            try:
                result = fn(df)
                if result is not None and not result.empty:
                    all_features.append(result)
            except Exception as e:
                logger.warning(f"Feature function {fn.__name__} failed: {e}")
        
        if not all_features:
            return pd.DataFrame()
        
        # Concatenate all features
        combined = pd.concat(all_features, axis=1)
        
        # Drop rows with all NaN
        combined = combined.dropna(how='all')
        
        return combined


def generate_walk_forward_splits(
    df: pd.DataFrame,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    train_window: int = 252,
    step: int = 1,
) -> list[WalkForwardSplit]:
    """Generate walk-forward splits.
    
    Args:
        df: Full OHLCV DataFrame
        start_date: Start of walk-forward period
        end_date: End of walk-forward period
        train_window: Minimum training window (trading days)
        step: Step size (days)
        
    Returns:
        List of WalkForwardSplit objects
    """
    if isinstance(start_date, str):
        start_date = pd.Timestamp(start_date)
    if isinstance(end_date, str):
        end_date = pd.Timestamp(end_date)
    
    splits = []
    current_date = start_date
    
    while current_date <= end_date:
        # Find the index position of current_date
        dates = df.index
        mask = dates <= current_date
        train_count = mask.sum()
        
        if train_count < train_window:
            current_date += pd.Timedelta(days=step)
            continue
        
        train_end = dates[mask][-1]
        
        splits.append(WalkForwardSplit(
            train_start=dates[0],
            train_end=train_end,
            test_date=current_date,
            X_train=pd.DataFrame(),
            X_test=pd.DataFrame()
        ))
        
        current_date += pd.Timedelta(days=step)
    
    return splits
