"""Data validation layer for VRMS."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

import pandas as pd

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates data quality before feature computation."""
    
    def __init__(
        self,
        max_missing_pct: float = 0.05,
        max_stale_days: int = 3,
        min_non_zero_days: int = 100
    ):
        self.max_missing_pct = max_missing_pct
        self.max_stale_days = max_stale_days
        self.min_non_zero_days = min_non_zero_days
    
    def validate_ohlcv(self, df: pd.DataFrame) -> tuple[bool, str]:
        """Validate OHLCV data.
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            Tuple of (is_valid, reason)
        """
        if df is None or df.empty:
            return False, "Empty DataFrame"
        
        # Check required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            return False, f"Missing columns: {set(required_cols) - set(df.columns)}"
        
        # Check for missing values
        missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns))
        if missing_pct > self.max_missing_pct:
            return False, f"Too many missing values: {missing_pct:.1%}"
        
        # Check for stale data
        last_date = pd.Timestamp(df.index.max())
        days_since_last = (pd.Timestamp.now() - last_date).days
        if days_since_last > self.max_stale_days:
            return False, f"Data is {days_since_last} days stale"
        
        # Check for zero-volume days
        zero_vol_days = (df['Volume'] == 0).sum()
        if zero_vol_days > len(df) * 0.5:
            return False, f"Too many zero-volume days: {zero_vol_days}"
        
        # Check for flat line (delisted?)
        if df['Close'].std() == 0:
            return False, "Flat line - possibly delisted"
        
        # Check for OHLC consistency
        if not (df['High'] >= df['Low']).all():
            return False, "High < Low detected"
        
        if not ((df['Close'] >= df['Low']) & (df['Close'] <= df['High'])).all():
            return False, "Close outside High-Low range"
        
        return True, "OK"
    
    def validate_fii_dii(self, df: pd.DataFrame) -> tuple[bool, str]:
        """Validate FII/DII data."""
        if df is None or df.empty:
            return False, "Empty DataFrame"
        
        required_cols = ['FII', 'DII', 'Net']
        if not all(col in df.columns for col in required_cols):
            return False, f"Missing columns: {set(required_cols) - set(df.columns)}"
        
        # Check for stale data
        last_date = pd.Timestamp(df.index.max())
        days_since_last = (pd.Timestamp.now() - last_date).days
        if days_since_last > self.max_stale_days:
            return False, f"FII/DII data is {days_since_last} days stale"
        
        return True, "OK"
    
    def validate_features(self, df: pd.DataFrame) -> tuple[bool, str]:
        """Validate feature DataFrame."""
        if df is None or df.empty:
            return False, "Empty feature DataFrame"
        
        # Check for NaN features
        nan_features = df.columns[df.isnull().any()].tolist()
        if len(nan_features) > len(df.columns) * 0.5:
            return False, f"Too many NaN features: {len(nan_features)}"
        
        # Check for infinite values
        if df.abs().max().max() > 1e10:
            return False, "Infinite values detected"
        
        return True, "OK"
