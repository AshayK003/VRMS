"""Test feature engineering module."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.engineering import (
    compute_adx,
    compute_atr,
    compute_garch_vol,
    compute_momentum,
    compute_realized_vol,
    compute_relative_strength,
    compute_rsi,
    compute_volume_features,
)


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Create sample OHLCV DataFrame for testing."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.5)
    low = close - np.abs(np.random.randn(n) * 0.5)
    open_price = close + np.random.randn(n) * 0.3
    volume = np.random.randint(100000, 1000000, n)
    
    return pd.DataFrame({
        "Open": open_price,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    }, index=dates)


class TestComputeRealizedVol:
    def test_returns_dataframe(self, sample_ohlcv):
        result = compute_realized_vol(sample_ohlcv)
        assert isinstance(result, pd.DataFrame)
    
    def test_correct_columns(self, sample_ohlcv):
        result = compute_realized_vol(sample_ohlcv, windows=[5, 10, 20])
        assert "vol_5d" in result.columns
        assert "vol_10d" in result.columns
        assert "vol_20d" in result.columns
    
    def test_values_non_negative(self, sample_ohlcv):
        result = compute_realized_vol(sample_ohlcv).dropna()
        assert (result >= 0).all().all()
    
    def test_nan_for_initial_rows(self, sample_ohlcv):
        result = compute_realized_vol(sample_ohlcv, windows=[20])
        # First 19 rows should be NaN
        assert result["vol_20d"].iloc[:19].isna().all()


class TestComputeMomentum:
    def test_returns_dataframe(self, sample_ohlcv):
        result = compute_momentum(sample_ohlcv)
        assert isinstance(result, pd.DataFrame)
    
    def test_correct_columns(self, sample_ohlcv):
        result = compute_momentum(sample_ohlcv, windows=[21, 63])
        assert "mom_21d" in result.columns
        assert "mom_63d" in result.columns
    
    def test_momentum_calculation(self, sample_ohlcv):
        result = compute_momentum(sample_ohlcv, windows=[5])
        # Momentum should be (close / close.shift(5)) - 1
        expected = sample_ohlcv["Close"].pct_change(5)
        pd.testing.assert_series_equal(result["mom_5d"], expected, check_names=False)


class TestComputeADX:
    def test_returns_series(self, sample_ohlcv):
        result = compute_adx(sample_ohlcv)
        assert isinstance(result, pd.Series)
    
    def test_first_value_nan(self, sample_ohlcv):
        result = compute_adx(sample_ohlcv)
        assert np.isnan(result.iloc[0])
    
    def test_values_in_range(self, sample_ohlcv):
        result = compute_adx(sample_ohlcv)
        # ADX should be between 0 and 100
        valid = result.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()


class TestComputeRSI:
    def test_returns_series(self, sample_ohlcv):
        result = compute_rsi(sample_ohlcv)
        assert isinstance(result, pd.Series)
    
    def test_values_in_range(self, sample_ohlcv):
        result = compute_rsi(sample_ohlcv)
        # RSI should be between 0 and 100
        valid = result.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()


class TestComputeATR:
    def test_returns_series(self, sample_ohlcv):
        result = compute_atr(sample_ohlcv)
        assert isinstance(result, pd.Series)
    
    def test_values_positive(self, sample_ohlcv):
        result = compute_atr(sample_ohlcv)
        valid = result.dropna()
        assert (valid > 0).all()


class TestComputeVolumeFeatures:
    def test_returns_dataframe(self, sample_ohlcv):
        result = compute_volume_features(sample_ohlcv)
        assert isinstance(result, pd.DataFrame)
    
    def test_correct_columns(self, sample_ohlcv):
        result = compute_volume_features(sample_ohlcv)
        assert "volume_ratio" in result.columns
        assert "circuit_flag" in result.columns
    
    def test_circuit_flag_binary(self, sample_ohlcv):
        result = compute_volume_features(sample_ohlcv)
        assert result["circuit_flag"].isin([0, 1]).all()


class TestComputeGarchVol:
    def test_returns_series(self, sample_ohlcv):
        result = compute_garch_vol(sample_ohlcv)
        assert isinstance(result, pd.Series)
    
    def test_values_non_negative(self, sample_ohlcv):
        result = compute_garch_vol(sample_ohlcv)
        valid = result.dropna()
        assert (valid >= 0).all()


class TestComputeRelativeStrength:
    def test_returns_dataframe(self, sample_ohlcv):
        benchmark = sample_ohlcv.copy()
        result = compute_relative_strength(sample_ohlcv, benchmark)
        assert isinstance(result, pd.DataFrame)
    
    def test_correct_columns(self, sample_ohlcv):
        benchmark = sample_ohlcv.copy()
        result = compute_relative_strength(sample_ohlcv, benchmark, windows=[21])
        assert "rs_21d" in result.columns
