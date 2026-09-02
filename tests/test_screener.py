"""Test screener module."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.screener.multi_asset import (
    NIFTY_50,
    compute_conviction,
    compute_momentum,
    get_vix_regime,
)


class TestNifty50:
    def test_50_stocks(self):
        assert len(NIFTY_50) == 50
    
    def test_tmcv_present(self):
        assert "TMCV.NS" in NIFTY_50
    
    def test_tatamotors_absent(self):
        # Old ticker should not be in the list
        assert "TATAMOTORS.NS" not in NIFTY_50
    
    def test_all_have_ns_suffix(self):
        assert all(s.endswith(".NS") for s in NIFTY_50)


class TestGetVixRegime:
    def test_fear_regime(self):
        vix_df = pd.DataFrame({"VIX": [20.0]}, index=["2024-01-01"])
        regime, vix = get_vix_regime(vix_df, {"vix_high": 18, "vix_low": 14})
        assert regime == "fear"
        assert vix == 20.0
    
    def test_complacency_regime(self):
        vix_df = pd.DataFrame({"VIX": [11.0]}, index=["2024-01-01"])
        regime, vix = get_vix_regime(vix_df, {"vix_high": 18, "vix_low": 14})
        assert regime == "complacency"
    
    def test_neutral_regime(self):
        vix_df = pd.DataFrame({"VIX": [16.0]}, index=["2024-01-01"])
        regime, vix = get_vix_regime(vix_df, {"vix_high": 18, "vix_low": 14})
        assert regime == "neutral"
    
    def test_empty_dataframe(self):
        vix_df = pd.DataFrame()
        regime, vix = get_vix_regime(vix_df, {"vix_high": 18, "vix_low": 14})
        assert regime == "neutral"
        assert vix == 14.0


class TestComputeMomentum:
    def test_returns_dict(self):
        # Need at least 50 rows for compute_momentum
        np.random.seed(42)
        n = 60
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        df = pd.DataFrame({
            "Close": close,
            "High": close + 0.5,
            "Low": close - 0.5,
        }, index=dates)
        result = compute_momentum(df)
        assert isinstance(result, dict)
        assert "price" in result
        assert "ma20" in result
        assert "adx" in result
    
    def test_insufficient_data(self):
        df = pd.DataFrame({
            "Close": [100] * 10,
            "High": [101] * 10,
            "Low": [99] * 10,
        })
        result = compute_momentum(df)
        assert result == {}


class TestComputeConviction:
    def test_fear_zone_scoring(self):
        mom = {
            "price": 100,
            "ma20": 95,
            "ma50": 90,
            "above_ma20": True,
            "above_ma50": True,
            "ma_cross": True,
            "adx": 30,
            "roc_20": 0.05,
        }
        score, reason = compute_conviction(mom, "fear", {"adx_threshold": 25})
        assert score > 0
        assert isinstance(reason, str)
    
    def test_complacency_zone_scoring(self):
        mom = {
            "price": 100,
            "ma20": 105,
            "ma50": 110,
            "above_ma20": False,
            "above_ma50": False,
            "ma_cross": False,
            "adx": 30,
            "roc_20": -0.05,
        }
        score, reason = compute_conviction(mom, "complacency", {"adx_threshold": 25})
        assert score > 0
    
    def test_empty_mom(self):
        score, reason = compute_conviction({}, "fear", {"adx_threshold": 25})
        assert score == 0.0
        assert reason == "Insufficient data"
