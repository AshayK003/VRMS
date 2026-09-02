"""Test corporate actions module."""
from __future__ import annotations

import pandas as pd
import pytest

from src.data.corporate_actions import (
    CORPORATE_ACTIONS,
    filter_corporate_action_days,
    is_corporate_action_day,
)


class TestCorporateActions:
    def test_tmcv_demerger_date(self):
        assert "2025-11-14" in CORPORATE_ACTIONS["TMCV"]["DEMERGER"]
    
    def test_tatamotors_demerger_date(self):
        assert "2025-11-14" in CORPORATE_ACTIONS["TATAMOTORS"]["DEMERGER"]
    
    def test_vedl_demerger_date(self):
        assert "2026-05-01" in CORPORATE_ACTIONS["VEDL"]["DEMERGER"]
    
    def test_heg_demerger_date(self):
        assert "2026-09-01" in CORPORATE_ACTIONS["HEG"]["DEMERGER"]
    
    def test_hdfc_bank_bonus(self):
        assert "2025-08-27" in CORPORATE_ACTIONS["HDFCBANK"]["BONUS"]
    
    def test_reliance_bonus(self):
        assert "2024-10-28" in CORPORATE_ACTIONS["RELIANCE"]["BONUS"]
    
    def test_bajaj_finance_split(self):
        assert "2025-06-25" in CORPORATE_ACTIONS["BAJFINANCE"]["SPLIT"]
    
    def test_kotak_bank_split(self):
        assert "2026-01-14" in CORPORATE_ACTIONS["KOTAKBANK"]["SPLIT"]


class TestIsCorporateActionDay:
    def test_demerger_day_flagged(self):
        is_action, reason = is_corporate_action_day("TMCV", "2025-11-14")
        assert is_action is True
        assert "DEMERGER" in reason
    
    def test_day_before_demerger_flagged(self):
        is_action, reason = is_corporate_action_day("TMCV", "2025-11-13")
        assert is_action is True
    
    def test_normal_day_not_flagged(self):
        is_action, reason = is_corporate_action_day("TMCV", "2025-06-15")
        assert is_action is False
    
    def test_unknown_symbol_not_flagged(self):
        is_action, reason = is_corporate_action_day("UNKNOWN", "2025-11-14")
        assert is_action is False


class TestFilterCorporateActionDays:
    def test_filters_demerger_day(self):
        dates = pd.date_range("2025-11-12", "2025-11-18", freq="B")
        df = pd.DataFrame({"Close": range(len(dates))}, index=dates)
        filtered = filter_corporate_action_days(df, "TMCV")
        # Nov 14 (demerger) and Nov 13 (day before) should be removed
        assert pd.Timestamp("2025-11-14") not in filtered.index
        assert pd.Timestamp("2025-11-13") not in filtered.index
    
    def test_keeps_normal_days(self):
        dates = pd.date_range("2025-06-01", "2025-06-10", freq="B")
        df = pd.DataFrame({"Close": range(len(dates))}, index=dates)
        filtered = filter_corporate_action_days(df, "TMCV")
        assert len(filtered) == len(df)
